#!/usr/bin/env python3
"""mcc_deck.py — the GPU-box side of the NAS Model Cassette Changer.

Your models live on a NAS. Your GPU box has fast local disk and limited memory. The deck treats every model on the NAS
like a cassette on a shelf:

  SHELF    the NAS index (library.json, built there by nas/mcc_index.py), copied here by `mcc_deck.py sync` every few
           minutes. The sync never wakes a sleeping NAS: while it is unreachable the last copy is shown with its age.
  TIERS    every runnable unit (a GGUF quant, or a whole safetensors folder) gets an estimated number of GPU nodes it needs.
  INSERT   copies the unit from the NAS to local disk (rsync, resumable) and starts it: GGUF -> Ollama, safetensors -> a vLLM or
           SGLang container, a model listed in recipes.json -> its own recipe lane (e.g. TensorFold's start.sh / stop.sh).
           Nothing is ever served straight off the network share.
  GUARD    an INSERT never stops anything you did not name. Protected models keep their memory reserved even when they are
           unloaded. If an insert would stop or evict anything, the server answers 409 with the exact list and does nothing
           until the request confirms every name. The plan is re-checked after the copy, and if a protected model was pushed
           out anyway the new cassette is removed again.
  EJECT    stops what the deck is playing; optionally deletes the local copy (separate confirm). The NAS copy is never touched.

Stdlib only (Python 3.9+). Talks to: ssh + rsync (to the NAS), the Ollama HTTP API, docker (vLLM / SGLang), recipe scripts, /proc/meminfo.

Usage:
  mcc_deck.py serve  [--config FILE] [--demo] [--port N] [--listen ADDR]
  mcc_deck.py sync   [--config FILE]          # what the timer runs
  mcc_deck.py summary [--config FILE]
  mcc_deck.py fit GB                          # how many nodes a GB-sized unit needs under your config
  mcc_deck.py archs vllm|sglang [--config FILE]   # read the architectures the engine image can load (for the ARCHITECTURE CHECK)
"""
import argparse, collections, glob, hmac, json, os, re, secrets, shlex, shutil, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERSION = "1.3.0"
HOME = os.path.expanduser("~")

DEFAULTS = {
    "nas_ssh": "nas",                                   # an ssh alias (BatchMode, key auth) for the NAS
    "nas_library_dir": "/srv/models-nas/library",       # the NAS library_dir (absolute path on the NAS)
    "local_dir": "~/models/deck",                       # where cassettes are copied to on this box
    "state_dir": "~/.local/state/mcc",
    "listen": "127.0.0.1", "port": 8099,
    "token_file": "~/.config/mcc/token",
    "node_name": "node-1",
    "fleet": [{"name": "node-1", "self": True}],        # more nodes: {"name": "node-2", "ssh": "user@host"}
    "fit": {"gpu_budget_gib": 95.69, "rank_overhead_gib": 5.6, "kv_floor_gib": 8.0, "weight_factor": 1.15, "max_nodes": 4},
    "protected": [],                                    # Ollama tags that must never be pushed out, e.g. ["qwen3:14b"]
    "servable_categories": ["LLMs", "Coding", "Small-On-Device"],
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_max_loaded": 3,                             # OLLAMA_MAX_LOADED_MODELS on this box
    "num_ctx": 8192,                                    # every deck GGUF is created and warm-loaded at this context
    "kv_band_gib": 4.0, "mem_floor_gib": 10.0,
    "big_lanes": [],                                    # systemd --user units that hold the GPU (stopped, by name, before a vLLM insert)
    "engines": ["vllm", "sglang"],                      # offered for safetensors models (when the model's runtime lists them)
    "vllm": {"image": "vllm/vllm-openai:latest", "port": 8010, "gpu_memory_utilization": 0.85, "max_model_len": 32768,
             "trust_remote_code": False, "extra_args": [], "container": "mcc-vllm"},
    "sglang": {"image": "lmsysorg/sglang:latest", "port": 30000, "mem_fraction_static": 0.85, "context_length": 32768,
               "trust_remote_code": False, "extra_args": [], "container": "mcc-sglang"},
    "recipes_file": "~/.config/mcc/recipes.json",       # recipe lanes (e.g. TensorFold): {library id: {label, start, stop, port, need_gib, …}}
    "new_days": 7,
    # FORM · FIT · FUNCTION (v1.2): the shelf is sectioned by what a model DOES (function), ordered inside each section by whether it runs
    # here now (fit), and every row shows what it physically is (form). Categories not listed land in "Other".
    "functions": [
        {"key": "chat", "label": "Chat & reasoning", "categories": ["LLMs"]},
        {"key": "coding", "label": "Coding", "categories": ["Coding"]},
        {"key": "ondevice", "label": "Small & on-device", "categories": ["Small-On-Device"]},
        {"key": "image", "label": "Image", "categories": ["Image"]},
        {"key": "video", "label": "Video", "categories": ["Video"]},
        {"key": "audio", "label": "Music & audio", "categories": ["Music-Audio"]},
        {"key": "speech", "label": "Speech", "categories": ["Speech-TTS", "Speech-ASR"]},
        {"key": "ocr", "label": "Documents & OCR", "categories": ["OCR"]},
        {"key": "embed", "label": "Embeddings & search", "categories": ["Embeddings-Reranking"]},
    ],
    "apps": {},                                         # function key -> {"label": "ComfyUI", "url": "http://127.0.0.1:8188"}: an OPEN button for models the deck does not serve
    # v1.3 ARCHITECTURE CHECK: per engine, a JSON file listing the architectures its image can load ({"image", "read", "architectures": [...]}),
    # written by `mcc_deck.py archs vllm` / `archs sglang`. A missing file = no check for that engine (the 1.2 behaviour).
    "engine_archs": {"vllm": "~/.config/mcc/vllm_archs.json", "sglang": "~/.config/mcc/sglang_archs.json"},
}
ENGINE_LABEL = {"vllm": "vLLM", "sglang": "SGLang"}
ARCH_PROBE = {   # prints one architecture per line from inside the engine's own image
    "vllm": ["python3", "-c", "from vllm import ModelRegistry\nfor a in sorted(ModelRegistry.get_supported_archs()): print(a)"],
    "sglang": ["python3", "-c", "from sglang.srt.models.registry import ModelRegistry\nfor a in sorted(ModelRegistry.models): print(a)"]}
RECIPE_DEFAULTS = {"nodes": 1, "health_path": "/health", "requires_ollama_stopped": False, "start_timeout_s": 1800, "stop_timeout_s": 180,
                   "copy": True}                      # copy: false = the recipe keeps its own weights (e.g. TensorFold's HF cache); INSERT only starts it
RUNNING = {"active", "activating", "reloading"}
QUANT_RX = re.compile(r"(UD-)?(IQ\d_[A-Z]+|Q\d_K_[A-Z]+|Q\d_K|Q\d_\d|Q\d_[A-Z]+|BF16|F16|F32|FP8|MXFP4|NVFP4)", re.I)
SHARD_RX = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path and os.path.exists(path):
        with open(path) as f: user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict): cfg[k].update(v)
            else: cfg[k] = v
    for k in ("local_dir", "state_dir", "token_file", "recipes_file"): cfg[k] = os.path.expanduser(cfg[k])
    cfg["engine_archs"] = {e: os.path.expanduser(p) for e, p in (cfg.get("engine_archs") or {}).items() if p}
    return cfg


def load_recipes(path):
    """recipes.json -> {library id: recipe}. Keys starting with '_' are comments. A missing file = no recipe lanes."""
    raw = _rjson(path, {}) if path else {}
    out = {}
    for k, v in (raw.items() if isinstance(raw, dict) else []):
        if k.startswith("_") or not isinstance(v, dict) or not v.get("start") or not v.get("port") or not v.get("need_gib"): continue
        out[k] = {**RECIPE_DEFAULTS, "label": v.get("engine") or "recipe", **v, "key": k}
    return out


def _rjson(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default


def _wjson(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f: json.dump(obj, f)
    os.replace(tmp, path)


def _sh(cmd, timeout=20):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception: return ""


def _slug(s): return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


# ── pure logic (unit-tested) ───────────────────────────────────────────────────────────────────────────────────────
def nodes_needed(nbytes, fit):
    """smallest k with max(W + overhead·k + kv_floor, factor·W) <= budget·k, W = weights in GiB. An estimate and a floor."""
    w = nbytes / 2**30
    for k in range(1, int(fit["max_nodes"]) + 1):
        if max(w + fit["rank_overhead_gib"] * k + fit["kv_floor_gib"], fit["weight_factor"] * w) <= fit["gpu_budget_gib"] * k: return k
    return None


def quant_of(name):
    m = QUANT_RX.search(os.path.basename(name)) or QUANT_RX.search(name)
    return (m.group(0).upper() if m else os.path.basename(name).rsplit(".", 1)[0])[:28]


def build_units(listing):
    """[(bytes, 'Maker/Model/Variant/…/file.gguf')] -> {id: [unit…]}: shards grouped, the smallest mmproj attached to every unit"""
    by_id, mmproj = {}, {}
    for size, rel in listing:
        parts = rel.split("/")
        if len(parts) < 4: continue
        mid, inner = "/".join(parts[:3]), "/".join(parts[3:])
        if "mmproj" in os.path.basename(inner).lower(): mmproj.setdefault(mid, []).append((size, inner)); continue
        m = SHARD_RX.match(inner); key = m.group(1) if m else inner[:-5]
        u = by_id.setdefault(mid, {}).setdefault(key, {"key": key, "quant": quant_of(key), "files": [], "bytes": 0})
        u["files"].append(inner); u["bytes"] += size
    out = {}
    for mid, units in by_id.items():
        mp = sorted(mmproj.get(mid, []))[:1]
        for u in units.values():
            u["files"].sort()
            if mp: u["mmproj"] = mp[0][1]; u["bytes"] += mp[0][0]
        out[mid] = sorted(units.values(), key=lambda u: u["bytes"])
    return out


def _engines_busy(res, but=None):
    return [e["name"] for e in (res.get("engines_running") or []) if e.get("name") != but]


def plan_recipe(m, unit, res, cfg, recipe, busy=False):
    """A recipe lane (e.g. TensorFold) runs ALONE on the GPU. It states its own fit (need_gib, nodes); the deck checks it, names everything
    it would stop, and never stops a system service: a recipe that needs Ollama stopped is refused while Ollama answers."""
    label = recipe.get("label") or "recipe"
    if m.get("status") != "complete": return {"refuse": f"not complete on the NAS ({m.get('status')})"}
    if int(recipe.get("nodes") or 1) > 1: return {"refuse": f"the {label} recipe needs {recipe['nodes']} nodes — launch it from its kit"}
    if busy: return {"refuse": "the deck is busy (an insert is running)"}
    if recipe.get("requires_ollama_stopped") and res.get("ollama_up"):
        return {"refuse": f"Ollama is running and the {label} recipe needs the whole GPU — stop Ollama first (e.g. sudo systemctl stop ollama); "
                          "the deck never stops system services"}
    other = _engines_busy(res, but=label)
    if other: return {"refuse": f"{', '.join(other)} is playing — EJECT it first"}
    w = unit["bytes"] / 2**30
    if recipe.get("copy", True) and res.get("free_disk_gib") is not None and res["free_disk_gib"] < w * 1.1:
        return {"refuse": f"not enough disk: needs ~{w*1.1:.0f} GiB free, {res['free_disk_gib']:.0f} GiB free"}
    resident = {o["name"]: o["gib"] for o in res["ollama"]} if res.get("ollama_up") else {}
    ev = list(res["lanes"]) + list(resident)
    free = res["mem_avail_gib"] + sum(resident.values())          # big lanes' memory is not counted: a conservative (fail-closed) forecast
    need = float(recipe["need_gib"])
    if free < need:
        return {"refuse": f"the {label} recipe needs {need:.0f} GiB; about {free:.0f} GiB would be free" + (" after stopping " + ", ".join(ev) if ev else "")}
    return {"path": "recipe", "would_evict": ev, "need_gib": need, "free_gib": round(free, 1), "note": f"the {label} recipe runs alone on the GPU"}


def plan_insert(m, unit, res, cfg, busy=False, recipe=None):
    """What an INSERT of this unit would do, given a residents snapshot. Never acts. Fails closed."""
    if recipe: return plan_recipe(m, unit, res, cfg, recipe, busy)
    prot = cfg["protected"]; cat = m.get("category"); fmt = (m.get("format") or "").lower(); k = unit.get("nodes")
    rt = [str(x).lower() for x in (m.get("runtime") or [])]; eng = unit.get("engine") or "vllm"
    if cat not in cfg["servable_categories"]: return {"refuse": f"{cat} model — the deck serves text models (Ollama / vLLM / SGLang); run it in its own app"}
    if fmt not in ("gguf", "safetensors"): return {"refuse": f"format {fmt or '?'} — the deck serves gguf (Ollama) and safetensors (vLLM / SGLang), or a recipe lane"}
    if fmt == "gguf" and not ({"ollama", "llama.cpp"} & set(rt)): return {"refuse": f"runtime {', '.join(rt) or '?'} — not an Ollama model"}
    if fmt == "safetensors" and eng not in rt: return {"refuse": f"runtime {', '.join(rt) or '?'} — not a {ENGINE_LABEL.get(eng, eng)} model"}
    if fmt == "safetensors" and eng not in cfg["engines"]: return {"refuse": f"{ENGINE_LABEL.get(eng, eng)} is not enabled in 'engines'"}
    if fmt == "safetensors":
        aw = arch_refusal(m, eng, (cfg.get("_archs") or {}).get(eng) if "_archs" in cfg else read_archs((cfg.get("engine_archs") or {}).get(eng)))
        if aw: return {"refuse": aw}
    if m.get("status") != "complete": return {"refuse": f"not complete on the NAS ({m.get('status')})"}
    if k is None: return {"refuse": f"beyond the fleet (needs more than {cfg['fit']['max_nodes']} nodes by the estimate)"}
    if k > 1: return {"refuse": f"needs {k} nodes — multi-node serving is launched from its kit, not from the deck"}
    if busy: return {"refuse": "the deck is busy (an insert is running)"}
    up = res.get("ollama_up", res.get("ollama_ok"))
    if fmt == "gguf" and not up: return {"refuse": "Ollama is not running — GGUF cassettes play in Ollama (start it, or pick a vLLM / SGLang model)"}
    if up and not res.get("ollama_ok"): return {"refuse": "cannot read Ollama right now — refusing rather than guessing what would be evicted"}
    missing = [p for p in prot if p not in res["tags_gib"]] if up else []      # Ollama stopped: nothing of it is resident, nothing to protect
    if missing: return {"refuse": f"protected model(s) {', '.join(missing)} not found in Ollama — fix 'protected' in the config"}
    w = unit["bytes"] / 2**30; disk_need = w * (2.2 if fmt == "gguf" else 1.1)
    if res.get("free_disk_gib") is not None and res["free_disk_gib"] < disk_need:
        return {"refuse": f"not enough disk: needs ~{disk_need:.0f} GiB free, {res['free_disk_gib']:.0f} GiB free"}
    resident = {o["name"]: o["gib"] for o in res["ollama"]}
    if fmt == "safetensors":
        busy_e = _engines_busy(res)
        if busy_e: return {"refuse": f"{', '.join(busy_e)} is already playing — EJECT it first"}
        ev = list(res["lanes"]) + list(resident)
        if up: ev += [p for p in prot if p not in ev]        # the engine takes most of GPU memory: protected models cannot run beside it
        return {"path": eng, "would_evict": ev, "note": f"a {ENGINE_LABEL.get(eng, eng)} cassette takes the GPU: every big lane and every Ollama model stops"}
    need = w * 1.2 + 3 + cfg["kv_band_gib"]
    obs = res.get("observed_gib") or {}
    reserve = sum(max(obs.get(p, 0), res["tags_gib"].get(p, 0) * 1.3 + 1.5) for p in prot if p not in resident)
    free = res["mem_avail_gib"] - cfg["mem_floor_gib"] - reserve
    droppable = sorted(n for n in resident if n.startswith("deck/")) + \
        sorted((n for n in resident if n not in prot and not n.startswith("deck/")), key=lambda n: -resident[n])
    ev = []; over = len(resident) + 1 - res.get("max_loaded", cfg["ollama_max_loaded"])
    if over > len(droppable): return {"refuse": f"Ollama holds {len(resident)} of {res.get('max_loaded')} models and only protected ones could make room"}
    ev += droppable[:max(over, 0)]
    got = free + sum(resident[n] for n in ev)
    for n in droppable:
        if got >= need: break
        if n not in ev: ev.append(n); got += resident[n]
    if got < need:
        return {"refuse": f"does not fit: needs ~{need:.0f} GiB at {cfg['num_ctx']} ctx, ~{max(free, 0):.0f} GiB free after the protected reserve"}
    return {"path": "ollama", "would_evict": ev, "need_gib": round(need, 1), "free_gib": round(free, 1)}


def arch_refusal(m, engine, known):
    """ARCHITECTURE CHECK (v1.3), before anything is copied or stopped. known = the architectures the engine's image can load (None = no
    list for this engine: no check). m["architectures"] comes from the NAS index: absent = an index older than 1.3 (no check);
    None = no config.json at the model's root; "ERR" = unreadable; [] = declares none. All but "absent" fail closed."""
    if not known or "architectures" not in m: return None
    a, lab = m["architectures"], ENGINE_LABEL.get(engine, engine)
    if a is None: return f"no config.json at the model's root — {lab} needs one; refusing rather than copying it to find out"
    if a == "ERR": return "its config.json could not be read on the NAS — refusing rather than guessing"
    if not a: return f"its config.json declares no architecture — refusing rather than copying it to find out"
    if not set(a) & set(known): return f"architecture {', '.join(a)} is not in this {lab} image — refused before anything is copied or stopped"
    return None


def read_archs(path):
    """{"architectures": [...]} file -> set, or None when there is no usable list (= no check)"""
    d = _rjson(path, None) if path else None
    a = (d or {}).get("architectures") if isinstance(d, dict) else None
    return set(a) if isinstance(a, list) and a else None


SELECTOR_ORDER = ["Ollama", "vLLM", "SGLang", "EXL3", "Apps"]      # recipe lanes (by their label) go before EXL3


def engines_of(row):
    """the engines that can play a shelf row — what the ENGINE selector filters on (a safetensors model may list vLLM and SGLang)"""
    if row.get("recipe"): return [row["recipe"].get("label") or "recipe"]
    if (row.get("fit") or {}).get("class") == "app" or not row.get("servable"): return ["Apps"]
    if str(row.get("quant") or "").upper().startswith("EXL") or {"exllama", "tabbyapi"} & {str(x).lower() for x in row.get("runtime") or []}: return ["EXL3"]
    if (row.get("format") or "").lower() == "gguf": return ["Ollama"]
    e = [ENGINE_LABEL.get(u.get("engine"), u.get("engine")) for u in row.get("units") or [] if u.get("engine")]
    return list(dict.fromkeys(e)) or ["vLLM"]


def engine_bar(rows, res, playing):
    """the ENGINE selector: one entry per engine that has models, with a count and a state dot.
    state: serving (it is playing now) · stopped (Ollama not running) · idle · unwired (listed, the deck cannot play it)"""
    n = {}
    for r in rows:
        for e in engines_of(r): n[e] = n.get(e, 0) + 1
    run = {e.get("name") for e in res.get("engines_running") or []}
    deck_oll = any(o["name"].startswith("deck/") for o in res.get("ollama") or [])
    recipes = [e for e in n if e not in SELECTOR_ORDER]
    out = []
    for e in SELECTOR_ORDER[:3] + sorted(recipes) + SELECTOR_ORDER[3:]:
        if not n.get(e): continue
        if e == "Ollama":
            st = ("stopped", "Ollama is not running") if not res.get("ollama_up", True) else \
                 ("serving", "models loaded") if res.get("ollama") else ("idle", "running, nothing loaded")
            if deck_oll: st = ("serving", "a deck cassette is playing in Ollama")
        elif e == "EXL3": st = ("unwired", "listed so you can see them — the deck does not play EXL3 yet (it needs an exllamav3 / TabbyAPI player)")
        elif e == "Apps": st = ("unwired", "image, video, speech … models run in their own apps")
        else: st = ("serving", "playing now") if e in run else ("idle", "idle")
        out.append({"key": e, "n": n[e], "state": st[0], "why": st[1]})
    return out


def last_fail(history, pull_log_tail=""):
    """the newest failed INSERT that no later good INSERT superseded -> {id, unit, ts, rc, why}; else None"""
    for a in reversed(history or []):
        if a.get("action") != "INSERT": continue
        if a.get("rc") in (0, None): return None
        return {"id": a.get("id"), "unit": a.get("unit"), "ts": a.get("ts"), "rc": a.get("rc"),
                "why": (a.get("note") or pull_log_tail or "see the deck log")[:300]}
    return None


# ── BENCHMARKS (v1.3): live tok/s from the engine's own counters + on-demand Decode / Prefill runs ─────────────────────────────────
PROM = {"gen": ("vllm:generation_tokens_total", "sglang:generation_tokens_total"),
        "prompt": ("vllm:prompt_tokens_total", "sglang:prompt_tokens_total"),
        "running": ("vllm:num_requests_running", "sglang:num_running_reqs"),
        "waiting": ("vllm:num_requests_waiting", "sglang:num_queue_reqs")}


def parse_prom(text):
    """Prometheus text -> {gen, prompt, running, waiting} (each summed over its label sets; a missing metric stays absent)"""
    out = {}
    for line in (text or "").splitlines():
        if not line or line[0] == "#": continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for k, names in PROM.items():
            if name in names:
                try: out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError: pass
    return out


def rates(prev, cur):
    """two (t, counters) samples -> (decode tok/s, prefill tok/s); None when they cannot be compared (first sample, counter reset)"""
    if not prev or not cur: return None
    dt = cur[0] - prev[0]; a, b = prev[1], cur[1]
    if dt <= 0 or "gen" not in a or "gen" not in b: return None
    dg, dp = b["gen"] - a["gen"], b.get("prompt", 0) - a.get("prompt", 0)
    if dg < 0 or dp < 0: return None                                  # the engine restarted
    return round(dg / dt, 1), round(dp / dt, 1)


def bump_peak(peaks, day, kind, v, keep=30):
    """daily peak tok/s per kind ({day: {decode, prefill}}), the last `keep` days"""
    if v is None or v <= 0: return peaks
    d = peaks.setdefault(day, {}); d[kind] = max(d.get(kind, 0), round(v, 1))
    for k in sorted(peaks)[:-keep]: peaks.pop(k)
    return peaks


BENCH_DECODE = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Keep going."
BENCH_FILL = "The deck copies a model from the shelf, starts it, and measures how fast it reads and writes. "   # ≈ 20 tokens


def bench_prompt(kind, nonce):
    """decode: a short prompt, 256 tokens out · prefill: ≈ 2,000 tokens in, 1 out. The nonce defeats prefix / prompt caches."""
    return f"[{nonce}] " + (BENCH_DECODE if kind == "decode" else BENCH_FILL * 100 + "\nSay OK.")


FIT_ORDER = ["ready", "switch", "blocked", "nodes", "app", "pending"]
FIT_LABEL = {"ready": "Ready now", "switch": "After a switch", "blocked": "Blocked right now", "nodes": "Needs more nodes",
             "app": "Runs in its own app", "pending": "Not on the shelf yet"}


def function_of(category, cfg):
    """-> (key, label) of the function section a NAS category belongs to; unknown categories land in Other."""
    for f in cfg.get("functions") or []:
        if category in (f.get("categories") or []): return f["key"], f.get("label") or f["key"]
    return "other", "Other"


def fit_of(m, units, plans, cfg):
    """FIT = can it run here, now? One class per row, from the plans the guard already made (never a new decision):
    ready (a unit fits beside what is running) · switch (a unit fits if the named models stop — fewest stops wins) · blocked (single-node
    but refused right now; the reason is the guard's) · nodes (needs more nodes than one) · app (a category the deck does not serve) ·
    pending (not complete on the NAS)."""
    if m.get("status") != "complete": return {"class": "pending", "why": f"not complete on the NAS ({m.get('status') or '?'})"}
    fk, _ = function_of(m.get("category"), cfg)
    if m.get("category") not in cfg["servable_categories"]:
        app = (cfg.get("apps") or {}).get(fk)
        return {"class": "app", "why": f"runs in {app.get('label')}" if app else "runs in its own app — the deck serves text models", **({"app": app} if app else {})}
    ok = [u for u in units if not (plans.get(u["key"]) or {}).get("refuse")]
    ready = [u for u in ok if not plans[u["key"]].get("would_evict")]
    if ready: return {"class": "ready", "unit": ready[0]["key"], "why": "fits beside what is running"}
    if ok:
        u = min(ok, key=lambda x: len(plans[x["key"]]["would_evict"]))
        return {"class": "switch", "unit": u["key"], "stops": plans[u["key"]]["would_evict"], "why": "stops " + ", ".join(plans[u["key"]]["would_evict"])}
    one = [u for u in units if u.get("nodes") == 1]
    if not one:
        ks = [u["nodes"] for u in units if u.get("nodes")]
        return {"class": "nodes", "why": f"needs {min(ks)} nodes" if ks else f"beyond the fleet (more than {cfg['fit']['max_nodes']} nodes)"}
    return {"class": "blocked", "unit": one[0]["key"], "why": (plans.get(one[0]["key"]) or {}).get("refuse") or "refused"}


# ── the machine: everything that touches the box, the NAS or a runtime ─────────────────────────────────────────────
class System:
    def __init__(self, cfg): self.cfg = cfg; self.lim = {"t": 0, "max": cfg["ollama_max_loaded"]}

    def ssh_nas(self, cmd, timeout):
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.cfg["nas_ssh"], cmd], capture_output=True, text=True, timeout=timeout)
            return r.stdout if r.returncode == 0 else None
        except Exception: return None

    def ollama(self, path, body=None, timeout=4):
        try:
            req = urllib.request.Request(self.cfg["ollama_url"] + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception: return None

    def engine_metrics(self, port, path="/metrics"):
        try: return urllib.request.urlopen(f"http://127.0.0.1:{int(port)}{path}", timeout=3).read().decode("utf-8", "replace")
        except Exception: return None

    def bench_openai(self, port, model, kind):
        """one timed /v1/completions call (streamed) -> {tok_s, tokens, secs}. decode = output tokens after the first one / time after it;
        prefill = prompt tokens / time to the first token (includes HTTP and queueing, so it reads a little low)."""
        body = {"model": model, "prompt": bench_prompt(kind, secrets.token_hex(4)), "stream": True, "temperature": 0,
                "max_tokens": 256 if kind == "decode" else 1, "ignore_eos": kind == "decode", "stream_options": {"include_usage": True}}
        req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time(); t1 = t2 = None; usage = {}
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                if any((c.get("text") or "") for c in d.get("choices") or []): t2 = time.time(); t1 = t1 or t2
        if kind == "decode":
            n = int(usage.get("completion_tokens") or 0)
            if n < 2 or not t1 or t2 <= t1: raise RuntimeError("the engine returned too few tokens to time")
            return {"tok_s": round((n - 1) / (t2 - t1), 1), "tokens": n, "secs": round(t2 - t0, 2)}
        n = int(usage.get("prompt_tokens") or 0)
        if not n or not t1: raise RuntimeError("the engine did not report prompt tokens")
        return {"tok_s": round(n / (t1 - t0), 1), "tokens": n, "secs": round(t1 - t0, 2)}

    def bench_ollama(self, name, kind):
        """Ollama times itself: eval_count / eval_duration (decode), prompt_eval_count / prompt_eval_duration (prefill)"""
        r = self.ollama("/api/generate", {"model": name, "prompt": bench_prompt(kind, secrets.token_hex(4)), "stream": False,
                                          "options": {"num_predict": 256 if kind == "decode" else 1, "temperature": 0, "num_ctx": self.cfg["num_ctx"]}}, timeout=600)
        if not r: raise RuntimeError("Ollama did not answer")
        c, d = (r.get("eval_count"), r.get("eval_duration")) if kind == "decode" else (r.get("prompt_eval_count"), r.get("prompt_eval_duration"))
        if not c or not d: raise RuntimeError("Ollama returned no timings")
        return {"tok_s": round(c / (d / 1e9), 1), "tokens": c, "secs": round((r.get("total_duration") or 0) / 1e9, 2)}

    def meminfo(self):
        mem = {}
        try:
            with open("/proc/meminfo") as f:
                for l in f: k, v = l.split(":"); mem[k] = int(v.split()[0]) / 1048576
        except Exception: pass
        return mem

    def container_up(self, name): return bool(_sh(["docker", "ps", "-q", "-f", f"name=^{name}$"], 8))

    def recipe_up(self, r):
        try: urllib.request.urlopen(f"http://127.0.0.1:{int(r['port'])}{r.get('health_path') or '/health'}", timeout=2).read(); return True
        except Exception: return False

    def engines_running(self, recipes):
        out = [{"engine": e, "name": ENGINE_LABEL[e]} for e in ("vllm", "sglang") if self.container_up(self.cfg[e]["container"])]
        out += [{"engine": r.get("engine") or "recipe", "name": r.get("label") or "recipe", "recipe": k} for k, r in recipes.items() if self.recipe_up(r)]
        return out

    def residents(self, state_dir, recipes=None):
        units = self.cfg["big_lanes"]
        states = (_sh(["systemctl", "--user", "is-active"] + list(units), 8) or "").splitlines() if units else []
        lanes = [u for u, st in zip(units, states) if st.strip() in RUNNING]
        ps, tg = self.ollama("/api/ps"), self.ollama("/api/tags")
        oll = [{"name": x.get("name"), "gib": round((x.get("size") or 0) / 2**30, 1)} for x in (ps or {}).get("models", [])]
        tags = {x.get("name"): round((x.get("size") or 0) / 2**30, 1) for x in (tg or {}).get("models", [])}
        obs_f = f"{state_dir}/observed.json"; obs = _rjson(obs_f, {}); changed = False
        for o in oll:
            if o["name"] in self.cfg["protected"] and o["gib"] > obs.get(o["name"], 0): obs[o["name"]] = o["gib"]; changed = True
        if changed: _wjson(obs_f, obs)
        mem = self.meminfo(); os.makedirs(self.cfg["local_dir"], exist_ok=True)
        return {"lanes": lanes, "ollama": oll, "tags_gib": tags, "ollama_ok": ps is not None and tg is not None, "ollama_up": ps is not None,
                "observed_gib": obs, "max_loaded": self.cfg["ollama_max_loaded"], "free_disk_gib": round(shutil.disk_usage(self.cfg["local_dir"]).free / 2**30, 1),
                "mem_avail_gib": round(mem.get("MemAvailable", 0), 1), "mem_total_gib": round(mem.get("MemTotal", 0), 1),
                "engines_running": self.engines_running(recipes or {})}

    def remote(self, node):
        r = _sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", node["ssh"], "awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo"], 6)
        return {"online": r.isdigit(), "mem_avail_gib": int(r) if r.isdigit() else None, "why": "" if r.isdigit() else "ssh unreachable"}

    def pull(self, mid, unit, f, job, state_dir):
        src = f"{self.cfg['nas_ssh']}:{self.cfg['nas_library_dir'].rstrip('/')}/{mid}/"; dst = f"{self.cfg['local_dir']}/{mid}/"
        os.makedirs(dst, exist_ok=True)
        cmd = ["rsync", "-rLt", "-s", "--partial", "--info=progress2", "--exclude", ".git"]
        files = list(unit.get("files") or []) + ([unit["mmproj"]] if unit.get("mmproj") else [])
        if files:
            lst = f"{state_dir}/pull_files.txt"
            with open(lst, "w") as lf: lf.write("\n".join(files) + "\n")
            cmd += [f"--files-from={lst}"]
        cmd += [src, dst]; f.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n"); f.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if "%" in line: job["progress"] = line.strip()[-80:]
            f.write(line); f.flush()
        p.wait()
        if p.returncode == 0:
            marker = f"{dst}.deck-complete-{_slug(unit['key'])}" if files else f"{dst}.deck-complete"
            with open(marker, "w") as mf: mf.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return p.returncode

    def unload(self, name):
        return self.ollama("/api/generate", {"model": name, "keep_alive": 0}, timeout=60) is not None

    def serve_gguf(self, mid, unit, f):
        path = f"{self.cfg['local_dir']}/{mid}"; files = sorted(unit.get("files") or [])
        first = f"{path}/{files[0]}" if files else ((sorted(glob.glob(f"{path}/**/*.gguf", recursive=True), key=os.path.getsize) or [None])[-1])
        if not first or not os.path.exists(first): f.write("[serve] no .gguf file in the cassette\n"); return 1, None
        name = "deck/" + (mid + ("_" + unit["quant"] if files else "")).replace("/", "_").lower()[:120]; mf = f"{path}/Modelfile.deck"
        with open(mf, "w") as m_: m_.write(f"FROM {first}\nPARAMETER num_ctx {self.cfg['num_ctx']}\n")
        with open(f"{path}/.deck-ollama-names", "a") as nf: nf.write(name + "\n")
        f.write(f"$ ollama create {name} -f {mf}\n"); f.flush()
        r = subprocess.run(["ollama", "create", name, "-f", mf], stdout=f, stderr=subprocess.STDOUT, text=True)
        if r.returncode: return r.returncode, None
        f.write("$ warm load (keep_alive 30m)\n"); f.flush()
        ok = self.ollama("/api/generate", {"model": name, "prompt": "hi", "keep_alive": "30m", "stream": False, "options": {"num_ctx": self.cfg["num_ctx"]}}, timeout=900)
        if ok is None: f.write("[warm load failed]\n"); return 1, None
        return 0, {"runtime": "ollama", "name": name}

    def make_room(self, f, evict):
        """stop exactly the confirmed names: big lanes (systemctl --user stop) and Ollama residents (unload)"""
        for u in self.cfg["big_lanes"]:
            if u in evict: f.write(f"$ systemctl --user stop {u}\n"); _sh(["systemctl", "--user", "stop", u], 180)
        for o in (self.ollama("/api/ps") or {}).get("models", []):
            if o.get("name") in evict: self.unload(o["name"]); f.write(f"[unloaded {o['name']}]\n")

    def engine_cmd(self, engine, mid):
        served = mid.split("/")[1]; vol = ["-v", f"{self.cfg['local_dir']}:/models:ro"]
        base = ["docker", "run", "-d", "--rm", "--gpus", "all", "--ipc=host", "--network", "host"]
        if engine == "vllm":
            v = self.cfg["vllm"]
            cmd = base + ["--name", v["container"]] + vol + [v["image"], "--model", f"/models/{mid}", "--served-model-name", served,
                   "--port", str(v["port"]), "--max-model-len", str(v["max_model_len"]), "--gpu-memory-utilization", str(v["gpu_memory_utilization"])]
        else:
            v = self.cfg["sglang"]
            cmd = base + ["--shm-size", "16g", "--name", v["container"]] + vol + [v["image"], "python3", "-m", "sglang.launch_server",
                   "--model-path", f"/models/{mid}", "--served-model-name", served, "--host", "127.0.0.1", "--port", str(v["port"]),
                   "--context-length", str(v["context_length"]), "--mem-fraction-static", str(v["mem_fraction_static"])]
        return cmd + (["--trust-remote-code"] if v.get("trust_remote_code") else []) + list(v.get("extra_args") or []), v, served

    def serve_engine(self, engine, mid, f, evict):
        """vLLM or SGLang in a container; both answer the OpenAI /v1/models route when ready"""
        self.make_room(f, evict); cmd, v, served = self.engine_cmd(engine, mid); c = v["container"]; L = ENGINE_LABEL[engine]
        _sh(["docker", "rm", "-f", c], 60); f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n"); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if r.returncode: return r.returncode, None
        for i in range(360):
            time.sleep(5)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{v['port']}/v1/models", timeout=3).read()
                f.write(f"[{L} {served} answering on :{v['port']} after {5*(i+1)} s]\n"); return 0, {"runtime": engine, "port": v["port"], "name": served}
            except Exception: pass
            if not self.container_up(c) and i > 2: f.write(f"[the {c} container exited — docker logs {c}]\n"); return 1, None
            if i % 12 == 11: f.write(f"  … waiting for /v1/models ({5*(i+1)} s)\n"); f.flush()
        f.write(f"[timed out waiting for {L}]\n"); return 1, None

    def recipe_env(self, r, mid):
        env = {**os.environ, "MCC_MODEL_DIR": f"{self.cfg['local_dir']}/{mid}", "MCC_MODEL_ID": mid, "MCC_PORT": str(r["port"])}
        env.update({k: str(v).replace("{model_dir}", env["MCC_MODEL_DIR"]) for k, v in (r.get("env") or {}).items()})
        return env

    def serve_recipe(self, r, mid, f, evict):
        """run the recipe's own start command (detached), then wait for its health URL; its stop command is EJECT"""
        self.make_room(f, evict); label = r.get("label") or "recipe"; cmd = r["start"]
        argv = ["bash", "-lc", cmd] if isinstance(cmd, str) else list(cmd)
        f.write(f"$ {cmd if isinstance(cmd, str) else ' '.join(shlex.quote(x) for x in cmd)}   ({label}, MCC_MODEL_DIR={self.cfg['local_dir']}/{mid})\n"); f.flush()
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, env=self.recipe_env(r, mid), start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < float(r.get("start_timeout_s") or 1800):
            time.sleep(5)
            if self.recipe_up(r):
                f.write(f"[{label} answering on :{r['port']} after {int(time.time()-t0)} s]\n")
                return 0, {"runtime": r.get("engine") or "recipe", "port": r["port"], "name": r.get("served") or mid.split("/")[1], "label": label, "recipe": r["key"]}
            if p.poll() not in (None, 0): f.write(f"[the {label} start command exited rc={p.returncode}]\n"); return p.returncode or 1, None
            if int(time.time() - t0) % 60 < 5: f.write(f"  … waiting for {label} on :{r['port']}{r.get('health_path')} ({int(time.time()-t0)} s)\n"); f.flush()
        f.write(f"[timed out waiting for {label}]\n"); return 1, None

    def stop_engines(self, recipes):
        done = []
        for e in ("vllm", "sglang"):
            c = self.cfg[e]["container"]
            if self.container_up(c): _sh(["docker", "rm", "-f", c], 120); done.append(f"stopped the {ENGINE_LABEL[e]} cassette")
        for k, r in recipes.items():
            if not self.recipe_up(r): continue
            cmd = r.get("stop")
            if not cmd: done.append(f"{r.get('label')} is running but its recipe has no 'stop' command — stop it yourself"); continue
            argv = ["bash", "-lc", cmd] if isinstance(cmd, str) else list(cmd)
            try: subprocess.run(argv, capture_output=True, timeout=float(r.get("stop_timeout_s") or 180), env=self.recipe_env(r, k))
            except Exception as e: done.append(f"{r.get('label')} stop command failed: {type(e).__name__}"); continue
            done.append(f"stopped the {r.get('label')} recipe" if not self.recipe_up(r) else f"{r.get('label')} still answers after its stop command — check it")
        return done

    def ollama_rm(self, name): _sh(["ollama", "rm", name], 60)


# ── the deck ───────────────────────────────────────────────────────────────────────────────────────────────────────
class Deck:
    def __init__(self, cfg, system=None):
        self.cfg = cfg; self.sys = system or System(cfg); s = cfg["state_dir"]
        for d in (s, cfg["local_dir"]): os.makedirs(d, exist_ok=True)
        self.LIB, self.UNITS, self.SEEN, self.NAS = f"{s}/library.json", f"{s}/units.json", f"{s}/seen.json", f"{s}/nas_status.json"
        self.ACTIONS, self.PULL_LOG, self.PLAYING = f"{s}/actions.jsonl", f"{s}/pull.log", f"{s}/playing.json"
        self.lock = threading.Lock(); self.job = {"running": False, "what": "", "t0": 0, "progress": ""}; self.syncing = False
        self.BENCH, self.PEAKS = f"{s}/bench.jsonl", f"{s}/peaks.json"
        self.live = collections.deque(maxlen=360); self.prev = None; self.served = {}      # 30 min of 5 s samples, per target
        self.bjob = {"running": False, "kind": "", "msg": ""}

    # sync ----------------------------------------------------------------------------------------------------------
    def sync(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%S"); st = _rjson(self.NAS, {}); lib = self.cfg["nas_library_dir"].rstrip("/")
        raw = self.sys.ssh_nas(f"cat {shlex.quote(lib + '/_index/library.json')}", 15)
        if raw is None:
            st.update(awake=False, checked=now); st["asleep_since"] = st.get("asleep_since") or now; _wjson(self.NAS, st)
            return {"ok": False, "nas": "asleep or unreachable", "since": st["asleep_since"]}
        try: d = json.loads(raw)
        except Exception: d = None
        if not isinstance(d, dict) or not isinstance(d.get("models"), list):
            st.update(awake=True, checked=now, bad_index=now, asleep_since=None); _wjson(self.NAS, st)
            return {"ok": False, "nas": "awake, index unreadable — cache kept"}
        lst = self.sys.ssh_nas(f"cd {shlex.quote(lib)} && find -L . -path ./_tiers -prune -o -path ./_kits -prune -o -name '*.gguf' -printf '%s\\t%P\\n' 2>/dev/null; true", 60)
        listing = []
        for line in (lst or "").splitlines():
            a, _, b = line.partition("\t")
            if a.isdigit() and b: listing.append((int(a), b))
        seen = _rjson(self.SEEN, {}); first = not seen
        for m in d["models"]: seen.setdefault(m.get("id"), "2000-01-01" if first else now[:10])   # the first sync marks nothing NEW
        _wjson(self.LIB, d); _wjson(self.SEEN, seen)
        if lst is not None: _wjson(self.UNITS, {"built": d.get("built"), "listed": now, "units": build_units(listing)})
        st.update(awake=True, checked=now, last_ok=now, asleep_since=None); st.pop("bad_index", None); _wjson(self.NAS, st)
        return {"ok": True, "models": len(d["models"]), "gguf_files": len(listing), "built": d.get("built")}

    def sync_bg(self):
        with self.lock:
            if self.syncing: return
            self.syncing = True
        def run():
            try: self.sync()
            finally: self.syncing = False
        threading.Thread(target=run, daemon=True).start()

    # state ---------------------------------------------------------------------------------------------------------
    def models(self):
        d = _rjson(self.LIB, {"models": []}); m = d.get("models") if isinstance(d, dict) else d
        return [x for x in (m or []) if isinstance(x, dict)], d

    def recipes(self):
        return self.cfg["_recipes"] if "_recipes" in self.cfg else load_recipes(self.cfg["recipes_file"])

    def units_for(self, m, units_idx, recipes=None):
        """GGUF: one unit per quant file. Safetensors: the whole folder once per enabled engine (vLLM / SGLang).
        A model listed in recipes.json: ONE unit, played by its recipe, which states its own fit."""
        r = (recipes if recipes is not None else self.recipes()).get(m.get("id"))
        if r:
            b = int(r.get("bytes") or m.get("size_bytes") or 0)
            return [{"key": f"{r['label']} recipe", "quant": r.get("quant") or m.get("quant") or "", "files": [], "bytes": b,
                     "nodes": int(r.get("nodes") or 1), "gb": round(b / 1e9, 1), "engine": r.get("engine") or "recipe", "recipe_key": r["key"]}], True
        fmt = (m.get("format") or "").lower()
        us = (units_idx or {}).get(m.get("id")) if fmt == "gguf" else None
        exact = bool(us)
        if us: us = [dict(u) for u in us]
        else:
            base = {"quant": m.get("quant") or m.get("format") or "", "files": [], "bytes": m.get("size_bytes") or 0}
            rt = [str(x).lower() for x in (m.get("runtime") or [])]
            engs = [e for e in self.cfg["engines"] if e in rt] if fmt == "safetensors" else []
            us = [{**base, "key": "(whole variant)" + ("" if i == 0 else f" · {ENGINE_LABEL.get(e, e)}"), "engine": e} for i, e in enumerate(engs)] \
                or [{**base, "key": "(whole variant)"}]
        for u in us: u["nodes"] = nodes_needed(u["bytes"], self.cfg["fit"]); u["gb"] = round(u["bytes"] / 1e9, 1)
        return us, exact

    def local_state(self, mid, units):
        p = f"{self.cfg['local_dir']}/{mid}"; whole = os.path.exists(f"{p}/.deck-complete")
        lu = [u["key"] for u in units if os.path.exists(f"{p}/.deck-complete-{_slug(u['key'])}")]
        return {"local": whole or bool(lu), "local_units": lu, "partial": os.path.isdir(p) and not whole and not lu}

    def kits_for(self, d, model):
        name = (model or "").lower()
        return [{"repo": k.get("repo"), "nodes": k.get("nodes"), "source": k.get("source")} for k in (d.get("kits") or [])
                if name and name in (k.get("repo") or "").lower()] if isinstance(d, dict) else []

    def rows(self, res):
        ms, d = self.models(); ui = (_rjson(self.UNITS, {}) or {}).get("units"); seen = _rjson(self.SEEN, {}); rcp = self.recipes()
        if "_archs_fixed" not in self.cfg: self.cfg["_archs"] = {e: read_archs(p) for e, p in (self.cfg.get("engine_archs") or {}).items()}
        cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - self.cfg["new_days"] * 86400)); out = []
        for m in ms:
            mid = m.get("id"); us, exact = self.units_for(m, ui, rcp); ks = [u["nodes"] for u in us if u["nodes"]]
            out.append({"id": mid, "maker": m.get("maker"), "model": m.get("model"), "variant": m.get("variant"), "category": m.get("category"),
                        "format": m.get("format"), "quant": m.get("quant"), "status": m.get("status"), "source": m.get("source"),
                        "size_gb": round((m.get("size_bytes") or 0) / 1e9, 1), "tier": min(ks) if ks else None, "units": us,
                        "units_exact": exact or (m.get("format") or "").lower() != "gguf", "kits": self.kits_for(d, m.get("model")),
                        "first_seen": seen.get(mid), "new": (seen.get(mid) or "") >= cutoff, "servable": m.get("category") in self.cfg["servable_categories"],
                        "plans": {u["key"]: plan_insert(m, u, res, self.cfg, self.job["running"], rcp.get(u.get("recipe_key"))) for u in us},
                        **({"recipe": {k: rcp[mid].get(k) for k in ("label", "engine", "need_gib", "license", "repo", "requires_ollama_stopped")}} if mid in rcp else {}),
                        **self.local_state(mid, us)})
            r = out[-1]; r["function"], r["function_label"] = function_of(r["category"], self.cfg); r["fit"] = fit_of(m, us, r["plans"], self.cfg)
            r["runtime"] = m.get("runtime") or []; r["engines"] = engines_of(r)
        fo = [f["key"] for f in self.cfg.get("functions") or []] + ["other"]
        return sorted(out, key=lambda r: (fo.index(r["function"]) if r["function"] in fo else 99, FIT_ORDER.index(r["fit"]["class"]), r["tier"] or 9,
                                          (r["maker"] or "").lower(), (r["model"] or "").lower(), r["variant"] or "")), d

    def local_cassettes(self, rows):
        ids = {r["id"]: r for r in rows}; out = []; L = self.cfg["local_dir"]
        for p in glob.glob(f"{L}/*/*/*"):
            if not os.path.isdir(p): continue
            mid = os.path.relpath(p, L); n = 0
            for root, _, files in os.walk(p):
                for f_ in files:
                    try: n += os.path.getsize(os.path.join(root, f_))
                    except OSError: pass
            r = ids.get(mid, {})
            out.append({"id": mid, "gb": round(n / 1e9, 1), "tier": r.get("tier"), "units": r.get("local_units", []),
                        "complete": os.path.exists(f"{p}/.deck-complete") or bool(r.get("local_units"))})
        return sorted(out, key=lambda x: x["id"])

    def slots(self, res):
        out = []
        for n in self.cfg["fleet"]:
            if n.get("self"):
                out.append({**n, "online": True, "free_disk_gb": round(res["free_disk_gib"] * 1.0737, 0), "mem_avail_gib": res["mem_avail_gib"],
                            "mem_total_gib": res["mem_total_gib"], "budget_gib": self.cfg["fit"]["gpu_budget_gib"], "lanes": res["lanes"],
                            "ollama": res["ollama"], "engines_running": res["engines_running"], "ollama_up": res.get("ollama_up"), "max_loaded": res.get("max_loaded")})
            elif not n.get("ssh"): out.append({**n, "online": False, "why": "not configured"})
            else: out.append({**n, **self.sys.remote(n)})
        return out

    def state(self, force=False):
        if force: self.sync_bg()
        res = self.sys.residents(self.cfg["state_dir"], self.recipes()); rows, d = self.rows(res); slots = self.slots(res)
        mx = int(self.cfg["fit"]["max_nodes"]); tiers = {str(k): sum(1 for r in rows if r["tier"] == k) for k in range(1, mx + 1)}
        tiers["beyond"] = sum(1 for r in rows if r["tier"] is None)
        hist = []
        if os.path.exists(self.ACTIONS):
            with open(self.ACTIONS) as f: hist = [json.loads(l) for l in f if l.strip()][-40:]
        tail = ""
        try:
            with open(self.PULL_LOG, errors="replace") as f: tail = next((l.strip() for l in reversed(f.read()[-4000:].splitlines()) if l.strip() and "%" not in l and not l.startswith("[done")), "")
        except OSError: pass
        lf = last_fail(hist, tail); playing = _rjson(self.PLAYING, {})
        f = self.cfg["fit"]
        return {"version": VERSION, "asof": time.strftime("%Y-%m-%dT%H:%M:%S"), "library_built": d.get("built") if isinstance(d, dict) else None,
                "nas": _rjson(self.NAS, {}), "syncing": self.syncing, "n_shelf": len(rows), "shelf": rows, "tiers": tiers, "max_nodes": mx,
                "n_new": sum(1 for r in rows if r["new"]), "slots": slots, "online_nodes": sum(1 for s in slots if s.get("online")),
                "protected": self.cfg["protected"], "residents": res, "job": dict(self.job), "playing": playing,
                "engine_bar": engine_bar(rows, res, playing), "bench": self.bench_state(res), "last_fail": lf, "arch_check": sorted(e for e, a in (self.cfg.get("_archs") or {}).items() if a),
                "local": self.local_cassettes(rows), "history": hist[-12:], "new_days": self.cfg["new_days"], "engines": self.cfg["engines"],
                "functions": [{"key": k, "label": lab, "n": sum(1 for r in rows if r["function"] == k),
                               "fit": {c: sum(1 for r in rows if r["function"] == k and r["fit"]["class"] == c) for c in FIT_ORDER}}
                              for k, lab in [(f["key"], f.get("label") or f["key"]) for f in self.cfg.get("functions") or []] + [("other", "Other")]],
                "fit_order": FIT_ORDER, "fit_label": FIT_LABEL,
                "fit": {**f, "num_ctx": self.cfg["num_ctx"],
                        "rule": f"smallest k with max(W + {f['rank_overhead_gib']}·k + {f['kv_floor_gib']}, {f['weight_factor']}·W) ≤ {f['gpu_budget_gib']}·k  (W = unit weights, GiB)"}}

    def summary(self):
        ms, d = self.models(); ui = (_rjson(self.UNITS, {}) or {}).get("units"); t = {}
        for m in ms:
            ks = [u["nodes"] for u in self.units_for(m, ui)[0] if u["nodes"]]; k = str(min(ks)) if ks else "beyond"; t[k] = t.get(k, 0) + 1
        nas = _rjson(self.NAS, {})
        return {"total": len(ms), "tiers": t, "built": d.get("built") if isinstance(d, dict) else None, "nas_awake": nas.get("awake"), "asleep_since": nas.get("asleep_since")}

    # benchmarks ----------------------------------------------------------------------------------------------------
    def target(self, res=None):
        """what the benchmark panel measures — the same thing the player shows: the deck cassette, else a running engine / recipe lane,
        else the first Ollama resident. -> {key, label, model, engine, port?, metrics?, context} or None"""
        p = _rjson(self.PLAYING, {}); rcp = self.recipes()
        res = res or self.sys.residents(self.cfg["state_dir"], rcp)
        def eng(e, name, label):
            if e in ("vllm", "sglang"):
                v = self.cfg[e]; return {"engine": ENGINE_LABEL[e], "port": v["port"], "metrics": "/metrics", "model": name, "label": label,
                                         "context": v.get("max_model_len") or v.get("context_length")}
            r = next((x for x in rcp.values() if (x.get("label") or "recipe") == label or x.get("engine") == e), None) or {}
            return {"engine": r.get("label") or label, "port": r.get("port"), "metrics": r.get("metrics_path", "/metrics"), "model": r.get("served") or name,
                    "label": label, "context": r.get("context")}
        if p.get("id") and p.get("runtime") == "ollama":
            t = {"engine": "Ollama", "model": p["name"], "label": p["id"], "context": self.cfg["num_ctx"]}
        elif p.get("id"): t = eng(p.get("runtime"), p.get("name"), p.get("label") or ENGINE_LABEL.get(p.get("runtime"), p.get("runtime")))
        elif res.get("engines_running"):
            e = res["engines_running"][0]; t = eng(e.get("engine"), e.get("name"), e.get("name"))
        elif res.get("ollama"): t = {"engine": "Ollama", "model": res["ollama"][0]["name"], "label": res["ollama"][0]["name"], "context": None}
        else: return None
        t["key"] = f"{t['engine']}:{t['model']}"; return t

    def sample(self, res=None):
        """one live sample of the target's own counters (vLLM / SGLang / a recipe with metrics_path). Ollama keeps no counters."""
        t = self.target(res); now = time.time()
        if not t or not t.get("port") or not t.get("metrics"): self.prev = None; return None
        c = parse_prom(self.sys.engine_metrics(t["port"], t["metrics"]))
        if "gen" not in c: self.prev = None; return None
        cur = (now, c); r = rates(self.prev, cur) if self.prev and self.prev[2] == t["key"] else None; self.prev = (now, c, t["key"])
        if r is None: return None
        self.live.append({"t": round(now), "key": t["key"], "decode": r[0], "prefill": r[1], "running": c.get("running"), "waiting": c.get("waiting"), "gen": c["gen"]})
        if r[0] > 0: self.served[t["key"]] = now
        day = time.strftime("%Y-%m-%d"); pk = _rjson(self.PEAKS, {})
        _wjson(self.PEAKS, bump_peak(bump_peak(pk, day, "decode", r[0]), day, "prefill", r[1]))
        return r

    def sampler(self, every=5):
        def run():
            while True:
                try: self.sample()
                except Exception: pass
                time.sleep(every)
        threading.Thread(target=run, daemon=True).start()

    def bench(self, kind):
        if kind not in ("decode", "prefill"): return {"error": "kind must be decode or prefill", "status": 400}
        with self.lock:
            if self.job["running"]: return {"error": "an insert is running — benchmark after it", "status": 409}
            if self.bjob["running"]: return {"error": "a benchmark is already running", "status": 409}
            t = self.target()
            if not t: return {"error": "nothing is loaded — insert a cassette first", "status": 409}
            if t["engine"] != "Ollama" and not t.get("port"): return {"error": f"no port known for {t['engine']}", "status": 409}
            self.bjob.update(running=True, kind=kind, msg=f"{kind} on {t['model']}…")
        def run():
            try:
                r = self.sys.bench_ollama(t["model"], kind) if t["engine"] == "Ollama" else self.sys.bench_openai(t["port"], t["model"], kind)
                row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": t["key"], "engine": t["engine"], "model": t["model"], "kind": kind, **r}
                with open(self.BENCH, "a") as f: f.write(json.dumps(row) + "\n")
                _wjson(self.PEAKS, bump_peak(_rjson(self.PEAKS, {}), row["ts"][:10], kind, r["tok_s"]))
                self.bjob["msg"] = f"{kind}: {r['tok_s']} tok/s ({r['tokens']} tokens)"
            except Exception as e: self.bjob["msg"] = f"{kind} failed: {type(e).__name__}: {str(e)[:160]}"
            finally: self.bjob["running"] = False
        threading.Thread(target=run, daemon=True).start()
        return {"started": True, "kind": kind, "model": t["model"]}

    def bench_state(self, res=None):
        t = self.target(res); out = {"target": t, "job": dict(self.bjob), "live": None, "last": {}, "daily": []}
        pk = _rjson(self.PEAKS, {}); days = [time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400)) for i in range(13, -1, -1)]
        out["daily"] = [{"day": d, "decode": (pk.get(d) or {}).get("decode"), "prefill": (pk.get(d) or {}).get("prefill")} for d in days]
        if not t: return out
        rows = []
        if os.path.exists(self.BENCH):
            with open(self.BENCH) as f: rows = [json.loads(l) for l in f if l.strip()][-200:]
        for k in ("decode", "prefill"):
            r = [x for x in rows if x.get("key") == t["key"] and x.get("kind") == k]
            if r: out["last"][k] = {**r[-1], "best": max(x["tok_s"] for x in r), "runs": len(r)}
        L = [x for x in self.live if x["key"] == t["key"]]
        if L:
            busy = [x for x in L if x["decode"] > 0] or L; bp = [x for x in L if x["prefill"] > 0] or L
            seen = self.served.get(t["key"])
            out["live"] = {"decode": [x["decode"] for x in L][-90:], "prefill": [x["prefill"] for x in L][-90:], "now_decode": L[-1]["decode"], "now_prefill": L[-1]["prefill"],
                           "avg_decode": round(sum(x["decode"] for x in busy) / len(busy), 1), "avg_prefill": round(sum(x["prefill"] for x in bp) / len(bp), 1),
                           "running": L[-1]["running"], "waiting": L[-1]["waiting"], "generated": int(L[-1]["gen"]),
                           "idle_s": round(time.time() - seen) if seen else None, "minutes": round((L[-1]["t"] - L[0]["t"]) / 60)}
        return out

    # actions -------------------------------------------------------------------------------------------------------
    def log(self, **r):
        r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.ACTIONS, "a") as f: f.write(json.dumps(r) + "\n")

    def insert(self, mid, unit_key=None, confirm_evict=None):
        ms = {m["id"]: m for m in self.models()[0]}
        if mid not in ms: return {"error": "unknown model", "status": 404}
        m = ms[mid]; rcp = self.recipes(); units, _ = self.units_for(m, (_rjson(self.UNITS, {}) or {}).get("units"), rcp)
        unit = next((u for u in units if u["key"] == unit_key), None) if unit_key else \
            (units[0] if len(units) == 1 or all(not u.get("files") for u in units) else None)   # whole folder: the first engine (vLLM) by default
        if not unit: return {"error": "choose which file / quant to insert", "units": [u["key"] for u in units], "status": 400}
        ok = {x for x in confirm_evict if isinstance(x, str)} if isinstance(confirm_evict, list) else set()
        with self.lock:
            if self.job["running"]: return {"error": "deck busy", "job": dict(self.job), "status": 409}
            recipe = rcp.get(unit.get("recipe_key")); res = self.sys.residents(self.cfg["state_dir"], rcp)
            plan = plan_insert(m, unit, res, self.cfg, recipe=recipe)
            if plan.get("refuse"): return {"error": plan["refuse"], "status": 409}
            ev = plan.get("would_evict") or []
            if [n for n in ev if n not in ok]:
                return {"error": "this insert would STOP: " + ", ".join(ev) + " — nothing was done. Confirm each one by name to go ahead.",
                        "would_evict": ev, "plan": plan, "status": 409}
            self.job.update(running=True, what=f"INSERT {mid} [{unit['key']}]", t0=time.time(), progress="")
        threading.Thread(target=self._insert_job, args=(m, unit, ok, recipe), daemon=True).start()
        return {"started": True, "what": self.job["what"], "evicting_after_copy": ev}

    def _insert_job(self, m, unit, ok, recipe=None):
        mid = m["id"]; rc, evicted, note = 1, [], ""
        try:
            with open(self.PULL_LOG, "w") as f:
                have = self.local_state(mid, [unit])
                if recipe and recipe.get("copy") is False: rc = 0; f.write(f"[{recipe.get('label')} keeps its own weights (copy: false) — nothing to copy]\n")
                else: rc = 0 if (unit["key"] in have["local_units"] or (not unit.get("files") and have["local"])) else self.sys.pull(mid, unit, f, self.job, self.cfg["state_dir"])
                if rc: f.write(f"[copy FAILED rc={rc}]\n")
                plan2 = {}
                if rc == 0:                                       # RE-CHECK after the (minutes-long) copy, before touching anything
                    with self.lock:
                        res2 = self.sys.residents(self.cfg["state_dir"], self.recipes()); plan2 = plan_insert(m, unit, res2, self.cfg, recipe=recipe)
                    ev2 = plan2.get("would_evict") or []
                    if plan2.get("refuse"): rc, note = 3, f"situation changed during the copy — {plan2['refuse']} — nothing evicted"
                    elif [n for n in ev2 if n not in ok]: rc, note = 3, f"situation changed during the copy — would now also stop {', '.join(n for n in ev2 if n not in ok)}; insert again to confirm"
                if rc == 0 and plan2.get("path") == "ollama":
                    before = [o["name"] for o in res2["ollama"] if o["name"] in self.cfg["protected"]]
                    for n in plan2["would_evict"]:
                        if self.sys.unload(n): evicted.append(n); f.write(f"[evicted {n} — confirmed by name]\n")
                    rc, playing = self.sys.serve_gguf(mid, unit, f)
                    if rc == 0:
                        after = [o["name"] for o in ((self.sys.ollama("/api/ps") or {}).get("models") or [])]
                        lost = [p for p in before if p not in after]
                        if lost:                                  # a protected model got pushed out anyway: undo, loudly
                            self.sys.unload(playing["name"]); rc = 4
                            note = f"protected model(s) {', '.join(lost)} were unloaded by Ollama — the new cassette was removed again"
                        else: _wjson(self.PLAYING, {"id": mid, "unit": unit["key"], **playing}); f.write(f"[playing: {playing['name']}]\n")
                elif rc == 0:                                     # vLLM / SGLang / a recipe: everything named was confirmed above
                    evicted = plan2["would_evict"]
                    if plan2["path"] == "recipe": rc, playing = self.sys.serve_recipe(recipe, mid, f, evicted)
                    else: rc, playing = self.sys.serve_engine(plan2["path"], mid, f, evicted)
                    if rc == 0: _wjson(self.PLAYING, {"id": mid, "unit": unit["key"], **playing}); f.write(f"[playing: {playing.get('label') or ENGINE_LABEL.get(playing['runtime'], playing['runtime'])} {playing['name']}]\n")
                if note: f.write(f"[{note}]\n")
                f.write(f"[done rc={rc}]\n")
        finally:
            self.log(action="INSERT", id=mid, unit=unit["key"], nodes=unit.get("nodes"), rc=rc, evicted=evicted, note=note, secs=round(time.time() - self.job["t0"]))
            self.job.update(running=False, what="", progress="")

    def deck_names(self, mid):
        try:
            with open(f"{self.cfg['local_dir']}/{mid}/.deck-ollama-names") as f: return [l.strip() for l in f if l.strip().startswith("deck/")]
        except Exception: return []

    def eject(self, mid=None, remove_local=False):
        if self.job["running"]: return {"error": "deck busy", "status": 409}
        playing = _rjson(self.PLAYING, {}); stop = not mid or playing.get("id") == mid
        mid = mid or playing.get("id"); done = []
        if remove_local and (not mid or len(mid.split("/")) != 3 or ".." in mid.split("/")): return {"error": "delete needs a full Maker/Model/Variant id", "status": 400}
        if stop:
            done += self.sys.stop_engines(self.recipes())
            for x in ((self.sys.ollama("/api/ps") or {}).get("models") or []):
                if x["name"].startswith("deck/") and self.sys.unload(x["name"]): done.append(f"unloaded {x['name']}")
            if os.path.exists(self.PLAYING): os.remove(self.PLAYING)
        if remove_local and mid:
            names = self.deck_names(mid)                         # exactly the names this cassette created — never a prefix match
            for n in names: self.sys.ollama_rm(n); done.append(f"ollama rm {n}")
            p = f"{self.cfg['local_dir']}/{mid}"; root = os.path.realpath(self.cfg["local_dir"]) + "/"
            if os.path.isdir(p) and os.path.realpath(p).startswith(root):
                shutil.rmtree(p); done.append(f"deleted the local copy of {mid} (NAS copy untouched)")
        self.log(action="EJECT", id=mid, remove_local=remove_local, done=done)
        return {"done": done or ["nothing was playing"]}


# ── demo mode: a fake NAS + a fake GPU box, for screenshots and for trying the UI ─────────────────────────────────
DEMO_SHELF = [  # (category, maker, model, variant, publisher, format, quant, GB, gguf quants)
    ("LLMs", "Qwen", "Qwen3-32B", "original", "Qwen", "safetensors", "BF16", 65.5, None),
    ("LLMs", "Qwen", "Qwen3-32B", "unsloth-GGUF", "unsloth", "gguf", "Q4_K_M", 0, [("Q4_K_M", 19.8), ("Q6_K", 26.9), ("Q8_0", 34.8)]),
    ("LLMs", "Qwen", "Qwen3-235B-A22B", "unsloth-GGUF", "unsloth", "gguf", "Q4_K_M", 0, [("UD-Q2_K_XL", 88.8), ("Q4_K_M", 142.6), ("Q8_0", 250.0)]),
    ("LLMs", "Qwen", "Qwen3-30B-A3B", "FP8", "Qwen", "safetensors", "FP8", 32.4, None),
    ("LLMs", "Qwen", "Qwen3.8-Flash-Next", "Vontra-MLX-4bit-MTP", "Vontra", "safetensors", "MLX-4bit", 113.2, None),
    ("Coding", "Qwen", "Qwen3-Coder-30B-A3B-Instruct", "unsloth-GGUF", "unsloth", "gguf", "Q4_K_M", 0, [("Q4_K_M", 18.6), ("Q8_0", 32.5)]),
    ("LLMs", "OpenAI", "gpt-oss-120b", "original", "openai", "safetensors", "MXFP4", 65.3, None),
    ("LLMs", "OpenAI", "gpt-oss-20b", "ggml-org-GGUF", "ggml-org", "gguf", "MXFP4", 0, [("MXFP4", 12.1)]),
    ("LLMs", "Google", "Gemma-3-27B-it", "original", "google", "safetensors", "BF16", 54.9, None),
    ("LLMs", "Google", "Gemma-3-27B-it", "unsloth-GGUF", "unsloth", "gguf", "Q4_K_M", 0, [("Q4_K_M", 16.5), ("Q8_0", 28.7)]),
    ("Small-On-Device", "Google", "Gemma-3-4B-it", "original", "google", "safetensors", "BF16", 8.6, None),
    ("LLMs", "Meta", "Llama-3.3-70B-Instruct", "original", "meta-llama", "safetensors", "BF16", 141.1, None),
    ("LLMs", "Meta", "Llama-3.3-70B-Instruct", "bartowski-GGUF", "bartowski", "gguf", "Q4_K_M", 0, [("Q4_K_M", 42.5), ("Q6_K", 57.9)]),
    ("LLMs", "Meta", "Llama-4-Scout-17B-16E", "original", "meta-llama", "safetensors", "BF16", 217.0, None),
    ("LLMs", "Mistral", "Mistral-Small-3.2-24B", "original", "mistralai", "safetensors", "BF16", 48.0, None),
    ("Coding", "Mistral", "Devstral-Small-2507", "unsloth-GGUF", "unsloth", "gguf", "Q4_K_M", 0, [("Q4_K_M", 14.3), ("Q8_0", 25.1)]),
    ("LLMs", "DeepSeek", "DeepSeek-R1-Distill-Llama-70B", "original", "deepseek-ai", "safetensors", "BF16", 141.1, None),
    ("LLMs", "DeepSeek", "DeepSeek-V3.1", "unsloth-GGUF", "unsloth", "gguf", "Q2_K_XL", 0, [("UD-TQ1_0", 170.0), ("UD-Q2_K_XL", 251.0)]),
    ("LLMs", "DeepSeek", "DeepSeek-R1-0528", "original", "deepseek-ai", "safetensors", "FP8", 688.0, None),
    ("LLMs", "Zhipu-GLM", "GLM-4.5-Air", "FP8", "zai-org", "safetensors", "FP8", 113.0, None),
    ("LLMs", "Qwen", "Qwen3-32B", "turboderp-EXL3", "turboderp", "safetensors", "EXL3", 19.6, None),
    ("LLMs", "Example", "Novel-Hybrid-9B", "original", "example-lab", "safetensors", "BF16", 18.4, None),
    ("LLMs", "Zhipu-GLM", "GLM-4.5-Air", "unsloth-GGUF", "unsloth", "gguf", "Q4_K_M", 0, [("Q4_K_M", 72.9)]),
    ("LLMs", "Microsoft", "Phi-4", "original", "microsoft", "safetensors", "BF16", 29.3, None),
    ("Small-On-Device", "HuggingFace", "SmolLM3-3B", "original", "HuggingFaceTB", "safetensors", "BF16", 6.2, None),
    ("LLMs", "Moonshot", "Kimi-K2-Instruct", "unsloth-GGUF", "unsloth", "gguf", "Q2_K_XL", 0, [("UD-Q2_K_XL", 381.0)]),
    ("Image", "BlackForestLabs", "FLUX.1-dev", "original", "black-forest-labs", "safetensors", "BF16", 57.9, None),
    ("Image", "StabilityAI", "stable-diffusion-3.5-large", "original", "stabilityai", "safetensors", "BF16", 27.6, None),
    ("Video", "Wan-AI", "Wan2.2-T2V-A14B", "original", "Wan-AI", "safetensors", "BF16", 126.0, None),
    ("Video", "Lightricks", "LTX-Video", "original", "Lightricks", "safetensors", "BF16", 30.2, None),
    ("Speech-ASR", "OpenAI", "whisper-large-v3-turbo", "original", "openai", "safetensors", "F16", 1.6, None),
    ("Speech-TTS", "Hexgrad", "Kokoro-82M", "original", "hexgrad", "other", None, 0.3, None),
    ("Embeddings-Reranking", "Qwen", "Qwen3-Embedding-8B", "original", "Qwen", "safetensors", "BF16", 15.1, None),
    ("Embeddings-Reranking", "BAAI", "bge-reranker-v2-m3", "original", "BAAI", "safetensors", "F32", 2.3, None),
    ("OCR", "Mistral", "Mistral-OCR-mini", "original", "mistralai", "safetensors", "BF16", 8.9, None),
]


class DemoSystem(System):
    """No ssh, no rsync, no docker, no Ollama: a believable box with 128 GB unified memory and two Ollama residents.
    ollama_up=False simulates a box where Ollama is stopped (to try a recipe lane such as TensorFold)."""
    def __init__(self, cfg, ollama_up=True):
        super().__init__(cfg); self.up = ollama_up
        self.oll = [{"name": "qwen3:14b", "gib": 10.6}, {"name": "nomic-embed-text:latest", "gib": 0.6}] if ollama_up else []
        self.tags = {"qwen3:14b": 9.3, "nomic-embed-text:latest": 0.3, "llama3.2:3b": 2.0}; self.engine = None

    def residents(self, state_dir, recipes=None):
        used = sum(o["gib"] for o in self.oll) + (self.engine["gib"] if self.engine else 0)
        return {"lanes": [], "ollama": [dict(o) for o in self.oll], "tags_gib": dict(self.tags) if self.up else {}, "ollama_ok": self.up, "ollama_up": self.up,
                "observed_gib": {"qwen3:14b": 10.6}, "max_loaded": 3, "free_disk_gib": 1490.0, "mem_avail_gib": round(116.0 - used, 1),
                "mem_total_gib": 121.7, "engines_running": [{k: v for k, v in self.engine.items() if k != "gib"}] if self.engine else []}

    def remote(self, node): return {"online": True, "mem_avail_gib": 109, "why": ""}

    def ollama(self, path, body=None, timeout=4):
        if not self.up: return None
        if path == "/api/ps": return {"models": [{"name": o["name"], "size": int(o["gib"] * 2**30)} for o in self.oll]}
        return {}

    def pull(self, mid, unit, f, job, state_dir):
        total = unit["bytes"]; f.write(f"$ rsync -rLt -s --partial --info=progress2 nas:/srv/models-nas/library/{mid}/ ~/models/deck/{mid}/\n")
        for pct in range(0, 101, 4):
            line = f"{total*pct/100/1e9:8.1f}G {pct:3d}%  1.08GB/s    0:00:{max(0, 25-pct//4):02d}"; job["progress"] = line; f.write(line + "\n"); f.flush(); time.sleep(0.35)
        d = f"{self.cfg['local_dir']}/{mid}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/.deck-complete-{_slug(unit['key'])}" if unit.get("files") else f"{d}/.deck-complete", "w") as mf: mf.write("demo")
        with open(f"{d}/{unit['key']}.gguf" if unit.get("files") else f"{d}/model.safetensors", "wb") as wf: wf.truncate(total)   # sparse
        return 0

    def unload(self, name): self.oll = [o for o in self.oll if o["name"] != name]; return True

    def serve_gguf(self, mid, unit, f):
        name = "deck/" + (mid + "_" + unit["quant"]).replace("/", "_").lower()[:120]
        f.write(f"$ ollama create {name} -f Modelfile.deck\nsuccess\n$ warm load (keep_alive 30m)\n"); time.sleep(1.5)
        self.oll.append({"name": name, "gib": round(unit["bytes"] / 2**30 * 1.12 + 1.8, 1)}); self.tags[name] = round(unit["bytes"] / 2**30, 1)
        with open(f"{self.cfg['local_dir']}/{mid}/.deck-ollama-names", "a") as nf: nf.write(name + "\n")
        return 0, {"runtime": "ollama", "name": name}

    def serve_engine(self, engine, mid, f, evict):
        self.oll = [o for o in self.oll if o["name"] not in evict]; cmd, v, served = self.engine_cmd(engine, mid)
        f.write("$ " + " ".join(cmd[:14]) + " …\n"); time.sleep(2); self.engine = {"engine": engine, "name": ENGINE_LABEL[engine], "gib": 88.0}
        f.write(f"[{ENGINE_LABEL[engine]} {served} answering on :{v['port']} after 95 s]\n")
        return 0, {"runtime": engine, "port": v["port"], "name": served}

    def serve_recipe(self, r, mid, f, evict):
        self.oll = [o for o in self.oll if o["name"] not in evict]
        f.write(f"$ {r['start']}   ({r['label']}, MCC_MODEL_DIR=~/models/deck/{mid})\n  … loading weights, compiling kernels\n"); time.sleep(2.5)
        self.engine = {"engine": r.get("engine") or "recipe", "name": r["label"], "recipe": r["key"], "gib": 104.0}
        f.write(f"[{r['label']} answering on :{r['port']} after 152 s]\n")
        return 0, {"runtime": r.get("engine") or "recipe", "port": r["port"], "name": r.get("served") or mid.split("/")[1], "label": r["label"], "recipe": r["key"]}

    def stop_engines(self, recipes):
        if not self.engine: return []
        n = self.engine["name"]; self.engine = None
        return [f"stopped the {n} " + ("recipe" if n not in ENGINE_LABEL.values() else "cassette")]

    def ollama_rm(self, name): self.tags.pop(name, None)

    def engine_metrics(self, port, path="/metrics"):
        """a busy-then-idle serving pattern: ~60 tok/s decode and ~1,800 tok/s prefill while requests run"""
        if not self.engine: return None
        c = self.__dict__.setdefault("ctr", {"t": time.time(), "gen": 296000.0, "prompt": 3.1e6, "i": 0})
        now = time.time(); dt = now - c["t"]; c["t"] = now; c["i"] += 1
        busy = (c["i"] // 6) % 4 != 3; wob = 1 + 0.12 * ((c["i"] * 7919) % 13 - 6) / 6
        run = 1 if busy else 0
        c["gen"] += 63 * wob * dt * run; c["prompt"] += 1830 * wob * dt * run * (1 if c["i"] % 3 == 0 else 0.15)
        return (f"# TYPE vllm:generation_tokens_total counter\nvllm:generation_tokens_total{{model_name=\"demo\"}} {c['gen']:.0f}\n"
                f"vllm:prompt_tokens_total{{model_name=\"demo\"}} {c['prompt']:.0f}\nvllm:num_requests_running{{model_name=\"demo\"}} {run}\n"
                f"vllm:num_requests_waiting{{model_name=\"demo\"}} 0\n")

    def bench_openai(self, port, model, kind):
        time.sleep(2.5)
        return {"tok_s": 61.8, "tokens": 256, "secs": 4.2} if kind == "decode" else {"tok_s": 1874.0, "tokens": 2014, "secs": 1.07}

    def bench_ollama(self, name, kind):
        time.sleep(2)
        return {"tok_s": 38.4, "tokens": 256, "secs": 6.7} if kind == "decode" else {"tok_s": 912.0, "tokens": 2011, "secs": 2.2}


DEMO_ARCH = {"Qwen": "Qwen3ForCausalLM", "OpenAI": "GptOssForCausalLM", "Google": "Gemma3ForConditionalGeneration", "Meta": "LlamaForCausalLM",
             "Mistral": "MistralForCausalLM", "DeepSeek": "DeepseekV3ForCausalLM", "Zhipu-GLM": "Glm4MoeForCausalLM", "Microsoft": "Phi3ForCausalLM",
             "HuggingFace": "SmolLM3ForCausalLM", "Example": "NovelHybridForCausalLM"}
DEMO_ENGINE_ARCHS = sorted(set(DEMO_ARCH.values()) - {"NovelHybridForCausalLM"} | {"Qwen3MoeForCausalLM", "Llama4ForConditionalGeneration"})

DEMO_RECIPES = {"Qwen/Qwen3.8-Flash-Next/Vontra-MLX-4bit-MTP": {
    "label": "TensorFold", "engine": "tensorfold", "served": "Qwen3.8-Flash-Next", "quant": "MLX 4-bit + MTP", "nodes": 1, "need_gib": 112, "port": 8888,
    "health_path": "/health", "metrics_path": "/metrics", "requires_ollama_stopped": True, "copy": False, "start": "~/recipes/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/start.sh",
    "stop": "~/recipes/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/stop.sh",
    "repo": "https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold",
    "license": "recipe: MIT (see its LICENSE) · weights: see the model card"}}


def demo_seed(cfg):
    """Write a fake library + units + local copies into the demo state dir."""
    s = cfg["state_dir"]; os.makedirs(s, exist_ok=True); models, units = [], {}; now = time.strftime("%Y-%m-%d")
    for cat, maker, model, var, pub, fmt, q, gb, quants in DEMO_SHELF:
        mid = f"{maker}/{model}/{var}"; size = int((sum(x[1] for x in quants) if quants else gb) * 1e9)
        rt = ["llama.cpp", "ollama"] if fmt == "gguf" else ["exllama", "tabbyAPI"] if (q or "").startswith("EXL") else \
            (["vllm", "sglang", "transformers"] if cat in ("LLMs", "Coding", "Small-On-Device") else ["diffusers", "transformers"])
        models.append({"id": mid, "maker": maker, "model": model, "variant": var, "publisher": pub, "category": cat, "format": fmt, "quant": q,
                       "runtime": rt, "size_bytes": size, "status": "complete", "source": f"https://huggingface.co/{pub}/{model}"})
        if fmt == "safetensors" and cat in ("LLMs", "Coding", "Small-On-Device") and mid not in DEMO_RECIPES:
            models[-1]["architectures"] = [{"Llama-4-Scout-17B-16E": "Llama4ForConditionalGeneration", "Qwen3-30B-A3B": "Qwen3MoeForCausalLM"}.get(model, DEMO_ARCH.get(maker, "UnknownForCausalLM"))]
        if quants:
            units[mid] = [{"key": f"{model}-{qq}", "quant": qq, "files": [f"{model}-{qq}.gguf"], "bytes": int(g * 1e9)} for qq, g in quants]
    models[-1]["status"] = "downloading"
    d = {"schema": 1, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": models,
         "kits": [{"repo": "Qwen3-235B-A22B-2x-node-recipe", "nodes": 2, "source": "https://github.com/example/recipes"}]}
    _wjson(f"{s}/library.json", d); _wjson(f"{s}/units.json", {"built": d["built"], "listed": d["built"], "units": units})
    seen = {m["id"]: "2000-01-01" for m in models}
    for nid in ("OpenAI/gpt-oss-20b/ggml-org-GGUF", "Qwen/Qwen3-30B-A3B/FP8", "Zhipu-GLM/GLM-4.5-Air/unsloth-GGUF"): seen[nid] = now
    _wjson(f"{s}/seen.json", seen); _wjson(f"{s}/nas_status.json", {"awake": True, "last_ok": d["built"], "checked": d["built"]})
    for mid, key, gb in (("Google/Gemma-3-27B-it/unsloth-GGUF", "Gemma-3-27B-it-Q4_K_M", 16.5), ("Mistral/Devstral-Small-2507/unsloth-GGUF", "Devstral-Small-2507-Q4_K_M", 14.3)):
        p = f"{cfg['local_dir']}/{mid}"; os.makedirs(p, exist_ok=True)
        with open(f"{p}/.deck-complete-{_slug(key)}", "w") as mf: mf.write("demo")
        with open(f"{p}/{key}.gguf", "wb") as wf: wf.truncate(int(gb * 1e9))          # sparse: shows the size, uses no disk
    hist = [{"action": "INSERT", "id": "Mistral/Devstral-Small-2507/unsloth-GGUF", "unit": "Devstral-Small-2507-Q4_K_M", "nodes": 1, "rc": 0, "evicted": [], "secs": 31, "ts": f"{now}T09:12:40"},
            {"action": "EJECT", "id": "Mistral/Devstral-Small-2507/unsloth-GGUF", "remove_local": False, "done": ["unloaded deck/mistral_devstral-small-2507"], "ts": f"{now}T11:03:02"},
            {"action": "INSERT", "id": "Mistral/Mistral-Small-3.2-24B/original", "unit": "(whole variant)", "nodes": 1, "rc": 1, "evicted": ["qwen3:14b"], "secs": 212, "ts": f"{now}T11:20:15",
             "note": "vLLM exited while loading: CUDA out of memory (max_model_len 32768 is too long for the memory left) — try a smaller max_model_len"}]
    with open(f"{s}/actions.jsonl", "w") as f: f.write("".join(json.dumps(h) + "\n" for h in hist))
    dec = [58, 61, 0, 47, 66, 64, 72, 0, 63, 68, 295, 70, 66, 64]; pre = [1710, 1802, 0, 1420, 1836, 1795, 1903, 0, 1760, 1880, 2140, 1850, 1820, 1836]
    _wjson(f"{s}/peaks.json", {time.strftime("%Y-%m-%d", time.localtime(time.time() - (13 - i) * 86400)): {"decode": d, "prefill": p}
                               for i, (d, p) in enumerate(zip(dec, pre)) if d})
    with open(f"{s}/bench.jsonl", "w") as f:
        for kind, v, n in (("decode", 63.3, 256), ("prefill", 1836.0, 2014)):
            f.write(json.dumps({"ts": f"{now}T08:40:00", "key": "TensorFold:Qwen3.8-Flash-Next", "engine": "TensorFold", "model": "Qwen3.8-Flash-Next",
                                "kind": kind, "tok_s": v, "tokens": n, "secs": 4.0}) + "\n")


# ── HTTP ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def read_token(cfg):
    p = cfg["token_file"]
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f: f.write(secrets.token_urlsafe(32) + "\n")
        print(f"created an access token in {p} (chmod 600) — paste it into the page once", file=sys.stderr)
    with open(p) as f: return f.read().strip()


def make_handler(deck, token, page):
    class H(BaseHTTPRequestHandler):
        server_version = f"mcc-deck/{VERSION}"

        def log_message(self, *a): pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer"); self.send_header("X-Frame-Options", "DENY"); self.end_headers(); self.wfile.write(b)

        def _authed(self):
            if token is None: return True
            got = self.headers.get("X-Token", "")
            return bool(got) and hmac.compare_digest(got.encode(), token.encode())

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 65536: return None
            try: return json.loads(self.rfile.read(n) or b"{}")
            except Exception: return None

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/deck"): return self._send(200, page, "text/html")
            if u.path == "/api/health": return self._send(200, {"ok": True, "version": VERSION})
            if not self._authed(): return self._send(403, {"error": "token required"})
            if u.path == "/api/deck": return self._send(200, deck.state(force="refresh" in parse_qs(u.query)))
            if u.path == "/api/deck/summary": return self._send(200, deck.summary())
            if u.path == "/api/deck/log":
                try:
                    with open(deck.PULL_LOG, errors="replace") as f: txt = f.read()[-20000:]
                except OSError: txt = ""
                return self._send(200, {"log": txt, "job": dict(deck.job)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed(): return self._send(403, {"error": "token required"})
            b = self._body()
            if not isinstance(b, dict): return self._send(400, {"error": "bad JSON body"})
            u = urlparse(self.path).path
            if u == "/api/deck/insert": r = deck.insert(str(b.get("id") or ""), b.get("unit"), b.get("confirm_evict"))
            elif u == "/api/deck/eject": r = deck.eject(b.get("id") or None, bool(b.get("remove_local")))
            elif u == "/api/deck/bench": r = deck.bench(str(b.get("kind") or ""))
            else: return self._send(404, {"error": "not found"})
            self._send(r.pop("status", 200) if "error" in r else 200, r)
    return H


def probe_archs(cfg, engine):
    """run the engine's own image once and record what it can load -> cfg["engine_archs"][engine]. Re-run after pulling a new image."""
    if engine not in ARCH_PROBE: return 2, f"unknown engine {engine} (vllm or sglang)"
    image = cfg[engine]["image"]; out_p = (cfg.get("engine_archs") or {}).get(engine)
    if not out_p: return 2, f"engine_archs.{engine} is not set in the config"
    r = subprocess.run(["docker", "run", "--rm", "--entrypoint", ARCH_PROBE[engine][0], image] + ARCH_PROBE[engine][1:],
                       capture_output=True, text=True, timeout=600)
    archs = sorted({l.strip() for l in r.stdout.splitlines() if re.fullmatch(r"[A-Za-z0-9_]+", l.strip() or "-")})
    if r.returncode or not archs: return 1, f"could not read the architectures from {image} (rc={r.returncode}): {r.stderr.strip()[-300:]}"
    _wjson(out_p, {"engine": engine, "image": image, "read": time.strftime("%Y-%m-%dT%H:%M:%S"), "architectures": archs})
    return 0, f"{len(archs)} architectures from {image} -> {out_p}"


def page_html():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "deck.html"), encoding="utf-8") as f: return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description="NAS Model Cassette Changer — the deck")
    ap.add_argument("cmd", choices=["serve", "sync", "summary", "fit", "archs"]); ap.add_argument("arg", nargs="?")
    ap.add_argument("--config", default=os.environ.get("MCC_DECK_CONFIG", "~/.config/mcc/deck.json"))
    ap.add_argument("--demo", action="store_true", help="fake NAS + fake GPU box (no ssh/rsync/docker/Ollama); no token")
    ap.add_argument("--demo-no-ollama", action="store_true", help="with --demo: simulate Ollama stopped, so the TensorFold recipe row can be played")
    ap.add_argument("--port", type=int); ap.add_argument("--listen")
    a = ap.parse_args(argv)
    cfg = load_config(os.path.expanduser(a.config))
    if a.cmd == "fit":
        try: gb = float(a.arg)
        except (TypeError, ValueError): ap.error("fit needs a size in GB")
        k = nodes_needed(gb * 1e9, cfg["fit"]); print(f"{gb:g} GB -> {k if k else 'beyond'} node(s)"); return 0
    if a.cmd == "archs":
        rc, msg = probe_archs(cfg, a.arg or "vllm"); print(msg, file=sys.stderr if rc else sys.stdout); return rc
    if a.demo:
        import tempfile
        root = tempfile.mkdtemp(prefix="mcc-demo-")
        cfg.update(state_dir=f"{root}/state", local_dir=f"{root}/deck", protected=["qwen3:14b"],
                   fleet=[{"name": "node-1", "self": True, "note": "this box"}, {"name": "node-2", "ssh": "demo", "note": "peer"},
                          {"name": "node-3", "note": "planned"}, {"name": "node-4", "note": "planned"}],
                   apps={"image": {"label": "ComfyUI", "url": "#comfyui"}, "video": {"label": "ComfyUI", "url": "#comfyui"},
                         "speech": {"label": "the speech service", "url": "#speech"}})
        cfg["_recipes"] = {k: {**RECIPE_DEFAULTS, **v, "key": k} for k, v in DEMO_RECIPES.items()}
        cfg["_archs"] = {"vllm": set(DEMO_ENGINE_ARCHS), "sglang": set(DEMO_ENGINE_ARCHS)}; cfg["_archs_fixed"] = True
        demo_seed(cfg); deck = Deck(cfg, DemoSystem(cfg, ollama_up=not a.demo_no_ollama))
    else: deck = Deck(cfg)
    if a.cmd == "sync": print(json.dumps(deck.sync())); return 0
    if a.cmd == "summary": print(json.dumps(deck.summary(), indent=1)); return 0
    token = None if a.demo else read_token(cfg)
    host, port = a.listen or cfg["listen"], a.port or cfg["port"]
    srv = ThreadingHTTPServer((host, port), make_handler(deck, token, page_html())); deck.sampler(2 if a.demo else 5)
    print(f"mcc deck {VERSION} on http://{host}:{port}/" + ("  (DEMO MODE — nothing real is touched" + (", Ollama simulated as stopped)" if a.demo_no_ollama else ")") if a.demo else ""), file=sys.stderr)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
