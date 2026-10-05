#!/usr/bin/env python3
"""mcc_deck.py — the GPU-box side of the NAS Model Cassette Changer.

Your models live on a NAS. Your GPU box has fast local disk and limited memory. The deck treats every model on the NAS
like a cassette on a shelf:

  SHELF    the NAS index (library.json, built there by nas/mcc_index.py), copied here by `mcc_deck.py sync` every few
           minutes. The sync never wakes a sleeping NAS: while it is unreachable the last copy is shown with its age.
  TIERS    every runnable unit (a GGUF quant, or a whole safetensors folder) gets an estimated number of GPU nodes it needs.
  INSERT   copies the unit from the NAS to local disk (rsync, resumable) and starts it: GGUF -> Ollama, safetensors -> a vLLM
           container. Nothing is ever served straight off the network share.
  GUARD    an INSERT never stops anything you did not name. Protected models keep their memory reserved even when they are
           unloaded. If an insert would stop or evict anything, the server answers 409 with the exact list and does nothing
           until the request confirms every name. The plan is re-checked after the copy, and if a protected model was pushed
           out anyway the new cassette is removed again.
  EJECT    stops what the deck is playing; optionally deletes the local copy (separate confirm). The NAS copy is never touched.

Stdlib only (Python 3.9+). Talks to: ssh + rsync (to the NAS), the Ollama HTTP API, docker (vLLM), /proc/meminfo.

Usage:
  mcc_deck.py serve  [--config FILE] [--demo] [--port N] [--listen ADDR]
  mcc_deck.py sync   [--config FILE]          # what the timer runs
  mcc_deck.py summary [--config FILE]
  mcc_deck.py fit GB                          # how many nodes a GB-sized unit needs under your config
"""
import argparse, glob, hmac, json, os, re, secrets, shlex, shutil, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERSION = "1.0.0"
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
    "vllm": {"image": "vllm/vllm-openai:latest", "port": 8010, "gpu_memory_utilization": 0.85, "max_model_len": 32768,
             "trust_remote_code": False, "extra_args": [], "container": "mcc-vllm"},
    "new_days": 7,
}
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
    for k in ("local_dir", "state_dir", "token_file"): cfg[k] = os.path.expanduser(cfg[k])
    return cfg


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


def plan_insert(m, unit, res, cfg, busy=False):
    """What an INSERT of this unit would do, given a residents snapshot. Never acts. Fails closed."""
    prot = cfg["protected"]; cat = m.get("category"); fmt = (m.get("format") or "").lower(); k = unit.get("nodes")
    rt = [str(x).lower() for x in (m.get("runtime") or [])]
    if cat not in cfg["servable_categories"]: return {"refuse": f"{cat} model — the deck serves text models (Ollama / vLLM); run it in its own app"}
    if fmt not in ("gguf", "safetensors"): return {"refuse": f"format {fmt or '?'} — the deck serves gguf (Ollama) and safetensors (vLLM)"}
    if fmt == "gguf" and not ({"ollama", "llama.cpp"} & set(rt)): return {"refuse": f"runtime {', '.join(rt) or '?'} — not an Ollama model"}
    if fmt == "safetensors" and "vllm" not in rt: return {"refuse": f"runtime {', '.join(rt) or '?'} — not a vLLM model"}
    if m.get("status") != "complete": return {"refuse": f"not complete on the NAS ({m.get('status')})"}
    if k is None: return {"refuse": f"beyond the fleet (needs more than {cfg['fit']['max_nodes']} nodes by the estimate)"}
    if k > 1: return {"refuse": f"needs {k} nodes — multi-node serving is launched from its kit, not from the deck"}
    if busy: return {"refuse": "the deck is busy (an insert is running)"}
    if not res.get("ollama_ok"): return {"refuse": "cannot read Ollama right now — refusing rather than guessing what would be evicted"}
    missing = [p for p in prot if p not in res["tags_gib"]]
    if missing: return {"refuse": f"protected model(s) {', '.join(missing)} not found in Ollama — fix 'protected' in the config"}
    w = unit["bytes"] / 2**30; disk_need = w * (2.2 if fmt == "gguf" else 1.1)
    if res.get("free_disk_gib") is not None and res["free_disk_gib"] < disk_need:
        return {"refuse": f"not enough disk: needs ~{disk_need:.0f} GiB free, {res['free_disk_gib']:.0f} GiB free"}
    resident = {o["name"]: o["gib"] for o in res["ollama"]}
    if fmt == "safetensors":
        if res.get("vllm_running"): return {"refuse": "a vLLM cassette is already playing — EJECT it first"}
        ev = list(res["lanes"]) + list(resident)
        ev += [p for p in prot if p not in ev]               # vLLM takes most of GPU memory: protected models cannot run beside it
        return {"path": "vllm", "would_evict": ev, "note": "a vLLM cassette takes the GPU: every big lane and every Ollama model stops"}
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

    def meminfo(self):
        mem = {}
        try:
            with open("/proc/meminfo") as f:
                for l in f: k, v = l.split(":"); mem[k] = int(v.split()[0]) / 1048576
        except Exception: pass
        return mem

    def residents(self, state_dir):
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
        return {"lanes": lanes, "ollama": oll, "tags_gib": tags, "ollama_ok": ps is not None and tg is not None, "observed_gib": obs,
                "max_loaded": self.cfg["ollama_max_loaded"], "free_disk_gib": round(shutil.disk_usage(self.cfg["local_dir"]).free / 2**30, 1),
                "mem_avail_gib": round(mem.get("MemAvailable", 0), 1), "mem_total_gib": round(mem.get("MemTotal", 0), 1),
                "vllm_running": bool(_sh(["docker", "ps", "-q", "-f", f"name=^{self.cfg['vllm']['container']}$"], 8))}

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

    def serve_vllm(self, mid, f, evict):
        v = self.cfg["vllm"]; c = v["container"]
        for u in self.cfg["big_lanes"]:
            if u in evict: f.write(f"$ systemctl --user stop {u}\n"); _sh(["systemctl", "--user", "stop", u], 180)
        for o in (self.ollama("/api/ps") or {}).get("models", []):
            if o.get("name") in evict: self.unload(o["name"]); f.write(f"[unloaded {o['name']}]\n")
        served = mid.split("/")[1]
        cmd = ["docker", "run", "-d", "--rm", "--name", c, "--gpus", "all", "--ipc=host", "--network", "host",
               "-v", f"{self.cfg['local_dir']}:/models:ro", v["image"], "--model", f"/models/{mid}", "--served-model-name", served,
               "--port", str(v["port"]), "--max-model-len", str(v["max_model_len"]), "--gpu-memory-utilization", str(v["gpu_memory_utilization"])]
        cmd += (["--trust-remote-code"] if v.get("trust_remote_code") else []) + list(v.get("extra_args") or [])
        _sh(["docker", "rm", "-f", c], 60); f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n"); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if r.returncode: return r.returncode, None
        for i in range(360):
            time.sleep(5)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{v['port']}/v1/models", timeout=3).read()
                f.write(f"[vLLM {served} answering on :{v['port']} after {5*(i+1)} s]\n"); return 0, {"runtime": "vllm", "port": v["port"], "name": served}
            except Exception: pass
            if not _sh(["docker", "ps", "-q", "-f", f"name=^{c}$"], 8) and i > 2: f.write(f"[the {c} container exited — docker logs {c}]\n"); return 1, None
            if i % 12 == 11: f.write(f"  … waiting for /v1/models ({5*(i+1)} s)\n"); f.flush()
        f.write("[timed out waiting for vLLM]\n"); return 1, None

    def stop_vllm(self):
        c = self.cfg["vllm"]["container"]
        if _sh(["docker", "ps", "-q", "-f", f"name=^{c}$"], 8): _sh(["docker", "rm", "-f", c], 120); return True
        return False

    def ollama_rm(self, name): _sh(["ollama", "rm", name], 60)


# ── the deck ───────────────────────────────────────────────────────────────────────────────────────────────────────
class Deck:
    def __init__(self, cfg, system=None):
        self.cfg = cfg; self.sys = system or System(cfg); s = cfg["state_dir"]
        for d in (s, cfg["local_dir"]): os.makedirs(d, exist_ok=True)
        self.LIB, self.UNITS, self.SEEN, self.NAS = f"{s}/library.json", f"{s}/units.json", f"{s}/seen.json", f"{s}/nas_status.json"
        self.ACTIONS, self.PULL_LOG, self.PLAYING = f"{s}/actions.jsonl", f"{s}/pull.log", f"{s}/playing.json"
        self.lock = threading.Lock(); self.job = {"running": False, "what": "", "t0": 0, "progress": ""}; self.syncing = False

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

    def units_for(self, m, units_idx):
        us = (units_idx or {}).get(m.get("id")) if (m.get("format") or "").lower() == "gguf" else None
        exact = bool(us)
        us = [dict(u) for u in us] if us else [{"key": "(whole variant)", "quant": m.get("quant") or m.get("format") or "", "files": [], "bytes": m.get("size_bytes") or 0}]
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
        ms, d = self.models(); ui = (_rjson(self.UNITS, {}) or {}).get("units"); seen = _rjson(self.SEEN, {})
        cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - self.cfg["new_days"] * 86400)); out = []
        for m in ms:
            mid = m.get("id"); us, exact = self.units_for(m, ui); ks = [u["nodes"] for u in us if u["nodes"]]
            out.append({"id": mid, "maker": m.get("maker"), "model": m.get("model"), "variant": m.get("variant"), "category": m.get("category"),
                        "format": m.get("format"), "quant": m.get("quant"), "status": m.get("status"), "source": m.get("source"),
                        "size_gb": round((m.get("size_bytes") or 0) / 1e9, 1), "tier": min(ks) if ks else None, "units": us,
                        "units_exact": exact or (m.get("format") or "").lower() != "gguf", "kits": self.kits_for(d, m.get("model")),
                        "first_seen": seen.get(mid), "new": (seen.get(mid) or "") >= cutoff, "servable": m.get("category") in self.cfg["servable_categories"],
                        "plans": {u["key"]: plan_insert(m, u, res, self.cfg, self.job["running"]) for u in us}, **self.local_state(mid, us)})
        return sorted(out, key=lambda r: (r["tier"] or 9, not r["servable"], (r["maker"] or "").lower(), (r["model"] or "").lower(), r["variant"] or "")), d

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
                            "ollama": res["ollama"], "vllm_running": res["vllm_running"], "max_loaded": res.get("max_loaded")})
            elif not n.get("ssh"): out.append({**n, "online": False, "why": "not configured"})
            else: out.append({**n, **self.sys.remote(n)})
        return out

    def state(self, force=False):
        if force: self.sync_bg()
        res = self.sys.residents(self.cfg["state_dir"]); rows, d = self.rows(res); slots = self.slots(res)
        mx = int(self.cfg["fit"]["max_nodes"]); tiers = {str(k): sum(1 for r in rows if r["tier"] == k) for k in range(1, mx + 1)}
        tiers["beyond"] = sum(1 for r in rows if r["tier"] is None)
        hist = []
        if os.path.exists(self.ACTIONS):
            with open(self.ACTIONS) as f: hist = [json.loads(l) for l in f if l.strip()][-12:]
        f = self.cfg["fit"]
        return {"version": VERSION, "asof": time.strftime("%Y-%m-%dT%H:%M:%S"), "library_built": d.get("built") if isinstance(d, dict) else None,
                "nas": _rjson(self.NAS, {}), "syncing": self.syncing, "n_shelf": len(rows), "shelf": rows, "tiers": tiers, "max_nodes": mx,
                "n_new": sum(1 for r in rows if r["new"]), "slots": slots, "online_nodes": sum(1 for s in slots if s.get("online")),
                "protected": self.cfg["protected"], "residents": res, "job": dict(self.job), "playing": _rjson(self.PLAYING, {}),
                "local": self.local_cassettes(rows), "history": hist, "new_days": self.cfg["new_days"],
                "fit": {**f, "num_ctx": self.cfg["num_ctx"],
                        "rule": f"smallest k with max(W + {f['rank_overhead_gib']}·k + {f['kv_floor_gib']}, {f['weight_factor']}·W) ≤ {f['gpu_budget_gib']}·k  (W = unit weights, GiB)"}}

    def summary(self):
        ms, d = self.models(); ui = (_rjson(self.UNITS, {}) or {}).get("units"); t = {}
        for m in ms:
            ks = [u["nodes"] for u in self.units_for(m, ui)[0] if u["nodes"]]; k = str(min(ks)) if ks else "beyond"; t[k] = t.get(k, 0) + 1
        nas = _rjson(self.NAS, {})
        return {"total": len(ms), "tiers": t, "built": d.get("built") if isinstance(d, dict) else None, "nas_awake": nas.get("awake"), "asleep_since": nas.get("asleep_since")}

    # actions -------------------------------------------------------------------------------------------------------
    def log(self, **r):
        r["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.ACTIONS, "a") as f: f.write(json.dumps(r) + "\n")

    def insert(self, mid, unit_key=None, confirm_evict=None):
        ms = {m["id"]: m for m in self.models()[0]}
        if mid not in ms: return {"error": "unknown model", "status": 404}
        m = ms[mid]; units, _ = self.units_for(m, (_rjson(self.UNITS, {}) or {}).get("units"))
        unit = next((u for u in units if u["key"] == unit_key), None) if unit_key else (units[0] if len(units) == 1 else None)
        if not unit: return {"error": "choose which file / quant to insert", "units": [u["key"] for u in units], "status": 400}
        ok = {x for x in confirm_evict if isinstance(x, str)} if isinstance(confirm_evict, list) else set()
        with self.lock:
            if self.job["running"]: return {"error": "deck busy", "job": dict(self.job), "status": 409}
            res = self.sys.residents(self.cfg["state_dir"]); plan = plan_insert(m, unit, res, self.cfg)
            if plan.get("refuse"): return {"error": plan["refuse"], "status": 409}
            ev = plan.get("would_evict") or []
            if [n for n in ev if n not in ok]:
                return {"error": "this insert would STOP: " + ", ".join(ev) + " — nothing was done. Confirm each one by name to go ahead.",
                        "would_evict": ev, "plan": plan, "status": 409}
            self.job.update(running=True, what=f"INSERT {mid} [{unit['key']}]", t0=time.time(), progress="")
        threading.Thread(target=self._insert_job, args=(m, unit, ok), daemon=True).start()
        return {"started": True, "what": self.job["what"], "evicting_after_copy": ev}

    def _insert_job(self, m, unit, ok):
        mid = m["id"]; rc, evicted, note = 1, [], ""
        try:
            with open(self.PULL_LOG, "w") as f:
                have = self.local_state(mid, [unit])
                rc = 0 if (unit["key"] in have["local_units"] or (not unit.get("files") and have["local"])) else self.sys.pull(mid, unit, f, self.job, self.cfg["state_dir"])
                if rc: f.write(f"[copy FAILED rc={rc}]\n")
                plan2 = {}
                if rc == 0:                                       # RE-CHECK after the (minutes-long) copy, before touching anything
                    with self.lock: res2 = self.sys.residents(self.cfg["state_dir"]); plan2 = plan_insert(m, unit, res2, self.cfg)
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
                elif rc == 0:
                    evicted = plan2["would_evict"]; rc, playing = self.sys.serve_vllm(mid, f, evicted)
                    if rc == 0: _wjson(self.PLAYING, {"id": mid, "unit": unit["key"], **playing}); f.write(f"[playing: vLLM {playing['name']}]\n")
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
            if self.sys.stop_vllm(): done.append("stopped the vLLM cassette")
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
    """No ssh, no rsync, no docker, no Ollama: a believable box with 128 GB unified memory and two Ollama residents."""
    def __init__(self, cfg):
        super().__init__(cfg); self.oll = [{"name": "qwen3:14b", "gib": 10.6}, {"name": "nomic-embed-text:latest", "gib": 0.6}]
        self.tags = {"qwen3:14b": 9.3, "nomic-embed-text:latest": 0.3, "llama3.2:3b": 2.0}; self.vllm = None

    def residents(self, state_dir):
        used = sum(o["gib"] for o in self.oll) + (88.0 if self.vllm else 0)
        return {"lanes": [], "ollama": [dict(o) for o in self.oll], "tags_gib": dict(self.tags), "ollama_ok": True, "observed_gib": {"qwen3:14b": 10.6},
                "max_loaded": 3, "free_disk_gib": 1490.0, "mem_avail_gib": round(116.0 - used, 1), "mem_total_gib": 121.7, "vllm_running": bool(self.vllm)}

    def remote(self, node): return {"online": True, "mem_avail_gib": 109, "why": ""}

    def ollama(self, path, body=None, timeout=4):
        if path == "/api/ps": return {"models": [{"name": o["name"], "size": int(o["gib"] * 2**30)} for o in self.oll]}
        return {}

    def pull(self, mid, unit, f, job, state_dir):
        total = unit["bytes"]; f.write(f"$ rsync -rLt -s --partial --info=progress2 nas:/srv/models-nas/library/{mid}/ ~/models/deck/{mid}/\n")
        for pct in range(0, 101, 4):
            line = f"{total*pct/100/1e9:8.1f}G {pct:3d}%  812.4MB/s    0:00:{max(0, 25-pct//4):02d}"; job["progress"] = line; f.write(line + "\n"); f.flush(); time.sleep(0.35)
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

    def serve_vllm(self, mid, f, evict):
        self.oll = [o for o in self.oll if o["name"] not in evict]; time.sleep(2); self.vllm = mid
        f.write(f"$ docker run -d --name mcc-vllm … vllm/vllm-openai:latest --model /models/{mid}\n[vLLM answering on :8010 after 95 s]\n")
        return 0, {"runtime": "vllm", "port": 8010, "name": mid.split("/")[1]}

    def stop_vllm(self):
        was = bool(self.vllm); self.vllm = None; return was

    def ollama_rm(self, name): self.tags.pop(name, None)


def demo_seed(cfg):
    """Write a fake library + units + local copies into the demo state dir."""
    s = cfg["state_dir"]; os.makedirs(s, exist_ok=True); models, units = [], {}; now = time.strftime("%Y-%m-%d")
    for cat, maker, model, var, pub, fmt, q, gb, quants in DEMO_SHELF:
        mid = f"{maker}/{model}/{var}"; size = int((sum(x[1] for x in quants) if quants else gb) * 1e9)
        rt = ["llama.cpp", "ollama"] if fmt == "gguf" else (["vllm", "sglang", "transformers"] if cat in ("LLMs", "Coding", "Small-On-Device") else ["diffusers", "transformers"])
        models.append({"id": mid, "maker": maker, "model": model, "variant": var, "publisher": pub, "category": cat, "format": fmt, "quant": q,
                       "runtime": rt, "size_bytes": size, "status": "complete", "source": f"https://huggingface.co/{pub}/{model}"})
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
            {"action": "EJECT", "id": "Mistral/Devstral-Small-2507/unsloth-GGUF", "remove_local": False, "done": ["unloaded deck/mistral_devstral-small-2507"], "ts": f"{now}T11:03:02"}]
    with open(f"{s}/actions.jsonl", "w") as f: f.write("".join(json.dumps(h) + "\n" for h in hist))


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
            else: return self._send(404, {"error": "not found"})
            self._send(r.pop("status", 200) if "error" in r else 200, r)
    return H


def page_html():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "deck.html"), encoding="utf-8") as f: return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description="NAS Model Cassette Changer — the deck")
    ap.add_argument("cmd", choices=["serve", "sync", "summary", "fit"]); ap.add_argument("gb", nargs="?", type=float)
    ap.add_argument("--config", default=os.environ.get("MCC_DECK_CONFIG", "~/.config/mcc/deck.json"))
    ap.add_argument("--demo", action="store_true", help="fake NAS + fake GPU box (no ssh/rsync/docker/Ollama); no token")
    ap.add_argument("--port", type=int); ap.add_argument("--listen")
    a = ap.parse_args(argv)
    cfg = load_config(os.path.expanduser(a.config))
    if a.cmd == "fit":
        if a.gb is None: ap.error("fit needs a size in GB")
        k = nodes_needed(a.gb * 1e9, cfg["fit"]); print(f"{a.gb:g} GB -> {k if k else 'beyond'} node(s)"); return 0
    if a.demo:
        import tempfile
        root = tempfile.mkdtemp(prefix="mcc-demo-")
        cfg.update(state_dir=f"{root}/state", local_dir=f"{root}/deck", protected=["qwen3:14b"],
                   fleet=[{"name": "node-1", "self": True, "note": "this box"}, {"name": "node-2", "ssh": "demo", "note": "peer"},
                          {"name": "node-3", "note": "planned"}, {"name": "node-4", "note": "planned"}])
        demo_seed(cfg); deck = Deck(cfg, DemoSystem(cfg))
    else: deck = Deck(cfg)
    if a.cmd == "sync": print(json.dumps(deck.sync())); return 0
    if a.cmd == "summary": print(json.dumps(deck.summary(), indent=1)); return 0
    token = None if a.demo else read_token(cfg)
    host, port = a.listen or cfg["listen"], a.port or cfg["port"]
    srv = ThreadingHTTPServer((host, port), make_handler(deck, token, page_html()))
    print(f"mcc deck {VERSION} on http://{host}:{port}/" + ("  (DEMO MODE — nothing real is touched)" if a.demo else ""), file=sys.stderr)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
