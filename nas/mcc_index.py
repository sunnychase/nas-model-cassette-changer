#!/usr/bin/env python3
"""mcc_index.py — the NAS side of the NAS Model Cassette Changer.

Walks the folders you download models into ("staging"), works out what each model is, and publishes:

  <library>/<Maker>/<Model>/<Variant>        relative symlinks into staging (valid over NFS / SMB)
  <library>/_kits/<repo>                     git clones of serving recipes ("kits")
  <library>/_tiers/<N>-node/<Maker>/...      the same models grouped by how many GPU nodes they need (optional)
  <library>/_index/library.json              the machine index the deck reads
  <library>/_index/library.md                the same as a human-readable table
  <model folder>/manifest.json               the record, next to the weights
  <model folder>/ABOUT-MODEL.md              what it is, which box it fits, how to run it (optional)

It never moves, renames or deletes a model file. The only things it removes are symlinks it created itself
that no longer point at anything. Stdlib only. Safe to run every few minutes (see systemd/mcc-index.timer).

Usage:
  mcc_index.py [--config /etc/mcc/nas.json] [--no-about] [--no-tiers] [--dry-run]
"""
import argparse, hashlib, html, json, os, re, sys, time

# ── configuration ──────────────────────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "data_root": "/srv/models-nas",                 # everything below is relative to this
    "library_dir": "library",                       # the generated maker-first view + _index/
    "staging": [                                    # where models are downloaded to; layout = folder levels below path
        {"path": "Models", "layout": "category/org/model"},
    ],
    "families": [],                                 # [regex on the folder name, Maker, canonical model or null]
    "org_maker": {},                                # extra Hugging Face org -> Maker names (merged over the built-ins)
    "node": {"name": "node", "usable_gb": 95, "max_nodes": 4},   # one GPU box; used for tiers and ABOUT fit tables
    "client_mount": "/mnt/nas",                     # where clients mount data_root (only used in ABOUT setup text)
    "tiers": True,
    "about": True,
}

ORG_MAKER = {  # public Hugging Face orgs -> a friendly maker name; anything else keeps its org name
    "Qwen": "Qwen", "deepseek-ai": "DeepSeek", "zai-org": "Zhipu-GLM", "THUDM": "Zhipu-GLM", "google": "Google",
    "moonshotai": "Moonshot", "mistralai": "Mistral", "openbmb": "OpenBMB", "black-forest-labs": "BlackForestLabs",
    "Lightricks": "Lightricks", "MiniMaxAI": "MiniMax", "Wan-AI": "Wan-AI", "nvidia": "NVIDIA", "openai": "OpenAI",
    "tencent": "Tencent", "stabilityai": "StabilityAI", "facebook": "Meta", "meta-llama": "Meta", "baidu": "Baidu",
    "jinaai": "Jina", "BAAI": "BAAI", "microsoft": "Microsoft", "ibm-granite": "IBM", "allenai": "AllenAI",
    "HuggingFaceTB": "HuggingFace", "CohereForAI": "Cohere", "CohereLabs": "Cohere", "xai-org": "xAI",
}

CAT_TASK = {"LLMs": "text-generation", "Coding": "text-generation", "Small-On-Device": "text-generation",
            "Image": "text-to-image", "Video": "text-to-video", "Speech-ASR": "automatic-speech-recognition",
            "Speech-TTS": "text-to-speech", "Music-Audio": "text-to-audio", "OCR": "image-to-text",
            "Embeddings-Reranking": "feature-extraction"}

TASK = {  # pipeline_tag -> (plain label, what it is for)
    "text-generation": ("Text LLM", "chat, reasoning, summarising, coding and tool use over text"),
    "image-text-to-text": ("Vision-language LLM", "chat and reasoning over text and images"),
    "any-to-any": ("Multimodal model", "mixed text / image / audio in and out"),
    "text-to-image": ("Image generator", "creating images from a text prompt"),
    "image-to-image": ("Image editor", "editing or transforming an existing image"),
    "text-to-video": ("Video generator", "creating short video clips from a text prompt"),
    "image-to-video": ("Video generator", "animating a still image into a clip"),
    "automatic-speech-recognition": ("Speech-to-text", "transcribing audio to text"),
    "text-to-speech": ("Text-to-speech", "turning text into spoken audio"),
    "text-to-audio": ("Audio / music generator", "generating music or sound from a prompt"),
    "sentence-similarity": ("Embedding model", "vectors for search, clustering and retrieval (RAG)"),
    "feature-extraction": ("Embedding model", "vectors for search, clustering and retrieval (RAG)"),
    "text-ranking": ("Reranker", "re-ordering search results by relevance"),
    "text-classification": ("Classifier", "scoring or labelling text"),
    "image-to-text": ("OCR / captioning model", "reading text out of images and documents"),
}

WEIGHT_EXT = (".safetensors", ".gguf", ".bin", ".pt", ".pth")
NOTES = "## Notes (hand-written)"
ABOUT = "ABOUT-MODEL.md"


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path and os.path.exists(path):
        with open(path) as f: user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict): cfg[k].update(v)
            else: cfg[k] = v
    elif path and path != "/etc/mcc/nas.json":
        sys.exit(f"config not found: {path}")
    cfg["org_maker_all"] = {**ORG_MAKER, **(cfg.get("org_maker") or {})}
    cfg["families_rx"] = [(re.compile(rx, re.I), maker, canon) for rx, maker, canon in (cfg.get("families") or [])]
    return cfg


# ── what a folder is ───────────────────────────────────────────────────────────────────────────────────────────────
def weight_files(d):
    out = []
    for root, dirs, fs in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        out += [os.path.relpath(os.path.join(root, f), d) for f in fs if f.endswith(WEIGHT_EXT)]
    return sorted(out)


def fmt_of(files):
    names = [f.lower() for f in files]
    if any(n.endswith(".gguf") for n in names): return "gguf"
    if any(n.endswith(".safetensors") for n in names): return "safetensors"
    return "other"


def quant_of(d, files, fmt):
    q = None
    cfgp = os.path.join(d, "config.json")
    if os.path.exists(cfgp):
        try:
            with open(cfgp) as f: c = json.load(f)
            qc = c.get("quantization_config") or {}
            q = (qc.get("quant_algo") or qc.get("quant_method") or "").upper() or None
            if not q and c.get("torch_dtype"): q = str(c["torch_dtype"]).replace("torch.", "")
        except Exception: pass
    n = os.path.basename(d).upper()
    for k in ("NVFP4", "MXFP4", "FP8", "INT8", "AWQ", "GPTQ", "EXL3", "EXL2", "MXFP8", "BF16"):
        if k in n and (not q or q in ("BFLOAT16", "FLOAT16")): q = k
    if fmt == "gguf" and not q:
        m = re.search(r"(Q\d_K_[A-Z]+|Q\d_K|Q\d_\d|IQ\d_[A-Z]+|F16|BF16)", " ".join(files).upper()); q = m.group(1) if m else None
    return q


def runtime_of(fmt, q, category):
    if fmt == "gguf":
        return ["llama.cpp", "ollama"] if CAT_TASK.get(category, "text-generation") == "text-generation" else ["llama.cpp", "ComfyUI-GGUF"]
    if q and q.startswith("EXL"): return ["exllama", "tabbyAPI"]
    if CAT_TASK.get(category, "text-generation") != "text-generation": return ["diffusers", "transformers"]
    return ["vllm", "sglang", "transformers"]


def status_of(d, files):
    partial = False
    for root, _, fs in os.walk(d):
        if any(f.endswith((".incomplete", ".part", ".aria2")) for f in fs): partial = True; break
    if partial: return "downloading"
    return "complete" if files else "empty"


def du(d):
    t = 0
    for root, _, fs in os.walk(d):
        for f in fs:
            try: t += os.lstat(os.path.join(root, f)).st_size
            except OSError: pass
    return t


def classify(name, org, cfg):
    for rx, maker, canon in cfg["families_rx"]:
        if rx.match(name): return maker, (canon or name)
    return cfg["org_maker_all"].get(org, org), name


def variant_of(name, org, maker, model, cfg):
    tail = name[len(model):].lstrip("-_ ") if name.lower().startswith(model.lower()) else name
    if cfg["org_maker_all"].get(org, org) == maker:             # the maker's own upload (an unmapped org is its own maker)
        return tail.upper() if tail else "original"
    return f"{org}-{tail}".strip("-") if tail else org


def source_of(d, org, name):
    try:
        with open(os.path.join(d, ".mcc-source")) as f: repo = f.read().strip().split("@")[0]
        if repo: return f"https://huggingface.co/{repo}", repo.split("/")[0]
    except OSError: pass
    return f"https://huggingface.co/{org}/{name}", org


def git_url(d):
    try:
        with open(os.path.join(d, ".git", "config")) as f:
            for ln in f:
                if ln.strip().startswith("url ="): return ln.split("=", 1)[1].strip()
    except OSError: pass
    return None


def kit_nodes(repo, contents):
    m = re.search(r"(\d)x", repo, re.I) or re.search(r"tp(\d)", " ".join(contents), re.I)
    return int(m.group(1)) if m else None


def nodes_needed(gb, cfg):
    per = float(cfg["node"]["usable_gb"])
    for k in range(1, int(cfg["node"]["max_nodes"]) + 1):
        if gb <= per * k: return k
    return None


# ── the walk ───────────────────────────────────────────────────────────────────────────────────────────────────────
def scan(cfg):
    data = cfg["data_root"]; records, kits = [], []
    for st in cfg["staging"]:
        base = os.path.join(data, st["path"]); layout = st.get("layout", "org/model").split("/"); depth = len(layout)
        if not os.path.isdir(base): continue
        for cur, dirs, _ in os.walk(base):
            dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in ("__pycache__", "runtime"))
            parts = os.path.relpath(cur, base).split("/")
            if parts == ["."]: continue
            sub = os.listdir(cur)
            if ".git" in sub and len(parts) <= depth and not any(x.endswith(WEIGHT_EXT) for x in sub):
                dirs[:] = []                                       # a git clone without weights = a serving kit (recipe), not a model
                kits.append({"id": f"_kits/{parts[-1]}", "repo": parts[-1], "nodes": kit_nodes(parts[-1], sub),
                             "path": os.path.relpath(cur, data), "source": git_url(cur),
                             "has_model": any(x in sub for x in ("model", "base-model"))})
                continue
            if len(parts) != depth: continue
            dirs[:] = []
            seg = dict(zip(layout, parts))
            cat = seg.get("category") or st.get("category", "LLMs")
            name = seg.get("model"); org = seg.get("org") or st.get("org") or "local"
            src, org = source_of(cur, org, name)
            files = weight_files(cur); fmt = fmt_of(files); q = quant_of(cur, files, fmt)
            maker, model = classify(name, org, cfg); var = variant_of(name, org, maker, model, cfg)
            size = du(cur)
            records.append({"id": f"{maker}/{model}/{var}", "maker": maker, "model": model, "variant": var, "publisher": org,
                            "category": cat, "format": fmt, "quant": q, "runtime": runtime_of(fmt, q, cat), "size_bytes": size,
                            "n_weight_files": len(files), "status": status_of(cur, files), "path": os.path.relpath(cur, data),
                            "source": src, "nodes_est": nodes_needed(size / 1e9, cfg), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
    seen, out = set(), []
    for r in records:                                    # two staging folders resolving to the same id: keep the first, suffix the rest
        i, n = r["id"], 2
        while r["id"] in seen: r["id"] = f"{i}-{n}"; r["variant"] = r["id"].split("/", 2)[2]; n += 1
        seen.add(r["id"]); out.append(r)
    return sorted(out, key=lambda r: r["id"].lower()), kits


# ── outputs ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _link(target_abs, dst, live):
    live.add(dst); os.makedirs(os.path.dirname(dst), exist_ok=True)
    rel = os.path.relpath(target_abs, os.path.dirname(dst))
    if os.path.islink(dst):
        if os.readlink(dst) == rel: return 0
        os.unlink(dst)
    elif os.path.exists(dst):
        return 0                                         # a real folder lives here: never touch it
    os.symlink(rel, dst); return 1


def build_views(cfg, records, kits):
    data = cfg["data_root"]; lib = os.path.join(data, cfg["library_dir"]); made = 0; live = set()
    for r in records:
        made += _link(os.path.join(data, r["path"]), os.path.join(lib, r["maker"], r["model"], r["variant"]), live)
    for k in kits:
        made += _link(os.path.join(data, k["path"]), os.path.join(lib, "_kits", k["repo"]), live)
    if cfg.get("tiers"):
        for r in records:
            t = f"{r['nodes_est']}-node" if r["nodes_est"] else "beyond"
            made += _link(os.path.join(data, r["path"]), os.path.join(lib, "_tiers", t, r["maker"], r["model"], r["variant"]), live)
    pruned = 0
    for cur, dirs, fs in os.walk(lib):                   # remove only OUR symlinks that are dangling or no longer produced
        if "/_index" in cur: continue
        for x in dirs + fs:
            p = os.path.join(cur, x)
            if os.path.islink(p) and (not os.path.exists(p) or p not in live): os.unlink(p); pruned += 1
    for cur, dirs, fs in os.walk(lib, topdown=False):    # and the empty folders that leaves behind
        if cur != lib and not os.listdir(cur) and "/_index" not in cur and not os.path.islink(cur):
            try: os.rmdir(cur)
            except OSError: pass
    return made, pruned


def write_index(cfg, records, kits):
    lib = os.path.join(cfg["data_root"], cfg["library_dir"]); idx_dir = os.path.join(lib, "_index"); os.makedirs(idx_dir, exist_ok=True)
    tiers = {}
    for r in records: tiers[str(r["nodes_est"] or "beyond")] = tiers.get(str(r["nodes_est"] or "beyond"), 0) + 1
    idx = {"schema": 1, "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "library_dir": cfg["library_dir"],
           "node": cfg["node"], "n_models": len(records), "n_complete": sum(r["status"] == "complete" for r in records),
           "total_bytes": sum(r["size_bytes"] for r in records), "makers": sorted({r["maker"] for r in records}),
           "tiers": tiers, "models": records, "kits": kits}
    tmp = os.path.join(idx_dir, "library.json.tmp")
    with open(tmp, "w") as f: json.dump(idx, f, indent=1)
    os.replace(tmp, os.path.join(idx_dir, "library.json"))
    with open(os.path.join(idx_dir, "library.md"), "w") as f:
        f.write(f"# Model library — {idx['built']} — {idx['n_models']} models ({idx['n_complete']} complete), {idx['total_bytes']/1e12:.2f} TB\n\n"
                "| maker | model | variant | format/quant | runtime | size | nodes | status |\n|---|---|---|---|---|---|---|---|\n")
        for r in records:
            f.write(f"| {r['maker']} | {r['model']} | {r['variant']} | {r['format']}/{r['quant'] or '-'} | {', '.join(r['runtime'])} | "
                    f"{r['size_bytes']/1e9:.1f} GB | {r['nodes_est'] or '>'+str(cfg['node']['max_nodes'])} | {r['status']} |\n")
    for r in records:                                    # manifest next to the weights (only rewritten when it changed)
        mp = os.path.join(cfg["data_root"], r["path"], "manifest.json")
        try:
            old = json.load(open(mp)) if os.path.exists(mp) else {}
            strip = lambda d: {k: v for k, v in d.items() if k != "updated"}
            if strip(old) != strip(r):
                with open(mp, "w") as f: json.dump(r, f, indent=1)
        except Exception: pass
    return idx


# ── ABOUT-MODEL.md ─────────────────────────────────────────────────────────────────────────────────────────────────
def front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n?", text, re.S); out = {}
    if not m: return out, text
    y = m.group(1)
    for k in ("pipeline_tag", "license", "license_name"):
        mm = re.search(rf"^{k}\s*:\s*(.+)$", y, re.M)
        if mm: out[k] = mm.group(1).strip().strip("'\"")
    for k in ("base_model", "language"):
        mm = re.search(rf"^{k}\s*:\s*(.*)$", y, re.M)
        if not mm: continue
        v = mm.group(1).strip()
        if v.startswith("["): out[k] = [x.strip().strip("'\"") for x in v.strip("[]").split(",") if x.strip()]
        elif v: out[k] = [v.strip("'\"")]
        else:
            blk = re.search(rf"^{k}\s*:\s*\n((?:\s*-\s*.+\n?)+)", y, re.M)
            out[k] = [re.sub(r"^\s*-\s*", "", l).strip().strip("'\"") for l in blk.group(1).splitlines()] if blk else []
    return out, text[m.end():]


def first_paragraph(body):
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S); body = re.sub(r"<[^>]+>", "", body)
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if not p or p.startswith(("#", "|", "```", "![", "[![", ">", "- ", "* ")) or len(p) < 60: continue
        p = html.unescape(p); p = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", p); p = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", p)
        p = re.sub(r"[*_`]", "", p); p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 60: return (p[:560].rsplit(" ", 1)[0] + " …") if len(p) > 560 else p
    return None


def cfg_facts(d):
    out = {}
    p = os.path.join(d, "config.json")
    if os.path.exists(p):
        try:
            with open(p) as f: c = json.load(f)
            t = c.get("text_config") or {}
            out["arch"] = (c.get("architectures") or [None])[0]
            ctx = c.get("max_position_embeddings") or t.get("max_position_embeddings")
            if ctx: out["ctx"] = int(ctx)
            if c.get("vision_config"): out["vision"] = True
        except Exception: pass
    if os.path.exists(os.path.join(d, "model_index.json")):
        try:
            with open(os.path.join(d, "model_index.json")) as f: out["diffusers"] = json.load(f).get("_class_name") or True
        except Exception: out["diffusers"] = True
    return out


def biggest(d, exts):
    best = (0, None)
    for r, ds, fs in os.walk(d):
        ds[:] = [x for x in ds if not x.startswith(".")]
        for f in fs:
            if f.endswith(exts):
                try: s = os.path.getsize(os.path.join(r, f))
                except OSError: continue
                if s > best[0]: best = (s, os.path.relpath(os.path.join(r, f), d))
    return best[1]


def setup_text(rec, d, cf, task, cfg):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{rec['model']}-{rec['variant']}").strip("-")
    lib = f"{cfg['client_mount']}/{cfg['library_dir']}/{rec['id']}"; local = f"~/models/{safe}"
    L = ["### 1. Copy it to the GPU box first (never serve a model straight off the network share)", "```bash",
         f"rsync -a --info=progress2 '{lib}/' {local}/", "```", "The cassette deck does this for you when you press INSERT.", ""]
    fmt = rec["format"]; q = (rec.get("quant") or "").upper()
    if fmt == "gguf":
        f = biggest(d, (".gguf",)) or "<file>.gguf"; m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", f)
        first = re.sub(r"-\d{5}-of-", "-00001-of-", f) if m else f
        mm = [x for x in os.listdir(d) if "mmproj" in x.lower()] if os.path.isdir(d) else []
        L += ["### 2. Run with llama.cpp", "```bash", f"llama-server -m {local}/{first} -c 8192 -ngl 99 --port 8080" + (f" --mmproj {local}/{mm[0]}" if mm else ""), "```"]
        if m: L += [f"Split GGUF: point at the first shard (`-00001-of-{m.group(2)}`); llama.cpp loads the rest."]
        L += ["", "### or with Ollama", "```bash", f"printf 'FROM {local}/{first}\\nPARAMETER num_ctx 8192\\n' > Modelfile && ollama create {safe.lower()} -f Modelfile", "```"]
    elif q.startswith("EXL"):
        L += ["### 2. Run with ExLlama / TabbyAPI (EXL formats are ExLlama-only)", "```bash", f"# TabbyAPI config.yml:  model_dir: ~/models   model_name: {safe}", "python main.py", "```"]
    elif cf.get("diffusers") or task in ("text-to-image", "image-to-image", "text-to-video", "image-to-video"):
        cls = cf.get("diffusers") if isinstance(cf.get("diffusers"), str) and cf["diffusers"].isidentifier() else "DiffusionPipeline"
        L += ["### 2. Run with diffusers", "```python", "import torch", f"from diffusers import {cls} as P",
              f"pipe = P.from_pretrained('{local}', torch_dtype=torch.bfloat16).to('cuda')", "out = pipe(prompt='a lighthouse at dusk, studio light')", "```",
              "ComfyUI works too: put the folder (or its single-file checkpoint) under `ComfyUI/models/`."]
    elif task == "automatic-speech-recognition":
        L += ["### 2. Run with transformers", "```python", "from transformers import pipeline",
              f"asr = pipeline('automatic-speech-recognition', model='{local}', device='cuda')", "print(asr('audio.wav')['text'])", "```"]
    elif task in ("sentence-similarity", "feature-extraction", "text-ranking", "text-classification"):
        L += ["### 2. Run", "```python", "from sentence_transformers import SentenceTransformer", f"m = SentenceTransformer('{local}')", "print(m.encode(['hello world']).shape)", "```"]
    else:
        ctx = cf.get("ctx"); mml = min(ctx, 32768) if ctx else 8192
        L += ["### 2. Serve with vLLM (OpenAI-compatible API)", "```bash", f"vllm serve {local} --served-model-name {safe.lower()} --max-model-len {mml} --port 8000", "```"]
        if q in ("NVFP4", "MXFP4", "FP8", "MXFP8"): L += [f"{q} needs a recent GPU generation with native {q} kernels; check your card before inserting."]
        if rec.get("nodes_est") and rec["nodes_est"] > 1:
            L += [f"Larger than one {cfg['node']['name']}: serve it tensor-parallel across {rec['nodes_est']} nodes, or pick a smaller quant."]
    return "\n".join(L)


def render_about(rec, d, cfg):
    fm, body = {}, ""
    rp = os.path.join(d, "README.md")
    if os.path.exists(rp):
        try:
            with open(rp, errors="ignore") as f: fm, body = front_matter(f.read(40000))
        except Exception: pass
    cf = cfg_facts(d); task = fm.get("pipeline_tag") or CAT_TASK.get(rec.get("category"), "text-generation")
    if cf.get("vision") and task == "text-generation": task = "image-text-to-text"
    label, purpose = TASK.get(task, ("Model", "see the model card"))
    gb = rec["size_bytes"] / 1e9; own = rec["variant"] == "original" or cfg["org_maker_all"].get(rec["publisher"]) == rec["maker"]
    L = [f"# {rec['model']} — {rec['variant']}", "",
         f"**{label}** from **{rec['maker']}**" + ("" if own else f", repackaged by **{rec['publisher']}**") +
         (f" as **{rec['quant']}**" if rec.get("quant") else "") + f" · {gb:,.1f} GB · for {purpose}.", ""]
    para = first_paragraph(body) if body else None
    if para: L += ["## What it is (from the model card)", para, ""]
    L += ["## Facts", "| | |", "|---|---|", f"| Library path | `{cfg['library_dir']}/{rec['id']}` |", f"| Source | {rec.get('source') or '—'} |",
          f"| Format / quant | {rec['format']} / {rec.get('quant') or '—'} |", f"| Size on disk | {gb:,.1f} GB in {rec['n_weight_files']} weight file(s) |"]
    if fm.get("base_model"): L += [f"| Base model | {', '.join(fm['base_model'])} |"]
    if cf.get("arch"): L += [f"| Architecture | `{cf['arch']}` |"]
    if cf.get("ctx"): L += [f"| Max context (config) | {cf['ctx']:,} tokens |"]
    lic = fm.get("license_name") or fm.get("license")
    if lic: L += [f"| License | {lic}" + (" — read the terms before commercial use" if re.search(r"non-?commercial|other|research|nc", str(lic), re.I) else "") + " |"]
    L += [f"| Runtimes | {', '.join(rec.get('runtime') or [])} |", f"| Status | {rec['status']} |", "", "## Which box it fits", "| Box | Fits? |", "|---|---|"]
    per = float(cfg["node"]["usable_gb"])
    for k in range(1, int(cfg["node"]["max_nodes"]) + 1):
        L += [f"| {k} × {cfg['node']['name']} (≈ {per*k:.0f} GB usable) | {'yes' if gb <= per*k else 'no'} |"]
    L += ["", "Weights only — leave headroom for the KV cache and activations.", "", "## Setup", setup_text(rec, d, cf, task, cfg), "",
          "The upstream `README.md` in this folder is the authority for prompts, chat template and licence.", ""]
    return "\n".join(L)


def write_abouts(cfg, records):
    wrote = kept = 0
    for rec in records:
        d = os.path.join(cfg["data_root"], rec["path"])
        if rec["status"] != "complete" or not os.path.isdir(d): continue
        text = render_about(rec, d, cfg); sig = hashlib.sha1(text.encode()).hexdigest()[:12]; out = os.path.join(d, ABOUT); notes = ""
        if os.path.exists(out):
            with open(out, errors="ignore") as f: old = f.read()
            if NOTES in old: notes = old[old.index(NOTES):].rstrip() + "\n"
            if f"sig:{sig}" in old: kept += 1; continue
        full = text + (notes or f"{NOTES}\n_(anything you add below this line is kept when the file is regenerated)_\n") + f"\n<!-- generated by mcc_index.py · sig:{sig} -->\n"
        tmp = out + ".tmp"
        with open(tmp, "w") as f: f.write(full)
        os.replace(tmp, out); wrote += 1
    return wrote, kept


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the NAS model library index and views.")
    ap.add_argument("--config", default=os.environ.get("MCC_NAS_CONFIG", "/etc/mcc/nas.json"))
    ap.add_argument("--no-about", action="store_true"); ap.add_argument("--no-tiers", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="scan and print a summary; write nothing")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.no_tiers: cfg["tiers"] = False
    t0 = time.time(); records, kits = scan(cfg)
    if a.dry_run:
        for r in records: print(f"{r['id']:<70} {r['format']:<12} {r['size_bytes']/1e9:8.1f} GB  nodes={r['nodes_est']}  {r['status']}")
        print(f"{len(records)} models, {len(kits)} kits (dry run, nothing written)"); return 0
    made, pruned = build_views(cfg, records, kits); idx = write_index(cfg, records, kits)
    wrote = kept = 0
    if cfg.get("about") and not a.no_about: wrote, kept = write_abouts(cfg, records)
    print(f"index: {idx['n_models']} models ({idx['n_complete']} complete), {idx['total_bytes']/1e12:.2f} TB, {len(kits)} kits, "
          f"links +{made} -{pruned}, about wrote {wrote} kept {kept}, {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
