#!/usr/bin/env python3
"""mcc_bench.py — long-context benchmarks, a quality test, metrics history and phone alerts for the deck (v1.5).
Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).

Imported by mcc_deck.py; standard library only (Python 3.9+).

BENCHMARKS (one job at a time, shared with the v1.3 Decode / Prefill buttons)
  decode2  Every stream is first given a context of `ctx` tokens (warm: [user: context], 16 tokens out). Then all the streams of a
           level decode `out` tokens at once ([user: context, assistant: the warm reply VERBATIM, user: ask]). Levels 1 → 2 → 4 → 8
           run on the same conversations, each level extending the last, so an engine that resumes only a strictly longer prompt
           (a prompt cache keyed on prefixes) reuses the context. Thinking off. Per stream: the client's time to the first token and the
           decode rate (completion_tokens − 1) / (t_last − t_first), where t_first is the first content OR reasoning delta. Per level:
           the aggregate and a "steady" window [max t_first, min t_last] while every stream is decoding.
           Flags: "recomputed" (the engine's cached-token count is under 90 % of the warm prompt; without a count, a TTFT over
           max(0.3 s, 25 % of the warm prefill), labelled "unchecked"), "short" (an early end), "contaminated" (another client's request
           was seen while it ran).
  sweep    Prefill at 1k … 256k (up to the lane's window), a fresh nonce first, 1 token out: client TTFT and prompt_tokens / TTFT.
  quality  A seeded built-in suite (6 categories × 10, graded by rule on an "ANSWER:" line), temperature 0; the last 30 runs per model;
           two runs of the same suite compare with an exact McNemar test. Engine errors are counted; more than 2 fail the run.
  Guards   Refused during an INSERT and while another benchmark runs. Limits come from the deck's config and the engine itself
           (vLLM / SGLang context and KV pool, a recipe's `context` / `parallel` / `kv_bytes_per_token`, Ollama's resident context),
           unknown = 32k and one stream. A recipe whose KV cache grows per request plans its memory: streams × (ctx + replies) ×
           bytes/token × a margin against MemAvailable − headroom. Another client's request refuses the start and stops the run (or
           flags it, with "anyway"). MemAvailable under the floor stops it. CANCEL closes every open socket.
           Ollama: its native /api/chat only, num_ctx = the resident's own context, keep_alive never shorter, one stream.

HISTORY    the deck's live samples → 1-minute means for 48 h (metrics_1m.jsonl) → 15-minute means for 30 days (metrics_15m.jsonl).
HEALTH     memory low, a hot GPU, GPU errors and out-of-memory kills in the kernel log (when it is readable), the playing cassette's
           engine gone with no EJECT, a nearly full disk. A condition must last `hold_s` before it notifies; you get a note when it
           clears; at most one raise per kind per 30 minutes.
NOTIFY     ntfy, OFF until you create a topic on the page (state_dir/notify.json, 0600). https only; the host is resolved when a
           message is sent and refused when it is loopback, private, link-local, CGNAT or reserved; the connection goes to that
           address; redirects are never followed; 10 s timeout. A message names the alert kind only: no host, address, port, path,
           model or token."""
import http.client, ipaddress, json, math, os, random, re, secrets, socket, ssl, subprocess, threading, time, urllib.parse

KINDS = ("decode2", "sweep", "quality")
CTX_CHOICES = (0, 16384, 32768, 65536, 131072, 262144)
LEVELS = (1, 2, 4, 8)
SWEEP = (1024, 4096, 16384, 32768, 65536, 131072, 262144)
OUTS = (256, 512, 1024)
MARGIN = 512                     # the warm reply + the ask + the chat template, beside each level's reply
UNKNOWN_CTX = 32768
MEM_ABORT_GIB = 8.0
KV_MARGIN = 1.5                  # measured on a TensorFold lane (16k × 2 streams: 1.06× the formula; 64k × 2: 1.29×), plus room
KV_MARGIN_LONG = 2.0             # 128k and longer: not measured yet
PROMPT_CACHE_GIB = 16.0          # a typical kept-prompt budget, when a recipe does not state `prompt_cache_gib`
SUITE, SUITE_SEED = "q1", 1025
ASK0 = ("Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Never conclude it: keep adding "
        "new scenes and dialogue until you are stopped.")
ASK = "Continue the story from where it stopped. Do not summarise and do not end it: keep going with new scenes."
TAIL = "\n\nEnd your reply with one final line in exactly this form: ANSWER: <answer>"


class Stop(Exception):
    pass


def _now(): return time.perf_counter()
def _ts(): return time.strftime("%Y-%m-%dT%H:%M:%S")
def kv_margin(ctx): return KV_MARGIN_LONG if ctx >= 131072 else KV_MARGIN


def mem_avail_gib():
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemAvailable:"): return int(l.split()[1]) / 1048576
    except OSError: pass
    return None


def _rjson(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default


def _cached_of(u):
    d = (u or {}).get("prompt_tokens_details")
    if isinstance(d, dict) and isinstance(d.get("cached_tokens"), int): return d["cached_tokens"]
    for k in ("cached_tokens", "prompt_cache_hit_tokens", "cache_read_input_tokens"):
        if isinstance((u or {}).get(k), int): return u[k]
    return None


# ── engine I/O on a real box ────────────────────────────────────────────────────────────────────────────────────────────
class RealIO:
    """HTTP to the engines on 127.0.0.1. Every open connection is registered, so a cancel can close it."""
    def __init__(self, ollama_url="http://127.0.0.1:11434"):
        u = urllib.parse.urlsplit(ollama_url); self.ollama_host, self.ollama_port = u.hostname or "127.0.0.1", u.port or 11434
        self.live = set(); self.lock = threading.Lock(); self.caps = {}

    # connections
    def _conn(self, port, timeout, host="127.0.0.1"):
        c = http.client.HTTPConnection(host, int(port), timeout=timeout)
        with self.lock: self.live.add(c)
        return c

    def _drop(self, c):
        with self.lock: self.live.discard(c)
        try: c.close()
        except Exception: pass

    def ours(self):
        with self.lock: return len(self.live)

    def kill_all(self):
        with self.lock: cs = list(self.live)
        for c in cs:
            try:
                if c.sock: c.sock.shutdown(socket.SHUT_RDWR)
            except Exception: pass

    def get(self, port, path, timeout=4, text=False):
        c = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout)
        try:
            c.request("GET", path); r = c.getresponse(); b = r.read()
            if r.status != 200: return None
            return b.decode("utf-8", "replace") if text else json.loads(b)
        except Exception: return None
        finally: c.close()

    # OpenAI-compatible chat (vLLM, SGLang, recipe lanes)
    def chat(self, ctl, port, model, messages, max_tokens, pin=False, thinking=False, timeout=1800):
        if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
        cap = self.caps.setdefault(int(port), {})
        body = {"model": model, "messages": messages, "max_tokens": int(max_tokens), "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        if cap.get("ctk", True): body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        else: ctl["no_ctk"] = True
        pinned = bool(pin and cap.get("pin", True))
        if pinned: body.update(ignore_eos=True, min_tokens=int(max_tokens))
        c = self._conn(port, timeout); t0 = _now()
        try:
            c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = c.getresponse()
            if r.status != 200:
                txt = r.read()[:300].decode("utf-8", "replace")
                if "chat_template_kwargs" in body and r.status in (400, 422) and "chat_template_kwargs" in txt:
                    cap["ctk"] = False; self._drop(c); return self.chat(ctl, port, model, messages, max_tokens, pin, thinking, timeout)
                if pinned and r.status in (400, 422) and re.search(r"ignore_eos|min_tokens|unknown|unrecognized|extra", txt, re.I):
                    cap["pin"] = False; self._drop(c); return self.chat(ctl, port, model, messages, max_tokens, pin, thinking, timeout)
                raise RuntimeError(f"HTTP {r.status}: {txt[:160]}")
            content, reasoning, usage, chunks, marks, t_first, t_last = [], 0, {}, 0, [], None, None
            while True:
                line = r.readline()
                if not line: break
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try: d = json.loads(data)
                except ValueError: continue
                if d.get("error"): raise RuntimeError(f"engine error: {str(d['error'])[:160]}")
                if d.get("usage"): usage = d["usage"]
                for ch in d.get("choices") or []:
                    dl = ch.get("delta") or {}
                    txt = dl.get("content") or ""; rs = dl.get("reasoning_content") or dl.get("reasoning") or ""
                    if txt or rs:
                        t = _now(); t_first = t_first or t; t_last = t; chunks += 1; marks.append(t)
                        if txt: content.append(txt)
                        reasoning += len(rs)
        except (OSError, http.client.HTTPException) as e:
            if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
            raise RuntimeError(f"{type(e).__name__} talking to the engine")
        finally:
            self._drop(c)
        if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
        if t_first is None: raise RuntimeError("the engine sent no tokens")
        n = usage.get("completion_tokens")
        return {"content": "".join(content), "reasoning_chars": reasoning, "n": int(n) if isinstance(n, int) else None, "chunks": chunks,
                "prompt_tokens": usage.get("prompt_tokens"), "cached": _cached_of(usage), "t0": t0, "t_first": t_first, "t_last": t_last,
                "marks": marks, "pinned": pinned}

    # Ollama (native API only: /v1 cannot carry num_ctx, and a context mismatch reloads the model)
    def ollama_get(self, path):
        c = http.client.HTTPConnection(self.ollama_host, self.ollama_port, timeout=4)
        try:
            c.request("GET", path); r = c.getresponse(); b = r.read()
            return json.loads(b) if r.status == 200 else None
        except Exception: return None
        finally: c.close()

    def ollama_opts(self, name, expires_left):
        row = next((m for m in ((self.ollama_get("/api/ps") or {}).get("models") or []) if m.get("name") == name), None)
        if not row: raise RuntimeError(f"{name} is not resident in Ollama — a benchmark never loads a model")
        ctx = row.get("context_length")
        if not isinstance(ctx, int) or ctx <= 0: raise RuntimeError(f"Ollama does not report {name}'s context — refusing rather than risk a reload")
        left = expires_left(row.get("expires_at"))
        if left is None: raise RuntimeError(f"{name}'s expiry is unreadable — refusing rather than change its keep_alive")
        return {"ctx": ctx, "ka": -1 if left > 3e7 else f"{max(300, int(left) + 1)}s"}

    def ollama_chat(self, ctl, name, messages, num_predict, opts, timeout=1800):
        if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
        body = {"model": name, "messages": messages, "stream": False, "keep_alive": opts["ka"],
                "options": {"num_predict": int(num_predict), "temperature": 0, "num_ctx": opts["ctx"]}}
        c = self._conn(self.ollama_port, timeout, self.ollama_host); t0 = _now()
        try:
            c.request("POST", "/api/chat", body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = c.getresponse(); b = r.read()
            if r.status != 200: raise RuntimeError(f"HTTP {r.status}: {b[:160].decode('utf-8', 'replace')}")
            d = json.loads(b)
        except (OSError, http.client.HTTPException) as e:
            if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
            raise RuntimeError(f"{type(e).__name__} talking to Ollama")
        finally:
            self._drop(c)
        if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
        ec, ed, pc, pd = d.get("eval_count"), d.get("eval_duration"), d.get("prompt_eval_count"), d.get("prompt_eval_duration")
        if not ec or not ed: raise RuntimeError("Ollama returned no timings")
        if (d.get("load_duration") or 0) > 1e9: ctl["reloaded"] = True       # the reload the rules exist to prevent: said on the result
        return {"content": (d.get("message") or {}).get("content") or "", "n": ec, "decode_s": ed / 1e9, "prompt_eval": pc or 0,
                "prefill_s": (pd or 0) / 1e9, "ttft": ((d.get("load_duration") or 0) + (pd or 0)) / 1e9, "wall": _now() - t0}

    # what else the engine is serving right now (None = it cannot tell)
    def busy(self, t):
        port = t.get("port")
        if not port: return None
        h = self.get(port, t.get("health_path") or "/health", 2)
        if isinstance(h, dict) and isinstance(h.get("live"), dict):
            return int(h["live"].get("connections") or 0) + int(h["live"].get("waiting") or 0)
        txt = self.get(port, t.get("metrics") or "/metrics", 2, text=True)
        if not txt: return None
        n, seen = 0, False
        for k in ("vllm:num_requests_running", "vllm:num_requests_waiting", "sglang:num_running_reqs", "sglang:num_queue_reqs"):
            for m in re.finditer(r"^" + re.escape(k) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", txt, re.M):
                n += int(float(m.group(1))); seen = True
        return n if seen else None

    def kv_pool(self, t):
        """the engine's KV pool in tokens: vLLM cache_config_info (num_gpu_blocks × block_size), SGLang max_total_num_tokens"""
        txt = self.get(t["port"], t.get("metrics") or "/metrics", 3, text=True) or ""
        m = re.search(r"^vllm:cache_config_info\{([^}]*)\}", txt, re.M)
        if m:
            lab = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            try: return int(lab["num_gpu_blocks"]) * int(lab["block_size"])
            except (KeyError, ValueError): pass
        info = self.get(t["port"], "/get_server_info", 3)
        if isinstance(info, dict) and isinstance(info.get("max_total_num_tokens"), int): return info["max_total_num_tokens"]
        return None

    def max_model_len(self, port):
        mm = self.get(port, "/v1/models", 3) or {}
        d0 = (mm.get("data") or [{}])[0] if isinstance(mm.get("data"), list) and mm.get("data") else {}
        v = d0.get("max_model_len")
        return int(v) if isinstance(v, int) else None

    def gpu(self):
        """{util, temp, power} from nvidia-smi, or {} when there is none"""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=4).stdout.strip().splitlines()
        except Exception: return {}
        if not out: return {}
        v = [x.strip() for x in out[0].split(",")]
        def num(x):
            try: return float(x)
            except ValueError: return None
        return {"util": num(v[0]) if len(v) > 0 else None, "temp": num(v[1]) if len(v) > 1 else None, "power": num(v[2]) if len(v) > 2 else None}

    def kernel_log(self, since):
        """kernel messages since `since` (epoch s), or None when the kernel log cannot be read by this user"""
        try:
            r = subprocess.run(["journalctl", "-k", "--since", f"@{int(since)}", "--no-pager", "-q", "-o", "cat"], capture_output=True, text=True, timeout=3)
            return r.stdout if r.returncode == 0 else None
        except Exception: return None

    def mem_avail(self): return mem_avail_gib()

    # ntfy
    def notify(self, cfg, lvl, msg):
        host, port = server_ok(cfg.get("server") or "https://ntfy.sh")
        fam, sa = public_addr(host, port)
        raw = socket.socket(fam, socket.SOCK_STREAM); raw.settimeout(10); raw.connect(sa)
        ctx = ssl.create_default_context(); sock = ctx.wrap_socket(raw, server_hostname=host)
        c = http.client.HTTPSConnection(host, port, timeout=10, context=ctx); c.sock = sock
        h = {"Title": "Lion AI Models" if lvl == "info" else "Lion AI Models alert", "Priority": {"crit": "high", "warn": "default"}.get(lvl, "low"),
             "Tags": {"crit": "rotating_light", "warn": "warning"}.get(lvl, "white_check_mark"), "Content-Type": "text/plain; charset=utf-8"}
        if cfg.get("token"): h["Authorization"] = f"Bearer {cfg['token']}"
        c.request("POST", "/" + urllib.parse.quote(cfg["topic"], safe=""), body=msg.encode(), headers=h)
        r = c.getresponse(); r.read(); c.close()
        if not 200 <= r.status < 300: raise RuntimeError(f"HTTP {r.status}")     # a 3xx is NOT followed (http.client never follows)


def server_ok(server):
    u = urllib.parse.urlsplit(server or "")
    if u.scheme != "https" or not u.hostname or u.username or u.password or u.path not in ("", "/") or u.query or u.fragment:
        raise ValueError("the server must be a plain https://host[:port] address")
    return u.hostname, u.port or 443


def public_addr(host, port):
    """the first address of host (resolved now); ValueError when ANY address is not public"""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not infos: raise ValueError("unresolvable")
    out = []
    for fam, _, _, _, sa in infos:
        ip = ipaddress.ip_address(sa[0])
        if ip.version == 6 and ip.ipv4_mapped: ip = ip.ipv4_mapped
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                or (ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"))):
            raise ValueError("a non-public address")
        out.append((fam, sa))
    return out[0]


# ── demo engine: believable numbers, no network ─────────────────────────────────────────────────────────────────────────
class DemoIO(RealIO):
    """What --demo uses: a fake engine with a prefix cache and stream-count-dependent speed, so every benchmark, the history and the
    alerts can be tried without a GPU. Timings are scaled down so a run takes seconds; the numbers are demo numbers."""
    SPEED = 0.004                                       # seconds slept per simulated token (decode) — the run is fast, the rates are not
    def __init__(self):
        super().__init__(); self.kept = []; self.rng = random.Random(5)

    def _flat(self, msgs): return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def chat(self, ctl, port, model, messages, max_tokens, pin=False, thinking=False, timeout=1800):
        if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
        with self.lock: self.live.add(id(messages))
        try:
            text = self._flat(messages); pt = max(1, len(text) // 4)
            hit = max((k for k in self.kept if text.startswith(k) and len(k) < len(text)), key=len, default="")
            self.kept = (self.kept + [text])[-64:]; cached = len(hit) // 4
            k = max(1, len([x for x in self.live]))
            pre = (pt - cached) / 2300.0 + 0.25 + 0.06 * (k - 1); dec = 41.0 / (1 + 0.32 * (k - 1)) * (0.97 + 0.06 * self.rng.random())
            t0 = _now(); time.sleep(min(1.5, pre * 0.01) + 0.05)
            t_first = _now(); n = int(max_tokens)
            q = next((it for it in SUITE_ITEMS() if it["q"] in text), None)
            content = ("ANSWER: " + (q["a"] if self.rng.random() < 0.9 else "0")) if q else ("word " * min(n, 16))
            marks = [t_first + i * (1.0 / dec) * 0 + i * self.SPEED for i in range(n)]
            time.sleep(n * self.SPEED)
            if ctl["cancel"]: raise Stop(ctl["why"] or "cancelled")
            span = (n - 1) / dec                              # report the simulated rate, not the scaled sleep
            return {"content": content, "reasoning_chars": 0, "n": n, "chunks": n, "prompt_tokens": pt, "cached": cached,
                    "t0": t_first - pre, "t_first": t_first, "t_last": t_first + span, "marks": [t_first + i * span / max(1, n - 1) for i in range(n)],
                    "pinned": True}
        finally:
            with self.lock: self.live.discard(id(messages))

    def busy(self, t): return 0
    def kv_pool(self, t): return None
    def max_model_len(self, port): return None
    def ollama_opts(self, name, expires_left): return {"ctx": 32768, "ka": "1800s"}

    def ollama_chat(self, ctl, name, messages, num_predict, opts, timeout=1800):
        r = self.chat(ctl, 0, name, messages, num_predict)
        return {"content": r["content"], "n": r["n"], "decode_s": (r["t_last"] - r["t_first"]) or 0.1, "prompt_eval": r["prompt_tokens"] - (r["cached"] or 0),
                "prefill_s": r["t_first"] - r["t0"], "ttft": r["t_first"] - r["t0"], "wall": 1.0}

    def gpu(self):
        b = (int(time.time()) // 120) % 3 != 0
        return {"util": 72.0 if b else 4.0, "temp": 61.0 if b else 43.0, "power": 88.0 if b else 24.0}

    def kernel_log(self, since): return ""
    def mem_avail(self): return 58.0
    def notify(self, cfg, lvl, msg): time.sleep(0.2)     # demo: nothing leaves the machine


# ── prompts ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_W = ("panel", "relay", "cable", "harness", "station", "fixture", "sensor", "bracket", "channel", "module", "rack", "adapter",
      "keeper", "harbor", "lantern", "signal", "valve", "shelf", "ledger", "survey", "north", "quiet", "copper", "amber", "steady")
_V = ("checked", "logged", "moved", "tested", "noted", "replaced", "cleaned", "measured", "tightened", "labelled")


def filler(chars, seed):
    g = random.Random(seed); out, n = [], 0
    while n < chars:
        s = f"At {g.randint(0, 23):02d}:{g.randint(0, 59):02d} the {g.choice(_W)} {g.choice(_W)} was {g.choice(_V)} near the {g.choice(_W)}, reading {g.randint(1, 999)}. "
        out.append(s); n += len(s)
    return "".join(out)


def context_text(tokens, seed, cpt):
    nonce = secrets.token_hex(6)                          # first, so no two streams share a prefix past the chat template
    return (f"[{nonce}] Here is a long station log. Read it; afterwards you will be asked to write a story.\n\n"
            + filler(int(tokens * cpt), seed) + "\n\nReply with: OK, ready.")


# ── the quality suite: seeded, built in, graded by rule ───────────────────────────────────────────────────────────────────
_NAMES = ("Ava", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun")
_THINGS = ("apple", "key", "coin", "stone", "shell", "ring", "button", "marble", "feather", "pencil")
_ANIMALS = ("falcon", "otter", "badger", "heron", "lynx", "marten", "osprey", "bison", "gecko", "walrus", "ibis", "puffin")
_WORDS = ("copper", "lantern", "harbor", "violet", "meadow", "anchor", "pepper", "quartz", "summit", "timber", "glacier", "willow")


def build_suite(seed=SUITE_SEED):
    out = []
    for i in range(10):                                   # arithmetic chains (exact integers)
        g = random.Random(seed * 100 + i); v = v0 = g.randint(10, 60); steps = []
        for _ in range(5):
            op = g.choice(("add", "sub", "mul", "half"))
            if op == "add": n = g.randint(2, 30); v += n; steps.append(f"add {n}")
            elif op == "sub": n = g.randint(2, 20); v -= n; steps.append(f"subtract {n}")
            elif op == "mul": n = g.randint(2, 4); v *= n; steps.append(f"multiply by {n}")
            else: v = v // 2; steps.append("halve it, rounding down")
        out.append({"cat": "arithmetic", "q": f"Start with the number {v0}. Then, in order: {', '.join(steps)}. What is the result?", "a": str(v), "g": "int"})
    for i in range(10):                                   # state tracking
        g = random.Random(seed * 200 + i); boxes = list("ABCD"); first = g.sample(_THINGS, 4); held = dict(zip(boxes, first)); ops = []
        for _ in range(5):
            x, y = g.sample(boxes, 2)
            if g.random() < 0.6: held[x], held[y] = held[y], held[x]; ops.append(f"swap the contents of box {x} and box {y}")
            else:
                new = g.choice([w for w in _THINGS if w not in held.values()]); held[x] = new; ops.append(f"replace what is in box {x} with a {new}")
        q = g.choice(boxes); init = ", ".join(f"box {b} holds a {o}" for b, o in zip(boxes, first))
        out.append({"cat": "state", "q": f"At the start {init}. Then: {'; then '.join(ops)}. What is in box {q} now? Answer with one word.", "a": held[q], "g": "word"})
    for i in range(10):                                   # instruction following (format-checked)
        g = random.Random(seed * 300 + i); kind = i % 4
        if kind == 0:
            ws = g.sample(_WORDS, 5)
            out.append({"cat": "instructions", "q": f"Write these words in reverse alphabetical order, in UPPERCASE, separated by commas: {', '.join(ws)}.",
                        "a": ",".join(sorted((w.upper() for w in ws), reverse=True)), "g": "list_upper"})
        elif kind == 1:
            ns = g.sample(range(2, 99), 5)
            out.append({"cat": "instructions", "q": f"Write these numbers in ascending order joined by hyphens, with no spaces: {' '.join(map(str, ns))}.",
                        "a": "-".join(map(str, sorted(ns))), "g": "exact"})
        elif kind == 2:
            w = g.choice(_WORDS) + g.choice(_ANIMALS)
            out.append({"cat": "instructions", "q": f"Write the word '{w}' backwards, in lowercase letters only.", "a": w[::-1], "g": "exact"})
        else:
            ws = g.sample(_WORDS + _ANIMALS, 6)
            out.append({"cat": "instructions", "q": f"Give the first letter of each of these words, in UPPERCASE, with no spaces: {' '.join(ws)}.",
                        "a": "".join(w[0].upper() for w in ws), "g": "exact"})
    for i in range(10):                                   # logic ordering
        g = random.Random(seed * 400 + i); ppl = g.sample(_NAMES, 4); facts = [f"{ppl[j]} is taller than {ppl[j + 1]}" for j in range(3)]; g.shuffle(facts)
        ask = g.choice(("shortest", "tallest", "second tallest", "second shortest"))
        a = {"tallest": ppl[0], "second tallest": ppl[1], "second shortest": ppl[2], "shortest": ppl[3]}[ask]
        out.append({"cat": "logic", "q": f"{'. '.join(facts)}. Who is the {ask}? Answer with the name only.", "a": a, "g": "word"})
    conv = (("hours", "minutes", 60, 2), ("minutes", "seconds", 60, 1), ("days", "hours", 24, 1), ("kilometres", "metres", 1000, 10),
            ("kilograms", "grams", 1000, 4), ("litres", "millilitres", 1000, 4), ("metres", "centimetres", 100, 4), ("weeks", "days", 7, 1))
    for i in range(10):                                   # unit conversion (whole-number answers)
        g = random.Random(seed * 500 + i); a_, b_, f, den = conv[i % len(conv)]; num = g.randint(3, 40); val = num / den
        out.append({"cat": "units", "q": f"Convert {val:g} {a_} to {b_}. Give a whole number.", "a": str(int(round(val * f))), "g": "int"})
    for i in range(10):                                   # long-context recall (~5k tokens)
        g = random.Random(seed * 600 + i); keys = g.sample(_ANIMALS, 8); codes = [g.randint(1000, 9999) for _ in keys]
        parts = [filler(2400, seed * 600 + i * 10 + j) + f"The access code for the {k} locker is {c}. " for j, (k, c) in enumerate(zip(keys, codes))]
        q = g.randrange(8)
        out.append({"cat": "recall", "q": "".join(parts) + filler(1200, seed + i) + f"\n\nWhat is the access code for the {keys[q]} locker? Give the four digits.",
                    "a": str(codes[q]), "g": "int"})
    return out


_SUITE = {}
def SUITE_ITEMS():
    if SUITE not in _SUITE: _SUITE[SUITE] = build_suite()
    return _SUITE[SUITE]


def grade(item, content):
    """True / False from the reply's LAST 'ANSWER:' line (any reasoning is never read)"""
    m = None
    for m in re.finditer(r"ANSWER\s*[:：]\s*(.+)", content or "", re.I): pass
    if not m: return False
    v = re.sub(r"[.。]\s*$", "", m.group(1).strip().strip("*`'\"").strip()).strip()
    a, kind = item["a"], item["g"]
    if kind == "int":
        if item.get("cat") == "units": v = re.sub(r"\s+[A-Za-z]+$", "", v)       # "900 minutes"
        n = re.sub(r"[,\s]", "", v); return bool(re.fullmatch(r"-?\d+", n)) and int(n) == int(a)
    if kind == "word": return v.lower().strip(" .!") in (a.lower(), f"a {a.lower()}", f"an {a.lower()}")
    if kind == "list_upper":
        parts = [x.strip() for x in v.split(",")]; return all(x == x.upper() for x in parts) and ",".join(parts) == a
    return v == a


def mcnemar(a, b):
    """exact two-sided McNemar on two runs' per-item strings ('1' right / '0' wrong): (right only in a, right only in b, p)"""
    if len(a) != len(b): raise ValueError("the runs have different items")
    x = sum(1 for p, q in zip(a, b) if p == "1" and q == "0"); y = sum(1 for p, q in zip(a, b) if p == "0" and q == "1"); n = x + y
    if n == 0: return x, y, 1.0
    return x, y, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


# ── metrics history ────────────────────────────────────────────────────────────────────────────────────────────────────
HF = ("cpu", "mem", "avail", "util", "power", "temp")
RANGES = {"6h": (6 * 3600, "1m"), "24h": (86400, "1m"), "48h": (2 * 86400, "1m"), "7d": (7 * 86400, "15m"), "30d": (30 * 86400, "15m")}


class History:
    """1-minute means for 48 h and 15-minute means for 30 days, written by the one sampler thread (which also trims)."""
    def __init__(self, state_dir):
        self.M1, self.M15 = f"{state_dir}/metrics_1m.jsonl", f"{state_dir}/metrics_15m.jsonl"
        self.acc = {"min": None, "sum": {}, "cnt": {}, "max": {}, "q": [], "hour": None, "n": 0}

    @staticmethod
    def _append(path, obj):
        with open(path, "a") as f: f.write(json.dumps(obj) + "\n")

    @staticmethod
    def _trim(path, keep_s, now):
        try:
            with open(path) as f: rows = [l for l in f if l.strip()]
        except OSError: return
        keep = [l for l in rows if json.loads(l).get("t", 0) >= now - keep_s]
        if len(keep) == len(rows): return
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as f: f.writelines(keep)
        os.replace(tmp, path)

    def add(self, p, now):
        """p = one live sample {cpu, mem, avail, util, power, temp, tok, ptok}"""
        A, mi = self.acc, int(now // 60)
        if A["min"] is None: A["min"] = mi
        if mi != A["min"] and A["n"]:
            row = {"t": A["min"] * 60}
            for k in HF:
                if A["cnt"].get(k): row[k] = round(A["sum"][k] / A["cnt"][k], 1)
            for k in ("tok", "ptok"):
                if A["max"].get(k): row[k] = A["max"][k]
            self._append(self.M1, row); A["q"].append(row)
            q15 = (A["min"] * 60) // 900
            if (mi * 60) // 900 != q15:
                rs = [r for r in A["q"] if r["t"] // 900 == q15]
                if rs: self._append(self.M15, merge(rs, q15 * 900))
                A["q"] = [r for r in A["q"] if r["t"] // 900 != q15]
            if A["hour"] != mi // 60:
                A["hour"] = mi // 60; self._trim(self.M1, 48 * 3600, now); self._trim(self.M15, 30 * 86400, now)
            A.update(min=mi, n=0, sum={}, cnt={}, max={})
        A["n"] += 1
        for k in HF:
            v = p.get(k)
            if isinstance(v, (int, float)): A["sum"][k] = A["sum"].get(k, 0) + v; A["cnt"][k] = A["cnt"].get(k, 0) + 1
        for k in ("tok", "ptok"):
            v = p.get(k)
            if isinstance(v, (int, float)) and v > A["max"].get(k, 0): A["max"][k] = v

    def read(self, rng="24h", now=None):
        span, src = RANGES.get(rng, RANGES["24h"]); now = now or time.time(); path = self.M1 if src == "1m" else self.M15
        try:
            with open(path) as f: rows = [json.loads(l) for l in f if l.strip()]
        except OSError: rows = []
        rows = [r for r in rows if r.get("t", 0) >= now - span]
        if len(rows) > 720:
            step = span / 720; out, cur, b = [], [], None
            for r in rows:
                k = int((r["t"] - (now - span)) // step)
                if b is not None and k != b and cur: out.append(merge(cur, cur[0]["t"])); cur = []
                b = k; cur.append(r)
            if cur: out.append(merge(cur, cur[0]["t"]))
            rows = out
        return {"range": rng if rng in RANGES else "24h", "span_s": span, "step": src, "points": rows}


def merge(rs, t):
    o = {"t": t}
    for k in HF:
        v = [r[k] for r in rs if r.get(k) is not None]
        if v: o[k] = round(sum(v) / len(v), 1)
    for k in ("tok", "ptok", "tmax"):
        v = [r.get(k) or (r.get("temp") if k == "tmax" else None) for r in rs]
        v = [x for x in v if x]
        if v: o[k] = max(v)
    return o


# ── the benchmark job, health and notifications ──────────────────────────────────────────────────────────────────────────
class Bench:
    EVENT_HOLD_S = 600; RAISE_EVERY_S = 1800

    def __init__(self, deck, io=None):
        self.deck = deck; self.io = io or RealIO(deck.cfg.get("ollama_url") or "http://127.0.0.1:11434"); s = deck.cfg["state_dir"]
        self.QUAL, self.NOTIFY, self.NLOG = f"{s}/quality.jsonl", f"{s}/notify.json", f"{s}/notify_log.jsonl"
        self.ctl = {"cancel": False, "why": "", "external": 0, "settling": False, "reloaded": False, "no_ctk": False}
        self.cpt = {}; self.hist = History(s); self.cpu_prev = None
        self.H = {"since": None, "kt": 0.0, "events": {}, "prev_play": None, "cand": None}
        self.N = {"active": {}, "last_raise": {}}

    # limits -----------------------------------------------------------------------------------------------------------
    def limits(self, t):
        cfg, eng = self.deck.cfg, t.get("engine")
        if eng == "Ollama":
            o = self.io.ollama_opts(t["model"], self.deck.expires_left)
            return {"ctx": o["ctx"], "streams": 1, "kv_bpt": None, "headroom_gib": MEM_ABORT_GIB, "pool": None, "visible": False,
                    "basis": f"the resident's own context ({o['ctx']:,} tokens); Ollama here answers one request at a time"}
        r = self.deck.recipe_for_target(t)
        if r:
            ctx = int(r.get("context") or 0) or UNKNOWN_CTX; par = int(r.get("parallel") or 1); bpt = r.get("kv_bytes_per_token")
            res = float(r.get("memory_reserve_gib") or 0)
            return {"ctx": ctx, "streams": max(1, min(8, par)), "kv_bpt": int(bpt) if bpt else None, "headroom_gib": MEM_ABORT_GIB + res + 2,
                    "prompt_cache_gib": float(r.get("prompt_cache_gib") or PROMPT_CACHE_GIB), "pool": None,
                    "basis": (f"the {r.get('label') or 'recipe'} recipe: context {ctx:,} · parallel {par}"
                              + (f" · {int(bpt):,} B/token KV" if bpt else " · KV pre-allocated (no kv_bytes_per_token)")
                              + (f" · reserve {res:g} GiB" if res else ""))}
        e = "vllm" if eng == "vLLM" else "sglang" if eng == "SGLang" else None
        if e:
            v = cfg[e]; ml = self.io.max_model_len(t["port"]) or int(v.get("max_model_len") or v.get("context_length") or UNKNOWN_CTX)
            pool = self.io.kv_pool(t); par = int((cfg.get("bench") or {}).get("max_streams") or 8)
            return {"ctx": ml, "streams": max(1, min(8, par)), "kv_bpt": None, "headroom_gib": MEM_ABORT_GIB, "pool": pool,
                    "basis": f"{eng}: max_model_len {ml:,}" + (f" · KV pool {pool:,} tokens" if pool else " · KV pool not reported")}
        return {"ctx": UNKNOWN_CTX, "streams": 1, "kv_bpt": None, "headroom_gib": MEM_ABORT_GIB, "pool": None,
                "basis": f"the engine states no limits: {UNKNOWN_CTX:,} tokens, one stream"}

    def external(self, t):
        if t.get("engine") == "Ollama": return None
        n = self.io.busy(t)
        return None if n is None else max(0, n - self.io.ours())

    def _rates(self, key):
        pre, dec = None, None
        try:
            with open(self.deck.BENCH) as f: rows = [json.loads(l) for l in f if l.strip()][-400:]
        except OSError: rows = []
        for x in rows:
            if x.get("key") != key: continue
            if x.get("kind") == "prefill" and x.get("tok_s"): pre = x["tok_s"]
            if x.get("kind") == "sweep" and x.get("points"): pre = max((p.get("tok_s") or 0) for p in x["points"]) or pre
            if x.get("kind") == "decode" and x.get("tok_s"): dec = x["tok_s"]
        return pre or 2000.0, dec or 40.0

    # plan ---------------------------------------------------------------------------------------------------------------
    def plan(self, kind, p):
        p = p if isinstance(p, dict) else {}
        if kind not in KINDS: return {"ok": False, "refuse": f"kind must be one of {', '.join(KINDS)}"}
        t = self.deck.target()
        if not t: return {"ok": False, "refuse": "nothing is loaded — insert a cassette first"}
        if t["engine"] != "Ollama" and not t.get("port"): return {"ok": False, "refuse": f"no port known for {t['engine']}"}
        try: lim = self.limits(t)
        except Exception as e: return {"ok": False, "refuse": str(e)[:240]}
        notes = []; pre, dec = self._rates(t["key"])
        if self.external(t) is None: notes.append("other clients are not visible on this engine: the run cannot give way to a chat that starts meanwhile")
        avail = self.io.mem_avail() or 0; room = avail - lim["headroom_gib"]
        try: out = int(p.get("out") or 512)
        except (TypeError, ValueError): out = 0
        if out not in OUTS: return {"ok": False, "refuse": "out must be 256, 512 or 1024 tokens"}
        base = {"target": {k: t.get(k) for k in ("engine", "model", "port")}, "limits": lim, "avail_gib": round(avail, 1)}
        if kind == "decode2":
            try: ctx = int(p.get("ctx") or 0); lv = sorted({int(x) for x in (p.get("levels") or [1]) if int(x) in LEVELS})
            except (TypeError, ValueError): return {"ok": False, "refuse": "ctx and levels must be numbers"}
            if not 0 <= ctx <= 1048576: return {"ok": False, "refuse": "ctx must be 0 … 1,048,576 tokens"}
            if not lv: return {"ok": False, "refuse": "pick at least one stream level (1, 2, 4, 8)"}
            if max(lv) > lim["streams"]: return {"ok": False, **base, "refuse": f"{max(lv)} streams is more than this lane serves at once ({lim['streams']})"}
            top = lim["ctx"] - out * len(lv) - MARGIN * (len(lv) + 1)
            if ctx >= lim["ctx"] and top > 0:
                notes.append(f"filled to {top:,} tokens: the {lim['ctx']:,}-token window less the replies"); ctx = top
            elif ctx > top:
                return {"ok": False, **base, "refuse": f"{ctx:,} tokens plus {len(lv)} level(s) of {out}-token replies is past this lane's {lim['ctx']:,}-token "
                                                       f"window per stream: the most it takes is {max(top, 0):,}"}
            per = ctx + out * len(lv) + MARGIN * (len(lv) + 1); need = 0.0
            if lim["kv_bpt"]:
                need = kv_margin(ctx) * max(lv) * per * lim["kv_bpt"] / 2 ** 30
                if need > room:
                    return {"ok": False, **base, "need_gib": round(need, 1),
                            "refuse": (f"needs ~{need:.1f} GiB ({max(lv)} × {per:,} tokens × {lim['kv_bpt']:,} B × {kv_margin(ctx):g} margin), "
                                       f"{max(room, 0):.1f} GiB free beyond the {lim['headroom_gib']:g} GiB headroom ({avail:.0f} GiB available)")}
                kept = max(lv) * (ctx + MARGIN) * lim["kv_bpt"] / 2 ** 30; pc = lim.get("prompt_cache_gib", PROMPT_CACHE_GIB)
                if kept > pc: notes.append(f"the warm prompts (~{kept:.1f} GiB) exceed the engine's {pc:g} GiB prompt cache: some streams will recompute (flagged)")
                if ctx: notes.append(f"afterwards the engine may keep up to ~{min(kept, pc):.1f} GiB of warm prompts until other prompts replace them")
            if lim["pool"] and max(lv) * per > lim["pool"]:
                return {"ok": False, **base, "refuse": f"{max(lv)} × {per:,} tokens is more than the engine's KV pool ({lim['pool']:,} tokens)"}
            if not ctx: notes.append("empty context: no warm-up, the streams start decoding at once")
            eta = 4 + (max(lv) * ctx) / pre + sum(out / max(dec / max(1, k ** 0.5), 1) for k in lv) + 5 * len(lv)
            return {"ok": True, **base, "eta_s": round(eta), "need_gib": round(need, 1), "notes": notes, "params": {"ctx": ctx, "levels": lv, "out": out}}
        if kind == "sweep":
            sz = [s for s in SWEEP if s + 64 <= lim["ctx"]]
            if lim["kv_bpt"]:
                pc = lim.get("prompt_cache_gib", PROMPT_CACHE_GIB); ok = []
                for s in sz:
                    kept = min(sum(x + 64 for x in ok) * lim["kv_bpt"] / 2 ** 30, pc)
                    if kept + kv_margin(s) * (s + 64) * lim["kv_bpt"] / 2 ** 30 > room: break
                    ok.append(s)
                if len(ok) < len(sz) and ok: notes.append(f"stops at {ok[-1]:,} tokens: larger points need more memory than is free beyond the headroom")
                sz = ok
            if lim["pool"]: sz = [s for s in sz if s + 64 <= lim["pool"]]
            if not sz: return {"ok": False, **base, "refuse": "no sweep point fits this lane's window and the free memory"}
            return {"ok": True, **base, "eta_s": round(sum(sz) / pre + 2 * len(sz)), "notes": notes, "params": {"sizes": sz}}
        thinking = bool(p.get("thinking"))
        if lim["ctx"] < 7000: return {"ok": False, **base, "refuse": f"the suite's recall questions need ~7,000 tokens; this lane has {lim['ctx']:,}"}
        conc = 1 if t["engine"] == "Ollama" else min(2, lim["streams"])
        notes.append(f"{len(SUITE_ITEMS())} questions, temperature 0, thinking {'on' if thinking else 'off'}, {conc} at a time")
        return {"ok": True, **base, "eta_s": round(len(SUITE_ITEMS()) * (1500 if thinking else 120) / dec / conc + 10 * 6000 / pre), "notes": notes,
                "params": {"thinking": thinking, "conc": conc}}

    # run ----------------------------------------------------------------------------------------------------------------
    def start(self, kind, p):
        p = p if isinstance(p, dict) else {}; d = self.deck
        if d.job["running"]: return {"error": "an insert is running — benchmark after it", "status": 409}
        pl = self.plan(kind, p)
        if not pl.get("ok"): return {"error": pl.get("refuse") or "refused", "status": 409, "plan": pl}
        t = d.target(); anyway = p.get("anyway") is True; ext = self.external(t)
        if ext and not anyway:
            return {"error": f"the engine is serving {ext} other request(s) right now — run it when it is idle, or tick 'run anyway' (the result is then flagged)", "status": 409}
        with d.lock:
            if d.bjob["running"]: return {"error": "a benchmark is already running", "status": 409}
            d.bjob.update(running=True, kind=kind, msg=f"{kind} on {t['model']}: starting…")
        self.ctl.update(cancel=False, why="", external=0, settling=False, reloaded=False, no_ctk=False)
        threading.Thread(target=self._run, args=(kind, t, pl, anyway), daemon=True).start()
        return {"started": True, "kind": kind, "model": t["model"], "plan": pl}

    def cancel(self):
        b = self.deck.bjob
        if not b["running"]: return {"error": "no benchmark is running", "status": 409}
        if b.get("kind") not in KINDS: return {"error": "the short Decode / Prefill run finishes by itself", "status": 409}
        self.ctl.update(cancel=True, why="cancelled from the page"); self.io.kill_all()
        return {"cancelling": True}

    def _msg(self, s): self.deck.bjob["msg"] = s

    def _watch(self, t, anyway, done):
        strikes = 0
        while not done.is_set():
            a = self.io.mem_avail()
            if a is not None and a < MEM_ABORT_GIB:
                self.ctl.update(cancel=True, why=f"stopped: memory fell to {a:.1f} GiB available (under {MEM_ABORT_GIB:g})"); self.io.kill_all(); return
            e = None if self.ctl["settling"] else self.external(t)
            strikes = strikes + 1 if e else 0
            if strikes >= 2:
                self.ctl["external"] = max(self.ctl["external"], e)
                if not anyway:
                    self.ctl.update(cancel=True, why="stopped: another client started using the engine — the run gave way"); self.io.kill_all(); return
            done.wait(1.0)

    def _settle(self, t, anyway):
        self.ctl["settling"] = True
        try:
            e = None
            for _ in range(6):
                e = self.external(t)
                if not e: return
                time.sleep(0.5)
            self.ctl["external"] = max(self.ctl["external"], e or 0)
            if not anyway: raise Stop("stopped: another client is using the engine — the run gave way")
        finally: self.ctl["settling"] = False

    def _run(self, kind, t, pl, anyway):
        done = threading.Event(); threading.Thread(target=self._watch, args=(t, anyway, done), daemon=True).start()
        row = {"ts": _ts(), "key": t["key"], "engine": t["engine"], "model": t["model"], "kind": kind, "params": pl["params"],
               "limits_basis": pl["limits"].get("basis"), "anyway": anyway, "others_visible": self.external(t) is not None}
        t0 = _now()
        try:
            row.update(self._decode2(t, pl["params"], anyway) if kind == "decode2" else self._sweep(t, pl["params"], anyway) if kind == "sweep"
                       else self._quality(t, pl["params"], anyway))
            row["status"] = "done"
        except Stop as e: row.update(status="stopped", why=str(e)[:240])
        except Exception as e: row.update(status="failed", why=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            done.set(); row["secs"] = round(_now() - t0, 1); row["contaminated"] = bool(self.ctl["external"])
            if self.ctl["reloaded"]: row["ollama_reloaded"] = True
            if self.ctl["no_ctk"]: row["thinking_set"] = False
            try:
                if kind == "quality" and row.get("status") == "done": self._save_quality(row)
                else:
                    with open(self.deck.BENCH, "a") as f: f.write(json.dumps(row) + "\n")
            except OSError: pass
            self._msg(summary(row)); self.deck.bjob["running"] = False

    def _call(self, t, messages, out, pin=False, thinking=False):
        if t["engine"] == "Ollama": return "o", self.io.ollama_chat(self.ctl, t["model"], messages, out, self.io.ollama_opts(t["model"], self.deck.expires_left))
        return "c", self.io.chat(self.ctl, t["port"], t["model"], messages, out, pin=pin, thinking=thinking)

    def _calibrate(self, t):
        if t["key"] in self.cpt: return self.cpt[t["key"]]
        def n(text):
            typ, r = self._call(t, [{"role": "user", "content": text}], 1)
            return r["prompt_eval"] if typ == "o" else (r["prompt_tokens"] or 0)
        a = n(f"[{secrets.token_hex(4)}] OK?"); b = n(f"[{secrets.token_hex(4)}] OK? " + filler(6000, 7))
        if b - a < 200: raise RuntimeError("the engine did not report prompt token counts — cannot size the context")
        self.cpt[t["key"]] = 6000 / (b - a); return self.cpt[t["key"]]

    def _decode2(self, t, P, anyway):
        ctx, lv, out = P["ctx"], P["levels"], P["out"]; smax = max(lv); convs, warm = [], []
        self._settle(t, anyway)
        if ctx:
            self._msg("sizing the context…"); cpt = self._calibrate(t)
            for i in range(smax):
                self._msg(f"warming stream {i + 1}/{smax} with {ctx:,} tokens…")
                m = [{"role": "user", "content": context_text(ctx, 1000 + i, cpt)}]
                typ, r = self._call(t, m, 16)
                warm.append({"prompt": r["prompt_eval"], "ttft": r["ttft"]} if typ == "o" else {"prompt": r["prompt_tokens"] or 0, "ttft": r["t_first"] - r["t0"]})
                convs.append(m + [{"role": "assistant", "content": r["content"]}])     # the warm reply back, byte for byte
        else:
            convs = [[] for _ in range(smax)]; warm = [{"prompt": 0, "ttft": 0} for _ in range(smax)]
        levels, unpinned = [], False
        for li, k in enumerate(lv):
            self._settle(t, anyway); self._msg(f"{k} stream{'s' if k > 1 else ''}: decoding {out} tokens each…")
            res, errs = [None] * k, []
            def one(i):
                try:
                    ask = (ASK0 if li == 0 else ASK) if ctx else f"[{secrets.token_hex(4)}] " + ASK0
                    m = convs[i] + [{"role": "user", "content": ask}]; typ, r = self._call(t, m, out, pin=True); res[i] = (typ, r, m)
                except Exception as e: errs.append(e)
            th = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(k)]
            for x in th: x.start()
            for x in th: x.join()
            if self.ctl["cancel"]: raise Stop(self.ctl["why"] or "cancelled")
            if errs: raise errs[0] if isinstance(errs[0], Stop) else RuntimeError(f"a stream failed: {errs[0]}")
            streams = []
            for i, (typ, r, m) in enumerate(res):
                w = warm[i]
                if typ == "o":
                    rate = r["n"] / r["decode_s"] if r["decode_s"] else None
                    s = {"ttft": round(r["ttft"], 3), "tok_s": round(rate, 1) if rate else None, "tokens": r["n"], "prompt": r["prompt_eval"],
                         "recomputed": bool(ctx) and r["prompt_eval"] > 0.5 * max(1, w["prompt"]), "check": "engine count" if ctx else "no context", "short": r["n"] < out}
                else:
                    n = r["n"] or r["chunks"]; span = (r["t_last"] - r["t_first"]) if r["t_last"] and r["t_first"] else 0
                    rate = (n - 1) / span if span > 0 and n > 1 else None
                    if not ctx: rec, chk = False, "no context"
                    elif r["cached"] is not None: rec, chk = r["cached"] < 0.9 * max(1, w["prompt"]), "engine count"
                    else: rec, chk = (r["t_first"] - r["t0"]) > max(0.3, 0.25 * w["ttft"]), "unchecked (timing only)"
                    s = {"ttft": round(r["t_first"] - r["t0"], 3), "tok_s": round(rate, 1) if rate else None, "tokens": n, "approx": r["n"] is None,
                         "cached": r["cached"], "prompt": r["prompt_tokens"], "recomputed": rec, "check": chk, "short": n < out, "reasoning": r["reasoning_chars"] > 0}
                    unpinned = unpinned or not r["pinned"]
                streams.append(s); convs[i] = m + [{"role": "assistant", "content": r["content"]}]      # the next level extends this one
            lvl = {"streams": k, "per_stream": mean([s["tok_s"] for s in streams]), "ttft": mean([s["ttft"] for s in streams]),
                   "recomputed": sum(1 for s in streams if s["recomputed"]), "short": sum(1 for s in streams if s["short"]), "check": streams[0]["check"], "detail": streams}
            if res[0][0] == "c":
                tf = min(r["t_first"] for _, r, _ in res); tl = max(r["t_last"] for _, r, _ in res)
                lvl["aggregate"] = round(sum((s["tokens"] or 1) - 1 for s in streams) / (tl - tf), 1) if tl > tf else None
                a, b = max(r["t_first"] for _, r, _ in res), min(r["t_last"] for _, r, _ in res)
                if b - a > 0.5:
                    lvl["steady"] = round(sum(sum(1 for x in r["marks"] if a <= x <= b) * ((s["tokens"] or r["chunks"]) / max(1, r["chunks"]))
                                              for (_, r, _), s in zip(res, streams)) / (b - a), 1)
            else: lvl["aggregate"] = lvl["per_stream"]
            levels.append(lvl)
        return {"levels": levels, "warm": [{"prompt": w["prompt"], "ttft": round(w["ttft"], 2)} for w in warm], "pinned": not unpinned, "thinking": False,
                "tok_s": levels[-1].get("aggregate"), "tokens": sum(s["tokens"] or 0 for l in levels for s in l["detail"])}

    def _sweep(self, t, P, anyway):
        pts = []; self._settle(t, anyway); self._msg("sizing the context…"); cpt = self._calibrate(t)
        for i, size in enumerate(P["sizes"]):
            self._settle(t, anyway); self._msg(f"prefill {size:,} tokens ({i + 1}/{len(P['sizes'])})…")
            typ, r = self._call(t, [{"role": "user", "content": context_text(size, 5000 + i, cpt)}], 1)
            if typ == "o":
                rate = r["prompt_eval"] / r["prefill_s"] if r["prefill_s"] else None
                pts.append({"size": size, "tokens": r["prompt_eval"], "ttft": round(r["ttft"], 3), "tok_s": round(rate, 1) if rate else None, "timing": "engine"})
            else:
                ttft = r["t_first"] - r["t0"]; n = r["prompt_tokens"] or 0
                if not n: raise RuntimeError("the engine reported no prompt_tokens")
                pts.append({"size": size, "tokens": n, "ttft": round(ttft, 3), "tok_s": round(n / ttft, 1) if ttft > 0 else None, "timing": "client TTFT",
                            "cache_hit": bool(r["cached"] and r["cached"] > 0.1 * n)})
        return {"points": pts, "tok_s": max((p["tok_s"] or 0) for p in pts) if pts else None, "tokens": sum(p["tokens"] for p in pts)}

    def _quality(self, t, P, anyway):
        items = SUITE_ITEMS(); res = [None] * len(items); errs = []; lock = threading.Lock(); nxt = [0]
        mt = 4096 if P["thinking"] else 768; self._settle(t, anyway)
        def worker():
            while True:
                with lock:
                    if nxt[0] >= len(items) or self.ctl["cancel"]: return
                    i = nxt[0]; nxt[0] += 1
                try: _, r = self._call(t, [{"role": "user", "content": items[i]["q"] + TAIL}], mt, thinking=P["thinking"]); res[i] = grade(items[i], r["content"])
                except Stop: return
                except Exception: res[i] = False; errs.append(i)
                self._msg(f"quality: {sum(1 for x in res if x is not None)}/{len(items)} answered…")
        th = [threading.Thread(target=worker, daemon=True) for _ in range(P["conc"])]
        for x in th: x.start()
        for x in th: x.join()
        if self.ctl["cancel"]: raise Stop(self.ctl["why"] or "cancelled")
        if len(errs) > 2: raise RuntimeError(f"{len(errs)} of {len(items)} questions failed at the engine (HTTP / timeout / dropped stream): not saved as a quality run")
        cats = {}
        for it, ok in zip(items, res):
            c = cats.setdefault(it["cat"], [0, 0]); c[1] += 1; c[0] += 1 if ok else 0
        right = sum(1 for x in res if x)
        return {"suite": SUITE, "seed": SUITE_SEED, "thinking": P["thinking"], "n": len(items), "right": right, "score": round(100 * right / len(items), 1),
                "cats": cats, "items": "".join("1" if x else "0" for x in res), "errors": len(errs), "tok_s": None}

    def _save_quality(self, row):
        rows = self.quality_runs(); rows.append(row); keep, count = [], {}
        for r in reversed(rows):
            k = r.get("key"); count[k] = count.get(k, 0) + 1
            if count[k] <= 30: keep.append(r)
        tmp = f"{self.QUAL}.tmp{os.getpid()}"
        with open(tmp, "w") as f: f.write("".join(json.dumps(r) + "\n" for r in reversed(keep)))
        os.replace(tmp, self.QUAL)

    def quality_runs(self, key=None):
        try:
            with open(self.QUAL) as f: rows = [json.loads(l) for l in f if l.strip()]
        except OSError: rows = []
        return [r for r in rows if key is None or r.get("key") == key]

    def compare(self, key, i, j):
        rs = self.quality_runs(key)
        try: a, b = rs[int(i)], rs[int(j)]
        except (IndexError, ValueError, TypeError): return {"error": "no such runs", "status": 400}
        if (a.get("suite"), a.get("seed")) != (b.get("suite"), b.get("seed")): return {"error": "the runs used different suites", "status": 409}
        if a.get("errors") or b.get("errors"): return {"error": "a run had engine errors — run it again before comparing", "status": 409}
        x, y, p = mcnemar(a["items"], b["items"])
        return {"a": {k: a.get(k) for k in ("ts", "score", "thinking")}, "b": {k: b.get(k) for k in ("ts", "score", "thinking")}, "a_only": x, "b_only": y,
                "p": round(p, 4), "cats": {c: [a["cats"].get(c), b["cats"].get(c)] for c in a.get("cats", {})},
                "note": "p is for the whole suite; the categories are raw counts (10 questions each is too few for a test)"}

    def state(self, t):
        rows = []
        try:
            with open(self.deck.BENCH) as f: rows = [json.loads(l) for l in f if l.strip()][-300:]
        except OSError: pass
        key = (t or {}).get("key"); last = {}
        for x in rows:
            if x.get("key") == key and x.get("kind") in ("decode2", "sweep"): last[x["kind"]] = x
        q = [{k: r.get(k) for k in ("ts", "score", "right", "n", "thinking", "cats", "contaminated", "anyway", "errors", "ollama_reloaded", "thinking_set", "secs")}
             for r in self.quality_runs(key)][-10:] if key else []
        return {"last": last, "quality": q, "quality_n": len(self.quality_runs(key)) if key else 0, "ctx_choices": CTX_CHOICES, "levels": LEVELS, "outs": OUTS}

    # history + health (called by the deck's sampler) ------------------------------------------------------------------------
    def _cpu(self):
        try:
            with open("/proc/stat") as f: v = [int(x) for x in f.readline().split()[1:]]
        except Exception: return None
        idle, tot = v[3] + (v[4] if len(v) > 4 else 0), sum(v); prev = self.cpu_prev; self.cpu_prev = (idle, tot)
        if not prev or tot <= prev[1]: return None
        return round(100 * (1 - (idle - prev[0]) / (tot - prev[1])), 1)

    def tick(self, live=None, res=None, now=None):
        """one sampler tick: a history point, the health findings, the notifications. Never raises."""
        now = now or time.time()
        try:
            g = self.io.gpu(); mem = self.deck.sys.meminfo() if hasattr(self.deck.sys, "meminfo") else {}
            avail = (res or {}).get("mem_avail_gib") or round(mem.get("MemAvailable", 0), 1) or None
            total = (res or {}).get("mem_total_gib") or round(mem.get("MemTotal", 0), 1) or None
            p = {"cpu": self._cpu(), "avail": avail, "mem": round(total - avail, 1) if total and avail else None,
                 "util": g.get("util"), "temp": g.get("temp"), "power": g.get("power"),
                 "tok": (live or {}).get("decode"), "ptok": (live or {}).get("prefill")}
            self.hist.add(p, now)
        except Exception: p = {}
        try: self.notify_tick(self.health(p, res, now), now)
        except Exception: pass

    def health(self, p, res=None, now=None):
        now = now or time.time(); f = []; hc = self.deck.cfg.get("health") or {}
        hot = float(hc.get("gpu_hot_c") or 85); low = float(hc.get("mem_low_gib") or MEM_ABORT_GIB)
        if isinstance(p.get("avail"), (int, float)) and p["avail"] < low: f.append(("mem_low", "crit", f"memory low (under {low:g} GiB free)", False))
        if isinstance(p.get("temp"), (int, float)) and p["temp"] >= hot: f.append(("gpu_hot", "crit", "GPU hot", False))
        if res and res.get("free_disk_gib") is not None and res["free_disk_gib"] < float(hc.get("disk_low_gib") or 50):
            f.append(("disk_low", "warn", "local disk nearly full", False))
        if hc.get("kernel_log", True) and now - self.H["kt"] >= 30:
            since = self.H["since"] or now - 60; self.H["kt"] = now; self.H["since"] = now
            log = self.io.kernel_log(since) or ""
            if re.search(r"NVRM: Xid", log): self.H["events"]["gpu_xid"] = (now, "crit", "GPU error (NVIDIA Xid) in the kernel log")
            if re.search(r"Out of memory: Killed process|oom-kill:", log): self.H["events"]["oom_kill"] = (now, "crit", "the kernel killed a process (out of memory)")
        playing = bool(_rjson(self.deck.PLAYING, {}).get("id")); up = bool((res or {}).get("engines_running")) or bool((res or {}).get("ollama"))
        if self.deck.job["running"]: self.H["cand"] = None
        elif playing and not up and self.H["cand"] is None: self.H["cand"] = now
        if up or not playing: self.H["cand"] = None
        if self.H["cand"] is not None and now - self.H["cand"] >= 60:
            try: recent = now - os.path.getmtime(self.deck.ACTIONS) < now - self.H["cand"] + 120
            except OSError: recent = False
            if not recent: self.H["events"]["lane_down"] = (now, "warn", "the playing cassette's engine stopped with no EJECT")
            self.H["cand"] = None
        for k, (t0, lvl, txt) in list(self.H["events"].items()):
            if now - t0 > self.EVENT_HOLD_S: self.H["events"].pop(k); continue
            f.append((k, lvl, txt, True))
        self.findings = [{"key": k, "lvl": l, "text": t} for k, l, t, _ in f]
        return f

    # notifications -------------------------------------------------------------------------------------------------------
    def notify_cfg(self):
        d = _rjson(self.NOTIFY, {}); return d if isinstance(d, dict) else {}

    def _save_cfg(self, d):
        tmp = f"{self.NOTIFY}.tmp{os.getpid()}"; fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f: json.dump(d, f)
        os.replace(tmp, self.NOTIFY); os.chmod(self.NOTIFY, 0o600)

    def notify_tick(self, findings, now=None):
        now = now or time.time(); cfg = self.notify_cfg()
        if not (cfg.get("enabled") and cfg.get("topic")): self.N["active"] = {}; return []
        hold = int(cfg.get("hold_s") or 60); sent, seen = [], set()
        for key, lvl, txt, ev in findings:
            seen.add(key); a = self.N["active"].setdefault(key, {"t0": now, "sent": False, "txt": txt, "ev": ev})
            if not a["sent"] and (ev or now - a["t0"] >= hold) and now - self.N["last_raise"].get(key, float("-inf")) >= self.RAISE_EVERY_S:
                a["sent"] = True; self.N["last_raise"][key] = now; sent.append((key, lvl, txt))
        for key in [k for k in self.N["active"] if k not in seen]:
            a = self.N["active"].pop(key)
            if a["sent"] and not a["ev"]: sent.append((key, "info", f"{a['txt']} (cleared)"))
        for key, lvl, msg in sent: threading.Thread(target=self.send, args=(cfg, key, lvl, msg), daemon=True).start()
        return sent

    def send(self, cfg, key, lvl, msg):
        """logs the alert key, ok and an exception CLASS only: a reason or a traceback could carry the server or the topic"""
        ok, err = False, ""
        try: self.io.notify(cfg, lvl, msg); ok = True
        except Exception as e: err = str(e) if re.fullmatch(r"HTTP \d{3}", str(e)) else type(e).__name__
        try:
            with open(self.NLOG, "a") as f: f.write(json.dumps({"ts": _ts(), "key": key, "lvl": lvl, "ok": ok, "err": err}) + "\n")
        except OSError: pass
        return ok, err

    def notify_view(self):
        c = self.notify_cfg(); tp = c.get("topic") or ""
        try:
            with open(self.NLOG) as f: log = [json.loads(l) for l in f if l.strip()][-20:]
        except OSError: log = []
        return {"enabled": bool(c.get("enabled")), "server": c.get("server") or "https://ntfy.sh", "topic_set": bool(tp), "topic_hint": ("…" + tp[-4:]) if tp else "",
                "token_set": bool(c.get("token")), "hold_s": int(c.get("hold_s") or 60), "log": log[::-1], "findings": getattr(self, "findings", []),
                "active": [{"kind": a["txt"], "since": int(a["t0"]), "sent": a["sent"]} for a in self.N["active"].values()]}

    def notify_post(self, b):
        b = b if isinstance(b, dict) else {}; act = b.get("action"); c = self.notify_cfg()
        if act == "new_topic":
            c["topic"] = "lion-EXAMPLE-demo-topic" if isinstance(self.io, DemoIO) else "lion-" + secrets.token_urlsafe(18); self._save_cfg(c)
            return {"topic": c["topic"], "server": c.get("server") or "https://ntfy.sh"}
        if act == "reveal":
            if not c.get("topic"): return {"error": "no topic yet — create one", "status": 409}
            return {"topic": c["topic"], "server": c.get("server") or "https://ntfy.sh"}
        if act == "save":
            sv = str(b.get("server") or c.get("server") or "https://ntfy.sh").strip().rstrip("/")
            try: server_ok(sv)
            except ValueError as e: return {"error": str(e), "status": 400}
            hold = b.get("hold_s", c.get("hold_s", 60))
            if not isinstance(hold, int) or isinstance(hold, bool) or not 30 <= hold <= 3600: return {"error": "hold must be 30 … 3600 seconds", "status": 400}
            c.update(server=sv, hold_s=hold, enabled=b.get("enabled") is True)
            if "token" in b:
                tk = str(b.get("token") or "").strip()
                if tk and not re.fullmatch(r"tk_[A-Za-z0-9]{20,64}", tk): return {"error": "an ntfy access token looks like tk_…", "status": 400}
                c["token"] = tk
            if c["enabled"] and not c.get("topic"): return {"error": "create a topic first", "status": 409}
            self._save_cfg(c); return self.notify_view()
        if act == "test":
            if not c.get("topic"): return {"error": "create a topic first", "status": 409}
            if not isinstance(self.io, DemoIO):
                try: public_addr(*server_ok(c.get("server") or "https://ntfy.sh"))
                except Exception as e: return {"error": f"the server is refused ({e if isinstance(e, ValueError) else type(e).__name__})", "status": 400}
            ok, err = self.send(c, "test", "info", "test notification from Lion AI Models")
            return {"ok": ok, "error": None if ok else f"not delivered ({err})", "status": 200 if ok else 502}
        return {"error": "unknown action", "status": 400}


def summary(r):
    if r.get("status") != "done": return f"{r['kind']} {r.get('status')}: {r.get('why', '')}"
    fl = " · flagged: another client used the engine" if r.get("contaminated") else ""
    if r["kind"] == "decode2":
        top = (r.get("levels") or [{}])[-1]
        return f"decode at {r['params']['ctx']:,} context: {top.get('streams')} × {top.get('per_stream')} tok/s · {top.get('aggregate')} tok/s total{fl}"
    if r["kind"] == "sweep":
        return f"prefill sweep: {len(r.get('points') or [])} points, up to {max((p['tok_s'] or 0) for p in r.get('points') or [{'tok_s': 0}]):,} tok/s{fl}"
    return f"quality: {r.get('right')}/{r.get('n')} ({r.get('score')}%){fl}"


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return round(sum(v) / len(v), 1) if v else None
