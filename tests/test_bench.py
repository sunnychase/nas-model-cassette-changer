"""v1.5 tests for deck/mcc_bench.py: long-context decode, the prefill sweep, the quality suite, history and the ntfy sender, against
stub engines on 127.0.0.1 (standard library only; nothing real is touched)."""
import json, os, shutil, socket, ssl, subprocess, sys, tempfile, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deck"))
import mcc_bench as B   # noqa: E402
import mcc_deck as M    # noqa: E402


def flat(msgs): return "".join(f"<{m['role']}>{m['content']}" for m in msgs)


class Eng:
    """a stub OpenAI engine: strict-prefix prompt cache, usage with cached tokens, SSE; modes switch behaviours on"""
    mode = set(); kept = []; active = 0; extra = 0; bodies = []; rate = 400.0; answer = None; nq = 0; lock = threading.Lock()


class EH(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _j(self, code, o):
        b = json.dumps(o).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self._j(200, {"status": "ok", "live": {"connections": Eng.active + Eng.extra, "waiting": 0}})
        self._j(404, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(n)); Eng.bodies.append(body)
        if "no_pin" in Eng.mode and ("ignore_eos" in body or "min_tokens" in body): return self._j(400, {"error": "unknown field ignore_eos"})
        if "no_ctk" in Eng.mode and "chat_template_kwargs" in body: return self._j(400, {"error": "Extra inputs are not permitted: chat_template_kwargs"})
        if body["max_tokens"] == 768 and "fail_q" in Eng.mode:
            Eng.nq += 1
            if Eng.nq % 10 == 1: return self._j(500, {"error": "boom"})
        text = flat(body["messages"]); pt = len(text) // 4
        with Eng.lock:
            hit = "" if "nocache" in Eng.mode else max((k for k in Eng.kept if text.startswith(k) and len(k) < len(text)), key=len, default="")
            Eng.kept.append(text); Eng.active += 1
        try:
            time.sleep((pt - len(hit) // 4) / 200000)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Connection", "close"); self.end_headers()
            want = body["max_tokens"] if (body.get("ignore_eos") or "early_eos" not in Eng.mode) else min(body["max_tokens"], 100)
            def send(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n"); self.wfile.flush()
            rs = 30 if "reason_first" in Eng.mode else 0
            for _ in range(rs): send({"choices": [{"delta": {"reasoning_content": "hmm "}}]}); time.sleep(1 / Eng.rate)
            k = 1 if Eng.answer else want
            for _ in range(k): send({"choices": [{"delta": {"content": Eng.answer or "word "}}]}); time.sleep(1 / Eng.rate)
            u = {"prompt_tokens": pt, "completion_tokens": k + rs}
            if "nocount" not in Eng.mode: u["prompt_tokens_details"] = {"cached_tokens": len(hit) // 4}
            send({"choices": [{"delta": {}, "finish_reason": "length"}], "usage": u}); self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
        finally:
            with Eng.lock: Eng.active -= 1


class OH(BaseHTTPRequestHandler):
    """a stub Ollama: /api/ps, /api/chat with its own timings"""
    protocol_version = "HTTP/1.1"; bodies = []; paths = []; last = ""; load = 1000000
    def log_message(self, *a): pass
    def _j(self, o):
        b = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        exp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 600))
        self._j({"models": [{"name": "m:7b", "context_length": 32768, "expires_at": exp}]})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); b = json.loads(self.rfile.read(n)); OH.bodies.append(b); OH.paths.append(self.path)
        text = flat(b["messages"]); pe = len(text) // 4 - (len(OH.last) // 4 if text.startswith(OH.last) and OH.last else 0); OH.last = text
        np_ = b["options"]["num_predict"]
        self._j({"message": {"content": "ok"}, "eval_count": np_, "eval_duration": int(np_ / 50 * 1e9), "prompt_eval_count": max(pe, 1),
                 "prompt_eval_duration": int(max(pe, 1) / 3000 * 1e9), "load_duration": OH.load})


def serve(h):
    s = ThreadingHTTPServer(("127.0.0.1", 0), h); s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start(); return s


class FakeDeck:
    def __init__(self, state, target, recipe):
        self.cfg = M.load_config(None); self.cfg["state_dir"] = state; self.t = target; self.r = recipe
        self.job = {"running": False}; self.bjob = {"running": False, "kind": "", "msg": ""}; self.lock = threading.Lock()
        self.BENCH, self.PLAYING, self.ACTIONS = f"{state}/bench.jsonl", f"{state}/playing.json", f"{state}/actions.jsonl"
        self.expires_left = M.expires_left; self.sys = M.System(self.cfg)
    def target(self): return dict(self.t) if self.t else None
    def recipe_for_target(self, t): return self.r if t.get("engine") == "TensorFold" else None


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E = serve(EH); cls.O = serve(OH)
    @classmethod
    def tearDownClass(cls):
        cls.E.shutdown(); cls.O.shutdown()

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="mcc-bench-test-")
        Eng.mode, Eng.kept, Eng.extra, Eng.bodies, Eng.answer, Eng.nq, Eng.rate = set(), [], 0, [], None, 0, 400.0
        self.T = {"key": "TensorFold:stub", "engine": "TensorFold", "model": "stub", "port": self.E.server_address[1], "health_path": "/health"}
        self.R = {"label": "TensorFold", "port": self.T["port"], "context": 262144, "parallel": 8, "kv_bytes_per_token": 17888, "memory_reserve_gib": 10}
        self.deck = FakeDeck(self.d, self.T, self.R); self.mem = {"v": 200.0}
        io = B.RealIO(f"http://127.0.0.1:{self.O.server_address[1]}"); io.mem_avail = lambda: self.mem["v"]; io.gpu = lambda: {}
        self.b = B.Bench(self.deck, io)

    def tearDown(self): shutil.rmtree(self.d, ignore_errors=True)

    def run_job(self, kind, p, wait=120):
        r = self.b.start(kind, p)
        if not r.get("started"): return r
        for _ in range(wait * 10):
            if not self.deck.bjob["running"]: break
            time.sleep(0.1)
        with open(self.deck.BENCH) if os.path.exists(self.deck.BENCH) else open(os.devnull) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        return rows[-1] if rows else None


class TestDecode(Base):
    def test_clean_run_reuses_the_context(self):
        row = self.run_job("decode2", {"ctx": 16384, "levels": [1, 2, 4], "out": 256})
        self.assertEqual(row["status"], "done", row)
        self.assertLess(abs(row["warm"][0]["prompt"] - 16384) / 16384, 0.06)
        self.assertEqual([l["streams"] for l in row["levels"]], [1, 2, 4])
        self.assertEqual(sum(l["recomputed"] for l in row["levels"]), 0)
        self.assertEqual(row["levels"][0]["check"], "engine count")
        self.assertTrue(row["levels"][-1].get("aggregate") and row["levels"][-1].get("steady"))
        self.assertTrue(row["pinned"])
        wave = [b for b in Eng.bodies if b["max_tokens"] == 256]
        self.assertTrue(all(b["chat_template_kwargs"] == {"enable_thinking": False} and b["temperature"] == 0 for b in wave))
        self.assertEqual(wave[0]["messages"][1], {"role": "assistant", "content": "word " * 16})     # the warm reply, verbatim
        self.assertTrue(all(b["messages"][0]["content"].startswith("[") for b in wave))            # the nonce first

    def test_cache_miss_is_flagged(self):
        Eng.mode = {"nocache"}
        row = self.run_job("decode2", {"ctx": 16384, "levels": [2], "out": 256})
        self.assertEqual(row["levels"][0]["recomputed"], 2)

    def test_no_cached_count_is_unchecked(self):
        Eng.mode = {"nocount"}
        row = self.run_job("decode2", {"ctx": 16384, "levels": [1], "out": 256})
        self.assertTrue(row["levels"][0]["check"].startswith("unchecked"))

    def test_reasoning_starts_the_clock(self):
        Eng.mode = {"reason_first"}
        row = self.run_job("decode2", {"ctx": 0, "levels": [1], "out": 256})
        self.assertLess(row["levels"][0]["ttft"], 0.2); self.assertTrue(row["levels"][0]["detail"][0]["reasoning"])

    def test_unpinned_and_short(self):
        Eng.mode = {"no_pin", "early_eos"}
        row = self.run_job("decode2", {"ctx": 0, "levels": [1], "out": 256})
        self.assertEqual(row["status"], "done"); self.assertFalse(row["pinned"]); self.assertEqual(row["levels"][0]["short"], 1)

    def test_no_chat_template_kwargs(self):
        Eng.mode = {"no_ctk"}
        row = self.run_job("decode2", {"ctx": 0, "levels": [1], "out": 256})
        self.assertEqual(row["status"], "done"); self.assertIs(row.get("thinking_set"), False)

    def test_another_client_stops_the_run_or_flags_it(self):
        Eng.rate = 150
        threading.Thread(target=lambda: (time.sleep(2.5), setattr(Eng, "extra", 1)), daemon=True).start()
        row = self.run_job("decode2", {"ctx": 0, "levels": [1, 2, 4, 8], "out": 1024})
        self.assertEqual(row["status"], "stopped"); self.assertIn("another client", row["why"])
        Eng.extra = 0
        threading.Thread(target=lambda: (time.sleep(2.5), setattr(Eng, "extra", 1)), daemon=True).start()
        row = self.run_job("decode2", {"ctx": 0, "levels": [1, 2], "out": 1024, "anyway": True})
        self.assertEqual(row["status"], "done"); self.assertTrue(row["contaminated"] and row["anyway"])
        r = self.b.start("decode2", {"ctx": 0, "levels": [1], "out": 256})
        self.assertEqual(r.get("status"), 409); self.assertIn("other request", r["error"])

    def test_cancel(self):
        threading.Thread(target=lambda: (time.sleep(1.5), self.b.cancel()), daemon=True).start()
        row = self.run_job("decode2", {"ctx": 0, "levels": [4], "out": 1024})
        self.assertEqual(row["status"], "stopped"); self.assertEqual(self.b.io.ours(), 0)

    def test_cancel_refuses_the_short_benchmark(self):
        self.deck.bjob.update(running=True, kind="decode")
        self.assertEqual(self.b.cancel().get("status"), 409)

    def test_memory_watch_stops_the_run(self):
        threading.Thread(target=lambda: (time.sleep(1.5), self.mem.update(v=6.0)), daemon=True).start()
        row = self.run_job("decode2", {"ctx": 0, "levels": [2], "out": 1024})
        self.assertEqual(row["status"], "stopped"); self.assertIn("memory", row["why"])


class TestPlan(Base):
    def test_memory_plan(self):
        self.mem["v"] = 28.0
        pl = self.b.plan("decode2", {"ctx": 131072, "levels": [8], "out": 512})
        self.assertFalse(pl["ok"]); self.assertIn("needs ~", pl["refuse"])
        self.assertTrue(self.b.plan("decode2", {"ctx": 65536, "levels": [1], "out": 512})["ok"])

    def test_window(self):
        pl = self.b.plan("decode2", {"ctx": 262144, "levels": [1], "out": 512})
        self.assertTrue(pl["ok"]); self.assertEqual(pl["params"]["ctx"], 262144 - 512 - 1024)
        self.assertIn("the most it takes", self.b.plan("decode2", {"ctx": 262000, "levels": [1, 2], "out": 1024}).get("refuse", ""))

    def test_refusals(self):
        self.R["parallel"] = 4
        self.assertIn("more than this lane", self.b.plan("decode2", {"ctx": 0, "levels": [8]}).get("refuse", ""))
        self.assertFalse(self.b.plan("decode2", {"ctx": 0, "levels": [1], "out": 300})["ok"])
        self.assertFalse(self.b.plan("nope", {})["ok"])
        self.deck.t = None
        self.assertIn("nothing is loaded", self.b.plan("decode2", {}).get("refuse", ""))

    def test_sweep_counts_the_kept_points(self):
        self.mem["v"] = 24.0
        self.assertEqual(max(self.b.plan("sweep", {})["params"]["sizes"]), 65536)

    def test_sweep_runs(self):
        row = self.run_job("sweep", {})
        self.assertEqual(row["status"], "done"); self.assertGreaterEqual(len(row["points"]), 5)
        self.assertFalse(any(p["cache_hit"] for p in row["points"]))


class TestOllama(Base):
    def setUp(self):
        super().setUp()
        self.deck.t = {"key": "Ollama:m:7b", "engine": "Ollama", "model": "m:7b"}; OH.bodies, OH.paths, OH.last, OH.load = [], [], "", 1000000

    def test_native_api_resident_context(self):
        row = self.run_job("decode2", {"ctx": 16384, "levels": [1], "out": 256})
        self.assertEqual(row["status"], "done", row)
        self.assertTrue(OH.paths and all(p == "/api/chat" for p in OH.paths))
        self.assertTrue(all(b["options"]["num_ctx"] == 32768 for b in OH.bodies))
        self.assertTrue(all(b["keep_alive"] == -1 or int(str(b["keep_alive"]).rstrip("s")) >= 300 for b in OH.bodies))
        self.assertEqual(row["levels"][0]["recomputed"], 0); self.assertIs(row["others_visible"], False)
        self.assertFalse(self.b.plan("decode2", {"ctx": 0, "levels": [2]})["ok"])
        self.assertTrue(any("not visible" in n for n in self.b.plan("decode2", {"ctx": 0, "levels": [1]})["notes"]))

    def test_reload_is_said(self):
        OH.load = int(2e9)
        row = self.run_job("decode2", {"ctx": 0, "levels": [1], "out": 256})
        self.assertIs(row.get("ollama_reloaded"), True)


class TestQuality(Base):
    def test_suite(self):
        items = B.SUITE_ITEMS()
        self.assertEqual(len(items), 60); self.assertEqual(len({i["cat"] for i in items}), 6)
        self.assertEqual(B.build_suite()[33]["a"], items[33]["a"])
        self.assertTrue(all(B.grade(i, "text\nANSWER: " + i["a"]) for i in items))
        self.assertFalse(any(B.grade(i, "ANSWER: zzz-wrong") for i in items))
        self.assertTrue(all(i["a"].lstrip("-").isdigit() for i in items if i["cat"] == "units"))

    def test_grader_rules(self):
        self.assertTrue(B.grade({"a": "5", "g": "int"}, "ANSWER: 4\nno\nANSWER: 5"))
        self.assertTrue(B.grade({"a": "1234", "g": "int"}, "ANSWER: 1,234"))
        self.assertTrue(B.grade({"a": "900", "g": "int", "cat": "units"}, "ANSWER: 900 minutes"))
        self.assertFalse(B.grade({"a": "900", "g": "int", "cat": "arithmetic"}, "ANSWER: 900 minutes"))
        self.assertFalse(B.grade({"a": "abc", "g": "exact"}, "ANSWER: ABC"))
        self.assertFalse(B.grade({"a": "B,A", "g": "list_upper"}, "ANSWER: b, a")); self.assertTrue(B.grade({"a": "B,A", "g": "list_upper"}, "ANSWER: B, A"))
        self.assertFalse(B.grade({"a": "5", "g": "int"}, "5"))

    def test_mcnemar(self):
        self.assertEqual(B.mcnemar("1100", "0011"), (2, 2, 1.0))
        self.assertAlmostEqual(B.mcnemar("1" * 10, "0" * 10)[2], 2 / 1024)
        self.assertEqual(B.mcnemar("1010", "1010")[2], 1.0)

    def test_run_errors_and_compare(self):
        Eng.answer = "ANSWER: 42"; self.run_job("quality", {})
        q = self.b.quality_runs("TensorFold:stub")
        self.assertEqual((q[-1]["n"], len(q[-1]["items"]), q[-1]["errors"]), (60, 60, 0))
        Eng.mode = {"fail_q"}; n0 = len(self.b.quality_runs()); self.run_job("quality", {})
        self.assertEqual(len(self.b.quality_runs()), n0); self.assertIn("failed", self.deck.bjob["msg"])
        for i in range(32):
            self.b._save_quality({"ts": f"2026-10-25T00:00:{i:02d}", "key": "TensorFold:stub", "suite": "q1", "seed": 1025, "items": "1" * 30 + "0" * 30, "cats": {}, "score": 50, "errors": i % 2})
        self.assertEqual(len(self.b.quality_runs("TensorFold:stub")), 30)
        self.assertEqual(self.b.compare("TensorFold:stub", 28, 29).get("status"), 409)       # one of them had an engine error
        self.assertEqual(self.b.compare("TensorFold:stub", 0, 99).get("status"), 400)


class TestHistory(unittest.TestCase):
    def test_rollups_and_trim(self):
        d = tempfile.mkdtemp(); h = B.History(d); t0 = (int(time.time()) // 900) * 900 - 3600
        try:
            for i in range(0, 40 * 60, 2):
                h.add({"cpu": 10, "mem": 50, "avail": 70, "util": 30, "power": 40, "temp": 40 + (i // 60) % 5, "tok": 5 if i % 120 == 0 else 0}, t0 + i)
            m1 = [json.loads(l) for l in open(h.M1)]; m15 = [json.loads(l) for l in open(h.M15)]
            self.assertEqual((len(m1), len(m15)), (39, 2))
            self.assertEqual((m1[0]["util"], m1[0]["avail"]), (30, 70)); self.assertTrue(any(r.get("tok") == 5 for r in m1))
            self.assertTrue(all(r.get("tmax") for r in m15))
            self.assertEqual(len(h.read("48h", now=t0 + 2400)["points"]), 39)
            with open(h.M1, "a") as f: f.write(json.dumps({"t": t0 - 3 * 86400}) + "\n")
            B.History._trim(h.M1, 48 * 3600, t0 + 2400)
            self.assertTrue(all(json.loads(l)["t"] >= t0 - 48 * 3600 for l in open(h.M1)))
            with open(h.M15, "w") as f: f.writelines(json.dumps({"t": t0 + i * 60, "temp": 40, "tmax": 50 + i % 3}) + "\n" for i in range(3000))
            r = h.read("30d", now=t0 + 3000 * 60)
            self.assertLessEqual(len(r["points"]), 721); self.assertEqual(max(p.get("tmax", 0) for p in r["points"]), 52)
        finally: shutil.rmtree(d, ignore_errors=True)


class TestNotify(Base):
    def test_server_rules(self):
        for bad in ("http://ntfy.sh", "https://ntfy.sh/topic", "https://u:p@ntfy.sh", "ftp://x", "https://ntfy.sh?x=1"):
            with self.assertRaises(ValueError): B.server_ok(bad)
        for host in ("127.0.0.1", "localhost", "fd12:3456::1", "169.254.1.1", "100.64.3.4", "::1", "0.0.0.0"):
            with self.assertRaises((ValueError, OSError)): B.public_addr(host, 443)

    def test_settings(self):
        self.assertEqual(self.b.notify_post({"action": "save", "enabled": True}).get("status"), 409)
        r = self.b.notify_post({"action": "new_topic"})
        self.assertTrue(r["topic"].startswith("lion-") and len(r["topic"]) >= 29)
        self.assertEqual(oct(os.stat(self.b.NOTIFY).st_mode & 0o777), "0o600")
        v = self.b.notify_view(); self.assertNotIn(r["topic"], json.dumps(v))
        self.assertEqual(self.b.notify_post({"action": "save", "token": "nope"}).get("status"), 400)
        self.assertEqual(self.b.notify_post({"action": "save", "hold_s": 5}).get("status"), 400)

    def test_state_machine(self):
        self.b.notify_post({"action": "new_topic"}); self.b.notify_post({"action": "save", "enabled": True, "hold_s": 60})
        sent = []; self.b.send = lambda c, k, l, m: sent.append((k, m))
        f = [("mem_low", "crit", "memory low", False)]
        self.b.notify_tick(f, now=1000); self.b.notify_tick(f, now=1030); time.sleep(0.1); self.assertEqual(sent, [])
        self.b.notify_tick(f, now=1061); time.sleep(0.2); self.assertEqual(sent, [("mem_low", "memory low")])
        self.b.notify_tick(f, now=1100); self.b.notify_tick([], now=1200); time.sleep(0.2); self.assertEqual(sent[-1][1], "memory low (cleared)")
        self.b.notify_tick(f, now=1300); self.b.notify_tick(f, now=1400); time.sleep(0.1); self.assertEqual(len(sent), 2)        # one raise per 30 min
        self.b.notify_tick([("gpu_xid", "crit", "GPU error", True)], now=1401); time.sleep(0.2); self.assertEqual(sent[-1][1], "GPU error")
        self.b.notify_tick([], now=1500); time.sleep(0.2); self.assertNotIn("GPU error (cleared)", [m for _, m in sent])

    @unittest.skipUnless(shutil.which("openssl"), "needs openssl to make a test certificate")
    def test_redirect_not_followed_and_no_topic_in_the_log(self):
        d = tempfile.mkdtemp()
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", f"{d}/k.pem", "-out", f"{d}/c.pem", "-days", "1", "-subj", "/CN=localhost"], capture_output=True, check=True)
        hits = []
        class RH(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                hits.append(self.path); self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(302); self.send_header("Location", "https://127.0.0.1:1/x"); self.send_header("Content-Length", "0"); self.end_headers()
        rs = ThreadingHTTPServer(("127.0.0.1", 0), RH); sc = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); sc.load_cert_chain(f"{d}/c.pem", f"{d}/k.pem")
        rs.socket = sc.wrap_socket(rs.socket, server_side=True); threading.Thread(target=rs.serve_forever, daemon=True).start()
        real_pub, real_ctx = B.public_addr, ssl.create_default_context
        try:
            B.public_addr = lambda host, port: (socket.AF_INET, ("127.0.0.1", rs.server_address[1]))
            def unverified():
                c = real_ctx(); c.check_hostname = False; c.verify_mode = ssl.CERT_NONE; return c
            ssl.create_default_context = unverified
            ok, err = self.b.send({"server": "https://ntfy.example", "topic": "lion-SECRETTOPIC123"}, "t", "info", "x")
        finally:
            B.public_addr, ssl.create_default_context = real_pub, real_ctx; rs.shutdown(); shutil.rmtree(d, ignore_errors=True)
        self.assertEqual((ok, err, len(hits)), (False, "HTTP 302", 1))
        with open(self.b.NLOG) as f: log = f.read()
        self.assertNotIn("SECRETTOPIC", log); self.assertNotIn("ntfy.example", log)


class TestDemoDeck(unittest.TestCase):
    def test_demo_round_trip(self):
        """--demo: a TensorFold recipe plays, a long-context decode and a quality run go through the deck, history is served"""
        root = tempfile.mkdtemp()
        try:
            cfg = M.load_config(None); cfg.update(state_dir=f"{root}/state", local_dir=f"{root}/deck", protected=["qwen3:14b"])
            cfg["_recipes"] = {k: {**M.RECIPE_DEFAULTS, **v, "key": k} for k, v in M.DEMO_RECIPES.items()}
            cfg["_archs"] = {"vllm": set(M.DEMO_ENGINE_ARCHS), "sglang": set(M.DEMO_ENGINE_ARCHS)}; cfg["_archs_fixed"] = True
            M.demo_seed(cfg); deck = M.Deck(cfg, M.DemoSystem(cfg, ollama_up=False))
            mid = "Qwen/Qwen3.8-Flash-Next/Vontra-MLX-4bit-MTP"
            self.assertTrue(deck.insert(mid, "TensorFold recipe").get("started"))
            for _ in range(200):
                if not deck.job["running"]: break
                time.sleep(0.05)
            pl = deck.bench2.plan("decode2", {"ctx": 16384, "levels": [1, 2]}); self.assertTrue(pl["ok"], pl)
            self.assertTrue(deck.bench2.start("decode2", {"ctx": 16384, "levels": [1, 2]}).get("started"))
            for _ in range(600):
                if not deck.bjob["running"]: break
                time.sleep(0.05)
            st = deck.state(); self.assertEqual(st["bench"]["v2"]["last"]["decode2"]["status"], "done")
            self.assertGreater(len(deck.bench2.hist.read("48h")["points"]), 100)
            self.assertIn("findings", deck.bench2.notify_view())
        finally: shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
