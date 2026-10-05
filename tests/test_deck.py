import json, os, sys, tempfile, time, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deck"))
import mcc_deck as D  # noqa: E402

GiB = 2**30


def cfg(**kw):
    c = json.loads(json.dumps(D.DEFAULTS)); c.update(kw); return c


def res(mem=100.0, ollama=(), tags=None, max_loaded=3, vllm=False, lanes=()):
    oll = [{"name": n, "gib": g} for n, g in ollama]
    return {"lanes": list(lanes), "ollama": oll, "tags_gib": tags if tags is not None else {n: g for n, g in ollama}, "ollama_ok": True,
            "observed_gib": {}, "max_loaded": max_loaded, "free_disk_gib": 2000.0, "mem_avail_gib": mem, "mem_total_gib": 121.7, "vllm_running": vllm}


GGUF = {"id": "Qwen/Qwen3-32B/unsloth-GGUF", "category": "LLMs", "format": "gguf", "runtime": ["llama.cpp", "ollama"], "status": "complete"}
ST = {"id": "Qwen/Qwen3-32B/original", "category": "LLMs", "format": "safetensors", "runtime": ["vllm"], "status": "complete"}


class Fit(unittest.TestCase):
    F = D.DEFAULTS["fit"]

    def test_small_fits_one(self): self.assertEqual(D.nodes_needed(20e9, self.F), 1)

    def test_boundary(self):
        # one node holds W with W + 5.6 + 8 <= 95.69  ->  W <= 82.09 GiB
        self.assertEqual(D.nodes_needed(82.0 * GiB, self.F), 1)
        self.assertEqual(D.nodes_needed(82.2 * GiB, self.F), 2)

    def test_beyond(self): self.assertIsNone(D.nodes_needed(700e9, self.F))

    def test_monotonic(self):
        last = 1
        for gb in range(1, 400, 3):
            k = D.nodes_needed(gb * 1e9, self.F) or 99
            self.assertGreaterEqual(k, last); last = k


class Units(unittest.TestCase):
    def test_shards_grouped_and_mmproj_attached(self):
        L = [(10, "M/Mod/V/Mod-Q4_K_M-00001-of-00002.gguf"), (12, "M/Mod/V/Mod-Q4_K_M-00002-of-00002.gguf"),
             (30, "M/Mod/V/Mod-Q8_0.gguf"), (2, "M/Mod/V/mmproj-F16.gguf"), (1, "M/Mod/V/mmproj-BF16.gguf"), (5, "toplevel.gguf")]
        u = D.build_units(L)["M/Mod/V"]
        self.assertEqual([x["quant"] for x in u], ["Q4_K_M", "Q8_0"])
        self.assertEqual(u[0]["bytes"], 10 + 12 + 1)            # smallest mmproj added
        self.assertEqual(len(u[0]["files"]), 2)
        self.assertEqual(u[0]["mmproj"], "mmproj-BF16.gguf")

    def test_quant_names(self):
        self.assertEqual(D.quant_of("x/Model-UD-Q2_K_XL"), "UD-Q2_K_XL")
        self.assertEqual(D.quant_of("Model-IQ4_XS"), "IQ4_XS")


class Guard(unittest.TestCase):
    def unit(self, gb, nodes=1): return {"key": "k", "bytes": gb * GiB, "nodes": nodes}

    def test_refuses_non_text(self):
        self.assertIn("refuse", D.plan_insert({**GGUF, "category": "Image"}, self.unit(10), res(), cfg()))

    def test_refuses_multi_node_and_incomplete(self):
        self.assertIn("2 nodes", D.plan_insert(GGUF, self.unit(10, nodes=2), res(), cfg())["refuse"])
        self.assertIn("not complete", D.plan_insert({**GGUF, "status": "downloading"}, self.unit(10), res(), cfg())["refuse"])

    def test_fails_closed_when_ollama_unreadable(self):
        r = res(); r["ollama_ok"] = False
        self.assertIn("cannot read Ollama", D.plan_insert(GGUF, self.unit(10), r, cfg())["refuse"])

    def test_missing_protected_refuses(self):
        self.assertIn("protected", D.plan_insert(GGUF, self.unit(10), res(), cfg(protected=["qwen3:14b"]))["refuse"])

    def test_fits_without_evicting(self):
        p = D.plan_insert(GGUF, self.unit(20), res(mem=100, ollama=[("qwen3:14b", 10)]), cfg(protected=["qwen3:14b"]))
        self.assertEqual(p["path"], "ollama"); self.assertEqual(p["would_evict"], [])

    def test_never_evicts_protected_for_gguf(self):
        p = D.plan_insert(GGUF, self.unit(60), res(mem=60, ollama=[("qwen3:14b", 10), ("other:7b", 6)]), cfg(protected=["qwen3:14b"]))
        self.assertIn("refuse", p)                               # would need the protected model's memory -> refuse, not evict
        self.assertNotIn("qwen3:14b", p.get("would_evict", []))

    def test_evicts_unprotected_when_needed(self):
        p = D.plan_insert(GGUF, self.unit(40), res(mem=55, ollama=[("qwen3:14b", 10), ("big:32b", 25)]), cfg(protected=["qwen3:14b"]))
        self.assertEqual(p["would_evict"], ["big:32b"])

    def test_slot_limit_evicts_deck_first(self):
        p = D.plan_insert(GGUF, self.unit(5), res(mem=100, ollama=[("a:1b", 1), ("deck/x", 3), ("b:1b", 1)]), cfg())
        self.assertEqual(p["would_evict"], ["deck/x"])

    def test_unloaded_protected_keeps_its_reserve(self):
        base = res(mem=70, ollama=[], tags={"qwen3:14b": 9.3})
        self.assertIn("refuse", D.plan_insert(GGUF, self.unit(40), base, cfg(protected=["qwen3:14b"])))
        self.assertNotIn("refuse", D.plan_insert(GGUF, self.unit(40), base, cfg()))

    def test_vllm_names_everything_it_stops(self):
        p = D.plan_insert(ST, self.unit(30), res(ollama=[("qwen3:14b", 10)], lanes=["big-lane.service"]), cfg(protected=["qwen3:14b"]))
        self.assertEqual(p["path"], "vllm"); self.assertEqual(p["would_evict"], ["big-lane.service", "qwen3:14b"])

    def test_second_vllm_refused(self):
        self.assertIn("already playing", D.plan_insert(ST, self.unit(30), res(vllm=True), cfg())["refuse"])


class DemoRoundTrip(unittest.TestCase):
    def test_insert_confirm_eject(self):
        root = tempfile.mkdtemp()
        c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d", protected=["qwen3:14b"])
        D.demo_seed(c); deck = D.Deck(c, D.DemoSystem(c)); deck.sys.pull = lambda *a: self._fake_pull(c, *a)
        st = deck.state(); self.assertEqual(st["n_shelf"], len(D.DEMO_SHELF))
        r = deck.insert("Qwen/Qwen3-30B-A3B/FP8")                    # vLLM: must name what it stops
        self.assertEqual(r["status"], 409); self.assertIn("qwen3:14b", r["would_evict"])
        r = deck.insert("OpenAI/gpt-oss-20b/ggml-org-GGUF"); self.assertTrue(r.get("started"))
        for _ in range(100):
            if not deck.job["running"]: break
            time.sleep(0.05)
        self.assertEqual(deck.state()["playing"]["id"], "OpenAI/gpt-oss-20b/ggml-org-GGUF")
        self.assertIn("qwen3:14b", [o["name"] for o in deck.sys.oll])   # the protected model survived
        out = deck.eject(); self.assertTrue(any("unloaded" in x for x in out["done"]))
        out = deck.eject("OpenAI/gpt-oss-20b/ggml-org-GGUF", remove_local=True)
        self.assertFalse(os.path.exists(f"{root}/d/OpenAI/gpt-oss-20b/ggml-org-GGUF"))

    @staticmethod
    def _fake_pull(c, mid, unit, f, job, state_dir):
        d = f"{c['local_dir']}/{mid}"; os.makedirs(d, exist_ok=True)
        with open(f"{d}/.deck-complete-{D._slug(unit['key'])}", "w") as mf: mf.write("t")
        return 0

    def test_eject_refuses_path_escape(self):
        root = tempfile.mkdtemp(); c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d")
        deck = D.Deck(c, D.DemoSystem(c))
        self.assertIn("error", deck.eject("../../etc", remove_local=True))


if __name__ == "__main__":
    unittest.main()
