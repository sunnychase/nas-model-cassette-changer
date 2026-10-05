import json, os, sys, tempfile, time, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deck"))
import mcc_deck as D  # noqa: E402

GiB = 2**30


def cfg(**kw):
    c = json.loads(json.dumps(D.DEFAULTS)); c.update(kw); return c


def res(mem=100.0, ollama=(), tags=None, max_loaded=3, vllm=False, lanes=(), ollama_up=True, engines=None):
    oll = [{"name": n, "gib": g} for n, g in ollama] if ollama_up else []
    eng = engines if engines is not None else ([{"engine": "vllm", "name": "vLLM"}] if vllm else [])
    return {"lanes": list(lanes), "ollama": oll, "tags_gib": (tags if tags is not None else {n: g for n, g in ollama}) if ollama_up else {},
            "ollama_ok": ollama_up, "ollama_up": ollama_up, "observed_gib": {}, "max_loaded": max_loaded, "free_disk_gib": 2000.0,
            "mem_avail_gib": mem, "mem_total_gib": 121.7, "engines_running": eng}


GGUF = {"id": "Qwen/Qwen3-32B/unsloth-GGUF", "category": "LLMs", "format": "gguf", "runtime": ["llama.cpp", "ollama"], "status": "complete"}
ST = {"id": "Qwen/Qwen3-32B/original", "category": "LLMs", "format": "safetensors", "runtime": ["vllm", "sglang"], "status": "complete"}
TF = {"id": "Qwen/Qwen3.8-Flash-Next/Vontra-MLX-4bit-MTP", "category": "LLMs", "format": "safetensors", "runtime": [], "status": "complete"}
RECIPE = {**D.RECIPE_DEFAULTS, "label": "TensorFold", "engine": "tensorfold", "need_gib": 112, "port": 8089, "start": "true",
          "requires_ollama_stopped": True, "key": TF["id"]}


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
        r = res(); r["ollama_ok"] = False                       # up but half-readable
        self.assertIn("cannot read Ollama", D.plan_insert(GGUF, self.unit(10), r, cfg())["refuse"])
        self.assertIn("cannot read Ollama", D.plan_insert(ST, self.unit(10), r, cfg())["refuse"])

    def test_ollama_stopped_blocks_gguf_not_engines(self):
        r = res(ollama_up=False)
        self.assertIn("Ollama is not running", D.plan_insert(GGUF, self.unit(10), r, cfg(protected=["qwen3:14b"]))["refuse"])
        p = D.plan_insert(ST, self.unit(30), r, cfg(protected=["qwen3:14b"]))
        self.assertEqual(p["path"], "vllm"); self.assertEqual(p["would_evict"], [])

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

    def test_second_engine_refused(self):
        self.assertIn("already playing", D.plan_insert(ST, self.unit(30), res(vllm=True), cfg())["refuse"])
        u = {**self.unit(30), "engine": "sglang"}
        self.assertIn("vLLM is already playing", D.plan_insert(ST, u, res(vllm=True), cfg())["refuse"])

    def test_sglang_path_names_everything(self):
        p = D.plan_insert(ST, {**self.unit(30), "engine": "sglang"}, res(ollama=[("qwen3:14b", 10)]), cfg(protected=["qwen3:14b"]))
        self.assertEqual(p["path"], "sglang"); self.assertEqual(p["would_evict"], ["qwen3:14b"])

    def test_engine_must_be_in_runtime_and_enabled(self):
        self.assertIn("not a SGLang model", D.plan_insert({**ST, "runtime": ["vllm"]}, {**self.unit(30), "engine": "sglang"}, res(), cfg())["refuse"])
        self.assertIn("not enabled", D.plan_insert(ST, {**self.unit(30), "engine": "sglang"}, res(), cfg(engines=["vllm"]))["refuse"])


class Recipes(unittest.TestCase):
    U = {"key": "TensorFold recipe", "bytes": 113 * 1e9, "nodes": 1, "recipe_key": TF["id"]}

    def test_refused_while_ollama_runs(self):
        p = D.plan_insert(TF, self.U, res(ollama=[("qwen3:14b", 10)]), cfg(), recipe=RECIPE)
        self.assertIn("stop Ollama first", p["refuse"])

    def test_plays_when_ollama_stopped_and_names_lanes(self):
        p = D.plan_insert(TF, self.U, res(mem=116, ollama_up=False, lanes=["big-lane.service"]), cfg(), recipe=RECIPE)
        self.assertEqual(p["path"], "recipe"); self.assertEqual(p["would_evict"], ["big-lane.service"])

    def test_memory_forecast_fails_closed(self):
        self.assertIn("needs 112 GiB", D.plan_insert(TF, self.U, res(mem=90, ollama_up=False), cfg(), recipe=RECIPE)["refuse"])

    def test_other_engine_blocks_and_multi_node_refused(self):
        self.assertIn("EJECT it first", D.plan_insert(TF, self.U, res(mem=116, ollama_up=False, vllm=True), cfg(), recipe=RECIPE)["refuse"])
        self.assertIn("needs 2 nodes", D.plan_insert(TF, self.U, res(mem=116, ollama_up=False), cfg(), recipe={**RECIPE, "nodes": 2})["refuse"])

    def test_recipe_without_ollama_requirement_evicts_by_name(self):
        r = {**RECIPE, "requires_ollama_stopped": False}
        p = D.plan_insert(TF, self.U, res(mem=100, ollama=[("qwen3:14b", 10), ("x:7b", 5)]), cfg(), recipe=r)
        self.assertEqual(p["would_evict"], ["qwen3:14b", "x:7b"])

    def test_load_recipes_skips_comments_and_incomplete(self):
        import tempfile
        fp = tempfile.mktemp(suffix=".json")
        with open(fp, "w") as f: json.dump({"_doc": "x", "a/b/c": {"start": "s", "port": 1, "need_gib": 2}, "d/e/f": {"port": 1}}, f)
        r = D.load_recipes(fp); self.assertEqual(list(r), ["a/b/c"]); self.assertEqual(r["a/b/c"]["health_path"], "/health")


class DemoRoundTrip(unittest.TestCase):
    def test_insert_confirm_eject(self):
        root = tempfile.mkdtemp()
        c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d", protected=["qwen3:14b"])
        D.demo_seed(c); deck = D.Deck(c, D.DemoSystem(c)); deck.sys.pull = lambda *a: self._fake_pull(c, *a)
        st = deck.state(); self.assertEqual(st["n_shelf"], len(D.DEMO_SHELF))
        units = [u["key"] for u in next(x for x in st["shelf"] if x["id"] == "Qwen/Qwen3-30B-A3B/FP8")["units"]]
        self.assertEqual(units, ["(whole variant)", "(whole variant) · SGLang"])
        r = deck.insert("Qwen/Qwen3-30B-A3B/FP8", "(whole variant) · SGLang")   # SGLang: must name what it stops
        self.assertEqual(r["status"], 409); self.assertIn("qwen3:14b", r["would_evict"])
        r = deck.insert("Qwen/Qwen3-30B-A3B/FP8")                    # no unit given: vLLM by default
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

    def test_tensorfold_recipe_round_trip(self):
        root = tempfile.mkdtemp()
        c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d", protected=["qwen3:14b"])
        c["_recipes"] = {k: {**D.RECIPE_DEFAULTS, **v, "key": k} for k, v in D.DEMO_RECIPES.items()}
        D.demo_seed(c); deck = D.Deck(c, D.DemoSystem(c, ollama_up=False)); deck.sys.pull = lambda *a: self._fake_pull(c, *a)
        tf = "Qwen/Qwen3.8-Flash-Next/Vontra-MLX-4bit-MTP"
        row = next(x for x in deck.state()["shelf"] if x["id"] == tf)
        self.assertEqual(row["recipe"]["label"], "TensorFold"); self.assertEqual(row["tier"], 1)
        self.assertTrue(deck.insert(tf).get("started"))
        for _ in range(200):
            if not deck.job["running"]: break
            time.sleep(0.05)
        self.assertEqual(deck.state()["playing"]["label"], "TensorFold")
        self.assertIn("EJECT it first", deck.insert("Qwen/Qwen3-30B-A3B/FP8")["error"])   # one GPU engine at a time
        self.assertIn("Ollama is not running", deck.insert("OpenAI/gpt-oss-20b/ggml-org-GGUF")["error"])
        self.assertIn("stopped the TensorFold recipe", deck.eject()["done"])

    def test_eject_refuses_path_escape(self):
        root = tempfile.mkdtemp(); c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d")
        deck = D.Deck(c, D.DemoSystem(c))
        self.assertIn("error", deck.eject("../../etc", remove_local=True))


if __name__ == "__main__":
    unittest.main()
