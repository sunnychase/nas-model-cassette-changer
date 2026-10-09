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



class FormFitFunction(unittest.TestCase):
    """v1.2: sections by FUNCTION, rows ordered by FIT (taken from the guard's own plans), FORM shown per row."""
    def u(self, key="Q4", gb=20, nodes=1): return {"key": key, "quant": key, "bytes": int(gb * 1e9), "nodes": nodes}

    def test_function_of_maps_categories_and_falls_back(self):
        c = cfg()
        self.assertEqual(D.function_of("LLMs", c)[0], "chat"); self.assertEqual(D.function_of("Speech-ASR", c)[0], "speech")
        self.assertEqual(D.function_of("Robotics", c), ("other", "Other"))

    def test_ready_when_a_unit_fits_without_stops(self):
        us = [self.u("Q8", 40), self.u("Q4", 20)]; plans = {"Q8": {"path": "ollama", "would_evict": ["a"]}, "Q4": {"path": "ollama", "would_evict": []}}
        f = D.fit_of(GGUF, us, plans, cfg()); self.assertEqual((f["class"], f["unit"]), ("ready", "Q4"))

    def test_switch_picks_the_fewest_stops_and_names_them(self):
        us = [self.u("A"), self.u("B")]; plans = {"A": {"would_evict": ["x", "y"]}, "B": {"would_evict": ["x"]}}
        f = D.fit_of(GGUF, us, plans, cfg()); self.assertEqual((f["class"], f["unit"], f["stops"]), ("switch", "B", ["x"]))

    def test_blocked_keeps_the_guards_reason(self):
        f = D.fit_of(GGUF, [self.u()], {"Q4": {"refuse": "Ollama is not running"}}, cfg())
        self.assertEqual((f["class"], f["why"]), ("blocked", "Ollama is not running"))

    def test_nodes_when_no_single_node_unit(self):
        f = D.fit_of(GGUF, [self.u("Q8", 300, 3)], {"Q8": {"refuse": "needs 3 nodes"}}, cfg()); self.assertEqual(f["class"], "nodes")
        f = D.fit_of(GGUF, [self.u("X", 900, None)], {"X": {"refuse": "beyond"}}, cfg()); self.assertEqual(f["class"], "nodes")

    def test_app_and_pending_precede_plans(self):
        img = {**ST, "category": "Image"}
        f = D.fit_of(img, [self.u()], {"Q4": {"refuse": "Image model"}}, cfg(apps={"image": {"label": "ComfyUI", "url": "http://x"}}))
        self.assertEqual((f["class"], f["app"]["label"]), ("app", "ComfyUI"))
        self.assertEqual(D.fit_of({**GGUF, "status": "downloading"}, [self.u()], {"Q4": {"would_evict": []}}, cfg())["class"], "pending")

    def test_state_sections_and_counts_add_up(self):
        with tempfile.TemporaryDirectory() as t:
            c = D.load_config(None); c.update(state_dir=f"{t}/s", local_dir=f"{t}/d", protected=["qwen3:14b"], fleet=[{"name": "n1", "self": True}])
            c["_recipes"] = {k: {**D.RECIPE_DEFAULTS, **v, "key": k} for k, v in D.DEMO_RECIPES.items()}
            D.demo_seed(c); st = D.Deck(c, D.DemoSystem(c)).state()
            self.assertEqual(sum(f["n"] for f in st["functions"]), st["n_shelf"])
            for f in st["functions"]: self.assertEqual(sum(f["fit"].values()), f["n"])
            order = [r["function"] for r in st["shelf"]]; keys = [f["key"] for f in st["functions"]]
            self.assertEqual(order, sorted(order, key=keys.index))        # rows come out grouped in function order

class CassetteSelector(unittest.TestCase):
    """v1.3: ENGINE selector, ARCHITECTURE CHECK before anything is copied or stopped, last load failure."""
    K = {"Qwen3ForCausalLM", "LlamaForCausalLM"}

    def test_arch_refusal_fails_closed(self):
        self.assertIsNone(D.arch_refusal({**ST}, "vllm", self.K))                                   # pre-1.3 index: no field, no check
        self.assertIsNone(D.arch_refusal({**ST, "architectures": ["Qwen3ForCausalLM"]}, "vllm", self.K))
        self.assertIsNone(D.arch_refusal({**ST, "architectures": ["Novel"]}, "vllm", None))          # no list for this engine: no check
        for a, word in ((["NovelHybridForCausalLM"], "not in this vLLM image"), (None, "no config.json"), ("ERR", "could not be read"), ([], "declares no")):
            self.assertIn(word, D.arch_refusal({**ST, "architectures": a}, "vllm", self.K))

    def test_plan_insert_refuses_unknown_arch_before_copy(self):
        c = cfg(); c["_archs"] = {"vllm": self.K}
        p = D.plan_insert({**ST, "architectures": ["NovelHybridForCausalLM"]}, {"key": "w", "bytes": 20 * GiB, "nodes": 1, "engine": "vllm"}, res(), c)
        self.assertIn("not in this vLLM image", p["refuse"]); self.assertNotIn("would_evict", p)
        p = D.plan_insert({**ST, "architectures": ["Qwen3ForCausalLM"]}, {"key": "w", "bytes": 20 * GiB, "nodes": 1, "engine": "vllm"}, res(), c)
        self.assertEqual(p["path"], "vllm")
        c["_archs"] = {"vllm": self.K, "sglang": None}                                                # SGLang has no list: not checked
        p = D.plan_insert({**ST, "architectures": ["Novel"]}, {"key": "w", "bytes": 20 * GiB, "nodes": 1, "engine": "sglang"}, res(), c)
        self.assertEqual(p["path"], "sglang")

    def test_read_archs(self):
        d = tempfile.mkdtemp(); p = f"{d}/a.json"
        self.assertIsNone(D.read_archs(p)); self.assertIsNone(D.read_archs(None))
        with open(p, "w") as f: json.dump({"architectures": []}, f)
        self.assertIsNone(D.read_archs(p))                                                           # an empty list is no list
        with open(p, "w") as f: json.dump({"architectures": ["A", "B"]}, f)
        self.assertEqual(D.read_archs(p), {"A", "B"})

    def test_engines_of(self):
        self.assertEqual(D.engines_of({"format": "gguf", "servable": True}), ["Ollama"])
        self.assertEqual(D.engines_of({"format": "safetensors", "servable": True, "units": [{"engine": "vllm"}, {"engine": "sglang"}]}), ["vLLM", "SGLang"])
        self.assertEqual(D.engines_of({"format": "safetensors", "servable": True, "quant": "EXL3", "runtime": ["exllama"]}), ["EXL3"])
        self.assertEqual(D.engines_of({"recipe": {"label": "TensorFold"}, "servable": True}), ["TensorFold"])
        self.assertEqual(D.engines_of({"format": "safetensors", "servable": False, "fit": {"class": "app"}}), ["Apps"])

    def test_engine_bar_states(self):
        rows = [{"format": "gguf", "servable": True}, {"format": "safetensors", "servable": True, "units": [{"engine": "vllm"}]},
                {"recipe": {"label": "TensorFold"}, "servable": True}, {"quant": "EXL3", "servable": True, "format": "safetensors"}]
        bar = {e["key"]: e for e in D.engine_bar(rows, res(ollama=[("qwen3:14b", 10)]), {})}
        self.assertEqual([e["key"] for e in D.engine_bar(rows, res(), {})], ["Ollama", "vLLM", "TensorFold", "EXL3"])
        self.assertEqual(bar["Ollama"]["state"], "serving"); self.assertEqual(bar["vLLM"]["state"], "idle"); self.assertEqual(bar["EXL3"]["state"], "unwired")
        bar = {e["key"]: e for e in D.engine_bar(rows, res(ollama_up=False, engines=[{"engine": "tensorfold", "name": "TensorFold"}]), {})}
        self.assertEqual(bar["Ollama"]["state"], "stopped"); self.assertEqual(bar["TensorFold"]["state"], "serving")

    def test_last_fail_is_superseded_by_a_good_insert(self):
        bad = {"action": "INSERT", "id": "A/B/C", "rc": 1, "ts": "t1", "note": "vLLM exited"}
        self.assertEqual(D.last_fail([bad])["why"], "vLLM exited")
        self.assertEqual(D.last_fail([bad, {"action": "EJECT"}])["id"], "A/B/C")                    # an eject does not clear it
        self.assertIsNone(D.last_fail([bad, {"action": "INSERT", "rc": 0}]))
        self.assertEqual(D.last_fail([{**bad, "note": ""}], "copy FAILED")["why"], "copy FAILED")
        self.assertIsNone(D.last_fail([]))

    def test_demo_state_has_selector_and_failure(self):
        root = tempfile.mkdtemp(); c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d", protected=["qwen3:14b"])
        c["_archs"] = {"vllm": set(D.DEMO_ENGINE_ARCHS), "sglang": set(D.DEMO_ENGINE_ARCHS)}; c["_archs_fixed"] = True
        D.demo_seed(c); st = D.Deck(c, D.DemoSystem(c)).state()
        keys = [e["key"] for e in st["engine_bar"]]
        for k in ("Ollama", "vLLM", "SGLang", "EXL3", "Apps"): self.assertIn(k, keys)
        self.assertEqual(sum(1 for r in st["shelf"] if "Ollama" in r["engines"]), next(e["n"] for e in st["engine_bar"] if e["key"] == "Ollama"))
        nov = next(r for r in st["shelf"] if r["model"] == "Novel-Hybrid-9B")
        self.assertTrue(all("not in this" in p["refuse"] for p in nov["plans"].values()))
        self.assertEqual(st["last_fail"]["id"], "Mistral/Mistral-Small-3.2-24B/original"); self.assertEqual(st["arch_check"], ["sglang", "vllm"])


class Benchmarks(unittest.TestCase):
    """v1.3: live tok/s from the engine's counters, Decode / Prefill runs, daily peaks."""
    PROM = ('# HELP x\nvllm:generation_tokens_total{model_name="a"} 1000\nvllm:generation_tokens_total{model_name="b"} 500\n'
            'vllm:prompt_tokens_total{model_name="a"} 9000\nvllm:num_requests_running{model_name="a"} 2\nother_metric 7\n')

    def test_parse_prom_sums_label_sets(self):
        self.assertEqual(D.parse_prom(self.PROM), {"gen": 1500.0, "prompt": 9000.0, "running": 2.0})
        self.assertEqual(D.parse_prom("sglang:generation_tokens_total 12\nsglang:num_running_reqs 1"), {"gen": 12.0, "running": 1.0})
        self.assertEqual(D.parse_prom(None), {})

    def test_rates(self):
        self.assertEqual(D.rates((0, {"gen": 100, "prompt": 1000}), (5, {"gen": 400, "prompt": 10000})), (60.0, 1800.0))
        self.assertIsNone(D.rates(None, (5, {"gen": 1})))
        self.assertIsNone(D.rates((0, {"gen": 500}), (5, {"gen": 10})))                    # counter reset = engine restart
        self.assertIsNone(D.rates((5, {"gen": 1}), (5, {"gen": 2})))

    def test_bump_peak_keeps_max_and_window(self):
        pk = {}
        D.bump_peak(pk, "2026-10-01", "decode", 50); D.bump_peak(pk, "2026-10-01", "decode", 40); D.bump_peak(pk, "2026-10-01", "decode", 0)
        self.assertEqual(pk["2026-10-01"]["decode"], 50)
        for i in range(2, 40): D.bump_peak(pk, f"2026-10-{i:02d}", "prefill", i)
        self.assertEqual(len(pk), 30); self.assertNotIn("2026-10-01", pk)

    def test_expires_left_parses_ollama_timestamps(self):
        self.assertEqual(D.expires_left("2026-10-09T13:05:12.123456789-07:00", now=1791576312), 0)
        self.assertEqual(D.expires_left("2026-10-09T20:05:12Z", now=1791576312 - 60), 60)
        self.assertGreater(D.expires_left("2318-01-01T00:00:00.5-08:00"), 3e7)                    # keep_alive -1 = pinned
        for bad in (None, "", "soon", "2026-10-09 13:05:12"): self.assertIsNone(D.expires_left(bad))

    def test_ollama_bench_never_reloads_or_unpins(self):
        sent = []; c = cfg(); s = D.System(c)
        def fake(path, body=None, timeout=4):
            if path == "/api/ps": return {"models": [{"name": "qwen3:14b", "expires_at": "2318-01-01T00:00:00-08:00"}]}
            sent.append(body); return {"eval_count": 256, "eval_duration": 4e9, "total_duration": 5e9}
        s.ollama = fake
        self.assertEqual(s.bench_ollama("qwen3:14b", "decode")["tok_s"], 64.0)
        self.assertEqual(sent[0]["keep_alive"], -1); self.assertNotIn("num_ctx", sent[0]["options"])
        self.assertRaises(RuntimeError, s.bench_ollama, "not-resident:1b", "decode"); self.assertEqual(len(sent), 1)

    def test_bench_prompts_defeat_caches(self):
        self.assertNotEqual(D.bench_prompt("prefill", "a"), D.bench_prompt("prefill", "b"))
        self.assertGreater(len(D.bench_prompt("prefill", "a").split()), 1500)

    def _tf_deck(self):
        root = tempfile.mkdtemp(); c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d", protected=["qwen3:14b"])
        c["_recipes"] = {k: {**D.RECIPE_DEFAULTS, **v, "key": k} for k, v in D.DEMO_RECIPES.items()}
        D.demo_seed(c); deck = D.Deck(c, D.DemoSystem(c, ollama_up=False))
        self.assertTrue(deck.insert("Qwen/Qwen3.8-Flash-Next/Vontra-MLX-4bit-MTP").get("started"))
        for _ in range(200):
            if not deck.job["running"]: break
            time.sleep(0.05)
        return deck

    def test_live_samples_and_bench_on_tensorfold(self):
        deck = self._tf_deck(); t = deck.target()
        self.assertEqual((t["engine"], t["model"], t["port"]), ("TensorFold", "Qwen3.8-Flash-Next", 8888))
        self.assertIsNone(deck.sample())                                                  # the first sample has nothing to compare with
        time.sleep(0.2); r = deck.sample(); self.assertIsNotNone(r); self.assertGreater(r[0], 0)
        b = deck.bench_state(); self.assertEqual(b["live"]["now_decode"], r[0]); self.assertEqual(b["last"]["decode"]["tok_s"], 63.3)
        self.assertEqual(len(b["daily"]), 14)
        self.assertTrue(deck.bench("decode").get("started")); self.assertEqual(deck.bench("decode")["status"], 409)   # one at a time
        for _ in range(100):
            if not deck.bjob["running"]: break
            time.sleep(0.05)
        last = deck.bench_state()["last"]["decode"]; self.assertEqual((last["tok_s"], last["runs"], last["best"]), (61.8, 2, 63.3))
        self.assertEqual(deck.bench("warp")["status"], 400)

    def test_bench_refuses_with_nothing_loaded(self):
        root = tempfile.mkdtemp(); c = cfg(state_dir=f"{root}/s", local_dir=f"{root}/d")
        deck = D.Deck(c, D.DemoSystem(c, ollama_up=False))
        self.assertIsNone(deck.target()); self.assertIn("nothing is loaded", deck.bench("decode")["error"])
        self.assertIsNone(deck.bench_state()["live"])


if __name__ == "__main__":
    unittest.main()
