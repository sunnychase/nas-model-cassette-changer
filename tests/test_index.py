import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nas"))
import mcc_index as I  # noqa: E402


def read(p):
    with open(p) as f: return f.read()


def touch(p, size=0, text=None):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w" if text is not None else "wb") as f:
        if text is not None: f.write(text)
        else: f.truncate(size)


class Index(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(); r = self.root
        touch(f"{r}/Models/LLMs/Qwen/Qwen3-32B/model-00001-of-00002.safetensors", 1000)
        touch(f"{r}/Models/LLMs/Qwen/Qwen3-32B/config.json", text=json.dumps({"torch_dtype": "bfloat16", "max_position_embeddings": 40960}))
        touch(f"{r}/Models/LLMs/unsloth/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf", 500)
        touch(f"{r}/Models/Image/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors", 700)
        touch(f"{r}/Models/LLMs/someone/Half-Done/model.safetensors.incomplete", 10)
        touch(f"{r}/Models/LLMs/someone/Half-Done/x.safetensors", 10)
        touch(f"{r}/Models/LLMs/recipes/serve-2x-recipe/.git/config", text='[remote "origin"]\n\turl = https://example.com/r.git\n')
        touch(f"{r}/Models/LLMs/recipes/serve-2x-recipe/start.sh", text="#!/bin/sh\n")
        cfgp = f"{r}/nas.json"
        touch(cfgp, text=json.dumps({"data_root": r, "staging": [{"path": "Models", "layout": "category/org/model"}],
                                     "families": [["^Qwen3-32B", "Qwen", "Qwen3-32B"]]}))
        self.cfg = I.load_config(cfgp)

    def test_scan_and_write(self):
        recs, kits = I.scan(self.cfg); ids = {r["id"]: r for r in recs}
        self.assertIn("Qwen/Qwen3-32B/original", ids)
        self.assertIn("Qwen/Qwen3-32B/unsloth-GGUF", ids)
        self.assertEqual(ids["Qwen/Qwen3-32B/unsloth-GGUF"]["format"], "gguf")
        self.assertEqual(ids["Qwen/Qwen3-32B/original"]["quant"], "bfloat16")
        self.assertIn("BlackForestLabs/FLUX.1-dev/original", ids)
        self.assertEqual(ids["someone/Half-Done/original"]["status"], "downloading")
        self.assertEqual([k["repo"] for k in kits], ["serve-2x-recipe"]); self.assertEqual(kits[0]["nodes"], 2)
        I.build_views(self.cfg, recs, kits); idx = I.write_index(self.cfg, recs, kits)
        lib = f"{self.root}/library"
        self.assertTrue(os.path.islink(f"{lib}/Qwen/Qwen3-32B/original"))
        self.assertTrue(os.path.exists(f"{lib}/Qwen/Qwen3-32B/original/config.json"))     # the link resolves
        self.assertTrue(os.path.islink(f"{lib}/_kits/serve-2x-recipe"))
        self.assertTrue(os.path.islink(f"{lib}/_tiers/1-node/Qwen/Qwen3-32B/original"))
        self.assertEqual(json.loads(read(f"{lib}/_index/library.json"))["n_models"], idx["n_models"])
        self.assertTrue(os.path.exists(f"{self.root}/Models/LLMs/Qwen/Qwen3-32B/manifest.json"))

    def test_prunes_only_its_own_dangling_links(self):
        recs, kits = I.scan(self.cfg); I.build_views(self.cfg, recs, kits)
        lib = f"{self.root}/library"; touch(f"{lib}/Qwen/keep-me.txt", text="mine")
        recs2 = [r for r in recs if r["maker"] != "BlackForestLabs"]; I.build_views(self.cfg, recs2, kits)
        self.assertFalse(os.path.lexists(f"{lib}/BlackForestLabs/FLUX.1-dev/original"))
        self.assertTrue(os.path.exists(f"{lib}/Qwen/keep-me.txt"))                         # a real file is never touched
        self.assertTrue(os.path.exists(f"{self.root}/Models/Image/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"))

    def test_about_keeps_hand_notes(self):
        recs, kits = I.scan(self.cfg); I.write_abouts(self.cfg, recs)
        p = f"{self.root}/Models/LLMs/Qwen/Qwen3-32B/{I.ABOUT}"; txt = read(p)
        self.assertIn("Serve with vLLM", txt)
        touch(p, text=txt.replace("_(anything you add", "my tuning notes\n_(anything you add"))
        touch(f"{self.root}/Models/LLMs/Qwen/Qwen3-32B/README.md", text="---\nlicense: apache-2.0\n---\n\n" + "A dense model for testing the generator, long enough to count as a paragraph here.\n")
        I.write_abouts(self.cfg, recs); txt2 = read(p)
        self.assertIn("apache-2.0", txt2); self.assertIn("my tuning notes", txt2)


    def test_architectures_recorded(self):
        d = tempfile.mkdtemp()
        self.assertIsNone(I.archs_of(d))                                                             # no config.json
        touch(f"{d}/config.json", text="{not json"); self.assertEqual(I.archs_of(d), "ERR")
        touch(f"{d}/config.json", text=json.dumps({"architectures": ["Qwen3ForCausalLM"]})); self.assertEqual(I.archs_of(d), ["Qwen3ForCausalLM"])
        touch(f"{d}/config.json", text=json.dumps({"torch_dtype": "bfloat16"})); self.assertEqual(I.archs_of(d), [])


if __name__ == "__main__":
    unittest.main()
