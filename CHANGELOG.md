# Changelog

## 1.2.0 (unreleased, branch form-fit-function)
- **Form · Fit · Function shelf.** Sections are now by FUNCTION (Chat & reasoning, Coding, Small & on-device, Image, Video, Music & audio,
  Speech, Documents & OCR, Embeddings & search; configurable in `functions`), with a tab per function and its count.
- Inside each section, rows are grouped by FIT, taken from the guard's own plans: **Ready now** (▶ PLAY), **After a switch** (⇄ SWITCH & PLAY,
  names what stops), **Blocked right now** (the guard's reason), **Needs more nodes**, **Runs in its own app** (an *Open* button when `apps`
  has a URL for that function), **Not on the shelf yet**. Fit chips filter the shelf.
- Every row shows its FORM: format, quant, and the engine(s) that can play it.
- Rows are sorted function → fit → node tier → name. New `fit_of()` / `function_of()` with tests.

## 1.1.0
- **SGLang** as a second engine for safetensors models, chosen per model in the unit menu (`engines`, `sglang.*` in the config).
- **Recipe lanes** (`recipes.json`): a model can be played by its own start/stop scripts, for example a **TensorFold** serving recipe. The guard still applies: one GPU engine at a time, `need_gib` is checked, everything it would stop is named, and Ollama is never stopped for you (`requires_ollama_stopped`). `copy: false` is for recipes that keep their own weights.
- An Ollama that is stopped or not installed no longer blocks vLLM / SGLang / recipe inserts; only GGUF needs it. A half-working Ollama still fails closed.
- Demo: a TensorFold recipe row, and `--demo-no-ollama` to play it.
- Docs: a 10 GbE section, engine and recipe guides, new screenshots. 32 tests.

## 1.0.0
- First release: NAS indexer, deck with Ollama + vLLM, eviction guard, demo mode.
