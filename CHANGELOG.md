# Changelog

## 1.1.0
- **SGLang** as a second engine for safetensors models, chosen per model in the unit menu (`engines`, `sglang.*` in the config).
- **Recipe lanes** (`recipes.json`): a model can be played by its own start/stop scripts, for example a **TensorFold** serving recipe. The guard still applies: one GPU engine at a time, `need_gib` is checked, everything it would stop is named, and Ollama is never stopped for you (`requires_ollama_stopped`). `copy: false` is for recipes that keep their own weights.
- An Ollama that is stopped or not installed no longer blocks vLLM / SGLang / recipe inserts; only GGUF needs it. A half-working Ollama still fails closed.
- Demo: a TensorFold recipe row, and `--demo-no-ollama` to play it.
- Docs: a 10 GbE section, engine and recipe guides, new screenshots. 32 tests.

## 1.0.0
- First release: NAS indexer, deck with Ollama + vLLM, eviction guard, demo mode.
