# Changelog

## 1.3.1
- **Fix: the Ollama benchmark no longer disturbs the model it measures.** 1.3.0 sent `num_ctx` (a different context makes Ollama reload the
  model) and no `keep_alive` (so a model pinned with `keep_alive: -1` fell back to the 5-minute default and could unload). It now sends no
  `num_ctx` and passes the resident's current remaining expiry as `keep_alive`; a model that is not resident is refused. 55 tests.

## 1.3.0 — the cassette selector
- **ENGINE selector bar** at the top: All · Ollama · vLLM · SGLang · each recipe lane (e.g. TensorFold) · EXL3 · Apps, each with a model count
  and a status dot (playing / Ollama stopped / idle / not playable from the deck). Picking one filters the shelf, the function tabs and the fit
  chips. Remembered per browser; `#engine=<name>` deep links.
- **New player:** cassette plus a text panel: what is playing, what else is resident, memory free. The reels turn only while a cassette is
  loading. A failed load stays on the player with its reason until dismissed or superseded by a good load (`last_fail` in the state).
- **Architecture check** (fail closed): the NAS index records `config.json` architectures; `mcc_deck.py archs vllm|sglang` records what the
  engine image can load; the guard refuses an unsupported model before anything is copied or stopped. No list = no check (1.2 behaviour).
- **Benchmarks beside the tape** (the *LLM server* card): live decode / prefill tok/s from the engine's Prometheus counters with sparklines
  and busy averages, running requests, context, tokens generated; **Decode** and **Prefill** benchmark runs (OpenAI-compatible engines and
  Ollama, cache-busting nonce); 14-day daily peaks. `POST /api/deck/bench`; `bench.jsonl` / `peaks.json` in the state dir.
- NAS awake/asleep shown as a green/red dot in the header.
- EXL3 models are listed under their own engine (not playable yet).
- Demo: 35 models, including an EXL3 row and a model whose architecture the demo image cannot load; a failed load in the history; the
  TensorFold lane (`--demo-no-ollama`) reports synthetic counters so the benchmark card can be tried.
- `mcc_deck.py fit` takes its size as before; new `mcc_deck.py archs`. 53 tests.

## 1.2.0
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
