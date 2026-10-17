# Changelog

## Unreleased
- **README media:** a 12-second silent clip of the cassette player on a working deck (`docs/media/nas-model-cassette-changer.mp4`, with an
  animated `preview.webp` shown inline at the top), a still of that deck (`docs/images/screenshot-live-deck.png`: 220 models,
  Qwen3.8-Flash-Next in a TensorFold lane), and the fit chart redrawn with recent models at their real unit sizes
  (`docs/images/fit-chart-recent-models.png`: Qwen3.8-27B, Nemotron 3.5 Bolt and Super, Qwen3.8-Flash-Next, GLM-5.3-Flash,
  DeepSeek-V4.1-Flash, GLM-5.3). Same formula, constants and colours; the x-axis now runs to 560 GB. No code change.
- **Repository: badges, donate link, issue forms, CI hygiene check, AGENTS guide.** No code change; the version stays 1.4.0.
- **README:** a centred header (logo, title, tagline, the creator credit), a badge row (tests, latest release, licence, Python 3.9+,
  standard library only, engines, platform, last commit, stars, donate), a navigation row, *How it works*, an FAQ, a Documentation
  index, Contributing and a star-history chart. Every existing section is kept; Quick start now groups download, the demo, real
  hardware, hardware required and prerequisites.
- **CI:** `tests.yml` keeps its test matrix (Python 3.9, 3.11, 3.12) and adds a hygiene job: `scripts/check.sh` compiles every
  Python file and fails on home-directory paths, private network addresses or token-shaped strings.
- **New files:** `AGENTS.md` (notes for coding agents, with the guard's promises), `NOTICE.md`, issue forms (bug, feature) with blank
  issues off, a pull-request template, and `.github/FUNDING.yml` (the Sponsor button: donate with Stripe).
- `CONTRIBUTING.md` and `SECURITY.md` expanded; their existing rules are unchanged. SECURITY.md now points remote access to User
  Guide §9 (it said §8).

## 1.4.0 — Lion AI Models · Professional Model Suite
- **New interface:** sidebar of sections (Now playing, Engines, GPU nodes, Models, Local copies, Deck log) with scroll highlighting, a clean
  header, charcoal-and-gold palette, calmer cards, tables and buttons. Same markup ids and the same script: every control and API is unchanged.
- **Phones:** the sidebar becomes a scrolling top bar; model rows become stacked cards with a full-width PLAY button.
- Still one self-contained `deck.html` (no build step, no external assets; Inter is used when installed, otherwise the system font).
- **README:** Hardware required and Prerequisites sections; credits. New screenshots of every view.
- Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic); credited in LICENSE and the code headers.

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
