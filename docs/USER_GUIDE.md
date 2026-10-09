# User Guide: NAS Model Cassette Changer

1. [What it does](#1-what-it-does)
2. [Requirements](#2-requirements)
3. [Set up the NAS](#3-set-up-the-nas)
4. [Set up the GPU box](#4-set-up-the-gpu-box)
5. [Using the page](#5-using-the-page)
6. [The guard, rule by rule](#6-the-guard-rule-by-rule)
7. [Engines and recipe lanes (SGLang, TensorFold)](#7-engines-and-recipe-lanes-sglang-tensorfold)
8. [Configuration reference](#8-configuration-reference)
9. [Remote access](#9-remote-access)
10. [More than one GPU node](#10-more-than-one-gpu-node)
11. [Troubleshooting](#11-troubleshooting)
12. [HTTP API](#12-http-api)
13. [FAQ](#13-faq)

---

## 1. What it does

| Part | Runs on | Job |
|---|---|---|
| `nas/mcc_index.py` | the NAS, every 10 min | Catalogues every model folder. Builds `library/<Maker>/<Model>/<Variant>` symlinks, groups models by node count under `library/_tiers/`, collects serving recipes under `library/_kits/`, and writes `library/_index/library.json`, plus a `manifest.json` and `ABOUT-MODEL.md` next to each model. |
| `deck/mcc_deck.py sync` | the GPU box, every 10 min | Copies `library.json` and a read-only list of `.gguf` files over ssh. If the NAS is asleep, it records that and keeps the last copy. It never wakes the NAS. |
| `deck/mcc_deck.py serve` | the GPU box, always on | The web page and API: shelf, tiers, insert, eject, log. |

A **cassette** is one runnable unit:

- for a GGUF repo, one quantisation (split shards are grouped, and a vision projector `mmproj` is attached);
- for safetensors, the whole folder.

**Insert** copies the unit to `~/models/deck/<Maker>/<Model>/<Variant>/` with `rsync --partial`, so an interrupted copy resumes. It then starts the unit:

- a GGUF is registered as `deck/<name>` in Ollama and warm-loaded;
- a safetensors folder runs in a **vLLM** (`vllm/vllm-openai`, port 8010) or **SGLang** (`lmsysorg/sglang`, port 30000) container, whichever you pick in the unit menu;
- a model listed in `recipes.json` is started by its **recipe lane**, for example a TensorFold serving recipe (§7).

## 2. Requirements

**NAS:** Linux, Python 3.9+, your model folders on local disk. It must be reachable by ssh from the GPU box (key auth). An NFS or SMB export is optional; it only matters if you want to browse the library by hand.

**GPU box:** Linux, Python 3.9+, `ssh`, `rsync`, plus [Ollama](https://ollama.com) for GGUF cassettes and/or Docker with the NVIDIA container toolkit for vLLM / SGLang cassettes. Recipe lanes need whatever their own scripts need. It needs enough local NVMe for the largest model you plan to insert, with room to spare.

### 2.1 Network: use 10 GbE

Every INSERT copies the whole model, so the link between the NAS and the GPU box decides the wait:

| Link | Copy rate | 65 GB model |
|---|---|---|
| 1 GbE | ~110 MB/s | ~10 min |
| 2.5 GbE | ~280 MB/s | ~4 min |
| **10 GbE** | **~800–1100 MB/s** | **~1–1.5 min** |

**Use 10 GbE if you can**: a 10 GbE NIC on each box, plus a 10 GbE switch or a direct cable. Some tips:

- **Disks:** the NAS disks must keep up, because one hard drive delivers only about 150–250 MB/s. Use several drives, an SSD/NVMe pool or a read cache.
- **Jumbo frames:** they are optional. If you enable them, every device on the path must support them.
- **Trouble:** if a 10 GbE copy crawls at a few MB/s while iperf looks fine one way, try turning off LRO/GRO on the NAS NIC (`ethtool -K <if> lro off gro off`). Some 10 GbE chipsets mishandle receive offload.
- **Eject:** ejecting never touches the network. It stops the model and, if you ask, deletes the local copy. With 10 GbE, deleting local copies you aren't using costs little, because putting one back takes about a minute.

## 3. Set up the NAS

### 3.1 Folder layout

Tell the indexer where your downloads live and how deep the model folders sit. Each entry in `staging` is a folder plus a `layout`:

| layout | example path | becomes |
|---|---|---|
| `category/org/model` | `Models/LLMs/Qwen/Qwen3-32B/` | `Qwen/Qwen3-32B/original` |
| `org/model` | `Downloads/unsloth/Qwen3-32B-GGUF/` (category from the entry) | `Qwen/Qwen3-32B/unsloth-GGUF` |
| `model` | `MyFinetunes/my-model/` (org from the entry, default `local`) | `local/my-model/original` |

**Categories** decide what the deck will serve. These are served: `LLMs`, `Coding`, `Small-On-Device`. These are catalogued only: `Image`, `Video`, `Speech-ASR`, `Speech-TTS`, `Music-Audio`, `OCR`, `Embeddings-Reranking`.

### 3.2 Makers, models and variants

The indexer turns a Hugging Face org into a friendly maker name (`meta-llama` → `Meta`, `deepseek-ai` → `DeepSeek`, …). It also groups repackaged uploads under the original model:

- **Re-uploads of the same model:** add a `families` rule `[regex, Maker, canonical model]` so that, for example, `gemma-3-27b-it-GGUF` from a quantiser lands next to Google's `Gemma-3-27B-it`.
- **Weights in a folder named after someone else:** put an `.mcc-source` file in the folder containing `org/repo`; that sets the publisher and source link.

### 3.3 Install and run

```bash
sudo scripts/install-nas.sh
sudoedit /etc/mcc/nas.json
sudoedit /etc/systemd/system/mcc-index.service     # set User= to the owner of the model folders
python3 /usr/local/bin/mcc_index.py --dry-run       # prints what it found, writes nothing
sudo systemctl enable --now mcc-index.timer
```

The indexer never moves, renames or deletes a model file. The only things it removes are its own symlinks that no longer point anywhere. If a real folder already sits where a link would go, it is left alone.

### 3.4 Kits

A git clone with no weights in it is treated as a **kit**: a serving recipe such as a multi-node launch script. It is linked under `library/_kits/`. A `2x` or `tp2` in its name sets its node count. The deck shows a kit next to the models whose name it contains.

### 3.5 ABOUT-MODEL.md

Each complete model gets an `ABOUT-MODEL.md`. It contains:

- what the model is (from its model card);
- facts (format, size, context length, licence);
- a "which box it fits" table;
- ready-to-paste run commands for llama.cpp, Ollama, vLLM, diffusers or transformers.

Write your own notes under the `## Notes (hand-written)` heading; regeneration keeps them.

## 4. Set up the GPU box

```bash
scripts/install-deck.sh
```

Then edit `~/.config/mcc/deck.json`:

1. **`nas_ssh`**: an ssh alias that logs into the NAS with no prompt. Add it to `~/.ssh/config` with `BatchMode yes`. Test it with `ssh -o BatchMode=yes nas true`.
2. **`nas_library_dir`**: the absolute library path on the NAS (`data_root` + `library_dir`).
3. **`protected`**: Ollama tags that must never be pushed out, such as the chat model you use all day.
4. **`ollama_max_loaded`**: match `OLLAMA_MAX_LOADED_MODELS` on this box.

Then:

```bash
python3 ~/.local/share/mcc/mcc_deck.py sync          # {"ok": true, "models": …}
systemctl --user enable --now mcc-sync.timer mcc-deck.service
loginctl enable-linger "$USER"                        # keep user services running after logout
```

The first start creates a random access token in `~/.config/mcc/token` (mode 0600).

Optional but recommended: record what your engine images can load, for the **architecture check** (§7.3). Re-run it after pulling a new image.

```bash
python3 ~/.local/share/mcc/mcc_deck.py archs vllm     # e.g. "365 architectures from vllm/vllm-openai:latest -> ~/.config/mcc/vllm_archs.json"
python3 ~/.local/share/mcc/mcc_deck.py archs sglang
```

## 5. Using the page

- **Header:** model counts per tier, how many models are new (first seen in the last 7 days), whether the NAS was awake at the last sync and when its index was built, and your protected models.
- **ENGINE bar (the cassette selector):** All · Ollama · vLLM · SGLang · each recipe lane · EXL3 · Apps, with counts and a status dot (green =
  playing now, red = Ollama stopped, grey = idle, hollow = listed but not playable from the deck). Picking one filters every section, tab and
  count below it. The choice is remembered per browser; `#engine=vLLM` in the URL opens the page on one engine.
- **Player:** what is loading or playing, what else is resident, and memory free. The reels turn only while a cassette is loading. A failed
  load shows *⚠ last load failed* with the reason until you dismiss it or a later load succeeds. **⏏ EJECT** stops the cassette.
- **GPU nodes:** memory and disk on this box, everything resident in Ollama (protected models in green), any big lanes, and the status of other nodes.
- **Filters:** search, category, format, tier, *new only*, and *insertable now* (hides everything the guard would refuse).
- **Shelf:** one table per tier. For a GGUF with several quants, choose one in the **unit → nodes** menu; the *if you insert it now* column updates for that quant. It shows one of three outcomes:
  - ✓ fits beside what is running;
  - **would stop: …** — the exact list;
  - ✗ a reason it is refused.
- **INSERT:** confirm, then:
  - if anything would stop, a dialog lists it and the button stays disabled until every name is ticked;
  - progress shows in the yellow bar and the deck log.
- **Ejector:** every local copy on this box. **Delete local copy** removes the folder and the exact Ollama names that cassette created. The NAS copy is never touched.

## 6. The guard, rule by rule

An insert is **refused** (HTTP 409, nothing done) when:

| Rule | Why |
|---|---|
| The category is not servable | Image, video and speech models need their own apps. |
| The format is not gguf or safetensors, or the runtime doesn't match | EXL3/EXL2 need ExLlama, for example. |
| The model is still downloading on the NAS | |
| The unit needs more than one node | Multi-node runs launch from their kit. |
| Another insert is running | One job at a time. |
| Ollama can't be read | The guard won't guess what would be evicted; it fails closed. |
| A protected model is not installed in Ollama | A typo in the config would otherwise silently protect nothing. |
| Not enough local disk | GGUF needs about 2.2 × its size, because `ollama create` writes a copy; safetensors needs about 1.1 ×. |
| It doesn't fit even after evicting every unprotected model | See below. |
| A vLLM, SGLang or recipe cassette is already playing | One GPU engine at a time: EJECT it first. |
| The engine isn't listed in the model's runtime, or isn't enabled in `engines` | |
| Ollama is stopped and the unit is a GGUF | GGUF cassettes play in Ollama. vLLM / SGLang / recipe inserts are still allowed. |

**Memory check for GGUF:** the cassette needs `1.2 × W + 3 + kv_band_gib` GiB, at `num_ctx`. Free memory is `MemAvailable − mem_floor_gib − reserve`. The *reserve* is the room for each protected model that is not loaded right now: its largest observed footprint, or 1.3 × its size + 1.5 GiB. If the cassette doesn't fit, the deck picks evictions in this order: older deck cassettes first, then unprotected models from largest to smallest. It also counts Ollama's own slot limit (`ollama_max_loaded`).

**For safetensors (vLLM or SGLang):** the container takes most of the GPU, so it would stop every big lane and every Ollama model, protected ones included. The dialog names them all.

**For a recipe lane:** see §7. It states its own `need_gib`, and is refused if the memory that would be free after the named stops is smaller.

**Any eviction needs every name confirmed.** Then:

1. The plan is computed again after the copy, which can take minutes. If anything changed, nothing is evicted and you are asked again.
2. After a GGUF loads, the deck checks that every protected model that was loaded before is still loaded. If Ollama dropped one anyway, the new cassette is unloaded and the log says so.

## 7. Engines and recipe lanes (SGLang, TensorFold)

### 7.1 vLLM or SGLang

Every complete safetensors text model gets one unit per enabled engine that its runtime lists. The indexer lists `vllm` and `sglang` for safetensors text models. In the **unit → nodes** menu, `· vLLM` and `· SGLang` mark the two choices. Both containers mount `local_dir` read-only and serve the OpenAI-compatible API:

| | vLLM | SGLang |
|---|---|---|
| image | `vllm/vllm-openai:latest` | `lmsysorg/sglang:latest` |
| port | 8010 | 30000 |
| memory knob | `gpu_memory_utilization` 0.85 | `mem_fraction_static` 0.85 |
| context | `max_model_len` 32768 | `context_length` 32768 |
| container | `mcc-vllm` | `mcc-sglang` |

To offer only one engine, set `"engines": ["vllm"]` (or `["sglang"]`). An API insert without a `unit` uses the first engine.

### 7.2 Recipe lanes

Some models run best under their own serving recipe: a container plus `start.sh`/`stop.sh` with settings tuned for one model on one machine. **TensorFold** recipes are the main example. A recipe lane lets the deck insert and eject such a model without knowing anything about how it is served.

Copy `examples/recipes.example.json` to `~/.config/mcc/recipes.json`. Each key is a **library id**, exactly as the shelf shows it:

```json
{
  "Qwen/Qwen3.8-Flash-Next/Vontra-MLX-4bit-MTP": {
    "label": "TensorFold",
    "engine": "tensorfold",
    "nodes": 1,
    "need_gib": 112,
    "port": 8888,
    "health_path": "/health",
    "requires_ollama_stopped": true,
    "copy": false,
    "start": "cd ~/recipes/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold && ./start.sh",
    "stop":  "cd ~/recipes/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold && ./stop.sh",
    "env": {"PORT": "8888"},
    "repo": "https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold",
    "license": "recipe: MIT · weights: see the model card"
  }
}
```

| key | required | meaning |
|---|---|---|
| `start` | yes | shell command, run detached with `MCC_MODEL_DIR`, `MCC_MODEL_ID`, `MCC_PORT` and `env` set |
| `stop` | recommended | shell command that EJECT runs (without one, EJECT tells you to stop it yourself) |
| `port`, `health_path` | yes / `/health` | the deck polls `http://127.0.0.1:<port><health_path>` until it answers (`start_timeout_s`, default 1800) |
| `need_gib` | yes | total memory the recipe needs. Its INSERT is refused if less would be free after the named stops |
| `nodes` | 1 | a recipe that needs more than one node is refused (launch it from its kit) |
| `requires_ollama_stopped` | false | if true, the INSERT is refused while Ollama answers. Stop Ollama yourself (`sudo systemctl stop ollama`); the deck never stops system services |
| `copy` | true | `false` = the recipe keeps its own weights (TensorFold uses its own Hugging Face cache, `HF_CACHE` in its `.env`), so INSERT only starts it. `true` = the deck copies the variant from the NAS first and passes the path as `MCC_MODEL_DIR` (use `"{model_dir}"` inside `env` values) |
| `label`, `engine`, `served`, `quant`, `repo`, `license` | | shown on the page |

**TensorFold, step by step:**

1. Clone the recipe repository for your model (for example [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) or [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold)) to `~/recipes/`.
2. Run its `start.sh` once by hand, following its README. The first run pulls the image and the weights and compiles kernels. Run its `stop.sh` when it answers.
3. Make sure the model's folder is on the NAS, so its row appears on the shelf, and use that row's id as the key in `recipes.json`.
4. Set `need_gib` from the recipe's README (a 128 GB unified-memory node: about 112 for Qwen3.8-Flash-Next, about 104 for Qwen3.8-27B).
5. If the recipe needs the whole GPU, stop Ollama, then press INSERT on the row. The deck starts it, waits for `/health`, and shows it as **NOW PLAYING**. EJECT runs `stop.sh`.

TensorFold and its recipes are separate projects with their own licences. This repository ships no TensorFold code, only an example `recipes.json` that calls the scripts from your own checkout.

Try it with no hardware: `python3 deck/mcc_deck.py serve --demo --demo-no-ollama`, then insert the **Qwen3.8-Flash-Next** row.

### 7.3 Architecture check

A vLLM or SGLang image can only load the model architectures it was built with. Without a check, an unsupported model is copied (minutes),
whatever it would stop is stopped, and only then does the engine refuse it. From 1.3:

1. The NAS indexer records each model's `config.json` `architectures` in `library.json`.
2. `mcc_deck.py archs vllm` (and `archs sglang`) runs the engine image once and writes the list it can load to `engine_archs.<engine>`.
3. Before an insert, the guard refuses a safetensors model whose architectures are all missing from that list, and says so on the row.

It fails closed: once a list exists, a model with no `config.json`, an unreadable one, or one that declares no architecture is refused too.
No list for an engine (or a NAS index older than 1.3) means no check, as before. Recipe lanes are not checked: the recipe states its own fit.

## 8. Configuration reference

### `nas.json`

| key | default | meaning |
|---|---|---|
| `data_root` | `/srv/models-nas` | everything is relative to this |
| `library_dir` | `library` | the generated view and `_index/` |
| `staging` | `[{"path":"Models","layout":"category/org/model"}]` | see §3.1 |
| `families` | `[]` | `[regex, Maker, canonical-or-null]` |
| `org_maker` | `{}` | extra org → maker names |
| `node.usable_gb`, `node.max_nodes`, `node.name` | `95`, `4`, `node` | tier view and ABOUT fit tables |
| `client_mount` | `/mnt/nas` | only used in ABOUT setup text |
| `tiers`, `about` | `true` | turn the tier view or the ABOUT files off |

### `deck.json`

| key | default | meaning |
|---|---|---|
| `nas_ssh` | `nas` | ssh alias (BatchMode) |
| `nas_library_dir` | `/srv/models-nas/library` | absolute path on the NAS |
| `local_dir` | `~/models/deck` | local cassettes |
| `state_dir` | `~/.local/state/mcc` | cache, log, history |
| `listen`, `port` | `127.0.0.1`, `8099` | web server |
| `token_file` | `~/.config/mcc/token` | created on first start |
| `fleet` | this box only | `[{"name","self":true}, {"name","ssh":"user@host"}]` |
| `fit.*` | 95.69 / 5.6 / 8 / 1.15 / 4 | budget per node, overhead per node, KV floor, weight factor, max nodes (GiB) |
| `protected` | `[]` | Ollama tags never pushed out |
| `servable_categories` | LLMs, Coding, Small-On-Device | |
| `ollama_url`, `ollama_max_loaded` | `http://127.0.0.1:11434`, `3` | |
| `num_ctx`, `kv_band_gib`, `mem_floor_gib` | `8192`, `4`, `10` | GGUF sizing |
| `big_lanes` | `[]` | systemd user units that hold the GPU; a vLLM / SGLang / recipe insert stops them by name |
| `engines` | `["vllm", "sglang"]` | engines offered for safetensors models |
| `vllm.*` | image, port 8010, 0.85, 32768, `trust_remote_code: false` | container settings |
| `sglang.*` | image, port 30000, `mem_fraction_static` 0.85, `context_length` 32768, `trust_remote_code: false` | container settings |
| `recipes_file` | `~/.config/mcc/recipes.json` | recipe lanes (§7.2); a missing file means there are none |
| `new_days` | `7` | how long the NEW badge lasts |
| `engine_archs` | `{"vllm": "~/.config/mcc/vllm_archs.json", "sglang": "~/.config/mcc/sglang_archs.json"}` | per engine, the architecture list written by `mcc_deck.py archs`; a missing file = no check (§7.3) |

**Tuning the fit for other hardware:**

1. Set `gpu_budget_gib` to the memory one model may use on one node.
2. For a 24 GB card that is about 22.
3. For a 128 GB unified-memory box, after the OS and other services, the default 95.69 is a good start.

`mcc_deck.py fit 70` prints the node count for a 70 GB unit under your config.

## 9. Remote access

The deck listens on localhost and requires the token on every API call. To use it from your phone, pick one:

- **SSH tunnel:** `ssh -L 8099:127.0.0.1:8099 gpu-box`, then open `http://127.0.0.1:8099/`.
- **VPN** (WireGuard, Tailscale, …): set `listen` to the VPN address only.
- **Zero-trust reverse proxy** (an identity-aware tunnel or proxy): put login in front of it and keep the token as a second factor.

Don't expose the port directly to the internet. The token is a single shared secret, and INSERT/EJECT can stop running models.

## 10. More than one GPU node

List the other nodes in `fleet` with an ssh target; the page shows each one's free memory. The tiers tell you which models need 2, 3 or 4 nodes. Inserting those is intentionally refused, because multi-node serving depends on your interconnect and launcher. Keep the launch recipe in a kit folder on the NAS so it shows next to the model.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Header says *not synced yet* | Run `mcc_deck.py sync` by hand and read its JSON. `ssh -o BatchMode=yes <nas_ssh> true` must succeed with no prompt. |
| *NAS asleep since …* | Expected while the NAS sleeps. The page shows the last index. Wake the NAS and press **↻ check the NAS**. |
| GGUF shows *per-quant sizes after the next sync* | The sync couldn't list `.gguf` files. Check that `find` can read the library over ssh. |
| *protected model(s) … not found in Ollama* | Fix the tag spelling in `protected` (compare with `ollama list`). |
| *cannot read Ollama right now* | Is `ollama serve` up at `ollama_url`? The guard refuses rather than guessing. |
| vLLM / SGLang cassette never answers | Run `docker logs mcc-vllm` (or `mcc-sglang`). Common causes: a model that needs `trust_remote_code`, a too-large context setting, or a quant format your GPU or that engine doesn't support. |
| Recipe row says *stop Ollama first* | The recipe has `requires_ollama_stopped`. Stop Ollama, insert, and start Ollama again after EJECT. |
| Row says *architecture … is not in this vLLM image* | The image can't load that model. Pull a newer image and re-run `mcc_deck.py archs vllm`, try SGLang, or add a recipe lane for it. |
| Selector shows *no architecture list yet* | Run `mcc_deck.py archs vllm` (needs Docker and the image). Until then nothing is checked. |
| Recipe never answers | Run its `start` command by hand and watch the deck log. Check that `port` and `health_path` match what the recipe actually serves. |
| Copy speed far below the link speed | The NAS disks, not the network, are the limit (one HDD ≈ 150–250 MB/s). See §2.1. |
| Copy is slow | rsync runs over ssh. Use a wired link to the NAS; a single HDD tops out around 150–250 MB/s. |
| Token prompt keeps coming back | The page stores the token in your browser. Clear the site data and paste the current contents of `~/.config/mcc/token`. |

The deck log on the page shows the last copy or start. `journalctl --user -u mcc-deck -u mcc-sync` shows the services.

## 12. HTTP API

Every call except `/` and `/api/health` needs the header `X-Token: <token>`.

| method | path | body | returns |
|---|---|---|---|
| GET | `/api/health` | | `{"ok": true, "version"}` |
| GET | `/api/deck[?refresh=1]` | | full state: shelf (each row has `engines`), plans, slots, playing, history, `engine_bar`, `last_fail`, `arch_check` (`refresh` starts a background sync) |
| GET | `/api/deck/summary` | | counts per tier, NAS status |
| GET | `/api/deck/log` | | the last copy/start log and the current job |
| POST | `/api/deck/insert` | `{"id", "unit", "confirm_evict": [names]}` (`unit` = a key from the row's `units`, e.g. `"(whole variant) · SGLang"`) | `{"started": true}`, or `409 {"error", "would_evict"}` |
| POST | `/api/deck/eject` | `{"id"?, "remove_local"?}` | `{"done": [...]}` |

Example:

```bash
T=$(cat ~/.config/mcc/token)
curl -s -H "X-Token: $T" http://127.0.0.1:8099/api/deck/summary
curl -s -H "X-Token: $T" -d '{"id":"OpenAI/gpt-oss-20b/ggml-org-GGUF"}' http://127.0.0.1:8099/api/deck/insert
```

## 13. FAQ

**Why not just run models straight off the NAS share?** Loading a model reads every byte, often more than once, and network page-cache misses make it slow and fragile. Copy once to NVMe, then run.

**Does the sync wake my NAS?** No. It makes one ssh attempt with a 5-second timeout. If the NAS doesn't answer, it records *asleep* and keeps the last index.

**Can I use it without Ollama, or without Docker?** Yes. GGUF needs only Ollama, vLLM/SGLang only Docker; whichever is missing just refuses that path.

**Why not run several engines at once?** On a single GPU box, vLLM, SGLang and most recipes each reserve most of the GPU memory up front. Running two at once fails at load time or runs out of memory later, so the deck allows one at a time.

**Where is the data?** On the GPU box: `~/.local/state/mcc/` (index copy, history, log) and `~/models/deck/` (cassettes). On the NAS: `library/` and the per-model `manifest.json` and `ABOUT-MODEL.md`.

**How do I regenerate the images in these docs?** Run `python3 docs/src/make_charts.py` and `python3 docs/src/make_architecture.py`. The screenshots come from `mcc_deck.py serve --demo`.
