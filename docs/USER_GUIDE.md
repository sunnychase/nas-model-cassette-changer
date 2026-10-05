# User Guide: NAS Model Cassette Changer

1. [What it does](#1-what-it-does)
2. [Requirements](#2-requirements)
3. [Set up the NAS](#3-set-up-the-nas)
4. [Set up the GPU box](#4-set-up-the-gpu-box)
5. [Using the page](#5-using-the-page)
6. [The guard, rule by rule](#6-the-guard-rule-by-rule)
7. [Configuration reference](#7-configuration-reference)
8. [Remote access](#8-remote-access)
9. [More than one GPU node](#9-more-than-one-gpu-node)
10. [Troubleshooting](#10-troubleshooting)
11. [HTTP API](#11-http-api)
12. [FAQ](#12-faq)

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

**Insert** copies the unit to `~/models/deck/<Maker>/<Model>/<Variant>/` with `rsync --partial`, so an interrupted copy resumes. It then starts the unit: a GGUF is registered as `deck/<name>` in Ollama and warm-loaded; a safetensors folder runs in a `vllm/vllm-openai` container on port 8010.

## 2. Requirements

**NAS:** Linux, Python 3.9+, your model folders on local disk. It must be reachable by ssh from the GPU box (key auth). An NFS or SMB export is optional; it only matters if you want to browse the library by hand.

**GPU box:** Linux, Python 3.9+, `ssh`, `rsync`, plus [Ollama](https://ollama.com) for GGUF cassettes and/or Docker with the NVIDIA container toolkit for vLLM cassettes. It needs enough local NVMe for the largest model you plan to insert, with room to spare.

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

## 5. Using the page

- **Header:** model counts per tier, how many models are new (first seen in the last 7 days), whether the NAS was awake at the last sync and when its index was built, and your protected models.
- **Deck:** what is loading or playing. **⏏ EJECT** stops it.
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

**Memory check for GGUF:** the cassette needs `1.2 × W + 3 + kv_band_gib` GiB, at `num_ctx`. Free memory is `MemAvailable − mem_floor_gib − reserve`. The *reserve* is the room for each protected model that is not loaded right now: its largest observed footprint, or 1.3 × its size + 1.5 GiB. If the cassette doesn't fit, the deck picks evictions in this order: older deck cassettes first, then unprotected models from largest to smallest. It also counts Ollama's own slot limit (`ollama_max_loaded`).

**For safetensors (vLLM):** the container takes most of the GPU, so it would stop every big lane and every Ollama model, protected ones included. The dialog names them all.

**Any eviction needs every name confirmed.** Then:

1. The plan is computed again after the copy, which can take minutes. If anything changed, nothing is evicted and you are asked again.
2. After a GGUF loads, the deck checks that every protected model that was loaded before is still loaded. If Ollama dropped one anyway, the new cassette is unloaded and the log says so.

## 7. Configuration reference

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
| `big_lanes` | `[]` | systemd user units that hold the GPU; a vLLM insert stops them by name |
| `vllm.*` | image, port 8010, 0.85, 32768, `trust_remote_code: false` | container settings |
| `new_days` | `7` | how long the NEW badge lasts |

**Tuning the fit for other hardware:**

1. Set `gpu_budget_gib` to the memory one model may use on one node.
2. For a 24 GB card that is about 22.
3. For a 128 GB unified-memory box, after the OS and other services, the default 95.69 is a good start.

`mcc_deck.py fit 70` prints the node count for a 70 GB unit under your config.

## 8. Remote access

The deck listens on localhost and requires the token on every API call. To use it from your phone, pick one:

- **SSH tunnel:** `ssh -L 8099:127.0.0.1:8099 gpu-box`, then open `http://127.0.0.1:8099/`.
- **VPN** (WireGuard, Tailscale, …): set `listen` to the VPN address only.
- **Zero-trust reverse proxy** (an identity-aware tunnel or proxy): put login in front of it and keep the token as a second factor.

Don't expose the port directly to the internet. The token is a single shared secret, and INSERT/EJECT can stop running models.

## 9. More than one GPU node

List the other nodes in `fleet` with an ssh target; the page shows each one's free memory. The tiers tell you which models need 2, 3 or 4 nodes. Inserting those is intentionally refused, because multi-node serving depends on your interconnect and launcher. Keep the launch recipe in a kit folder on the NAS so it shows next to the model.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Header says *not synced yet* | Run `mcc_deck.py sync` by hand and read its JSON. `ssh -o BatchMode=yes <nas_ssh> true` must succeed with no prompt. |
| *NAS asleep since …* | Expected while the NAS sleeps. The page shows the last index. Wake the NAS and press **↻ check the NAS**. |
| GGUF shows *per-quant sizes after the next sync* | The sync couldn't list `.gguf` files. Check that `find` can read the library over ssh. |
| *protected model(s) … not found in Ollama* | Fix the tag spelling in `protected` (compare with `ollama list`). |
| *cannot read Ollama right now* | Is `ollama serve` up at `ollama_url`? The guard refuses rather than guessing. |
| vLLM cassette never answers | Run `docker logs mcc-vllm`. Common causes: a model that needs `trust_remote_code`, a too-large `max_model_len`, or a quant format your GPU doesn't support. |
| Copy is slow | rsync runs over ssh. Use a wired link to the NAS; a single HDD tops out around 150–250 MB/s. |
| Token prompt keeps coming back | The page stores the token in your browser. Clear the site data and paste the current contents of `~/.config/mcc/token`. |

The deck log on the page shows the last copy or start. `journalctl --user -u mcc-deck -u mcc-sync` shows the services.

## 11. HTTP API

Every call except `/` and `/api/health` needs the header `X-Token: <token>`.

| method | path | body | returns |
|---|---|---|---|
| GET | `/api/health` | | `{"ok": true, "version"}` |
| GET | `/api/deck[?refresh=1]` | | full state: shelf, plans, slots, playing, history (`refresh` starts a background sync) |
| GET | `/api/deck/summary` | | counts per tier, NAS status |
| GET | `/api/deck/log` | | the last copy/start log and the current job |
| POST | `/api/deck/insert` | `{"id", "unit", "confirm_evict": [names]}` | `{"started": true}`, or `409 {"error", "would_evict"}` |
| POST | `/api/deck/eject` | `{"id"?, "remove_local"?}` | `{"done": [...]}` |

Example:

```bash
T=$(cat ~/.config/mcc/token)
curl -s -H "X-Token: $T" http://127.0.0.1:8099/api/deck/summary
curl -s -H "X-Token: $T" -d '{"id":"OpenAI/gpt-oss-20b/ggml-org-GGUF"}' http://127.0.0.1:8099/api/deck/insert
```

## 12. FAQ

**Why not just run models straight off the NAS share?** Loading a model reads every byte, often more than once, and network page-cache misses make it slow and fragile. Copy once to NVMe, then run.

**Does the sync wake my NAS?** No. It makes one ssh attempt with a 5-second timeout. If the NAS doesn't answer, it records *asleep* and keeps the last index.

**Can I use it without Ollama, or without Docker?** Yes. GGUF needs only Ollama, safetensors only Docker; whichever is missing just refuses that path.

**Where is the data?** On the GPU box: `~/.local/state/mcc/` (index copy, history, log) and `~/models/deck/` (cassettes). On the NAS: `library/` and the per-model `manifest.json` and `ABOUT-MODEL.md`.

**How do I regenerate the images in these docs?** Run `python3 docs/src/make_charts.py` and `python3 docs/src/make_architecture.py`. The screenshots come from `mcc_deck.py serve --demo`.
