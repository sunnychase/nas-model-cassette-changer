# Notes for coding agents

Guidance for AI coding agents (and people) changing this repository. Read [CONTRIBUTING.md](CONTRIBUTING.md) too.

## What this is

Lion AI Models (the NAS Model Cassette Changer): a NAS indexer that catalogues model folders, and a deck on the GPU box that
copies one model at a time to local NVMe and plays it in Ollama, a vLLM or SGLang container, or a recipe lane, behind a guard
that never stops anything the user did not name. See [How it works](README.md#how-it-works).

## Layout

| path | what it is |
|---|---|
| `nas/mcc_index.py` | the NAS indexer: library symlinks, tiers, kits, `library.json`, `manifest.json`, `ABOUT-MODEL.md`; standard library only |
| `deck/mcc_deck.py` | the GPU-box deck: sync, the guard (`plan_insert`, `plan_recipe`), insert / eject, benchmarks, the web server and API, demo mode; standard library only |
| `deck/deck.html` | the whole page: HTML, CSS and JavaScript in one file, no build step, no external assets |
| `nas/systemd/`, `deck/systemd/` | the index timer on the NAS; the sync timer and the deck service on the GPU box |
| `examples/` | `nas.example.json`, `deck.example.json`, `recipes.example.json` (TensorFold), with placeholders only |
| `scripts/` | `install-nas.sh`, `install-deck.sh`, and `check.sh` (the repository checks) |
| `tests/` | `test_deck.py` (fit, units, the guard, recipes, a demo round trip, the selector, benchmarks) and `test_index.py` |
| `docs/` | `USER_GUIDE.md`, the images, and `docs/src/` (the scripts that draw them; `make_charts.py` needs matplotlib, as doc tooling only) |

## Running it

```bash
python3 -m unittest discover -s tests -v       # the unit tests (CI: Python 3.9, 3.11 and 3.12)
bash scripts/check.sh                          # py_compile on every .py, and the hygiene grep
python3 deck/mcc_deck.py serve --demo          # fake 35-model NAS + 128 GB GPU box, no token; open http://127.0.0.1:8099/
python3 deck/mcc_deck.py serve --demo --demo-no-ollama   # Ollama "stopped", so the TensorFold recipe row can be played
```

Demo mode touches nothing on the machine: no ssh, rsync, Docker or Ollama. Use it, and the tests, for every change; never
exercise a change with a real INSERT or EJECT.

## The guard: semantics never to break

These are the promises the README and the User Guide make. A change that weakens one is a bug, even if every test passes.

1. **An insert never stops anything the user did not name.** If a plan would stop or evict anything, the API answers 409
   with the exact `would_evict` list and does nothing until `confirm_evict` names every one of them.
2. **Protected models** (`protected` in `deck.json`) keep their memory reserved even when they are unloaded, and a GGUF
   insert never evicts them. A protected tag that is not installed in Ollama refuses the insert (a typo must not silently
   protect nothing). vLLM and SGLang take most of the GPU, so they name every Ollama model they would stop, protected
   ones included.
3. **Fail closed.** If Ollama cannot be read, refuse rather than guess. Once an engine has an architecture list, a model whose
   architectures are missing, unreadable or undeclared is refused before anything is copied or stopped.
4. **Plan, then act, then re-check.** `plan_insert` never acts. The plan is computed again after the copy, and if anything
   changed nothing is evicted and the user is asked again. After a GGUF loads, a protected model that was dropped anyway
   means the new cassette is unloaded.
5. **One GPU engine at a time**, and the deck never stops a system service (such as Ollama) itself; a recipe with
   `requires_ollama_stopped` is refused while Ollama answers.
6. **The NAS is read-only to the deck** (`cat`, `find`, `rsync` from the NAS), and the sync never wakes a sleeping NAS. The
   indexer never moves, renames or deletes a model file. **Deleting a local copy** stays inside `local_dir`.

Any change to the guard needs a test in `tests/test_deck.py` that shows the new rule fails closed.

## Rules

1. **Standard library only**, Python 3.9+: nothing to `pip install` for the deck, the indexer or the tests. Avoid syntax
   newer than 3.9 (no `match`, no `X | Y` type unions at runtime).
2. **The page stays one self-contained file**: no build step, no external assets, no CDN. Element ids in `deck.html` are
   used by its script; do not rename them casually.
3. **Safe defaults stay safe**: listen on `127.0.0.1`, a token on every API call (compared in constant time),
   `trust_remote_code` off, recipe commands only from the user's own `recipes.json`.
4. **Nothing machine-specific or personal** in code, comments, docs, examples or commit messages: no home-directory paths,
   host names, private addresses, user names, e-mail addresses or tokens. Use `~`, `$HOME` or a config key with a
   placeholder default.
5. **Measured numbers only.** Copy rates, memory needs and benchmark figures in docs come from real runs.
6. **Document changes**: a CHANGELOG.md entry, and README / User Guide updates when behaviour, config keys or the API change.

## Do not

- Commit a real `deck.json`, `nas.json` or `token` (they are git-ignored), models, `__pycache__/` or state files.
- Start, stop or restart services, containers or Ollama, run `rsync` or `ssh` against real machines, or download models as a
  side effect of a code change.
- Add a dependency, a network listener beyond `127.0.0.1` by default, or a call to a model hub or cloud API.
