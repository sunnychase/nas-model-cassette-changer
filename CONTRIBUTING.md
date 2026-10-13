# Contributing

Thanks for helping. Bug reports, fixes, documentation and focused features are all welcome.

## Before you start

- **Bugs and ideas:** open an [issue](https://github.com/sunnychase/nas-model-cassette-changer/issues) with the bug or
  feature form. Search first; one problem or idea per issue.
- **Security problems:** do not open a public issue; see [SECURITY.md](SECURITY.md).
- **Bigger changes** (a new engine, a change to the guard, a new config key): open an issue first so the approach can be
  agreed before you spend time on it.

## Running it locally

No NAS or GPU is needed for most work. Demo mode fakes a 35-model NAS and a 128 GB GPU box and touches nothing on your
machine:

```bash
python3 deck/mcc_deck.py serve --demo                    # open http://127.0.0.1:8099/
python3 deck/mcc_deck.py serve --demo --demo-no-ollama   # Ollama "stopped": play the TensorFold recipe row
```

## House rules

- Keep it dependency-free: Python standard library only, plus the page in one HTML file with no build step.
- Any change to the guard (`plan_insert`) needs a test in `tests/test_deck.py` that shows the new rule fails closed.
- No personal or machine-specific data (home-directory paths, host names, private addresses, user names, tokens or e-mail
  addresses) in code, docs, examples or commit messages.
- Say what changed: add an entry to [CHANGELOG.md](CHANGELOG.md), and update the README or the
  [User Guide](docs/USER_GUIDE.md) when behaviour, config keys or the API change.

[AGENTS.md](AGENTS.md) lists the guard's promises in full; it is written for coding agents but is a good checklist for anyone.

## Checks

Run both before every commit. CI (`.github/workflows/tests.yml`) runs them on every push and pull request:

```bash
python3 -m unittest discover -s tests -v   # the tests, on Python 3.9, 3.11 and 3.12 in CI
bash scripts/check.sh                      # py_compile on every .py, and a grep for home paths, private addresses and tokens
```

UI changes: run `python3 deck/mcc_deck.py serve --demo` and check a desktop and a phone-width window.

## Pull requests

- One focused change per pull request, with a short imperative title.
- Fill in the template: what changed, why, how you tested it, and anything you could not test.
- If a change alters the UI, attach before/after screenshots from demo mode.
- Do not commit a real `deck.json`, `nas.json` or `token`, models or `__pycache__`.

By contributing you agree that your contribution is licensed under the [MIT licence](LICENSE).
