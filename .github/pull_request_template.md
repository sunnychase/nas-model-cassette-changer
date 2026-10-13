## What and why

<!-- What does this change, and what problem does it solve? Link the issue: "Fixes #123". -->

## How it was tested

<!-- Only what you actually ran, with the result. -->

- [ ] `python3 -m unittest discover -s tests -v` passes
- [ ] `bash scripts/check.sh` passes
- [ ] Demo mode (`python3 deck/mcc_deck.py serve --demo`), desktop and phone width, if it touches the page
- On real hardware: <!-- engine, model format, GPU box, NAS; or "not run" and why -->

## Checklist

- [ ] One focused change.
- [ ] Standard library only; the page is still one file with no build step and no external assets.
- [ ] A change to the guard has a test in `tests/test_deck.py` that shows the new rule fails closed.
- [ ] No personal paths, host names, addresses, tokens or e-mail addresses in code, docs, examples or commit messages.
- [ ] CHANGELOG.md has an entry, and the README or User Guide is updated if behaviour, config keys or the API changed.
- [ ] Before/after screenshots from demo mode if the UI changed.
- [ ] Anything not tested is said above.
