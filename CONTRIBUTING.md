# Contributing

- Keep it dependency-free: Python standard library only, plus the page in one HTML file with no build step.
- Run the tests: `python3 -m unittest discover -s tests -v`.
- Any change to the guard (`plan_insert`) needs a test in `tests/test_deck.py` that shows the new rule fails closed.
- UI changes: run `python3 deck/mcc_deck.py serve --demo` and check a desktop and a phone-width window.
- If a change alters the UI, attach before/after screenshots from demo mode.
