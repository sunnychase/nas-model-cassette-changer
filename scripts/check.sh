#!/bin/bash
# The repository's checks (CI runs the same script): every Python file compiles, and no tracked file carries a
# home-directory path, a private network address or something shaped like a Hugging Face token.
# No path is excluded from the hygiene grep: the examples and docs use placeholders (nas, node-2.local, /srv/models-nas, ~),
# never real addresses. The unit tests are separate: python3 -m unittest discover -s tests
# Needs python3 and git. Run from anywhere: bash scripts/check.sh [py|hygiene]   (no argument = both)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0; only="${1:-all}"
run() { [ "$only" = all ] || [ "$only" = "$1" ]; }
tmp=$(mktemp -d); trap 'rm -r "$tmp"' EXIT

if run py; then
echo "== python: py_compile"
n=0
while IFS= read -r f; do
  n=$((n + 1)); PYTHONPYCACHEPREFIX="$tmp/pyc" python3 -m py_compile "$f" || { echo "FAIL $f"; fail=1; }
done < <(git ls-files '*.py')
echo "$n files"
fi

if run hygiene; then
echo "== hygiene: home paths, private addresses, token-shaped strings"
hits=$(git grep -I -n -E '/home/[a-z]|192\.168\.[0-9]|\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b|hf_[A-Za-z0-9]{20,}' -- . | cut -c1-200)
if [ -n "$hits" ]; then echo "$hits"; echo "FAIL: use \$HOME, ~ or a config placeholder instead; never commit addresses or tokens"; fail=1; else echo "clean"; fi
fi

[ "$fail" = 0 ] && echo "ALL CHECKS PASSED" || echo "CHECKS FAILED"
exit "$fail"
