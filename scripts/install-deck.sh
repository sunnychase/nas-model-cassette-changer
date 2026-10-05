#!/usr/bin/env bash
# Install the deck for the current user on the GPU box (no root needed). Idempotent.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dst="$HOME/.local/share/mcc"; mkdir -p "$dst" "$HOME/.config/mcc" "$HOME/.config/systemd/user"
install -m 0755 "$here/deck/mcc_deck.py" "$dst/mcc_deck.py"; install -m 0644 "$here/deck/deck.html" "$dst/deck.html"
[ -f "$HOME/.config/mcc/deck.json" ] || install -m 0600 "$here/examples/deck.example.json" "$HOME/.config/mcc/deck.json"
install -m 0644 "$here"/deck/systemd/mcc-*.service "$here"/deck/systemd/mcc-*.timer "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
echo "Installed to $dst. Next:"
echo "  1. edit ~/.config/mcc/deck.json (nas_ssh alias, nas_library_dir, protected models)"
echo "  2. check the NAS link:  ssh -o BatchMode=yes <nas_ssh> true"
echo "  3. first sync:          python3 $dst/mcc_deck.py sync"
echo "  4. enable:              systemctl --user enable --now mcc-sync.timer mcc-deck.service"
echo "  5. open http://127.0.0.1:8099/ and paste the token from ~/.config/mcc/token"
