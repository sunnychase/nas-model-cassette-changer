#!/usr/bin/env bash
# Install the indexer on the NAS (run as root). Idempotent. Edit /etc/mcc/nas.json afterwards.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
install -m 0755 "$here/nas/mcc_index.py" /usr/local/bin/mcc_index.py
install -d -m 0755 /etc/mcc
[ -f /etc/mcc/nas.json ] || install -m 0644 "$here/examples/nas.example.json" /etc/mcc/nas.json
install -m 0644 "$here/nas/systemd/mcc-index.service" "$here/nas/systemd/mcc-index.timer" /etc/systemd/system/
systemctl daemon-reload
echo "Installed. Next:"
echo "  1. edit /etc/mcc/nas.json (data_root, staging folders) and set User= in /etc/systemd/system/mcc-index.service"
echo "  2. dry run:  python3 /usr/local/bin/mcc_index.py --dry-run"
echo "  3. enable:   systemctl enable --now mcc-index.timer"
