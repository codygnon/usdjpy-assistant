#!/bin/bash
# Run from repo root: ./deploy/setup_vps.sh
# Installs Python deps, creates data dirs, installs systemd units for API + loop.

set -e
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"
cd "$APP_ROOT"

echo "USDJPY Assistant VPS setup (app root: $APP_ROOT)"

# Install system packages
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-venv python3-pip git

# Data dirs (profiles + logs live here; USDJPY_DATA_DIR points at data/)
mkdir -p "$APP_ROOT/data/profiles/v1" "$APP_ROOT/data/logs"
echo "Data dirs: $APP_ROOT/data/profiles/v1, $APP_ROOT/data/logs"

# Virtual env and Python deps
if [ ! -d "$APP_ROOT/.venv" ]; then
  echo "Creating venv..."
  python3 -m venv "$APP_ROOT/.venv"
fi
echo "Installing Python dependencies..."
"$APP_ROOT/.venv/bin/pip" install -q -r "$APP_ROOT/requirements.txt"

# Install systemd units (replace __APP_ROOT__ with actual path)
echo "Installing systemd units..."
for f in usdjpy-api.service usdjpy-loop.service; do
  sed "s|__APP_ROOT__|$APP_ROOT|g" "$DEPLOY_DIR/$f" | sudo tee "/etc/systemd/system/$f" > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable usdjpy-api usdjpy-loop
sudo systemctl start usdjpy-api usdjpy-loop

echo "Done. API and loop are enabled and started."
echo "  API:  http://$(hostname -I | awk '{print $1}'):8000"
echo "  Status: sudo systemctl status usdjpy-api usdjpy-loop"
echo "  Logs:  sudo journalctl -u usdjpy-api -u usdjpy-loop -f"
