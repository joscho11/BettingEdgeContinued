#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.local/bin:${PATH}"
REPO="/mnt/c/Users/josep/Desktop/random_stuff/cowork_OS/JoSchoAnalytics"
VENV="/root/jsa-visual-venv"
cd "${REPO}"
export APP_OFFLINE=1
export PYTHONUNBUFFERED=1
# Do not export LOCALAPPDATA so Linux Playwright cannot pick a Windows Chrome.
unset LOCALAPPDATA || true
PY="${VENV}/bin/python"
"${PY}" -m pytest tests/test_visual_regression.py --update-visual -v --tb=line
"${PY}" -m pytest tests/test_visual_regression.py -v --tb=line
