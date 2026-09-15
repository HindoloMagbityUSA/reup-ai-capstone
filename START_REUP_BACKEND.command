#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "The local environment is not ready."
  echo "Double-click SETUP_REUP_BACKEND.command first."
  read "?Press Return to close..."
  exit 1
fi

source .venv/bin/activate
export PYTHONPATH=src

(sleep 2; open "http://127.0.0.1:8000/demo") &
python -m uvicorn api_app:app --host 0.0.0.0 --port 8000
