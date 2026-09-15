#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

PYTHON_CMD=""
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_CMD="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_CMD="python3.11"
else
  PYTHON_CMD="python3"
fi

echo "Using $PYTHON_CMD"
"$PYTHON_CMD" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-backend.txt

echo
echo "ReUP backend setup is complete."
echo "Double-click START_REUP_BACKEND.command to run the demo."
read "?Press Return to close..."
