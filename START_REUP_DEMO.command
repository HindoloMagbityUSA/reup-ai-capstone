#!/bin/zsh

set -e

PACKAGE_DIR="${0:A:h}"
cd "$PACKAGE_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ReUP demo setup is required once."
  echo "Creating the local Python environment..."
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_SETUP="python3.11"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_SETUP="python3.12"
  else
    echo "Python 3.11 or 3.12 is required for this packaged model."
    echo "Install Python 3.11, then start this launcher again."
    read "?Press Return to close."
    exit 1
  fi
  "$PYTHON_SETUP" -m venv .venv
fi

if ! .venv/bin/python -c "import torch, torchvision, streamlit" >/dev/null 2>&1; then
  echo "Installing the ReUP demo requirements..."
  .venv/bin/python -m pip install -r requirements.txt
fi

echo "Starting the ReUP AI Listing Assistant..."
PYTHONPATH=src .venv/bin/python -m streamlit run src/demo_app.py
