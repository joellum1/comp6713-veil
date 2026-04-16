#!/bin/bash

set -e  # stop on error

# ----- virtual environment ----------------

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# ----- upgrading pip ----------------------

pip install --upgrade pip

# ----- install requirements ---------------

echo "Installing dependencies..."
pip install -r requirements.txt


# ----- download datasets ------------------

echo "Downloading datasets..."
python3 data/retrieve.py


# ----- process datasets -------------------

echo "Processing datasets..."
python3 data/process.py


# ------------------------------------------

echo "Setup complete."