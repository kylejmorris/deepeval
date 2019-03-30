#!/usr/bin/env bash

echo "Starting E-Val"

# Figure out which virtualenv to activate, based on architecture.
# RPi3 is armv7l, RPi0 is armv6l.
source "$HOME/$(uname -m)/bin/activate"

export PYTHONPATH="$HOME/src"
export LD_LIBRARY_PATH="$HOME/lib"

python "$HOME/bin/e-val.py"
