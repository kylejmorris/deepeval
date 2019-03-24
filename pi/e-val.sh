#!/usr/bin/env bash

# Figure out which virtualenv to activate, based on architecture.
# RPi3 is armv7l, RPi0 is armv6l.
source "$HOME/$(uname -m)/bin/activate"

python "$HOME/bin/e-val.py"
