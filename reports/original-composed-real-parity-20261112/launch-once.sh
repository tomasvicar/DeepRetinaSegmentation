#!/bin/bash
set -u
cd /var/home/tom/code/DeepRetinaSegmentation_original_composed_parity_20261112
/var/home/tom/code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python observe.py
parity_observer_returncode=$?
printf '{"returncode":%d,"observed_returncode":true,"utc":"%s"}\n' "$parity_observer_returncode" "$(date -u +%FT%TZ)" > outer-exit.json
exit "$parity_observer_returncode"
