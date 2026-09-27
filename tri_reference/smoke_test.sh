#!/usr/bin/env bash
# Quick installation check on a shortened timeline; not part of the prespecified protocol.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY -m pytest -q tests
$PY -m tri.nullcal --out results_smoke_nullcal.json --n-sim 60
$PY -m tri.experiment --mode smoke --seeds 1 2 --out results_smoke
$PY -m tri.analysis --results results_smoke
