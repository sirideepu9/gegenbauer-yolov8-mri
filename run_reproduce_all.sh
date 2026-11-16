#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "Gegenbauer-YOLOv8 Reproducible Research Pipeline"
echo "Paper: Learnable Gegenbauer Activations and Direct Coordinate Regression"
echo "Authors: M. Deepthi, G. N. V. R. Vikram (2026)"
echo "=========================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ">>> 1. Standalone Table 3 and Section 4.2 Statistical Derivation:"
python3 experiments/compute_statistics.py

echo ">>> 2. Master Mathematical & Experimental Verification Suite:"
python3 reproduce_all_results.py
