#!/usr/bin/env bash
# reproduce_all.sh — End-to-end reproduction of all paper experiments.
#
# Modes:
#   ./scripts/reproduce_all.sh          — full reproduction (hours)
#   ./scripts/reproduce_all.sh --smoke  — quick smoke test (~2 min)
#
# Prerequisites: pip install -e ".[experiments]"
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=false
if [[ "${1:-}" == "--smoke" ]]; then SMOKE=true; fi

echo "=============================================="
echo " Phronesis Index — Full Experiment Suite"
if $SMOKE; then echo " (SMOKE-TEST MODE)"; fi
echo "=============================================="
echo ""

bash scripts/reproduce_logic_maze.sh ${SMOKE:+--smoke}
bash scripts/reproduce_safety_gym.sh ${SMOKE:+--smoke}
bash scripts/reproduce_multi_robot.sh ${SMOKE:+--smoke}
bash scripts/reproduce_scalability.sh ${SMOKE:+--smoke}

echo ""
echo "=============================================="
echo " All experiments completed successfully."
echo " Results saved under results/"
echo "=============================================="
