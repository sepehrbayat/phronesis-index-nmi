#!/usr/bin/env bash
# reproduce_logic_maze.sh — Logic Maze anomaly detection (Sec 5.1)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[Logic Maze] smoke test (grid=3, 2 runs, 50 steps)"
  python -m code.logic_maze.run --grid_size 3 --num_runs 2 --num_timesteps 50
else
  echo "[Logic Maze] full reproduction (grid=5, 10 runs, 150 steps)"
  python -m code.logic_maze.run --config code/logic_maze/config.yaml
fi
echo "✓ Logic Maze complete"
