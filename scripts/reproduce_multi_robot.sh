#!/usr/bin/env bash
# reproduce_multi_robot.sh — Multi-robot coordination (Sec 5.3)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[Multi-Robot] smoke test (5 robots, 2 runs, 30 steps)"
  python -m code.multi_robot.run --num_robots 5 --num_runs 2 --num_timesteps 30
else
  echo "[Multi-Robot] full reproduction (10 robots, 10 runs, 100 steps)"
  python -m code.multi_robot.run --config code/multi_robot/config.yaml
fi
echo "✓ Multi-Robot complete"
