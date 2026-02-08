#!/usr/bin/env bash
# reproduce_safety_gym.sh — Safety Gym safe RL (Sec 5.2)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[Safety Gym] smoke test (3 seeds)"
  python -m code.safety_gym.run --num_seeds 3
else
  echo "[Safety Gym] full reproduction (10 seeds)"
  python -m code.safety_gym.run --config code/safety_gym/config.yaml
fi
echo "✓ Safety Gym complete"
