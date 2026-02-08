#!/usr/bin/env bash
# reproduce_scalability.sh — Scalability test (Sec 5.4)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--smoke" ]]; then
  echo "[Scalability] smoke test (max 1000 agents, 2 trials)"
  python -m code.scalability.run --max_agents 1000 --num_trials 2
else
  echo "[Scalability] full reproduction (up to 50,000 agents, 5 trials)"
  python -m code.scalability.run --config code/scalability/config.yaml
fi
echo "✓ Scalability complete"
