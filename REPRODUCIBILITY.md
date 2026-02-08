# REPRODUCIBILITY.md — Claim-to-Artifact Mapping

**Paper:** "Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems"  
**Authors:** Sepehr Bayat  
**Submission:** Nature Machine Intelligence

---

## Overview

Every quantitative claim in the paper is backed by code that produces the
supporting artifact.  This document maps each claim to the exact command that
reproduces it.

**Quick smoke test (< 2 min):**

```bash
bash scripts/build_submission.sh --smoke
```

**Full reproduction (≈ 30 min on a modern laptop):**

```bash
bash scripts/run_all_experiments.sh        # run all experiments
bash scripts/build_submission.sh           # compile PDFs + validate
```

---

## Claim-to-Artifact Table

| # | Paper Claim (Section) | Command | Output Artifact | Verification |
|---|----------------------|---------|-----------------|--------------|
| 1 | Phronesis Index Φ = λ₁⁺/(h¹_ε + ε) with O(N log N) complexity (Sec 3) | `python code/phronesis_core.py` | stdout: Φ, λ₁⁺, h¹_ε, eigenvalues | Inspect printed metrics |
| 2 | Logic Maze: Φ drops when anomaly is injected (Sec 5.1) | `cd code/logic_maze && python run_logic_maze.py --grid_size 5 --num_runs 10 --num_timesteps 150` | `results/phi_timeseries.csv`, `results/figure1_timeseries.png` | Column `mean_phi` drops at anomaly\_time |
| 3 | Bellman Consistency: STPGC reduces cumulative cost vs standard Q-learning (Sec 5.2) | `cd code/safety_gym && python train_safety_gym.py --grid_size 8 --num_seeds 10 --episodes 500` | `results/training_curves.csv`, `results/phi_trajectory.csv`, `results/figure2_barchart.png` | Q-learning+STPGC has lower mean\_cost; t-test printed to stdout |
| 4 | Multi-Robot Coordination: Φ tracks spatial consistency (Sec 5.3) | `cd code/multi_robot && python run_multi_robot.py --num_robots 10 --num_timesteps 100 --num_runs 10` | `results/consistency_metrics.csv`, `results/robot_trajectories.png` | Φ increases as robots converge |
| 5 | Scalability: O(N log N) up to 50 000 agents (Sec 5.4) | `cd code/scalability && python run_scalability.py --max_agents 50000 --num_trials 5` | `results/scalability_data.csv`, `results/figure4_scalability.png` | Log-log slope ≈ 1 (N log N) |
| 6 | h¹_ε = #{λᵢ < ε} − dim(H⁰), dim(H⁰) = n\_comp × d (Sec 3, Theorem 1) | `python -c "from phronesis_core import *; ..."` (see example in phronesis_core.py) | Metrics dict contains `dim_h0` | Check dim\_h0 = 2 for d=2 connected graph |
| 7 | Error bound: \|h¹_ε − h¹_true\| ≤ ⌈2σ/δ⌉ (Theorem 2) | Proof in paper; code in `phronesis_core.py` (`compute()` method) | Eigenvalue array in metrics | Verify eigenvalue gap > ε |
| 8 | Main manuscript compiles with 0 undefined references | `bash scripts/build_submission.sh` | `paper/main_manuscript.pdf`, build log | Log shows 0 undefined references |
| 9 | Supplementary compiles with 0 undefined references | `bash scripts/build_submission.sh` | `supplementary/supplementary_information.pdf` | Log shows 0 undefined references |

---

## What Is NOT Claimed

- We do **not** claim results on the full Safety Gym benchmark (MuJoCo).  The
  Bellman consistency experiment uses a reproducible grid-world MDP.
- We do **not** pre-fill specific numerical values in the paper table for the
  safe-RL experiment.  The table directs the reader to run the script and
  inspect `results/training_curves.csv`.
- All experimental data are generated **synthetically** by the provided code.
  No external datasets are required.

---

## Environment

- Python ≥ 3.8
- numpy, scipy, networkx, matplotlib
- No GPU required
- No external datasets required

```bash
pip install numpy scipy networkx matplotlib
```

---

## Smoke Test Checklist

Run this to verify the package in < 2 minutes:

```bash
# 1. Syntax check
python -c "from code.phronesis_core import PhronesisIndex; print('OK')"

# 2. Logic Maze smoke
cd code/logic_maze && python run_logic_maze.py --grid_size 3 --num_runs 2 --num_timesteps 30

# 3. Safety / Bellman smoke
cd ../safety_gym && python train_safety_gym.py --smoke

# 4. Multi-Robot smoke
cd ../multi_robot && python run_multi_robot.py --num_robots 5 --num_timesteps 20 --num_runs 2

# 5. Scalability smoke
cd ../scalability && python run_scalability.py --max_agents 500 --num_trials 2
```

All commands should exit with code 0 and produce CSV + PNG files in their
respective `results/` directories.
