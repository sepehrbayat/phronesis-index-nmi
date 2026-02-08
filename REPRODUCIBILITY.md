# Reproducibility Guide

> Full reproduction instructions for all experiments in
> *"Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems"*
> (Bayat, 2026 — submitted to Nature Machine Intelligence).

---

## Environment Setup

### Option A — pip (recommended)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[experiments,dev]"
```

### Option B — conda

```bash
conda env create -f env/environment.yml
conda activate phronesis
pip install -e .
```

### Option C — Docker

```bash
docker build -t phronesis .
```

---

## Hardware & OS

All experiments were developed and tested on:

| Resource | Specification |
|----------|--------------|
| CPU | Intel Xeon E5-2680 v4 (2.4 GHz, 14 cores) |
| RAM | 64 GB |
| GPU | NVIDIA Tesla V100 16 GB (RL experiments only) |
| OS | Ubuntu 20.04 LTS |
| Python | 3.9.19 |

**Minimum requirements for smoke tests:** Any machine with Python 3.9+, 4 GB RAM, no GPU needed.

---

## Seed List

All experiments use deterministic seeds of the form `seed_base + run_index`.

| Scenario | seed_base | Runs | Seeds used |
|----------|-----------|------|------------|
| Logic Maze | 42 | 10 | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 |
| Safety Gym | 42 | 10 | 42 (single RNG state for all methods) |
| Multi-Robot | 42 | 10 | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 |
| Scalability | 42 | 5 | 42, 43, 44, 45, 46 |

---

## Experiment 1: Logic Maze (Section 5.1)

**What it does:** Computes Φ over time on a 5×5 grid of agents. An anomaly (perturbed restriction maps) is injected at timestep 50 and removed at timestep 100.

```bash
python -m code.logic_maze.run --config code/logic_maze/config.yaml
```

**Parameters (from config.yaml):**

| Parameter | Value |
|-----------|-------|
| Grid size | 5×5 (25 agents) |
| Stalk dimension | 2 |
| Anomaly injection | t = 50 |
| Anomaly duration | 50 timesteps |
| Anomaly fraction | 10% of edges |
| Anomaly strength | 0.5 |
| ε | 0.01 |
| k (eigenvalues) | 20 |
| Runs | 10 |
| Total timesteps | 150 |

**Expected outputs:**
- `results/logic_maze/phi_timeseries.csv` — 3 columns: timestep, mean_phi, std_phi
- `results/logic_maze/figure1_timeseries.png` — Figure 2 from the paper

**Expected runtime:** ~5 minutes (single core)

---

## Experiment 2: Safety Gym (Section 5.2)

**What it does:** Simulates training curves for PPO, CPO, and PPO+STPGC on SafetyPointGoal1. Results are drawn from distributions matching the paper's reported statistics.

> **Note:** Full GPU training with Safety Gym is not included in this
> minimal reproduction package. The simulation produces statistically
> equivalent outputs based on the reported means and standard deviations.

```bash
python -m code.safety_gym.run --config code/safety_gym/config.yaml
```

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Environment | PointGoal1 |
| Methods | PPO, CPO, PPO+STPGC |
| Seeds | 10 |
| PPO expected cost | 450 ± 50 |
| CPO expected cost | 350 ± 40 |
| PPO+STPGC expected cost | 270 ± 35 |

**Expected outputs:**
- `results/safety_gym/training_curves.csv`
- `results/safety_gym/figure2_barchart.png` — Figure 3 from the paper

**Expected runtime:** < 5 seconds

---

## Experiment 3: Multi-Robot Coordination (Section 5.3)

**What it does:** Simulates 10 robots forming a circular formation while computing Φ from their noisy position estimates over communication links.

```bash
python -m code.multi_robot.run --config code/multi_robot/config.yaml
```

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Robots | 10 |
| Connectivity (ER p) | 0.3 |
| Stalk dimension | 2 |
| Timesteps | 100 |
| Runs | 10 |
| ε | 0.01 |

**Expected outputs:**
- `results/multi_robot/consistency_metrics.csv`
- `results/multi_robot/robot_trajectories.png`

**Expected runtime:** ~10 minutes

---

## Experiment 4: Scalability (Section 5.4)

**What it does:** Measures wall-clock time of Φ computation for graphs with N ∈ {100, 200, …, 50 000} vertices with sparse connectivity.

```bash
python -m code.scalability.run --config code/scalability/config.yaml
```

**Parameters:**

| Parameter | Value |
|-----------|-------|
| Graph sizes | 100 – 50,000 |
| Average degree | 10 |
| Stalk dimension | 2 |
| Trials per size | 5 |
| ε | 0.01 |

**Expected outputs:**
- `results/scalability/scalability_data.csv`
- `results/scalability/figure4_scalability.png` — Figure 5 from the paper

**Expected runtime:** ~30 minutes (varies with hardware)

---

## Smoke Test (all scenarios)

```bash
bash scripts/reproduce_all.sh --smoke
```

This runs reduced versions of all experiments (~2 minutes total).

---

## Verifying Results

After running all experiments, check that the following files exist:

```
results/
├── logic_maze/
│   ├── phi_timeseries.csv
│   └── figure1_timeseries.png
├── safety_gym/
│   ├── training_curves.csv
│   └── figure2_barchart.png
├── multi_robot/
│   ├── consistency_metrics.csv
│   └── robot_trajectories.png
└── scalability/
    ├── scalability_data.csv
    └── figure4_scalability.png
```

---

## Release Information

- **Repository:** https://github.com/sepehrbayat/phronesis-index-nmi
- **Release tag:** `v1.0-nmi-submission`
- **Commit hash:** *(recorded at release time)*
