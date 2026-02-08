# Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems

This repository contains the code and data for the paper "Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems" by Sepehr Bayat, submitted to Nature Machine Intelligence.

## Overview

The **Phronesis Index** (Φ) is a computationally efficient spectral heuristic that quantifies the global consistency of a multi-agent system by approximating topological obstructions (cohomology) in cellular sheaves. This repository provides implementations of the Phronesis Index and the STPGC (Sheaf-Theoretic Policy Gradient with Consistency) algorithm across four experimental scenarios.

## Repository Structure

```
.
├── paper/              # LaTeX source and figures for the manuscript
├── code/               # Source code for all experiments
│   ├── phronesis_core.py  # Core Phronesis Index library
│   ├── logic_maze/     # Logic Maze anomaly detection
│   ├── safety_gym/     # Bellman consistency / safe RL (grid-world)
│   ├── multi_robot/    # Multi-robot spatial consistency
│   └── scalability/    # Scalability tests
├── scripts/            # Shell scripts for reproduction & Docker
├── env/                # Environment specifications (requirements.txt, environment.yml)
├── Dockerfile          # Reproducible Docker build
├── REPRODUCIBILITY.md  # Claim-to-artifact mapping
├── LICENSE             # MIT License
└── CITATION.cff        # Citation information
```

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- NetworkX (for graph operations)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/sepehrbayat/phronesis-index-nmi.git
cd phronesis-index-nmi
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r env/requirements.txt
```

3. (Optional) If using conda:
```bash
conda env create -f env/environment.yml
conda activate phronesis
```

## Reproducing the Results

Each experiment can be reproduced using the provided scripts.

**Recommended:** Use Docker for fully reproducible execution:
```bash
docker build -t phronesis-nmi .
docker run --rm phronesis-nmi
```

### 1. Logic Maze (Anomaly Detection)

```bash
cd code/logic_maze
python run_logic_maze.py --grid_size 5 --anomaly_time 50 --num_runs 10
```

This will generate:
- `results/phi_timeseries.csv` - Time series of Φ values
- `results/figure1_timeseries.png` - Plot matching Figure 2 in the paper

**Expected runtime:** ~5 minutes

### 2. Bellman Consistency (Safe Navigation)

```bash
cd code/safety_gym
python train_safety_gym.py --grid_size 8 --num_seeds 10 --episodes 500
# Or for a quick smoke test:
python train_safety_gym.py --smoke
```

This will train tabular Q-learning agents and generate:
- `results/training_curves.csv` - Cost comparison across methods
- `results/phi_trajectory.csv` - Phronesis Index during training
- `results/figure2_barchart.png` - Cost comparison bar chart

**Expected runtime:** ~5 minutes (full), ~30 seconds (smoke)

### 3. Multi-Robot Coordination

```bash
cd code/multi_robot
python run_multi_robot.py --num_robots 10 --num_timesteps 100 --num_runs 10
```

This will generate:
- `results/consistency_metrics.csv` - Consistency metrics over time
- `results/robot_trajectories.png` - Visualization of robot paths

**Expected runtime:** ~10 minutes

### 4. Scalability Tests

```bash
cd code/scalability
python run_scalability.py --max_agents 10000 --num_trials 3
```

This will generate:
- `results/scalability_data.csv` - Computation time vs. number of agents
- `results/figure4_scalability.png` - Scalability plot (Figure 5 in the paper)

**Expected runtime:** ~5 minutes

### Running All Experiments

To reproduce all experiments at once:
```bash
bash scripts/run_all_experiments.sh
```

This will run all four experiments sequentially and collect results in the `results/` directory.

## Code Structure

### Core Library

The core Phronesis Index computation is implemented in `code/phronesis_core.py`:

```python
from phronesis_core import PhronesisIndex

# Create a PhronesisIndex object
phi = PhronesisIndex(graph, stalks, restriction_maps)

# Compute the index
phi_value = phi.compute(epsilon=0.01)

# Get detailed metrics
metrics = phi.get_metrics()
print(f"Spectral gap: {metrics['lambda_1']}")
print(f"Cohomology dimension: {metrics['h1_epsilon']}")
```

### Bellman Consistency (STPGC)

The STPGC reward shaping is implemented inside `code/safety_gym/train_safety_gym.py`. It
trains tabular Q-learning agents on a grid-world MDP and uses the Phronesis
Index to shape rewards for safe navigation:

```python
# Run the experiment end-to-end
python code/safety_gym/train_safety_gym.py --grid_size 8 --num_seeds 10 --episodes 500
```

## Citation

If you use this code or the Phronesis Index in your research, please cite:

```bibtex
@article{bayat2026phronesis,
  title={Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems},
  author={Bayat, Sepehr},
  journal={Nature Machine Intelligence},
  year={2026},
  note={Submitted}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact:
- Sepehr Bayat (sepehrbayat@hooshex.com)
- Or open an issue on this repository

## Acknowledgments

This work was conducted at Hooshex AI Lab.
