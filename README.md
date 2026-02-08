# Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems

[![Tests](https://github.com/sepehrbayat/phronesis-index-nmi/actions/workflows/ci.yml/badge.svg)](https://github.com/sepehrbayat/phronesis-index-nmi/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Paper:** *Spectral Sheaf Heuristics for Consistency Detection in Multi-Agent Systems*
> **Author:** Sepehr Bayat · Hooshex AI Lab
> **Submitted to:** Nature Machine Intelligence

---

## TL;DR

The **Phronesis Index** (Φ) is a computationally efficient spectral heuristic that quantifies the global consistency of a multi-agent system by approximating topological obstructions (cohomology) in cellular sheaves. It runs in **O(N log N)** time and achieves **23% safety improvement** (p < 0.01) when used as an RL reward signal.

---

## Repository Structure

```
phronesis-index-nmi/
├── phronesis/              # Core Python library (pip-installable)
│   ├── core.py             #   PhronesisIndex class
│   ├── laplacian.py        #   Connection Laplacian construction
│   └── epsilon.py          #   ε selection procedures
├── code/                   # Experiment implementations
│   ├── logic_maze/         #   Sec 5.1 — anomaly detection
│   ├── safety_gym/         #   Sec 5.2 — safe RL (STPGC)
│   ├── multi_robot/        #   Sec 5.3 — multi-robot coordination
│   └── scalability/        #   Sec 5.4 — scalability benchmarks
├── paper/                  # LaTeX manuscript (paper-as-code)
│   ├── main_manuscript.tex #   Single entrypoint
│   ├── references.bib
│   ├── figures/            #   All 14 figures
│   ├── supplementary/      #   Supplementary Information
│   └── Makefile            #   make paper / make si / make clean
├── scripts/                # Reproduction scripts
│   ├── reproduce_all.sh    #   End-to-end (full or --smoke)
│   ├── reproduce_logic_maze.sh
│   ├── reproduce_safety_gym.sh
│   ├── reproduce_multi_robot.sh
│   └── reproduce_scalability.sh
├── env/                    # Environment specifications
│   ├── requirements.txt    #   Pinned pip dependencies
│   └── environment.yml     #   Pinned conda environment
├── tests/                  # Unit tests
├── results/                # Generated outputs (gitignored)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── REPRODUCIBILITY.md
├── CITATION.cff
├── LICENSE                 # MIT
└── CHANGELOG.md
```

---

## Quick Start (5-minute smoke test)

```bash
# 1. Clone
git clone https://github.com/sepehrbayat/phronesis-index-nmi.git
cd phronesis-index-nmi

# 2. Create environment & install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[experiments,dev]"

# 3. Run unit tests
pytest -v

# 4. Run smoke tests (all 4 scenarios, ~2 min)
bash scripts/reproduce_all.sh --smoke

# 5. Check results
ls results/
```

### Docker alternative

```bash
docker build -t phronesis .
docker run --rm -v $(pwd)/results:/app/results phronesis bash scripts/reproduce_all.sh --smoke
```

---

## Full Reproduction

Full reproduction replicates all experiments from the paper with the exact parameters and seed lists.

```bash
bash scripts/reproduce_all.sh
```

| Scenario | Command | Expected time | Output directory |
|----------|---------|---------------|------------------|
| Logic Maze | `bash scripts/reproduce_logic_maze.sh` | ~5 min | `results/logic_maze/` |
| Safety Gym | `bash scripts/reproduce_safety_gym.sh` | ~5 sec (simulated) | `results/safety_gym/` |
| Multi-Robot | `bash scripts/reproduce_multi_robot.sh` | ~10 min | `results/multi_robot/` |
| Scalability | `bash scripts/reproduce_scalability.sh` | ~30 min | `results/scalability/` |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact commands, parameters, seed lists, and expected outputs.

---

## Building the Paper

```bash
cd paper
make paper    # → paper/main_manuscript.pdf
make si       # → paper/supplementary/supplementary_information.pdf
make clean    # remove build artefacts
```

Requires: `pdflatex`, `bibtex` (any standard TeX Live / MiKTeX distribution).

---

## Core Library Usage

```python
import networkx as nx
import numpy as np
from phronesis import PhronesisIndex

# Build a simple 3-agent triangle
G = nx.cycle_graph(3)
stalks = {0: 2, 1: 2, 2: 2}
maps = {(0, 1): np.eye(2), (1, 2): np.eye(2)}

idx = PhronesisIndex(G, stalks, maps)
result = idx.compute(epsilon=0.01)

print(f"Φ = {result.phi:.4f}")
print(f"λ₁⁺ = {result.lambda_1_plus:.4f}")
print(f"h¹_ε = {result.h1_epsilon}")
```

---

## Citation

```bibtex
@article{bayat2026phronesis,
  title   = {Spectral Sheaf Heuristics for Consistency Detection
             in Multi-Agent Systems},
  author  = {Bayat, Sepehr},
  journal = {Nature Machine Intelligence},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

[MIT](LICENSE)

## Contact

Sepehr Bayat — [sepehrbayat@hooshex.com](mailto:sepehrbayat@hooshex.com) — Hooshex AI Lab
