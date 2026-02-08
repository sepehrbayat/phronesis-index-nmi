#!/usr/bin/env python3
"""
Logic Maze Anomaly Detection Experiment (Section 5.1).

Demonstrates how the Phronesis Index detects anomalies in a
multi-agent navigation grid.  An anomaly is injected at a
specified timestep by perturbing restriction maps on a fraction
of edges, and the time-series of Φ is recorded.

Usage:
    python -m code.logic_maze.run --config code/logic_maze/config.yaml
    python -m code.logic_maze.run --grid_size 3 --num_runs 2 --num_timesteps 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phronesis import PhronesisIndex


def load_config(path: str) -> dict:
    """Load YAML config, falling back to defaults if file missing."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def create_grid_graph(grid_size: int) -> nx.Graph:
    G = nx.grid_2d_graph(grid_size, grid_size)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)


def create_sheaf_structure(G: nx.Graph):
    stalks = {v: 2 for v in G.nodes()}
    restriction_maps = {(u, v): np.eye(2) for u, v in G.edges()}
    return stalks, restriction_maps


def inject_anomaly(restriction_maps, anomaly_edges, strength=0.5):
    perturbed = restriction_maps.copy()
    for edge in anomaly_edges:
        if edge in perturbed:
            perturbed[edge] = perturbed[edge] + np.random.randn(2, 2) * strength
    return perturbed


def run_experiment(
    grid_size: int = 5,
    anomaly_time: int = 50,
    anomaly_duration: int = 50,
    anomaly_fraction: float = 0.10,
    anomaly_strength: float = 0.5,
    num_runs: int = 10,
    num_timesteps: int = 150,
    epsilon: float = 0.01,
    k: int = 20,
    seed_base: int = 42,
    output_dir: str = "results/logic_maze",
):
    print(f"Logic Maze experiment: {grid_size}×{grid_size} grid, "
          f"{num_runs} runs, {num_timesteps} timesteps")

    all_phi = []
    for run in range(num_runs):
        np.random.seed(seed_base + run)
        G = create_grid_graph(grid_size)
        stalks, maps = create_sheaf_structure(G)
        edges = list(G.edges())
        n_anomaly = max(1, int(len(edges) * anomaly_fraction))

        phi_ts = []
        for t in range(num_timesteps):
            if anomaly_time <= t < anomaly_time + anomaly_duration:
                ae = [edges[i] for i in np.random.choice(len(edges), n_anomaly, replace=False)]
                cur_maps = inject_anomaly(maps, ae, anomaly_strength)
            else:
                cur_maps = maps
            phi = PhronesisIndex(G, stalks, cur_maps).compute(epsilon=epsilon, k=k)
            phi_ts.append(phi.phi)
        all_phi.append(phi_ts)
        print(f"  run {run+1}/{num_runs} done")

    mean_phi = np.mean(all_phi, axis=0)
    std_phi = np.std(all_phi, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(
        os.path.join(output_dir, "phi_timeseries.csv"),
        np.column_stack([np.arange(num_timesteps), mean_phi, std_phi]),
        delimiter=",", header="timestep,mean_phi,std_phi", comments="",
    )

    plt.figure(figsize=(10, 6))
    plt.plot(mean_phi, "b-", lw=2, label="Phronesis Index")
    plt.fill_between(range(num_timesteps), mean_phi - std_phi, mean_phi + std_phi, alpha=0.3)
    plt.axvline(x=anomaly_time, color="r", ls="--", label="Anomaly injected")
    plt.axvline(x=anomaly_time + anomaly_duration, color="g", ls="--", label="Anomaly removed")
    plt.xlabel("Timestep"); plt.ylabel("Phronesis Index (Φ)")
    plt.title("Logic Maze: Anomaly Detection via Phronesis Index")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure1_timeseries.png"), dpi=300)
    plt.close()

    print(f"Results → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Logic Maze experiment")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--seed_base", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    # Merge: CLI overrides > YAML > defaults
    kw = dict(
        grid_size=args.grid_size or cfg.get("grid", {}).get("size", 5),
        anomaly_time=cfg.get("anomaly", {}).get("injection_time", 50),
        anomaly_duration=cfg.get("anomaly", {}).get("duration", 50),
        anomaly_fraction=cfg.get("anomaly", {}).get("fraction", 0.10),
        anomaly_strength=cfg.get("anomaly", {}).get("strength", 0.5),
        num_runs=args.num_runs or cfg.get("simulation", {}).get("num_runs", 10),
        num_timesteps=args.num_timesteps or cfg.get("simulation", {}).get("num_timesteps", 150),
        epsilon=cfg.get("phronesis", {}).get("epsilon", 0.01),
        k=cfg.get("phronesis", {}).get("k", 20),
        seed_base=args.seed_base or cfg.get("simulation", {}).get("seed_base", 42),
        output_dir=args.output_dir or cfg.get("output", {}).get("dir", "results/logic_maze"),
    )
    run_experiment(**kw)


if __name__ == "__main__":
    main()
