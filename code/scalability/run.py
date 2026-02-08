#!/usr/bin/env python3
"""
Scalability Test for Phronesis Index (Section 5.4).

Measures wall-clock time of Φ computation as a function of the
number of agents N, demonstrating O(N log N) complexity.

Usage:
    python -m code.scalability.run --config code/scalability/config.yaml
    python -m code.scalability.run --max_agents 5000 --num_trials 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phronesis import PhronesisIndex


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def run_experiment(
    sizes: list[int] | None = None,
    avg_degree: int = 10,
    num_trials: int = 5,
    epsilon: float = 0.01,
    k: int = 20,
    seed_base: int = 42,
    output_dir: str = "results/scalability",
):
    if sizes is None:
        sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

    print(f"Scalability test: sizes={sizes}, trials={num_trials}")
    results = []

    for N in sizes:
        print(f"  N={N} ...", end=" ", flush=True)
        times = []
        for trial in range(num_trials):
            rng = np.random.RandomState(seed_base + trial)
            p = min(avg_degree / N, 1.0)
            G = nx.erdos_renyi_graph(N, p, seed=rng.randint(0, 2**31))

            # Ensure connectivity
            if not nx.is_connected(G):
                comps = list(nx.connected_components(G))
                for i in range(len(comps) - 1):
                    u = next(iter(comps[i]))
                    v = next(iter(comps[i + 1]))
                    G.add_edge(u, v)

            stalks = {v: 2 for v in G.nodes()}
            maps = {(u, v): np.eye(2) for u, v in G.edges()}

            t0 = time.perf_counter()
            PhronesisIndex(G, stalks, maps).compute(epsilon=epsilon, k=k)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        mean_t = np.mean(times)
        std_t = np.std(times)
        results.append((N, mean_t, std_t))
        print(f"{mean_t:.4f}s ± {std_t:.4f}s")

    arr = np.array(results)
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(
        os.path.join(output_dir, "scalability_data.csv"),
        arr, delimiter=",", header="num_agents,mean_time_s,std_time_s", comments="",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(arr[:, 0], arr[:, 1], "bo-", lw=2, ms=6)
    ax1.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.3)
    ax1.set(xlabel="N", ylabel="Time (s)", title="Scalability (linear)")
    ax1.grid(True, alpha=0.3)

    ax2.loglog(arr[:, 0], arr[:, 1], "ro-", lw=2, ms=6, label="Measured")
    c = np.median(arr[:, 1] / (arr[:, 0] * np.log(arr[:, 0])))
    fitted = c * arr[:, 0] * np.log(arr[:, 0])
    ax2.loglog(arr[:, 0], fitted, "g--", lw=2, label="O(N log N) fit")
    ax2.set(xlabel="N", ylabel="Time (s)", title="Scalability (log-log)")
    ax2.legend(); ax2.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure4_scalability.png"), dpi=300)
    plt.close()

    print(f"Results → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Scalability test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max_agents", type=int, default=None)
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--seed_base", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    if args.max_agents:
        sizes = np.unique(np.logspace(2, np.log10(args.max_agents), 15, dtype=int)).tolist()
    else:
        sizes = cfg.get("scalability", {}).get("sizes", None)

    kw = dict(
        sizes=sizes,
        avg_degree=cfg.get("scalability", {}).get("avg_degree", 10),
        num_trials=args.num_trials or cfg.get("scalability", {}).get("num_trials", 5),
        epsilon=cfg.get("phronesis", {}).get("epsilon", 0.01),
        k=cfg.get("phronesis", {}).get("k", 20),
        seed_base=args.seed_base or cfg.get("scalability", {}).get("seed_base", 42),
        output_dir=args.output_dir or cfg.get("output", {}).get("dir", "results/scalability"),
    )
    run_experiment(**kw)


if __name__ == "__main__":
    main()
