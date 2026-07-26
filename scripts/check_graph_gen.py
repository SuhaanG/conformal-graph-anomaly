"""
check_graph_gen.py

Sanity checks for src/graph_gen.py. Run this after any change to the generator
before trusting downstream FDR results — this is cheap insurance against
silently redefining "contamination" mid-project.

Checks:
1. Homophily knob (p_aa) moves empirical anomaly homophily monotonically.
2. Propagation mechanism actually shifts features of normal nodes adjacent
   to anomalies more than non-adjacent normal nodes.
3. Anomaly rate matches the configured rate.
4. No isolated nodes at default settings (would break message passing).

Run: python scripts/check_graph_gen.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from graph_gen import GraphGenConfig, ContaminatedGraphGenerator


def check_homophily_knob():
    print("=== Check 1: homophily knob monotonicity ===")
    results = []
    for p_aa in [0.01, 0.05, 0.15, 0.3]:
        cfg = GraphGenConfig(p_aa=p_aa, random_state=1)
        gen = ContaminatedGraphGenerator(cfg)
        gen.generate()
        stats = gen.summary_stats()
        h = stats["empirical_anomaly_homophily"]
        results.append(h)
        print(f"  p_aa={p_aa:.2f} -> empirical_homophily={h:.3f}")
    monotonic = all(results[i] < results[i + 1] for i in range(len(results) - 1))
    print(f"  PASS: monotonic increase" if monotonic else "  FAIL: not monotonic")
    return monotonic


def check_propagation(p_an=0.005, n_nodes=5000):
    print("\n=== Check 2: propagation shifts adjacent nodes more ===")
    cfg = GraphGenConfig(
        n_nodes=n_nodes, p_aa=0.3, p_an=p_an, feature_shift=3.0, random_state=2
    )
    gen = ContaminatedGraphGenerator(cfg)
    gen.generate()
    raw = gen.features.copy()
    propagated = gen.propagate_contamination(hops=2, mix_weight=0.5)

    normal_idx = np.where(gen.labels == 0)[0]
    adj_to_anomaly = [
        i for i in normal_idx
        if any(gen.labels[n] == 1 for n in gen.graph.neighbors(i))
    ]
    non_adjacent = np.setdiff1d(normal_idx, adj_to_anomaly)

    frac_adjacent = len(adj_to_anomaly) / len(normal_idx)
    print(f"  fraction of normal nodes adjacent to an anomaly: {frac_adjacent:.3f}")

    if len(adj_to_anomaly) == 0 or len(non_adjacent) == 0:
        print("  WARNING: no contrast group available, adjust p_an / n_nodes")
        return False

    shift_adjacent = np.mean(np.abs(propagated[adj_to_anomaly] - raw[adj_to_anomaly]))
    shift_nonadjacent = np.mean(np.abs(propagated[non_adjacent] - raw[non_adjacent]))
    print(f"  avg abs shift (adjacent):     {shift_adjacent:.4f}")
    print(f"  avg abs shift (non-adjacent): {shift_nonadjacent:.4f}")

    passed = shift_adjacent > shift_nonadjacent
    print("  PASS" if passed else "  FAIL")
    return passed


def check_anomaly_rate():
    print("\n=== Check 3: anomaly rate matches config ===")
    cfg = GraphGenConfig(anomaly_rate=0.08, random_state=3)
    gen = ContaminatedGraphGenerator(cfg)
    gen.generate()
    stats = gen.summary_stats()
    actual = stats["anomaly_rate_actual"]
    print(f"  configured=0.08, actual={actual:.4f}")
    passed = abs(actual - 0.08) < 0.005
    print("  PASS" if passed else "  FAIL")
    return passed


def check_no_isolated_nodes():
    print("\n=== Check 4: no isolated nodes at default settings ===")
    cfg = GraphGenConfig(random_state=4)
    gen = ContaminatedGraphGenerator(cfg)
    gen.generate()
    stats = gen.summary_stats()
    n_isolated = stats["n_isolated_nodes"]
    print(f"  isolated nodes: {n_isolated}")
    passed = n_isolated == 0
    print("  PASS" if passed else "  FAIL")
    return passed


if __name__ == "__main__":
    results = {
        "homophily_knob": check_homophily_knob(),
        "propagation": check_propagation(),
        "anomaly_rate": check_anomaly_rate(),
        "no_isolated_nodes": check_no_isolated_nodes(),
    }
    print("\n=== Summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    if all(results.values()):
        print("\nAll checks passed. Generator is safe to use for Step 2.")
    else:
        print("\nSome checks failed. Fix before proceeding to Step 2.")