"""
test_calibration_strategy_integration.py

End-to-end integration test for calibration_strategy_comparison.py's
run_seed(): a real (synthetic) graph, real detector training, all
strategies including "weighted". Catches integration bugs the isolated
weighted_conformal.py unit tests can't (mismatched array lengths between
eligible/degrees/unexposed inside run_seed itself).

This is an INTEGRATION test, not a test of the scientific finding at
scale -- the synthetic graph here is small (800 nodes, 15 epochs) purely
to run fast. gamma values from this run should not be interpreted as
evidence about real datasets.

Usage:
  python3 scripts/test_calibration_strategy_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import networkx as nx
from types import SimpleNamespace

from calibration_strategy_comparison import run_seed


def build_smoke_graph(seed=0, n=800, n_anomalies=40):
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, 3, seed=seed)
    labels = np.zeros(n, dtype=int)
    anomaly_idx = rng.choice(n, size=n_anomalies, replace=False)
    labels[anomaly_idx] = 1

    # connect anomalies preferentially to high-degree hubs, so exposure
    # correlates with degree -- mirrors the real amazon/tolokers pattern,
    # otherwise "clean" wouldn't produce any real degree shift to test
    degrees = dict(G.degree())
    hub_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
    for a in anomaly_idx:
        for h in rng.choice(hub_nodes, size=3, replace=False):
            G.add_edge(int(a), int(h))

    features = rng.standard_normal((n, 16)).astype(np.float32)
    return G, features, labels


def main():
    G, features, labels = build_smoke_graph()

    args = SimpleNamespace(
        detector="gae", n_epochs=15, device="cpu", use_sparse_prop=False,
        trim_pct=0.01, n_test_normal=150, dataset="synthetic_smoke",
        alpha=0.10, n_sim=50, n_degree_bins=8, n_calib_full=300,
    )

    print("Running run_seed() end-to-end (real graph, real detector training)...")
    out, err = run_seed(G, features, labels, seed=0, args=args)

    assert err is None, f"run_seed failed: {err}"

    strategies_present = [r["strategy"] for r in out]
    print(f"Strategies returned: {strategies_present}")
    assert "weighted" in strategies_present, "weighted strategy missing from output"

    w_row = [r for r in out if r["strategy"] == "weighted"][0]
    c_row = [r for r in out if r["strategy"] == "clean"][0]
    print(f"clean:    n_calib={c_row['n_calib']} gamma_hat={c_row['gamma_hat']:.3f} "
          f"fdr={c_row['realized_fdr']:.3f}")
    print(f"weighted: n_calib={w_row['n_calib']} gamma_hat={w_row['gamma_hat']:.3f} "
          f"fdr={w_row['realized_fdr']:.3f}")

    assert w_row["n_calib"] == c_row["n_calib"], (
        "weighted must reuse clean's exact n_calib (matched-frame requirement)"
    )
    assert w_row["m_test"] == c_row["m_test"], "weighted must share clean's test set"

    print("\nPASSED: end-to-end run_seed() with 'weighted' strategy works, "
          "matched frame confirmed")


if __name__ == "__main__":
    main()