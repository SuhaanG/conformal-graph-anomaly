"""
test_normalize_equivalence.py

Guards the large-graph performance path added for Yelp (n=45,954).

normalize_adj (src/detector.py) is frozen -- every already-validated result in
this project was produced through it, and it carries an explicit "do not
modify" comment. Two faster paths exist alongside it:

  normalize_adj_fast    -- O(n^2) broadcast instead of O(n^3) dense matmuls
  normalize_adj_sparse  -- torch sparse propagation matrix

Both claim to compute the same D^-1/2 (A+I) D^-1/2. This file asserts that
claim directly, because the whole Yelp result rests on it: if either path
differs numerically from the frozen one, every downstream FDR number computed
through it is suspect.

Run either way:
    python3 -m pytest tests/test_normalize_equivalence.py -v
    python3 tests/test_normalize_equivalence.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import networkx as nx
import torch

from detector import normalize_adj, normalize_adj_fast, normalize_adj_sparse


# float32 accumulation over a few hundred nodes; 1e-5 is comfortably tight
# enough to catch a real formulation error (a wrong axis, a missing self-loop,
# a transposed scale) while tolerating ordinary rounding.
ATOL = 1e-5


def _graphs():
    """Cases chosen for the structural edge conditions that actually differ
    between the dense and sparse constructions."""
    cases = {}

    g = nx.erdos_renyi_graph(500, 0.02, seed=0)
    g.add_nodes_from(range(500))
    cases["erdos_renyi_500"] = g

    # Isolated nodes: degree 0 before self-loops. The dense path relies on
    # A + I lifting them to degree 1; the sparse path relies on its explicit
    # range(n) self-loop term. Easy place for the two to disagree.
    g = nx.erdos_renyi_graph(200, 0.01, seed=1)
    g.add_nodes_from(range(300))  # nodes 200-299 isolated
    cases["with_isolated_nodes"] = g

    # Hub: one node adjacent to everything. Stresses the degree normalization
    # asymmetry between row and column scaling.
    g = nx.Graph()
    g.add_nodes_from(range(300))
    g.add_edges_from((0, i) for i in range(1, 300))
    g.add_edges_from(nx.erdos_renyi_graph(300, 0.01, seed=2).edges())
    cases["hub_dominated"] = g

    # Fully disconnected: every node isolated. Degenerate but legal.
    g = nx.Graph()
    g.add_nodes_from(range(100))
    cases["no_edges"] = g

    return cases


def _dense(graph):
    n = graph.number_of_nodes()
    return nx.to_numpy_array(graph, nodelist=range(n))


def test_fast_matches_frozen():
    """normalize_adj_fast must reproduce normalize_adj exactly enough that
    swapping it changes no result."""
    for name, graph in _graphs().items():
        A = _dense(graph)
        reference = normalize_adj(A).cpu().numpy()
        fast = normalize_adj_fast(A, device="cpu").cpu().numpy()
        assert np.allclose(reference, fast, atol=ATOL), (
            f"[{name}] normalize_adj_fast diverges from normalize_adj: "
            f"max abs diff = {np.abs(reference - fast).max():.3e}"
        )


def test_fast_does_not_mutate_input():
    """normalize_adj_fast adds self-loops in place on its own copy. If that
    copy is ever elided, it would silently corrupt the caller's adjacency --
    which train_dominant reuses as the structure-decoder target."""
    graph = _graphs()["erdos_renyi_500"]
    A = _dense(graph)
    before = A.copy()
    normalize_adj_fast(A, device="cpu")
    assert np.array_equal(A, before), "normalize_adj_fast mutated its input"


def test_sparse_matches_frozen():
    """normalize_adj_sparse is pre-existing but was only ever consumed by
    train_dominant_scalable, which is documented as broken. It is load-bearing
    for the Yelp run, so verify it against the frozen path directly."""
    for name, graph in _graphs().items():
        A = _dense(graph)
        reference = normalize_adj(A).cpu().numpy()
        sparse = normalize_adj_sparse(graph, "cpu").to_dense().cpu().numpy()
        assert np.allclose(reference, sparse, atol=ATOL), (
            f"[{name}] normalize_adj_sparse diverges from normalize_adj: "
            f"max abs diff = {np.abs(reference - sparse).max():.3e}"
        )


def test_row_sums_are_symmetric_normalized():
    """Sanity check on the property itself, independent of the reference
    implementation: D^-1/2 (A+I) D^-1/2 must be symmetric for an undirected
    graph. Catches a transposed scale that happened to match a transposed
    reference."""
    for name, graph in _graphs().items():
        A = _dense(graph)
        fast = normalize_adj_fast(A, device="cpu").cpu().numpy()
        assert np.allclose(fast, fast.T, atol=ATOL), (
            f"[{name}] normalized adjacency is not symmetric"
        )


if __name__ == "__main__":
    tests = [
        test_fast_matches_frozen,
        test_fast_does_not_mutate_input,
        test_sparse_matches_frozen,
        test_row_sums_are_symmetric_normalized,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
