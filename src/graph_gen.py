"""
graph_gen.py

Synthetic contaminated attributed graph generator for testing whether
finite-sample FDR control survives clustered, propagating calibration
contamination on graphs (H1/H4 in the research plan).

Design rationale:
- Two-block SBM (normal / anomalous) with independently tunable:
    * within-anomaly homophily (p_aa): how tightly anomalies cluster together
    * anomaly-normal edge rate (p_an): how much anomalies "touch" normal nodes
    * within-normal edge rate (p_nn): background graph density
  This lets us dial "clustering" and "propagation exposure" independently,
  which is exactly the mechanism H1 claims matters.
- Node features are drawn from two Gaussians (normal vs. anomalous) with a
  configurable mean-shift and shared covariance, so "detector task difficulty"
  is a separate knob from "graph structural contamination."
- A `contaminate_features_via_propagation` step lets us later simulate message
  passing pushing anomalous signal into neighboring normal nodes' *observed*
  features — this is the literal mechanism H1 needs to test, so it belongs in
  the generator, not bolted on downstream.

Everything here is intentionally simple and inspectable (no deep learning yet).
Sanity-check script lives in scripts/check_graph_gen.py.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field


@dataclass
class GraphGenConfig:
    n_nodes: int = 3000
    anomaly_rate: float = 0.05          # fraction of nodes that are anomalous
    p_nn: float = 0.01                  # normal-normal edge probability
    p_an: float = 0.01                  # anomaly-normal edge probability
    p_aa: float = 0.05                  # anomaly-anomaly edge probability (homophily knob)
    feature_dim: int = 16
    feature_shift: float = 1.5          # mean separation between normal/anomalous features
    feature_scale: float = 1.0          # shared std dev
    n_anomaly_clusters: int = 1         # 1 = one tight cluster; >1 = scattered clusters
    random_state: int | None = 0

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)


class ContaminatedGraphGenerator:
    """Generates an attributed graph with ground-truth anomaly labels and
    controllable structural + feature contamination, for stress-testing
    conformal FDR control under clustered, propagating contamination.
    """

    def __init__(self, config: GraphGenConfig):
        self.cfg = config
        self._rng = config.rng()
        self.graph: nx.Graph | None = None
        self.labels: np.ndarray | None = None   # 1 = anomalous, 0 = normal
        self.features: np.ndarray | None = None
        self.propagated_features: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Structure generation
    # ------------------------------------------------------------------
    def _assign_labels(self) -> np.ndarray:
        n = self.cfg.n_nodes
        n_anom = int(round(n * self.cfg.anomaly_rate))
        labels = np.zeros(n, dtype=int)

        if self.cfg.n_anomaly_clusters <= 1:
            anom_idx = self._rng.choice(n, size=n_anom, replace=False)
        else:
            # split anomalies into k roughly-equal scattered clusters
            k = self.cfg.n_anomaly_clusters
            per_cluster = n_anom // k
            chosen = set()
            for _ in range(k):
                remaining = np.setdiff1d(np.arange(n), list(chosen))
                block = self._rng.choice(remaining, size=per_cluster, replace=False)
                chosen.update(block.tolist())
            anom_idx = np.array(sorted(chosen))

        labels[anom_idx] = 1
        return labels

    def _build_graph(self, labels: np.ndarray) -> nx.Graph:
        n = self.cfg.n_nodes
        G = nx.Graph()
        G.add_nodes_from(range(n))

        normal_idx = np.where(labels == 0)[0]
        anom_idx = np.where(labels == 1)[0]

        # Normal-normal edges (Erdos-Renyi style within block)
        self._add_block_edges(G, normal_idx, normal_idx, self.cfg.p_nn, symmetric=True)
        # Anomaly-anomaly edges (homophily knob)
        self._add_block_edges(G, anom_idx, anom_idx, self.cfg.p_aa, symmetric=True)
        # Anomaly-normal cross edges (propagation exposure knob)
        self._add_block_edges(G, anom_idx, normal_idx, self.cfg.p_an, symmetric=False)

        return G

    def _add_block_edges(self, G, idx_a, idx_b, p, symmetric):
        # Vectorized-ish random edge sampling to avoid O(n^2) python loops at scale.
        if len(idx_a) == 0 or len(idx_b) == 0 or p <= 0:
            return
        if symmetric and np.array_equal(idx_a, idx_b):
            n = len(idx_a)
            # sample from upper triangle only
            n_possible = n * (n - 1) // 2
            n_edges = self._rng.binomial(n_possible, p)
            if n_edges == 0:
                return
            # sample without replacement from triangle indices (approx for large n)
            i_idx, j_idx = np.triu_indices(n, k=1)
            chosen = self._rng.choice(len(i_idx), size=min(n_edges, len(i_idx)), replace=False)
            for c in chosen:
                G.add_edge(idx_a[i_idx[c]], idx_a[j_idx[c]])
        else:
            # bipartite-style sampling between two disjoint sets
            n_possible = len(idx_a) * len(idx_b)
            n_edges = self._rng.binomial(n_possible, p)
            if n_edges == 0:
                return
            a_choices = self._rng.integers(0, len(idx_a), size=n_edges)
            b_choices = self._rng.integers(0, len(idx_b), size=n_edges)
            for a, b in zip(a_choices, b_choices):
                G.add_edge(idx_a[a], idx_b[b])

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------
    def _generate_features(self, labels: np.ndarray) -> np.ndarray:
        n, d = self.cfg.n_nodes, self.cfg.feature_dim
        base = self._rng.normal(loc=0.0, scale=self.cfg.feature_scale, size=(n, d))
        shift = np.zeros((n, d))
        shift[labels == 1] = self.cfg.feature_shift
        return base + shift

    def propagate_contamination(self, hops: int = 1, mix_weight: float = 0.3) -> np.ndarray:
        """Simulate message-passing contamination: each node's *observed* feature
        becomes a mixture of its own feature and its neighbors' features, averaged
        over `hops` steps. This is the literal mechanism H1 predicts corrupts
        normal nodes' calibration scores when anomalies cluster.

        mix_weight: how much weight is given to the neighbor average vs. self
        (0 = no propagation / raw features, 1 = fully replaced by neighbor average).
        """
        if self.graph is None or self.features is None:
            raise RuntimeError("Call generate() before propagate_contamination().")

        A = nx.to_numpy_array(self.graph, nodelist=range(self.cfg.n_nodes))
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0  # avoid div-by-zero for isolated nodes
        P = A / deg  # row-normalized adjacency (mean neighbor aggregator)

        feats = self.features.copy()
        for _ in range(hops):
            neighbor_avg = P @ feats
            feats = (1 - mix_weight) * feats + mix_weight * neighbor_avg

        self.propagated_features = feats
        return feats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
        labels = self._assign_labels()
        graph = self._build_graph(labels)
        features = self._generate_features(labels)

        self.labels = labels
        self.graph = graph
        self.features = features
        return graph, features, labels

    def summary_stats(self) -> dict:
        """Basic diagnostics to sanity-check that the knobs behave as intended
        before trusting any downstream FDR results."""
        if self.graph is None or self.labels is None:
            raise RuntimeError("Call generate() first.")

        G, labels = self.graph, self.labels
        anom_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]

        degrees = dict(G.degree())
        avg_deg_anom = np.mean([degrees[i] for i in anom_idx]) if len(anom_idx) else 0.0
        avg_deg_normal = np.mean([degrees[i] for i in normal_idx]) if len(normal_idx) else 0.0

        # empirical homophily: fraction of anomaly-incident edges that are anomaly-anomaly
        aa_edges = 0
        an_edges = 0
        for u, v in G.edges():
            if labels[u] == 1 and labels[v] == 1:
                aa_edges += 1
            elif labels[u] == 1 or labels[v] == 1:
                an_edges += 1
        total_anom_incident = aa_edges + an_edges
        empirical_homophily = aa_edges / total_anom_incident if total_anom_incident > 0 else float("nan")

        return {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_anomalies": int(labels.sum()),
            "anomaly_rate_actual": float(labels.mean()),
            "avg_degree_anomalous": avg_deg_anom,
            "avg_degree_normal": avg_deg_normal,
            "empirical_anomaly_homophily": empirical_homophily,
            "n_isolated_nodes": sum(1 for d in degrees.values() if d == 0),
        }


if __name__ == "__main__":
    cfg = GraphGenConfig()
    gen = ContaminatedGraphGenerator(cfg)
    gen.generate()
    stats = gen.summary_stats()
    for k, v in stats.items():
        print(f"{k}: {v}")