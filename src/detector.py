"""
detector.py

Step 3: baseline unsupervised GNN anomaly detector.

Uses a DOMINANT-style architecture: a shared GCN encoder feeds two decoders,
one reconstructing node features (attribute reconstruction) and one
reconstructing the adjacency matrix (structure reconstruction). Anomaly score
per node = weighted combination of attribute and structure reconstruction
error. High reconstruction error -> more anomalous.

Chosen over CoLA / GAD-NR for this stage because it is the simplest,
best-established unsupervised GAD baseline, trains fast enough to debug
locally (CPU, M2 MacBook Air), and its per-node reconstruction-error scores
plug directly into the conformal p-value machinery in the next step.

Runs on CPU by default. No CUDA assumptions — safe on the M2 MacBook Air.
When we move to multi-seed sweeps on Colab, the same code will pick up a GPU
automatically via the `device` argument.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


def normalize_adj(A: np.ndarray) -> torch.Tensor:
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2, standard GCN propagation matrix.
    DENSE version -- kept exactly as-is for backward compatibility with all
    already-validated results (synthetic experiments, Amazon real-data run).
    Do not modify; use normalize_adj_sparse for the scalable path instead."""
    A_hat = A + np.eye(A.shape[0])
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


def normalize_adj_fast(A: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """O(n^2) equivalent of normalize_adj, for large graphs.

    normalize_adj materializes D^-1/2 as a dense n x n diagonal matrix and does
    two dense n x n @ n x n matmuls, costing O(n^3). Since D is diagonal, the
    product D^-1/2 (A+I) D^-1/2 is exactly a row-then-column broadcast scale,
    which is O(n^2). At Yelp scale (n=45,954) that is ~3.9e14 flops vs ~4.2e9 --
    roughly 65 minutes per call vs seconds.

    Mathematically equivalent to normalize_adj; asserted directly against it in
    tests/test_normalize_equivalence.py. The remaining differences are
    cost-only: work happens in float32 (the dtype normalize_adj returns anyway)
    and self-loops are added in place instead of via np.eye, which avoids two
    extra n x n temporaries -- at Yelp scale that alone is ~34 GB of CPU RAM.

    normalize_adj is deliberately left untouched: every already-validated
    result in this project was produced through it.
    """
    A = np.array(A, dtype=np.float32, copy=True)
    np.fill_diagonal(A, A.diagonal() + 1.0)  # A_hat = A + I, without np.eye

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)

    A_t = torch.from_numpy(A).to(device)
    d_t = torch.from_numpy(deg_inv_sqrt).to(device)
    A_t.mul_(d_t.unsqueeze(1))  # row scale   -> D^-1/2 @ A_hat
    A_t.mul_(d_t.unsqueeze(0))  # column scale -> ... @ D^-1/2
    return A_t


def normalize_adj_sparse(graph: nx.Graph, device: str) -> torch.Tensor:
    """SCALABLE version of normalize_adj: builds the same D^-1/2 (A+I) D^-1/2
    propagation matrix but as a torch sparse tensor, avoiding the O(n^2)
    dense memory that made Yelp (45,954 nodes, ~2.1B dense entries)
    impractically slow. Mathematically identical normalization to
    normalize_adj -- only the representation differs."""
    n = graph.number_of_nodes()
    edges = list(graph.edges())
    src = [u for u, v in edges] + [v for u, v in edges] + list(range(n))  # symmetric + self-loops
    dst = [v for u, v in edges] + [u for u, v in edges] + list(range(n))
    src, dst = np.array(src), np.array(dst)

    deg = np.zeros(n)
    np.add.at(deg, src, 1.0)
    deg_inv_sqrt = np.zeros(n)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)

    vals = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    A_norm_sparse = torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce().to(device)
    return A_norm_sparse


def sample_structure_pairs(graph: nx.Graph, seed: int, neg_ratio: int = 1):
    """Samples positive (real edges) and negative (random non-edges) node
    pairs for edge-sampled structure reconstruction loss, replacing the
    dense Z @ Z.T computation. neg_ratio negatives are sampled per positive."""
    rng = np.random.default_rng(seed)
    n = graph.number_of_nodes()
    edges = np.array(list(graph.edges()))
    n_pos = len(edges)

    pos_src, pos_dst = edges[:, 0], edges[:, 1]

    neg_src = rng.integers(0, n, size=n_pos * neg_ratio)
    neg_dst = rng.integers(0, n, size=n_pos * neg_ratio)

    return (
        torch.tensor(pos_src, dtype=torch.long),
        torch.tensor(pos_dst, dtype=torch.long),
        torch.tensor(neg_src, dtype=torch.long),
        torch.tensor(neg_dst, dtype=torch.long),
    )


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A_norm, X):
        # A_norm @ ... works for both dense and sparse tensors in PyTorch
        # (sparse uses torch.sparse.mm under the hood via the @ operator).
        return A_norm @ self.linear(X)


class DOMINANT(nn.Module):
    """Shared GCN encoder + attribute decoder + structure decoder."""

    def __init__(self, in_dim, hidden_dim=64, encoder_layers=2):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * encoder_layers
        for i in range(encoder_layers):
            self.encoder_layers.append(GCNLayer(dims[i], dims[i + 1]))

        # attribute decoder: single GCN layer back to input dimension
        self.attr_decoder = GCNLayer(hidden_dim, in_dim)
        # structure decoder is just inner-product of embeddings (no params)

    def encode(self, A_norm, X):
        H = X
        for layer in self.encoder_layers:
            H = F.relu(layer(A_norm, H))
        return H

    def forward(self, A_norm, X):
        Z = self.encode(A_norm, X)
        X_hat = self.attr_decoder(A_norm, Z)
        A_hat = torch.sigmoid(Z @ Z.T)
        return X_hat, A_hat, Z


def train_dominant(
    graph: nx.Graph,
    features: np.ndarray,
    n_epochs: int = 100,
    hidden_dim: int = 64,
    lr: float = 0.01,
    alpha: float = 0.5,  # weight on attribute loss vs. structure loss
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
    use_sparse_prop: bool = False,
):
    """Trains DOMINANT on a single graph and returns per-node anomaly scores.

    alpha: weight given to attribute reconstruction error in the final score
    and loss (1 - alpha goes to structure reconstruction). DOMINANT's default
    favors attribute error since it's typically more informative; we keep it
    tunable since this parameter itself is a source of training randomness
    worth stress-testing later (ties back to the seed-instability motivation).

    use_sparse_prop: opt-in large-graph path, needed for Yelp (n=45,954) and
    off by default so every existing validated result reproduces byte for byte.
    It changes only HOW the propagation matrix is built, never the model, the
    loss, or the scoring:
      - the propagation matrix becomes a torch sparse tensor via
        normalize_adj_sparse (same D^-1/2 (A+I) D^-1/2 normalization, verified
        equivalent in tests/test_normalize_equivalence.py), which at Yelp scale
        is ~46 MB instead of 8.45 GB and skips normalize_adj's two dense
        n x n @ n x n matmuls (~65 min/call at that size);
      - the dense adjacency is materialized in float32 rather than float64,
        halving peak CPU RAM (16.9 GB -> 8.45 GB at Yelp scale).
    The structure DECODER stays dense either way -- that is what keeps the
    anomaly scores valid, and is precisely what train_dominant_scalable below
    got wrong.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = graph.number_of_nodes()
    if use_sparse_prop:
        A = nx.to_numpy_array(graph, nodelist=range(n), dtype=np.float32)
        A_norm = normalize_adj_sparse(graph, device)
        A_target = torch.from_numpy(A).to(device)
    else:
        A = nx.to_numpy_array(graph, nodelist=range(n))
        A_norm = normalize_adj(A).to(device)
        A_target = torch.tensor(A, dtype=torch.float32).to(device)
    X = torch.tensor(features, dtype=torch.float32).to(device)

    model = DOMINANT(in_dim=features.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        X_hat, A_hat, Z = model(A_norm, X)

        attr_loss = torch.mean(torch.sum((X - X_hat) ** 2, dim=1))
        struct_loss = torch.mean(torch.sum((A_target - A_hat) ** 2, dim=1))
        loss = alpha * attr_loss + (1 - alpha) * struct_loss

        loss.backward()
        optimizer.step()

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d} | loss={loss.item():.4f} "
                  f"(attr={attr_loss.item():.4f}, struct={struct_loss.item():.4f})")

    # final per-node anomaly scores: per-node reconstruction error, same weighting as loss
    model.eval()
    with torch.no_grad():
        X_hat, A_hat, Z = model(A_norm, X)
        attr_err = torch.sum((X - X_hat) ** 2, dim=1)
        struct_err = torch.sum((A_target - A_hat) ** 2, dim=1)
        scores = alpha * attr_err + (1 - alpha) * struct_err

    return scores.cpu().numpy(), model


def train_dominant_scalable(
    graph: nx.Graph,
    features: np.ndarray,
    n_epochs: int = 100,
    hidden_dim: int = 64,
    lr: float = 0.01,
    alpha: float = 0.5,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
    neg_ratio: int = 1,
):
    """
    *** NOT VALIDATED -- DO NOT USE FOR REAL EXPERIMENTS YET ***

    Attempted scalable variant of train_dominant for large graphs (e.g.
    Yelp, 45,954 nodes) where the dense structure decoder is impractically
    slow. Diagnostic testing on synthetic data found this scoring approach
    INVERTS the anomaly signal (structure-only AUROC = 0.0000, a perfect
    inversion, not noise) -- likely because in homophilous-anomaly setups
    (p_aa > p_nn), anomaly-anomaly edges are actually EASIER to reconstruct
    (tight, predictable clusters), giving anomalies LOWER reconstruction
    error, the opposite of the intended signal. Attribute-only path also
    underperforms the dense version (0.556 vs ~0.89 AUROC at a comparable
    config), suggesting the sparse propagation path itself needs further
    debugging too, not just the structure-loss scoring.

    Left in the codebase intentionally (not deleted) so the sparse-matrix
    infrastructure (normalize_adj_sparse, sample_structure_pairs) can be
    reused once the scoring design is fixed -- but the scoring logic below
    needs a redesign (e.g. comparing a node's edge-reconstruction quality
    against a null/random-pair baseline rather than raw NLL) before this
    is trustworthy. DO NOT wire this into real_data_experiment.py for Yelp
    until re-validated with a synthetic AUROC sanity check first.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = graph.number_of_nodes()
    A_norm = normalize_adj_sparse(graph, device)
    X = torch.tensor(features, dtype=torch.float32).to(device)

    model = DOMINANT(in_dim=features.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_src, pos_dst, neg_src, neg_dst = sample_structure_pairs(graph, seed, neg_ratio)
    pos_src, pos_dst = pos_src.to(device), pos_dst.to(device)
    neg_src, neg_dst = neg_src.to(device), neg_dst.to(device)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        Z = model.encode(A_norm, X)
        X_hat = model.attr_decoder(A_norm, Z)
        attr_loss = torch.mean(torch.sum((X - X_hat) ** 2, dim=1))

        pos_logits = torch.sum(Z[pos_src] * Z[pos_dst], dim=1)
        neg_logits = torch.sum(Z[neg_src] * Z[neg_dst], dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        struct_loss = pos_loss + neg_loss

        loss = alpha * attr_loss + (1 - alpha) * struct_loss
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d} | loss={loss.item():.4f} "
                  f"(attr={attr_loss.item():.4f}, struct={struct_loss.item():.4f})")

    model.eval()
    with torch.no_grad():
        Z = model.encode(A_norm, X)
        X_hat = model.attr_decoder(A_norm, Z)
        attr_err = torch.sum((X - X_hat) ** 2, dim=1)

        # per-node structural error: avg negative log-likelihood of the
        # node's own true edges (both directions, since graph is undirected).
        # VECTORIZED via scatter_add -- a Python loop over edges would be
        # catastrophically slow at Yelp's scale (~3.8M edges).
        edges = np.array(list(graph.edges()))
        u_idx = torch.tensor(edges[:, 0], dtype=torch.long, device=device)
        v_idx = torch.tensor(edges[:, 1], dtype=torch.long, device=device)

        logits_uv = torch.sum(Z[u_idx] * Z[v_idx], dim=1)
        nll_uv = F.binary_cross_entropy_with_logits(
            logits_uv, torch.ones_like(logits_uv), reduction="none"
        )

        struct_err = torch.zeros(n, device=device)
        counts = torch.zeros(n, device=device)
        struct_err.scatter_add_(0, u_idx, nll_uv)
        struct_err.scatter_add_(0, v_idx, nll_uv)
        counts.scatter_add_(0, u_idx, torch.ones_like(nll_uv))
        counts.scatter_add_(0, v_idx, torch.ones_like(nll_uv))
        counts[counts == 0] = 1.0
        struct_err = struct_err / counts

        scores = alpha * attr_err + (1 - alpha) * struct_err

    return scores.cpu().numpy(), model


if __name__ == "__main__":
    # quick smoke test using the locked-in Step 2 config
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
    from sklearn.metrics import roc_auc_score

    cfg = GraphGenConfig(
        n_nodes=3000, p_aa=0.3, p_an=0.01, p_nn=0.005,
        feature_shift=1.0, n_anomaly_clusters=3, random_state=0,
    )
    gen = ContaminatedGraphGenerator(cfg)
    graph, features, labels = gen.generate()

    print("Training DOMINANT on synthetic contaminated graph...")
    scores, model = train_dominant(graph, features, n_epochs=100, verbose=True)

    auroc = roc_auc_score(labels, scores)
    print(f"\nAUROC (sanity check — detector should separate anomalies well above chance): {auroc:.4f}")