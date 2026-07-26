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
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2, standard GCN propagation matrix."""
    A_hat = A + np.eye(A.shape[0])
    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, A_norm, X):
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
):
    """Trains DOMINANT on a single graph and returns per-node anomaly scores.

    alpha: weight given to attribute reconstruction error in the final score
    and loss (1 - alpha goes to structure reconstruction). DOMINANT's default
    favors attribute error since it's typically more informative; we keep it
    tunable since this parameter itself is a source of training randomness
    worth stress-testing later (ties back to the seed-instability motivation).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = graph.number_of_nodes()
    A = nx.to_numpy_array(graph, nodelist=range(n))
    A_norm = normalize_adj(A).to(device)
    X = torch.tensor(features, dtype=torch.float32).to(device)
    A_target = torch.tensor(A, dtype=torch.float32).to(device)

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