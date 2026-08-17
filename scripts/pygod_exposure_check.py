"""
pygod_exposure_check.py

THE FORK IN THE ROAD for this paper's framing.

Established so far:
  - Our DOMINANT encoder's final layer is dead (ReLU on the last layer forces
    Z >= 0; the structure loss then drives Z to exactly 0). Confirmed 6/6 across
    sizes and seeds. See DETECTOR_DIAGNOSTIC.md.
  - With that dead encoder, a normal node's exposure to anomalous neighbors has
    ~zero correlation with its score (|r| < 0.016 on synthetic; ~0 on all three
    real datasets once the degree confound is controlled). So the paper's
    contamination mechanism never reached the scores.
  - Removing our final ReLU revives the encoder AND restores the exposure
    correlation to +0.346 -- the mechanism is real when the encoder works. But
    it also inverts detection on our synthetic generator (AUROC 0.91 -> 0.12),
    because dense anomaly clusters (p_aa=0.3) are EASIER to reconstruct.
  - PyGOD's DOMINANT does NOT have the bug: its PyG backbone gates the
    activation behind `if i < self.num_layers - 1`, leaving the final embedding
    linear. Measured: frac_zero=0.0000, std=0.0518, min=-0.219 (negatives are
    impossible after a ReLU).

THE QUESTION THIS ANSWERS. PyGOD's encoder works. Does the contamination
mechanism operate under it? Specifically, on the same synthetic graphs:
  (a) does PyGOD's DOMINANT actually detect the anomalies (AUROC well above
      chance -- if it inverts like our no-ReLU variant did, the generator is
      the problem, not the detector), and
  (b) does a normal node's exposure correlate with its score?

WHY IT DECIDES THE PAPER.
  - AUROC good AND exposure correlates -> the original contamination question
    is testable after all. We asked the right question with a broken
    instrument. Swapping in a correct detector and re-running may recover the
    original framing.
  - AUROC inverted, or exposure does not correlate -> the reframe stands: the
    paper becomes the discovery-threshold result, and contamination goes.

Sweeps p_an so the contamination severity axis is visible, and reports our own
detector on the identical graphs as a side-by-side control.

Full batch, no NeighborLoader, so it needs no pyg-lib / torch-sparse:
  ~/envs/dgl311/bin/python scripts/pygod_exposure_check.py --n_nodes 3000 --epochs 100
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import torch
from scipy import stats

from graph_gen import GraphGenConfig, ContaminatedGraphGenerator
from detector import train_dominant


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def compute_exposure(graph, normal_idx, labels):
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        nb = list(graph.neighbors(i))
        if nb:
            exposure[j] = sum(1 for n in nb if labels[n] == 1) / len(nb)
    return exposure


def to_pyg(graph, features, labels):
    from torch_geometric.data import Data
    edges = np.array(list(graph.edges()), dtype=np.int64)
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    return Data(x=torch.tensor(features, dtype=torch.float32),
                edge_index=torch.tensor(both.T, dtype=torch.long),
                y=torch.tensor(labels, dtype=torch.long))


def pygod_scores(graph, features, labels, epochs, seed, hid_dim=64):
    """Trains pygod.nn.DOMINANTBase full-batch and returns per-node anomaly
    scores using PyGOD's own loss as the score, which is what its detector
    reports (double_recon_loss returns a per-node value)."""
    from pygod.nn import DOMINANTBase
    torch.manual_seed(seed)
    data = to_pyg(graph, features, labels)
    DOMINANTBase.process_graph(data)

    model = DOMINANTBase(in_dim=features.shape[1], hid_dim=hid_dim, num_layers=4,
                         dropout=0.0, act=torch.nn.functional.relu)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        x_, s_ = model(data.x, data.edge_index)
        loss = model.loss_func(data.x, x_, data.s, s_).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x_, s_ = model(data.x, data.edge_index)
        per_node = model.loss_func(data.x, x_, data.s, s_)
        Z = model.shared_encoder(data.x, data.edge_index)
    return per_node.cpu().numpy(), Z.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--p_an", type=float, nargs="+", default=[0.002, 0.01, 0.05])
    args = parser.parse_args()

    print(f"{'detector':10} {'p_an':>6} {'AUROC':>8} {'Z std':>10} {'mean exp':>9} "
          f"{'exposure r':>11} {'p-value':>10}")
    rows = []
    for p_an in args.p_an:
        acc = {"pygod": [], "ours": []}
        for seed in range(args.seeds):
            cfg = GraphGenConfig(n_nodes=args.n_nodes, p_aa=0.3, p_an=p_an, p_nn=0.005,
                                 feature_shift=1.0, n_anomaly_clusters=3, random_state=seed)
            graph, features, labels = ContaminatedGraphGenerator(cfg).generate()
            normal_idx = np.where(labels == 0)[0]
            exposure = compute_exposure(graph, normal_idx, labels)

            ps, pZ = pygod_scores(graph, features, labels, args.epochs, seed)
            r, p = stats.pearsonr(exposure, ps[normal_idx])
            acc["pygod"].append((auroc(ps, labels), pZ.std(), exposure.mean(), r, p))

            os_, model = train_dominant(graph, features, n_epochs=args.epochs, seed=seed,
                                        verbose=False, use_sparse_prop=True)
            r2, p2 = stats.pearsonr(exposure, os_[normal_idx])
            acc["ours"].append((auroc(os_, labels), 0.0, exposure.mean(), r2, p2))

        for name in ("pygod", "ours"):
            a = np.array(acc[name])
            print(f"{name:10} {p_an:6.3f} {a[:,0].mean():8.4f} {a[:,1].mean():10.6f} "
                  f"{a[:,2].mean():9.4f} {a[:,3].mean():+11.4f} {a[:,4].mean():10.4f}")
            rows.append((name, p_an, a[:, 0].mean(), a[:, 3].mean(), a[:, 4].mean()))
        print()

    pg = [r for r in rows if r[0] == "pygod"]
    best_auroc = max(r[2] for r in pg)
    max_abs_r = max(abs(r[3]) for r in pg)
    any_sig = any(r[4] < 0.05 and abs(r[3]) > 0.05 for r in pg)

    print(f"{'=' * 78}\nVERDICT\n{'=' * 78}")
    print(f"  PyGOD best AUROC across severities : {best_auroc:.4f}")
    print(f"  PyGOD largest |exposure r|         : {max_abs_r:.4f}")
    if best_auroc < 0.6:
        print("""
  PyGOD DOES NOT DETECT on this generator either. That points at the SYNTHETIC
  GENERATOR, not the detector: with p_aa=0.3 the anomaly clusters are dense and
  therefore easy to reconstruct, so structure-based scoring is inverted by
  construction. The contamination framing cannot be rescued by swapping
  detectors -- the generator would have to change. Proceed with the reframe.""")
    elif any_sig:
        print("""
  PyGOD DETECTS *AND* EXPOSURE CORRELATES. The contamination mechanism is real
  under a correct detector -- we asked the right question with a broken
  instrument. The original framing may be recoverable by swapping in PyGOD's
  DOMINANT and re-running. That is a larger job than the reframe (every number
  changes) but it restores the paper's original contribution. Decide
  deliberately, and note the real-data results would need re-running too.""")
    else:
        print("""
  PyGOD detects, but exposure still does not correlate with score. The
  mechanism does not operate even with a working encoder, so the reframe
  stands: the paper becomes the discovery-threshold result and contamination
  goes.""")


if __name__ == "__main__":
    main()
