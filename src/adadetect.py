"""
adadetect.py

AdaDetect (Marandon, Lei, Mary & Roquain, 2024, Annals of Statistics) as a
baseline for conformal graph anomaly detection.

MECHANISM. Split the known-null reference sample into D_train and D_cal. Fit a
probabilistic classifier separating D_train (label 0) from the ENTIRE test
sample (label 1). Its predicted class-1 probability becomes the nonconformity
score. Conformal p-values against D_cal, then BH -- identical to our procedure.
The point is that the score ADAPTS to the observed test mixture instead of
being fixed a priori, which is exactly the axis our reconstruction-error
detector does not vary along.

WHY THIS FILE EXISTS SEPARATELY. The comparison is only meaningful if the
statistical procedure is held fixed and the score is the sole free variable --
the principle scripts/baseline_comparison.py already commits to in its
docstring. Enforcing that requires care in three places, each of which has a
documented failure precedent in this project:

1. RESOLUTION FLOOR. conformal_p_values returns (count_ge+1)/(n_calib+1), so
   p_min = 1/(n_calib+1) and BH can only reject at sorted rank i when
   p_min <= alpha*i/m, i.e. bh_min_rank = ceil(m/(alpha*(n_calib+1))). A naive
   AdaDetect that splits the reference pool halves n_calib and roughly doubles
   that floor, so it would lose on power ARITHMETICALLY -- reproducing the
   Method B/C tautology this repo already had to retract once. Instead,
   build_matched_frame carves D_train out of normals the frozen protocol
   ALREADY DISCARDS (the max_normal_test=5000 cap throws away ~2,011 eligible
   normals per Amazon trial), so n_calib, m, and bh_min_rank are identical
   across arms -- and on the real-data path calib_idx/test_idx come out
   byte-identical to the frozen validated run.

2. TIES. conformal_p_values counts with >= and does NOT randomize (its
   docstring claims a randomized tie-break term; the body has none). Tree
   ensembles emit heavily discretized predict_proba, so blocks of test points
   share an exact p-value and BH accepts or rejects whole blocks. Handled by
   jitter_for_ties, applied UPSTREAM of the frozen p-value function.

3. LEAKAGE. The classifier's labels are partition membership, never anomaly
   labels. Separately, AdaDetect's validity needs the learned score to be
   independent of D_cal, so D_cal must not touch fitting in any way -- not via
   a validation split, not via early stopping, not via a fitted scaler. That is
   enforced structurally by adadetect_scores' signature.

src/conformal_fdr.py and src/detector.py are NOT modified by any of this; this
module imports and calls them.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
import torch
import torch.nn.functional as F

from conformal_fdr import conformal_p_values, benjamini_hochberg
from detector import normalize_adj, normalize_adj_sparse

VARIANTS = ("score1d", "embed", "feat", "embed_score")
CLASSIFIERS = ("logreg", "gbm", "rf")


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def build_matched_frame(
    graph,
    labels,
    scores,
    condition,
    seed,
    *,
    frame="synthetic",
    calib_frac=0.9,
    n_calib_cap=4000,
    max_normal_test=5000,
    trim_pct=0.01,
    n_train_target=2000,
    min_train=300,
    train_frac=0.25,
    min_test_normal=1000,
):
    """Builds ONE calibration/test/train partition shared by every arm.

    Steps 1-4 are a verbatim transcription of the corresponding frozen builder
    (_build_calib_test_split for frame="synthetic",_prepare_calib_test for
    frame="trimmed_real"), using the same RNG object in the same call order.
    Copied rather than refactored on purpose: the goal is bit-exact agreement
    with already-published numbers, and a "cleaner" rewrite is exactly how that
    silently breaks.

    Step 5 is new and APPENDED LAST, drawing D_train from a separate RNG stream
    so it cannot perturb the frozen draw sequence.

    CRITICAL: the frame is built from the DOMINANT `scores` only, and is frozen
    before AdaDetect's own score exists. Defining the trim or the partition
    using AdaDetect's learned score would be circular (its score depends on the
    test set) AND would reintroduce the asymmetric-trimming bug class that
    manufactured a spurious FDR of 0.51 here once already.

    Returns a dict, or None if the frame is infeasible (caller should skip and
    log `skip_reason`).
    """
    if condition not in ("clean", "contaminated", "adversarial"):
        raise ValueError(
            f"unknown condition {condition!r}; expected clean/contaminated/adversarial. "
            "(Note: the two frozen builders silently treat 'clean' as adversarial "
            "via a bare else-branch; this one refuses rather than guessing.)"
        )
    if frame not in ("synthetic", "trimmed_real"):
        raise ValueError(f"unknown frame {frame!r}")

    labels = np.asarray(labels)
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    exposure = compute_exposure(graph, normal_idx, labels)

    rng = np.random.default_rng(seed)

    # ---- steps 1-4: frozen logic, verbatim --------------------------------
    if frame == "synthetic":
        clean_pool = normal_idx[exposure == 0]
        if len(clean_pool) < 20:
            return None
        n_calib = int(round(calib_frac * len(clean_pool)))
        pool = normal_idx

        if condition == "contaminated":
            calib_idx = rng.choice(normal_idx, size=n_calib, replace=False)
        elif condition == "adversarial":
            order = np.argsort(-exposure)
            calib_idx = normal_idx[order][:n_calib]
        else:  # clean
            calib_idx = rng.choice(clean_pool, size=min(n_calib, len(clean_pool)),
                                   replace=False)

        remaining_normal = np.setdiff1d(pool, calib_idx)
        test_normal = remaining_normal  # frozen synthetic path applies no cap
    else:  # trimmed_real
        normal_scores_all = scores[normal_idx]
        score_cutoff = np.percentile(normal_scores_all, 100 * (1 - trim_pct))
        eligible_normal_idx = normal_idx[normal_scores_all <= score_cutoff]
        eligible_mask = np.isin(normal_idx, eligible_normal_idx)

        clean_pool = normal_idx[(exposure == 0) & eligible_mask]
        if len(clean_pool) < 20:
            return None
        pool = eligible_normal_idx

        if condition == "contaminated":
            n_calib = min(n_calib_cap, len(eligible_normal_idx))
            calib_idx = rng.choice(eligible_normal_idx, size=n_calib, replace=False)
        elif condition == "adversarial":
            n_calib = min(n_calib_cap, len(eligible_normal_idx))
            eligible_exposure = exposure[eligible_mask]
            order = np.argsort(-eligible_exposure)
            calib_idx = eligible_normal_idx[order][:n_calib]
        else:  # clean
            n_calib = len(clean_pool)
            calib_idx = clean_pool

        remaining_normal = np.setdiff1d(eligible_normal_idx, calib_idx)
        test_normal = remaining_normal
        if len(test_normal) > max_normal_test:
            test_normal = rng.choice(test_normal, size=max_normal_test, replace=False)

    # ---- step 5: NEW, appended, separate RNG stream ------------------------
    train_pool = np.setdiff1d(remaining_normal, test_normal)
    frame_exact_match_frozen = len(train_pool) > 0

    if not frame_exact_match_frozen:
        # No discarded slack (synthetic, or a real graph whose eligible pool is
        # small). Open the release valve: take D_train from the test normals.
        # n_calib NEVER moves -- shrinking it would raise the floor for both
        # arms and break comparability with every frozen number and every other
        # dataset. m is the valve, and it shrinks identically for both arms.
        budget = max(min_train, int(train_frac * len(test_normal)))
        n_train = min(n_train_target, budget)
        if len(test_normal) - n_train < min_test_normal or n_train < min_train:
            return None
        rng_train = np.random.default_rng((seed, 7))
        train_idx = rng_train.choice(test_normal, size=n_train, replace=False)
        test_normal = np.setdiff1d(test_normal, train_idx)
    else:
        n_train = min(n_train_target, len(train_pool))
        if n_train < min_train:
            return None
        rng_train = np.random.default_rng((seed, 7))
        train_idx = rng_train.choice(train_pool, size=n_train, replace=False)

    test_idx = np.concatenate([test_normal, anomaly_idx])
    test_labels = np.concatenate([
        np.zeros(len(test_normal), dtype=int),
        np.ones(len(anomaly_idx), dtype=int),
    ])

    return {
        "calib_idx": calib_idx,
        "test_idx": test_idx,
        "train_idx": train_idx,
        "test_labels": test_labels,
        "n_calib": int(len(calib_idx)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_null": int((test_labels == 0).sum()),
        "n_test_alt": int((test_labels == 1).sum()),
        "n_anomaly_total": int(len(anomaly_idx)),
        "frame_exact_match_frozen": bool(frame_exact_match_frozen),
    }


def compute_exposure(graph, normal_idx, labels):
    """Fraction of each normal node's neighbors that are anomalous. Identical
    to _compute_exposure in scripts/baseline_comparison.py; duplicated here so
    src/ has no dependency on scripts/."""
    exposure = np.zeros(len(normal_idx))
    for j, i in enumerate(normal_idx):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            continue
        exposure[j] = sum(1 for n in neighbors if labels[n] == 1) / len(neighbors)
    return exposure


def frame_floor_stats(n_calib, n_test, alpha):
    """The two numbers a reviewer needs to verify the arms really are matched."""
    p_floor = 1.0 / (n_calib + 1)
    bh_min_rank = math.ceil(n_test / (alpha * (n_calib + 1)))
    return {"p_floor": p_floor, "bh_min_rank": int(bh_min_rank)}


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------

def encoder_embedding(graph, features, model, device="cpu", use_sparse_prop=False):
    """The trained detector's node representation Z (hidden_dim per node).

    This is the PRIMARY covariate: it carries exactly the information our own
    score is computed from -- same trained model, same graph, same features --
    so the only thing that differs between arms is the readout (a learned
    classifier vs a fixed reconstruction error). That is what makes it the
    information-matched comparison.

    Branches on model type, since the frozen dominant_ours model and PyGOD
    models expose the embedding differently:

    - dominant_ours model: has .encode(A_norm, X). Rebuilds the normalized
      adjacency and calls encode() explicitly. `use_sparse_prop` MUST match
      whatever was passed to train_dominant -- the two propagation matrices
      are numerically equivalent (asserted in tests/test_normalize_equivalence.py)
      but reconstructing through a different path than training used is the
      kind of quiet inconsistency that is impossible to debug later, and the
      dense path costs O(n^3) (~65 min at Yelp scale) for no benefit.
    - PyGOD model: has .emb, populated by the final forward pass detectors.py's
      _train_pygod already ran during training/eval. No second forward pass
      needed -- read it directly.
    """
    if hasattr(model, "emb") and not hasattr(model, "encode"):
        # PyGOD model: .emb was already set by the last forward pass in
        # detectors.py's _train_pygod (model.eval() + one more forward call).
        if model.emb is None:
            raise RuntimeError(
                "PyGOD model.emb is None -- was score_nodes(..., return_model=True) "
                "actually called with a PyGOD detector, and did training complete? "
                "encoder_embedding cannot re-run a forward pass for a PyGOD model "
                "without the original edge_index, which is not passed to this function."
            )
        return model.emb.detach().cpu().numpy()

    import networkx as nx

    n = graph.number_of_nodes()
    if use_sparse_prop:
        A_norm = normalize_adj_sparse(graph, device)
    else:
        A = nx.to_numpy_array(graph, nodelist=range(n))
        A_norm = normalize_adj(A).to(device)
        del A
    X = torch.tensor(features, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        Z = model.encode(A_norm, X)
    return Z.cpu().numpy()


def build_covariates(variant, *, features, scores, embedding=None, degrees=None):
    """Assembles the classifier's design matrix for one variant.

    score1d      -- 1-d DOMINANT score. EXACTLY our method's information. A
                    monotone classifier on it is a strictly increasing
                    reparametrization of our score, and conformal p-values are
                    rank statistics, so this MUST reproduce our arm up to ties.
                    Its role is a correctness control, not a headline result.
    embed        -- encoder output Z. Information-matched; the pre-registered
                    primary.
    feat         -- raw standardized node features. STRICTLY LESS information
                    (never sees the graph). Diagnostic only; reporting it as
                    "AdaDetect loses" would be dishonest.
    embed_score  -- Z + score + log1p(degree). STRICTLY MORE information than
                    our method. The steelman upper bound.
    """
    if variant == "score1d":
        return scores.reshape(-1, 1)
    if variant == "feat":
        return np.asarray(features, dtype=np.float64)
    if variant == "embed":
        if embedding is None:
            raise ValueError("variant 'embed' requires embedding")
        return np.asarray(embedding, dtype=np.float64)
    if variant == "embed_score":
        if embedding is None or degrees is None:
            raise ValueError("variant 'embed_score' requires embedding and degrees")
        return np.column_stack([
            np.asarray(embedding, dtype=np.float64),
            scores.reshape(-1, 1),
            np.log1p(degrees).reshape(-1, 1),
        ])
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

def _fit_logreg_torch(X, y, *, seed, device, l2, max_iter):
    """L2 logistic regression in torch.

    Chosen over sklearn deliberately. sklearn is not in requirements.txt and is
    not importable on every dev box here, and this project's workflow is
    debug-locally-then-run-on-Colab -- an sklearn primary would be
    undebuggable locally. It also emits continuous float outputs, so exact ties
    are essentially impossible except for duplicate covariate rows.

    AdaDetect's FDR guarantee holds for ANY classifier, so this is a
    power/implementation choice, not a weakened baseline.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    d = Xt.shape[1]
    w = torch.zeros(d, 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)

    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = (Xt @ w + b).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + (1.0 / (2.0 * l2 * len(yt))) * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def _predict_logreg_torch(w, b, X, device):
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.sigmoid((Xt @ w + b).squeeze(-1)).cpu().numpy().astype(np.float64)


def adadetect_scores(
    X_train,
    X_test,
    X_calib,
    *,
    classifier="logreg",
    seed=0,
    device="cpu",
    l2=1.0,
    max_iter=500,
):
    """Fits P(class=1 | x) on X_train (y=0) vs X_test (y=1), then scores.

    X_calib is used ONLY for prediction and NEVER during fitting -- not for a
    validation split, not for early stopping, not for the feature scaler.
    AdaDetect's validity requires the learned score to be independent of the
    calibration set, so this is a correctness requirement, not hygiene. It is
    enforced by the signature: X_calib arrives as its own argument and is
    touched exactly once, at the end.

    Note the asymmetry worth stating in the paper: the DOMINANT score is
    transductive (it sees every node, including calibration), which is fine
    because a GNN is permutation-equivariant over nodes and a symmetric
    function of all nodes preserves exchangeability. AdaDetect's classifier is
    NOT symmetric in calib-vs-test, so it must never see calibration data.

    Returns (calib_scores, test_scores), higher = more anomalous.
    """
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    X_fit = np.vstack([X_train, X_test])

    # Scaler fit on D_train U test ONLY -- never on X_calib. See above.
    mu = X_fit.mean(axis=0, keepdims=True)
    sd = X_fit.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    X_fit_s = (X_fit - mu) / sd
    X_calib_s = (X_calib - mu) / sd
    X_test_s = (X_test - mu) / sd

    if classifier == "logreg":
        w, b = _fit_logreg_torch(X_fit_s, y, seed=seed, device=device,
                                 l2=l2, max_iter=max_iter)
        calib_scores = _predict_logreg_torch(w, b, X_calib_s, device)
        test_scores = _predict_logreg_torch(w, b, X_test_s, device)
        fit_scores = _predict_logreg_torch(w, b, X_fit_s, device)
    elif classifier in ("gbm", "rf"):
        # Lazily imported so the primary path never requires sklearn. Used only
        # for the tie ablation, which is Colab-only.
        if classifier == "gbm":
            from sklearn.ensemble import GradientBoostingClassifier as Clf
            clf = Clf(random_state=seed)
        else:
            from sklearn.ensemble import RandomForestClassifier as Clf
            clf = Clf(random_state=seed, n_jobs=-1)
        clf.fit(X_fit_s, y)
        calib_scores = clf.predict_proba(X_calib_s)[:, 1].astype(np.float64)
        test_scores = clf.predict_proba(X_test_s)[:, 1].astype(np.float64)
        fit_scores = clf.predict_proba(X_fit_s)[:, 1].astype(np.float64)
    else:
        raise ValueError(f"unknown classifier {classifier!r}; expected one of {CLASSIFIERS}")

    # Fit-quality diagnostic on the PARTITION label (train-vs-test), which uses
    # no anomaly labels. Reported, never used for selection.
    clf_partition_auc = score_auroc(fit_scores, y.astype(int))
    return calib_scores, test_scores, clf_partition_auc


# ---------------------------------------------------------------------------
# Ties, diagnostics, evaluation
# ---------------------------------------------------------------------------

def jitter_for_ties(concatenated_values, rng, scale=0.0):
    """Randomized tie-breaking, applied UPSTREAM of conformal_p_values.

    src/conformal_fdr.py is imported by seven-plus scripts that all produced
    committed results; adding randomization inside it would silently change
    every one of them. So the perturbation goes into the score instead.

    THE SIGNATURE IS THE SAFETY MECHANISM. It takes exactly ONE array -- the
    concatenated calibration+test values -- and draws ONE i.i.d. perturbation
    over it with ONE global scale. Drawing separately per set, or scaling per
    set, would make the perturbation depend on set membership: that is
    precisely the structural error behind the asymmetric trimming that
    manufactured a spurious FDR of 0.51 in this project. i.i.d. noise added to
    an exchangeable sequence preserves exchangeability exactly, so validity is
    untouched.

    Default scale=0.0 is a no-op, which also preserves the bit-exact agreement
    between our arm here and the frozen published runs.
    """
    values = np.asarray(concatenated_values, dtype=np.float64)
    if scale == 0.0:
        return values
    spread = np.ptp(values)
    if spread == 0.0:
        spread = 1.0
    return values + scale * spread * rng.random(len(values))


def tie_diagnostics(test_scores):
    """Turns "ties might be hurting the tree-ensemble arm" into a measurement."""
    _, counts = np.unique(test_scores, return_counts=True)
    return {
        "frac_unique_test_scores": float(len(counts) / len(test_scores)),
        "max_tied_block": int(counts.max()),
    }


def score_auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney U) in numpy, keeping the primary path
    sklearn-free. Verified against scipy.stats.mannwhitneyu including the
    heavy-tie, all-tied, and perfect-separation edge cases.

    DIAGNOSTIC ONLY -- never used to select a variant, classifier, or setting.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(counts))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / counts)[inv]
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate_arm(
    calib_scores,
    test_scores,
    test_labels,
    alpha,
    *,
    rng_jitter=None,
    jitter_scale=0.0,
    n_anomaly_total=None,
):
    """The ONLY place conformal_p_values and benjamini_hochberg are called.

    Every arm -- ours and every AdaDetect variant -- routes through this
    function, so procedure identity is guaranteed structurally rather than
    merely asserted in a docstring. There is no second code path to drift.
    """
    calib_scores = np.asarray(calib_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)

    if jitter_scale != 0.0:
        if rng_jitter is None:
            raise ValueError("jitter_scale != 0 requires rng_jitter")
        n_cal = len(calib_scores)
        combined = jitter_for_ties(
            np.concatenate([calib_scores, test_scores]), rng_jitter, jitter_scale)
        calib_scores, test_scores = combined[:n_cal], combined[n_cal:]

    p_values = conformal_p_values(calib_scores, test_scores)
    discoveries = benjamini_hochberg(p_values, alpha)

    n_discoveries = int(discoveries.sum())
    realized_fdr = (float(np.sum(discoveries & (test_labels == 0))) / n_discoveries
                    if n_discoveries > 0 else 0.0)
    denom = n_anomaly_total if n_anomaly_total else int((test_labels == 1).sum())
    power = (float(np.sum(discoveries & (test_labels == 1))) / denom
             if denom > 0 else 0.0)

    out = {
        "n_discoveries": n_discoveries,
        "realized_fdr": realized_fdr,
        "power": power,
        "score_auroc": score_auroc(test_scores, test_labels),
    }
    out.update(tie_diagnostics(test_scores))
    out["_p_values"] = p_values
    out["_discoveries"] = discoveries
    return out