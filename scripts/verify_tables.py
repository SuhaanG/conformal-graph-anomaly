"""Recompute every per-cell mean the TMLR draft reports from the strategy CSVs (stdlib only)."""
import csv, glob, os, statistics as st
from collections import defaultdict

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
files = sorted(glob.glob(os.path.join(root, "**", "calibration_strategy_*.csv"), recursive=True))

def fl(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

rows = []
for f in files:
    with open(f, newline="") as fh:
        rs = list(csv.DictReader(fh))
    for r in rs:
        r["src"] = os.path.relpath(f, root)
        rows.append(r)
    seeds = sorted({int(r["seed"]) for r in rs})
    strats = sorted({r["strategy"] for r in rs})
    print(f"FILE {os.path.relpath(f, root)} rows={len(rs)} seeds={seeds} strategies={strats}")

groups = defaultdict(list)
for r in rows:
    groups[(r["src"], r["dataset"], r["detector"], r["strategy"])].append(r)

cols = ["n_calib", "m_test", "n_null", "calib_mean_degree", "test_mean_degree",
        "spearman_score_degree", "score_gap_cohens_d", "gamma_t_lo", "t_lo",
        "n_discoveries", "realized_fdr", "power"]
hdr = f"{'dataset':9}{'detector':15}{'strategy':16}{'n':>2} " + " ".join(f"{c[:12]:>12}" for c in cols) + "   deg_ratio  null_p(list)  ndisc(list)  fdr(list)"
print("\nPER-CELL MEANS:")
print(hdr)
summary = {}
for key in sorted(groups):
    g = groups[key]
    means = {c: st.mean(fl(r[c]) for r in g) for c in cols}
    summary[key] = means
    ratio = means["test_mean_degree"] / means["calib_mean_degree"] if means["calib_mean_degree"] else float("nan")
    nullp = ",".join(f"{fl(r['mean_p_null_p']):.3g}" for r in g)
    nd = ",".join(str(int(fl(r["n_discoveries"]))) for r in g)
    fd = ",".join(f"{fl(r['realized_fdr']):.3f}" for r in g)
    src, ds, det, strat = key
    print(f"{ds:9}{det:15}{strat:16}{len(g):>2} " + " ".join(f"{means[c]:>12.4f}" for c in cols) + f"   {ratio:8.2f}  [{nullp}]  [{nd}]  [{fd}]")

print("\nDELTA GAMMA (weighted - clean):")
seen = defaultdict(dict)
for (src, ds, det, strat), m in summary.items():
    seen[(src, ds, det)][strat] = m
for (src, ds, det), s in sorted(seen.items()):
    if "clean" in s and "weighted" in s:
        c, w = s["clean"], s["weighted"]
        print(f"  {src} {ds}/{det}: gamma clean={c['gamma_t_lo']:.2f} weighted={w['gamma_t_lo']:.2f} delta={w['gamma_t_lo']-c['gamma_t_lo']:+.2f} | "
              f"FDR {c['realized_fdr']:.3f}->{w['realized_fdr']:.3f} | power {c['power']:.3f}->{w['power']:.3f}")

print("\nCLEAN CELLS (score gap, gamma, rho, degree ratio):")
for (src, ds, det, strat), m in sorted(summary.items()):
    if strat == "clean":
        print(f"  {src:55} {ds:9}{det:15} d={m['score_gap_cohens_d']:+.3f} gamma={m['gamma_t_lo']:.2f} rho={m['spearman_score_degree']:+.3f} "
              f"calib_deg={m['calib_mean_degree']:.1f} test_deg={m['test_mean_degree']:.1f} ratio={m['test_mean_degree']/m['calib_mean_degree']:.1f}")

# Kendall tau from the selection-bias matrix
print("\nKENDALL TAU (q_kendall_tau, eligibility vs degree) from selection_bias_matrix_v2.csv:")
mf = os.path.join(root, "published", "selection_bias_matrix_v2.csv")
kg = defaultdict(list)
with open(mf, newline="") as fh:
    for r in csv.DictReader(fh):
        kg[(r["dataset"], r["detector"])].append(r)
for (ds, det), g in sorted(kg.items()):
    taus = [fl(r["q_kendall_tau"]) for r in g]
    ps = [fl(r["q_kendall_p"]) for r in g]
    ncal = [fl(r["n_calib"]) for r in g]
    print(f"  {ds:9}{det:15} n={len(g)} tau mean={st.mean(taus):+.3f} min={min(taus):+.3f} max={max(taus):+.3f} "
          f"p max={max(ps):.3g} n_calib={st.mean(ncal):.0f} monotone={[r['q_is_monotone'] for r in g]}")
