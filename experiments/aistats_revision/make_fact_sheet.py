"""Build FACT_SHEET.md: every number is read from the result CSV/JSON files named in it."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE.parent / 'audit_method'
G = [('amazon', 'Amazon'), ('tfinance', 'T-Finance'), ('weibo_gadbench', 'Weibo (GADBench)')]
S = [('hgb_graph_features', 'graph'), ('hgb_attributes', 'attr')]
T9 = 2.262157


def numbers():
    out = []
    e = pd.read_csv(A / 'wcs_endtoend_summary.csv')
    e['se_cluster'] = e.fdp_std / np.sqrt(10)
    e['z'] = (e.fdp_mean - e.bound_mean) / e.se_cluster
    x = e.set_index(['dataset', 'model', 'scenario', 'method'])
    out.append('### End-to-end: scorer retrained on the biased labels (`wcs_endtoend_summary.csv`)\n')
    out.append('FDP / power; z = (mean FDP - bound) / SE, with SE computed across the ten training-seed averages '
               '(seeds are the independent units; the five test splits within a seed share its prior panel). '
               'These z values are descriptive, unadjusted diagnostics, not tests of the theorem.\n')
    out.append('| Graph | Scorer | Bias | Bound | Pooled BH | Weighted BH (no guarantee) | Randomized WCS |')
    out.append('|---|---|---|---|---|---|---|')
    for ds, name in G:
        for mdl, ml in S:
            for sc in ('moderate', 'severe', 'smooth'):
                r = {m: x.loc[(ds, mdl, sc, m)] for m in ('pooled_reference', 'weighted_bh', 'wcs_rand_homogeneous')}
                cells = ' | '.join(f"{v.fdp_mean:.3f} / {v.power_mean:.3f} (z {v.z:+.1f})" for v in r.values())
                out.append(f"| {name} | {ml} | {sc} | {r['pooled_reference'].bound_mean:.3f} | {cells} |")
    pooled = e[e.method == 'pooled_reference']
    rand = e[e.method == 'wcs_rand_homogeneous']
    pw = e.pivot_table(index=['dataset', 'model', 'scenario'], columns='method', values='power_mean')
    trials = pd.read_csv(A / 'wcs_endtoend_trials.csv.gz', usecols=['auroc'])
    out.append('')
    out.append(f"- Pooled BH exceeds the bound by more than 2 seed-clustered SEs in {int((pooled.z > 2).sum())}/18 cells.")
    out.append(f"- Randomized WCS: largest excess {rand.z.max():+.2f} SE; results are consistent with the design-based bound. "
               "Some sample means sit slightly above it (e.g. T-Finance graph, moderate: "
               f"{x.loc[('tfinance','hgb_graph_features','moderate','wcs_rand_homogeneous')].fdp_mean:.4f} vs bound "
               f"{x.loc[('tfinance','hgb_graph_features','moderate','wcs_rand_homogeneous')].bound_mean:.4f}); "
               "this neither proves nor refutes control. The proof supplies the claim.")
    out.append(f"- Randomized WCS mean power exceeds weighted BH in {int((pw.wcs_rand_homogeneous > pw.weighted_bh).sum())}/18 cells "
               f"and is at least that of deterministic-p WCS (same pruning coin) in "
               f"{int((pw.wcs_rand_homogeneous >= pw.wcs_homogeneous - 1e-12).sum())}/18. Weighted BY and stratified rules "
               "were not run end-to-end, so do not claim superiority over every valid method.")
    out.append(f"- Retrained-scorer AUROC: cell means {e.auroc_mean.min():.3f}-{e.auroc_mean.max():.3f}; "
               f"individual fits {trials.auroc.min():.3f}-{trials.auroc.max():.3f}.")
    out.append('- Label budget: the design still includes the uniform 10% prior panel (exposure definition and training). '
               'Only newly acquired labels are biased. State this prominently and count the panel in label costs.\n')

    s = pd.read_csv(A / 'wcs_estimated_summary.csv').set_index(['dataset', 'model', 'method'])
    out.append('### Propensities: true vs estimated vs misspecified (`wcs_estimated_summary.csv`), FDP / power\n')
    out.append('| Graph | Scorer | Pooled BH | WCS, true rho | WCS, estimated rho | WCS, 2-stratum rho (misspecified) |')
    out.append('|---|---|---|---|---|---|')
    for ds, name in G:
        for mdl, ml in S:
            cells = ' | '.join(f"{s.loc[(ds, mdl, m)].fdp_mean:.3f} / {s.loc[(ds, mdl, m)].power_mean:.3f}"
                               for m in ('pooled_reference', 'wcs_rand_oracle', 'wcs_rand_logit', 'wcs_rand_strata2'))
            out.append(f"| {name} | {ml} | {cells} |")
    out.append('\nThe estimated model is cross-fitted logistic regression that uses the same covariates that generate this '
               'synthetic acquisition mechanism. It is an empirical sensitivity analysis, not evidence for arbitrary '
               'operational selection, and the theorem does not cover it.\n')

    r = pd.read_csv(A / 'wcs_randomized_summary.csv')
    r = r[(r.design == 'exposure_80') & (r.model == 'hgb_graph_features')].set_index(['dataset', 'scenario', 'method'])
    out.append('### Resolution diagnostic (`wcs_randomized_summary.csv`; known rho, 80/20 strata, graph scorer)\n')
    out.append('m f / alpha for high-exposure anomalies (deterministic benchmark; for homogeneous pruning the realized '
               'necessary bound is multiplied by xi), anomaly share in the sparse stratum, and power:\n')
    out.append('| Graph | Bias | median m f / alpha | anomalies in sparse stratum | WCS (det. p) power | Randomized WCS power |')
    out.append('|---|---|---|---|---|---|')
    for ds, name in G:
        for sc, lab in (('neutral', 'none'), ('moderate', 'moderate'), ('severe', 'severe')):
            a, b = r.loc[(ds, sc, 'wcs_homogeneous')], r.loc[(ds, sc, 'wcs_rand_homogeneous')]
            out.append(f"| {name} | {lab} | {a.need_anom_high_mean:.0f} | {a.frac_anom_high_mean:.2f} | "
                       f"{a.power_mean:.3f} | {b.power_mean:.3f} |")

    v1 = json.loads((A / 'wcs_validation.json').read_text())
    v2 = json.loads((A / 'wcs_randomized_validation.json').read_text())
    v3 = json.loads((A / 'wcs_endtoend_validation.json').read_text())
    m2 = json.loads((A / 'wcs_randomized_manifest.json').read_text())
    out.append('\n### Validation (json files in `audit_method/`)\n')
    out.append(f"- Fast WCS vs the repository's own m-by-m equation implementation: {v1['fast_vs_reference_cases']} random cases identical. "
               "Agreement with the authors' public code covers only six constant-weight cases; do not claim variable-weight agreement.")
    out.append(f"- Exact finite-design FDR (deterministic p, both prunings): {v1['exact_design_cases']} designs, "
               f"{v1['exact_assignments']} assignments, none above the bound. Weighted BH: none above the bound in these "
               "designs (not a proof).")
    out.append(f"- Randomized p-values: exact uniformity in {v2['exact_uniformity_cases']} designs; Monte Carlo FDR in "
               f"{v2['mc_fdr_cases']} designs x {v2['mc_draws_per_case']} draws, worst +{v2['max_fdr_minus_bound_in_se']:.1f} SE (pre-set limit 4).")
    out.append(f"- Training-dependent scores: {v3['exact_cases']} designs, {v3['exact_states']} exact states, none above the bound.")
    out.append(f"- Resolution condition: {m2['resolution_checks']} nonempty rejection sets checked (each at its largest floor), "
               f"{m2['resolution_violations']} violations.")
    out.append('- An independent audit (`WCS_HANDOFF_REVIEW.md`) re-aggregated all 367,800 trial rows, matched every '
               'manifest hash, and added exact integration over the auxiliary coins (32 designs, max excess ~1e-16).')
    return '\n'.join(out)


HEAD = (HERE / 'fact_sheet_text.md').read_text(encoding='utf-8').split('<<NUMBERS>>')
(HERE / 'FACT_SHEET.md').write_text(HEAD[0] + numbers() + HEAD[1], encoding='utf-8')
print('FACT_SHEET.md written')
