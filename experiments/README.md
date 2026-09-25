# Experiment source bundle

This directory contains the newer experiment code and protocols developed after the
original tracked `scripts/` suite. It is intended to make the AISTATS revision
reproducible for collaborators.

- `audit_method/`: calibration, stratification, allocation, certification, partial-label,
  strong-scorer, Weibo, and T-Finance experiments with their protocols and manifests.
- `aistats_followup/`: independent-graph, H200, supervised-control, and headline
  evaluation runners, including the pinned vendor reference implementation.
- `aistats_revision/`: table builders, result integrators, validation checks, and
  revision utilities used to update the manuscript.

The bundle contains source, protocols, and compact manifests. It deliberately excludes
large downloaded datasets, the DGL runtime, generated PDFs/ZIPs, package-build copies,
and per-run GPU output directories. Dataset provenance and acquisition instructions are
recorded in the audit-method files; the GADBench data are re-downloadable from the
pinned upstream revision.

The manuscript source remains in Overleaf and in the local ignored `paper/` directory.
'

Compact CSV summaries and diagnostics are included beside the source. Large trial tables, model score arrays, graph archives, and GPU logs remain local and can be regenerated from the included protocols.
