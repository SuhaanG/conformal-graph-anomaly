# Provenance notes and corrections

## 2026-09-30: validity search chronology (correction to commit d95c75d)

The message of commit d95c75d says the validity-search protocol and script were
committed "before the search results". That is inaccurate. The protocol
(`WCS_COUNTEREXAMPLE_PROTOCOL.md`) was written, revised once (revision 2, framing and
reporting rules only) and hashed before the search was executed; the search was then
started, and the commit was made while it was running, after the ledger and the first
confirmation output already existed locally. Accurate description: "written and hashed
before execution, then committed while the search was running." The protocol file was not
changed after execution began except for one wording fix made before execution
("showed no excess" to "detected no clear violation"); its final hash is in
`WCS_COUNTEREXAMPLE_PROTOCOL.sha256` and in `wcs_counterexample_manifest.json`.

## 2026-09-30: uniform-acquisition baseline

The first version of `wcs_uniform.py` set the uniform rate from the realized non-test
nodes of each split, so acquisition probabilities depended on test identities, which
Theorem 1 does not allow. The run was stopped after four Amazon seeds with no summaries
written; its script and log are kept as `wcs_uniform_v0_exploratory.*` and are not used.
The protocol was revised (revision 2) to fix the rate from the deployment population before
test assignment. The corrected run was interrupted once by a machine shutdown after
Amazon and three T-Finance seeds (log kept as `wcs_uniform_interrupted_20260930.log`;
nothing was written or inspected) and was rerun from scratch with the same script.
