# AISTATS scope clarification (22 September 2026)

Keep only revisions relevant to AISTATS. The pilot experiment did not demonstrate
clear superiority, so its method, extra appendix, and table have been removed
from the submission manuscript and source ZIP. Research code/results remain
separate. The existing fixed-budget batch certificate stays as an operational
response to the main paper's selection problem. Current PDF: 24 total pages, eight main-text pages. Cloud build verified with zero errors and warnings. Full validation is in
FINAL_VALIDATION.json; historical notes below are superseded.

# Current revision: 22 September 2026

This update supersedes historical delivery and style statements below.
The PDF is 25 pages with eight main-text pages in the official 2027 style.
Workshop review feedback was used to add a concrete operational scenario,
foreground the batch-certification procedure, clarify the Weibo mechanism,
justify the benchmark/scorer subset, add original detector citations, define
the pooled standard deviation, and discuss other detector training paradigms.

A new pilot-selected audit is evaluated on every cached score: 72,000 trials,
with exact conditional proof, 256-population adaptive enumeration, and 2,000
synthetic implementation checks. It saves labels on average but has lower
yield at the same budget cap. It remains an exploratory appendix comparison;
no claim of a novel or superior graph-specific correction is made.

See WORKSHOP_FEEDBACK_ACTIONS.md, pilot_validation.json, and FINAL_VALIDATION.json
for details. Current synchronization status is recorded in OVERLEAF_SYNC.json.

---

# AISTATS revision notes - updated 21 September 2026

## Current update (supersedes earlier delivery status below)

The current PDF has 23 pages: eight main-text pages, two AI-disclosure/reference
pages, one checklist page, and twelve appendix pages in single-column format.
The vendor 2027 style and fancyhdr files are unmodified; STYLE_PROVENANCE.json
records their source and hashes. Main Table 2 now adds accurate supervised
controls on both graphs. The abstract and introduction lead with the demonstrated
selection failure on an accurate detector and useful full-reference power.
Appendix L gives full controls, SDs, observed-label results, and negative
normalization/reference-budget results. Appendix M reports the exploratory batch
certificate, proof, label costs, and failed stratified designs without claiming
a novel superior graph method. The checklist uses the official questions and
honestly records documentation gaps rather than answering everything Yes.

Synchronized and verified 2026-09-21: source matches local manuscript; cloud build has 23 pages, Errors 0 and Warnings 0; typesetting information inspected.
The working Overleaf project now contains the official 2027 style, checklist,
new supervised comparisons, and exploratory certification appendix.

The earlier notes below document the preceding GPU revision and are historical.

## Current deliverables

The revised paper is titled "Calibration Selection Can Inflate False Discoveries on Graphs". The PDF has 16 pages: seven main-text pages, one page of AI disclosure and references, and eight appendix pages. The original manuscript and Overleaf project are preserved. The working copy is https://www.overleaf.com/project/6aa4b3a8249d9edf82646a2b . Its main document is aistats.tex.

Deliverables are aistats.pdf, aistats_overleaf_source.zip, and aistats_reproducibility.zip. Use the explicit source ZIP for review materials: the Overleaf project also preserves older manuscripts outside this revision package.

## Evidence completed

- All 60 H200 model/seed jobs completed: two graphs, three learned detectors, ten training seeds. Two fixed degree controls also completed. Five crossed test splits and 200 calibration draws per comparison produce 2,325,000 evaluation rows. All 11,625 summary rows reproduce from the per-draw CSVs, and all job checksums pass.
- Primary evaluation has no score trimming and preserves labeled class prevalence. Amazon's 3,305 unlabeled nodes are excluded from calibration/testing while remaining in unsupervised training. Filtered DOMINANT mean FDP is 0.9357 on Amazon and 0.7400 on Tolokers at nominal 0.10; matched random values are 0 and 0.0038. Power, weak detector AUROCs, partial-label results, and mixture outcomes are reported.
- The conditional rank audit now directly covers those headline results. Exact null fractions below 0.01 are 0.536 versus 0.00545 on Amazon and 0.243 versus 0.01015 on Tolokers for filtered versus random DOMINANT calibration. An additional 300,000 fresh calibration draws across all 600 primary learned-detector conditions validate the calculations: maximum absolute error 0.000721; all 300 nonconstant comparisons within 2.98 MCSE, and all 300 deterministic comparisons exact.
- The independent graph study contains 2,500 unique generated graphs and 42,500 method evaluations. It separates degree heterogeneity, anomalous connectivity, score sensitivity, and neighborhood noise, including partial-label and all-null controls. The mixture exceeds nominal in two settings; this is reported rather than suppressed.
- Weighted conformal selection comparison contains 60,000 method evaluations. WCS follows the versioned Algorithm 2, with literal-equation checks and constant-weight reference-code comparisons. Variable-weight disagreement with the inspected reference code is documented. The method is evaluated under its independent-observation assumptions.
- The earlier independent score simulation (40,000 evaluations) and five-seed Weibo conditional audit remain separate supporting studies. CPU and GPU results are not pooled.

## Manuscript changes

The abstract and introduction lead with the untrimmed evidence, the conditional rank explanation, and the practical calibration-design lesson. Repeated qualifications were reduced without removing assumptions, unfavorable results, or material limitations. The main results compare DOMINANT, GAE, and Isolation Forest; partial labels and calibration-only trimming are included. Historical trimmed results are explicitly marked as provenance, not primary evidence. A malformed historical-table cross-reference was fixed during visual review.

The 20-cell degree audit and older calibration-distribution matrix do not constitute controlled matched evidence: historical label handling, changing reference sizes, changing test identities, and a custom detector implementation limit their interpretation. Their source data remain available, but they are not presented as corrected primary results.

## Reproduction and checks

- build_followup_tables.py validates GPU manifests and every summary mean, then regenerates follow-up tables and the independent-graph figure.
- audit_primary_ranks.py reconstructs all 600 primary rank conditions and performs the fresh calibration Monte Carlo check.
- revise_after_gpu.py builds the current manuscript from the frozen aistats_before_h200.tex backup. Edit that transformation or preserve direct source changes before rerunning it.
- package_revision.py builds explicit anonymized archives with per-file SHA-256 hashes. The source archive contains only required manuscript dependencies.
- Compile using pdflatex, bibtex, pdflatex, pdflatex. Rendering was inspected after changes; all citations and cross-references resolve. The official provisional style has one empty-box warning at the abstract, with no visible overflow.

## Submission status and remaining decisions

This is a working AISTATS revision, not a submitted or accepted paper. A percentage acceptance probability is not supported by these results. The empirical foundation is materially stronger; novelty and practical relevance remain reviewer judgments. The paper supplies a controlled selection study and an exact conditional diagnostic, not a new general graph FDR theorem.

The official AISTATS 2027 template was unavailable when checked. The unmodified 2026 style is provisional; replace it and recheck layout when the official package is released. The current CFP lists September 29, 2026 for abstracts and October 6 for full papers/supplements, AoE, with eight main-text pages: https://virtual.aistats.org/Conferences/2027/CallForPapers .

Before submission, authors must review scientific claims, proofs, author information, AI disclosure, reciprocal-reviewer eligibility, and the official checklist. Resolve the earlier TAE workshop status: the AISTATS concurrent-submission exception describes nonarchival extended abstracts of at most four pages; do not assume it covers an eight-page workshop manuscript. No venue submission, withdrawal, email, or public repository push was performed.

Final synchronization check: the revised aistats.tex and all required figures/tables are saved in the Overleaf working copy. Its latest build has 16 pages, Errors 0, and Warnings 0; typesetting information was inspected. Both final archives passed hash and targeted identity checks, and the extracted source archive compiled successfully with no unresolved references. See FINAL_VALIDATION.json.
