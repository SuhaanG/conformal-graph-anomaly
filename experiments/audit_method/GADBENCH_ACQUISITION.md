# GADBench acquisition handoff

Completed 2026-09-23 America/Chicago (2026-09-24 UTC). Acquisition and conversion
only: no model fitting, evaluation, sensitivity experiment, or protocol edits.

The actual [official GADBench archive](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing)
linked by its [README](https://github.com/squareRoot3/GADBench/blob/f9aa021ce9b6c6580427fb633b596843be76ddc6/readme.md)
was accessible. HTTP byte ranges retrieved only `datasets/weibo` and
`datasets/tfinance`, totaling 85,891,732 compressed bytes, from a 757,193,110-byte
ZIP. Each extracted member passed the archive's CRC32 and size checks. Full
member SHA256 and compressed-range SHA256 values are recorded; the entire ZIP
was not downloaded, so no whole-archive hash is claimed. No BWGNN fallback or
source deviation was needed. Raw DGL files are `data/gadbench/weibo` and
`data/gadbench/tfinance`.

## Ready for the parent task

- `tfinance_graph.npz`: 39,357 nodes, 10 float32 attributes, label 0 = 37,554,
  label 1 = 1,803. Raw: 42,484,443 directed entries including 39,357 self-loops,
  with no duplicate directed entries. Processed: 42,445,086 CSR entries,
  representing 21,222,543 undirected edges.
- `weibo_gadbench_graph.npz`: 8,405 nodes, 400 float32 attributes, label 0 =
  7,537, label 1 = 868. Raw: 416,368 directed entries including 8,405 self-loops,
  with no duplicate directed entries. Processed: 754,542 CSR entries,
  representing 377,271 undirected edges. The README's 407,963 Weibo edge count
  equals the raw number of directed entries after removing loops; it is not
  the number of undirected edges after symmetrization.

Both NPZs have exactly `features`, `labels`, `indices`, `indptr`, `degree`, matching
the prior cache's field names and dtypes (float32, int64, int64, int64, float64).
Features and labels are bitwise/value-identical to their distributed DGL node
arrays, and node order is retained. Labels were already scalar binary int64;
no inversion, argmax, thresholding, or recoding was applied. All features are
finite. Adjacency was constructed sparsely, binarized, unioned with its transpose,
and stripped of self-loops. Symmetry, canonical CSR structure, degree sums, zero
diagonal, and reloaded NPZ array equality passed. Neither graph has isolated nodes.
The DGL edge `count` attribute is intentionally ignored for binary adjacency.
Distributed train/validation/test masks were not used.

The [primary loader](https://github.com/squareRoot3/GADBench/blob/f9aa021ce9b6c6580427fb633b596843be76ddc6/utils.py#L7)
loads the first DGL graph without changing labels. The
[detector](https://github.com/squareRoot3/GADBench/blob/f9aa021ce9b6c6580427fb633b596843be76ddc6/models/detector.py#L24)
uses `ndata['label']` and evaluates class-1 probabilities as anomalies.
Saved README, loader, detector, and preprocessing-notebook bytes were checked
against that immutable upstream commit.

## Weibo source comparison

The local PyGOD raw file SHA256 matches the previously verified official source
hash `3827dca358a7bab33bef0b494db74e22642aad779c15da764d9ccec9806ce46c`.
Its labels are 8,058 zeros and 347 ones, and the existing `weibo_graph.npz`
labels equal those raw labels exactly. Thus the 347 versus 868 discrepancy is
present in the distributed source labels, not introduced by this conversion.

GADBench and PyGOD raw feature matrices are exactly equal at the same indices;
their symmetrized binary loop-free adjacency matrices are exactly equal, also
equal to the existing cache adjacency. However, only 8,404 feature rows are
unique: nodes 130 and 4108 have identical features and identical adjacency to
all other nodes (they are connected to each other). No external stable node IDs
were provided to distinguish these twins. Full node identity/order is therefore
not established, and no pointwise label-disagreement statistic is reported.

There is a feature-representation difference from the existing PyGOD cache:
that cache contains per-column z-scores, verified exactly against raw PyGOD
features; the new GADBench output retains the distributed raw float32 features.
Keep this as the separately named `weibo_gadbench` sensitivity, disclose that
representation difference, and do not attribute all future differences to
labels or count it as an independent graph replication. Original PyGOD files
and caches were not overwritten.

## Reproduction and provenance

Run from the workspace root with the existing local runtime:

```powershell
.venv-aistats/Scripts/python.exe paper/audit_method/acquire_gadbench.py --convert
# Optional reacquisition of the two members:
.venv-aistats/Scripts/python.exe paper/audit_method/acquire_gadbench.py --download
```

The small official DGL 1.1.2 Windows cp311 wheel was extracted into
`data/gadbench/dgl_runtime`. It contains a ctypes `dgl.dll` and no Python `.pyd`;
data-loading calls worked with the existing Python 3.14.2/PyTorch CPU environment.
This was a tested graph-I/O workaround, not a general compatibility claim.
No packages were installed into or replaced in the parent's environment.
The wheel, download URL, and hash are retained. The converter imports that
isolated directory; no training modules from GADBench are executed.

`GADBENCH_DATA_PROVENANCE.json` contains source URLs, immutable loader revision,
raw and processed SHA256 values, protocol/script hashes, exact runtime versions,
graph and label counts, comparison evidence, and validation results.
`data/gadbench/acquisition.json` records ZIP member byte ranges, compressed sizes,
CRC32 values, and raw-member hashes. `TFINANCE_PROTOCOL.md`, parent analysis
scripts, and manuscript TeX were not edited by this acquisition task.
