"""Integrate the completed frozen follow-up; preserve previous source snapshot."""
from pathlib import Path
import json,hashlib,shutil
D=Path(__file__).resolve().parent;A=D.parent/'audit_method'
def main():
    p=D/'aistats.tex';s=p.read_text(encoding='utf-8')
    backup=D/'before_allocation_integration_20260924.tex'
    if not backup.exists():backup.write_text(s,encoding='utf-8')
    if '\\input{allocation_main.tex}' not in s:
        s=s.replace('\\input{stratified_main_results.tex}','\\input{stratified_main_results.tex}\n\n\\input{allocation_main.tex}\n\\input{allocation_figure.tex}')
        s=s.replace('\\input{paired_uncertainty_appendix.tex}','\\clearpage\\input{allocation_proof.tex}\n\n\\input{paired_uncertainty_appendix.tex}')
        # Move the older headline plot and certification procedure to supporting
        # material, making room for the directly relevant remedy experiment.
        start=s.index('\\begin{figure*}[t]');end=s.index('\\end{figure*}',start)+len('\\end{figure*}')
        fig=s[start:end];s=s[:start]+s[end:]
        s=s.replace('\\label{app:unsupervised}','\\label{app:unsupervised}\n'+fig)
        start=s.index('If labels can instead be acquired for proposed discoveries,')
        end=s.index('\\section{DISCUSSION AND LIMITATIONS}',start)
        certificate=s[start:end]
        s=s[:start]+r'''When the available labels were acquired preferentially by exposure, use a remedy justified by that acquisition design, and evaluate its discovery power. Our stratified construction supplies such a guarantee but can be inefficient when the high-exposure reference is sparse. Known-propensity weighting requires correct acquisition probabilities; its marginal validity alone is not a BH guarantee. Direct finite-population auditing of a fixed discovery batch is another established option, with a different label budget and target (Appendix~\ref{app:deploymentaudit}).

'''+s[end:]
        s=s.replace('\\input{audit_extension.tex}',r'\section{DIRECT BATCH AUDITING AS AN ALTERNATIVE}'+'\n'+r'\label{app:deploymentaudit}'+'\n'+certificate+'\n'+r'\input{audit_extension.tex}')
    s=s.replace('Weibo supplies a third observed-exposure control with 8,405 nodes and 347 anomalous labels in the distributed file',
        'Weibo has 8,405 nodes; the original PyGOD control contains 347 anomalous labels in its distributed file')
    s=s.replace('We preserve the distributed labels and report source sensitivity separately.',
        'GADBench distributes 868 anomalous labels on the same graph. We retain both sources separately; full provenance appears in Appendix~\\ref{app:extensions}.')
    s=s.replace('No experiment in the primary comparison trims high-score normals.',
        'The prospective extension adds T-Finance (39,357 nodes, 1,803 anomalies) and GADBench Weibo~\\citep{tang2023gadbench}. No primary comparison trims high-score normals.')
    s=s.replace('A stratified remedy derives FDR control from randomized calibration/test roles, with a power cost relative to representative pooled calibration.',
        'Stratum-specific ranks restore FDR control under randomized roles, including exposure-biased label acquisition. Equal, size-proportional and pooled testing reveal that changing error allocation alone does not eliminate the power cost.')
    p.write_text(s,encoding='utf-8')
    provenance=A/'GADBENCH_DATA_PROVENANCE.json';obj=json.loads(provenance.read_text())
    obj['datasets']['weibo']['pygod_comparison'].pop('pygod_raw_path',None)
    provenance.write_text(json.dumps(obj,indent=2))
    # Future conversion should also avoid emitting an author-specific path.
    p=A/'acquire_gadbench.py';s=p.read_text();s=s.replace('dict(pygod_raw_path=str(pygod_path), pygod_url=prior[\'url\'],','dict(pygod_url=prior[\'url\'],')
    p.write_text(s)
    p=D/'SUPPLEMENT_README.md';s=p.read_text()
    if '## Allocation and acquisition follow-up' not in s:
        s+='''
## Allocation and acquisition follow-up (24 September)

Read ALLOCATION_ACQUISITION_PROTOCOL.md under paper/audit_method before interpreting the additions. It was frozen after equal-alpha outcomes, before testing these variants. This is explicitly an exploratory follow-up. Pilot-panel adaptation remains excluded.

```
python paper/audit_method/allocation_acquisition.py
python paper/audit_method/validate_allocation_additional.py
python paper/aistats_revision/build_allocation_results.py
```

These use cached scores on Amazon, T-Finance and GADBench Weibo and require no GPU. The 135,000-row ledger includes all allocation/acquisition variants. Existing arms reproduce 5,400 stored per-split outcomes. Independent reaggregation verifies seed summaries. Conditional block enumeration includes nonzero false discoveries; unequal-propensity enumeration checks weighted marginal ranks. Pooled stratified BH requires the stated product role law. Weighted BH is descriptive; weighted BY uses marginal validity. Uniform acquisition in the biased scenario is a counterfactual, with actual label costs recorded.

## Additional graphs and source provenance

TFINANCE_PROTOCOL.md and STRATIFIED_PROTOCOL.md were fixed before their corresponding outcomes. GADBENCH_DATA_PROVENANCE.json records official archive members, field counts, hashes and conversion. T-Finance and GADBench Weibo add 270,000 removal evaluations each; they are separate from the original 810,000. PyGOD and GADBench Weibo are alternative releases of one graph, not independent graph replications. Raw features and symmetrized adjacency match, but distributed labels contain 347 versus 868 anomalies. Raw feature processing also differs between the older processed cache and the GADBench extension. Never pool the two releases.

```
python paper/audit_method/benchmark_extension.py --datasets tfinance weibo_gadbench
python paper/audit_method/stratified_calibration.py --datasets amazon tolokers weibo
python paper/audit_method/stratified_calibration.py --datasets tfinance weibo_gadbench
python paper/aistats_revision/build_stratified_results.py
python paper/aistats_revision/build_extended_results.py
```

The processed sparse graph inputs are packaged, so routine reproduction does not require DGL or the >1GB raw T-Finance binary. acquire_gadbench.py provides optional official-archive acquisition and native-DGL conversion; provenance describes the original runtime. Complete dose curves and pointwise paired intervals are provided. Table/figure generators recreate numerical content; edited prose is maintained in the included source files. Author verification of proofs and scientific interpretations is required before submission.
'''
    p.write_text(s)
    p=D/'package_revision.py';s=p.read_text()
    s=s.replace("source += [p.name for p in sorted(D.glob('dose_paired_*.pdf'))]", "source += ['allocation_main.tex','allocation_figure.tex','allocation_proof.tex','allocation_results.tex','allocation_acquisition.pdf']\nsource += [p.name for p in sorted(D.glob('dose_paired_*.pdf'))]")
    s=s.replace("'build_extended_results.py']:","'build_extended_results.py','build_allocation_results.py']:")
    s=s.replace("'TFINANCE_PROTOCOL.md','STRATIFIED_PROTOCOL.md']:","'TFINANCE_PROTOCOL.md','STRATIFIED_PROTOCOL.md','ALLOCATION_ACQUISITION_PROTOCOL.md']:")
    p.write_text(s)
    print('Integrated completed allocation results and reproducibility documentation')
if __name__=='__main__':main()
