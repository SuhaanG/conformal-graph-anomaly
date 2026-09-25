"""Build explicit, anonymized source and reproducibility ZIPs (no git history)."""
from pathlib import Path
import zipfile,hashlib,json,csv
ROOT=Path(__file__).resolve().parents[2]; D=Path(__file__).resolve().parent
source=['aistats.tex','aistats_refs.bib','aistats2027.sty','fancyhdr.sty','checklist.tex','simulation_table.tex','selection_validation.pdf','graph_audit_full_table.tex']
source += ['untrimmed_table.tex','partial_table.tex','trimming_table.tex','wcs_full_table.tex','independent_full_table.tex','independent_graphs.pdf']
source += ['supervised_table.tex','supervised_full_table.tex','supervised_partial_table.tex','certificate_table.tex','audit_extension.tex']
source += ['headline_results.pdf','selection_dose.pdf','selection_dose_all_fdp.pdf','selection_dose_all_power.pdf',
           'dose_abstract.tex','selection_dose_main.tex','selection_dose_appendix.tex']
source += ['stratified_design.tex','stratified_appendix.tex','stratified_results.tex',
           'stratified_main_results.tex','paired_uncertainty_appendix.tex']
source += ['three_graph_results.tex','acquisition_main.pdf','allocation_full_figure.tex']
source += ['allocation_main.tex','allocation_figure.tex','allocation_proof.tex','allocation_results.tex','allocation_acquisition.pdf']
source += [p.name for p in sorted(D.glob('dose_paired_*.pdf'))]
source += [p.name for p in sorted(D.glob('extension_dose_*.pdf'))]
if (D/'extended_benchmarks.tex').exists():
    source += ['extended_benchmarks.tex','extended_benchmark_main.tex']

def make_zip(target,items):
    hashes={}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for file,name in items:
            assert file.is_file(),file
            data=file.read_bytes();z.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        z.writestr('FILE_HASHES.json',json.dumps(hashes,indent=2))
    return hashes

def main():
    audit=list(csv.DictReader((D/'graph_rank_aggregate.csv').open()))
    assert all(int(r['training_seeds'])==5 for r in audit),'Do not package incomplete graph runs'
    make_zip(D/'aistats_overleaf_source.zip',[(D/n,n) for n in source])
    files=[(D/n,'paper/aistats_revision/'+n) for n in source]
    artifact_names=['validate_revision.py','build_tables.py','graph_sensitivity.py','graph_rank_audit.py',
        'synthetic_trials.csv','synthetic_summary.csv','graph_summary.csv','paired_summary.csv',
        'analytic_rank_validation.csv','validation_manifest.json','graph_sensitivity_trials.csv',
        'graph_sensitivity_summary.csv','graph_sensitivity_manifest.json','graph_rank_trials.csv',
        'graph_rank_summary.csv','graph_rank_aggregate.csv','graph_rank_manifest.json']
    files += [(D/n,'paper/aistats_revision/'+n) for n in artifact_names]
    files += [(p,'paper/aistats_revision/'+p.name) for p in sorted(D.glob('weibo_gae_scores_*.npz'))]
    files += [(p,str(p.relative_to(ROOT)).replace('\\','/')) for p in sorted((ROOT/'results/published').glob('calibration_strategy_*.csv'))]
    files += [(ROOT/'results/published/degree_baseline_check.csv','results/published/degree_baseline_check.csv')]
    for n in ['detector.py','detectors.py','conformal_fdr.py','graph_gen.py']:
        files.append((ROOT/'src'/n,'src/'+n))
    files.append((ROOT/'scripts/real_data_experiment.py','scripts/real_data_experiment.py'))
    files.append((D/'SUPPLEMENT_README.md','README.md'))
    for name in ['build_followup_tables.py','followup_aggregate.csv','followup_validation.json',
                 'audit_primary_ranks.py','primary_rank_validation.csv','primary_rank_validation.json']:
        files.append((D/name,'paper/aistats_revision/'+name))
    f=ROOT/'paper/aistats_followup'
    for name in ['PROTOCOL.md','train_headlines.py','evaluate_headlines.py','h200_runner.py',
                 'independent_graphs.py','wcs_experiment.py','fetch_wcs_reference.py',
                 'independent_graph_trials.csv','independent_graph_summary.csv','independent_graph_manifest.json',
                 'wcs_trials.csv','wcs_summary.csv','wcs_manifest.json','wcs_validation.json']:
        files.append((f/name,'paper/aistats_followup/'+name))
    for p in sorted((f/'h200_results/paper/aistats_followup').iterdir()):
        if p.suffix in ('.json','.csv','.npz'):
            files.append((p,p.relative_to(ROOT).as_posix()))
    # Raw public inputs support a fresh training run; source URLs and hashes are documented.
    files.append((f/'Amazon.mat','paper/aistats_followup/Amazon.mat'))
    raw=ROOT/'tmp_tolokers/tolokers/raw/tolokers.npz'
    files.append((raw,raw.relative_to(ROOT).as_posix()))
    for name in ['build_focus_results.py','build_stratified_results.py','build_extended_results.py','build_allocation_results.py','build_editorial_display.py']:
        if not (D/name).exists():continue
        files.append((D/name,'paper/aistats_revision/'+name))
    for p in sorted((ROOT/'paper/audit_method').iterdir()):
        if (p.suffix in ('.py','.csv','.npz','.json','.gz') and not p.stem.endswith('_partial')
            and not p.stem.startswith(('pilot_','validate_pilot'))):
            files.append((p,p.relative_to(ROOT).as_posix()))
    for name in ['PROTOCOL.md','STRONG_SCORER_PROTOCOL.md','PARTIAL_LABEL_PROTOCOL.md','THEORY.md','DOSE_PROTOCOL.md','WEIBO_CONTROL_PROTOCOL.md','TFINANCE_PROTOCOL.md','STRATIFIED_PROTOCOL.md','ALLOCATION_ACQUISITION_PROTOCOL.md']:
        p=ROOT/'paper/audit_method'/name
        files.append((p,p.relative_to(ROOT).as_posix()))
    # GADBench raw DGL binaries are >1 GB. Include processed sparse inputs above
    # plus public acquisition instructions/hashes, not those large raw files.
    for name in ['acquisition.json','upstream_revision.json']:
        p=ROOT/'data/gadbench'/name
        if p.exists():files.append((p,p.relative_to(ROOT).as_posix()))
    make_zip(D/'aistats_reproducibility.zip',files)
    print('Packaged',len(files),'files; source ZIP and reproducibility ZIP')
if __name__=='__main__': main()
