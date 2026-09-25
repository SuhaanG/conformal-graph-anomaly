"""Recompute seed-level summaries and manuscript working figures."""
from pathlib import Path
import json,hashlib,sys,platform
import numpy as np
import pandas as pd
import scipy,sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent

def summarize(frame,keys,metrics,name):
    seed=frame.groupby(keys+['seed'])[metrics].mean()
    out=seed.groupby(keys).mean().join(seed.groupby(keys).std().add_suffix('_seed_sd')).reset_index()
    out.to_csv(D/name,index=False)
    return out

def main():
    a=pd.read_csv(D/'strong_certificate_trials.csv');b=pd.read_csv(D/'strong_baseline_trials.csv')
    p=pd.read_csv(D/'partial_label_trials.csv');c=pd.read_csv(D/'strong_diagnostics.csv')
    assert len(a)==96000 and len(b)==40400 and len(p)==160400
    assert a.groupby(['dataset','model','seed','method','budget']).size().eq(200).all()
    assert a.seed.nunique()==10 and a.groupby(['dataset','model']).seed.nunique().eq(10).all()
    assert a.audit_labels.eq(a.budget).all()
    assert (a.true_discoveries+a.false_discoveries).eq(a.discoveries).all()
    assert a.loc[a.discoveries>0,'fdp_bound'].le(.1+1e-12).all()
    assert a.certificate_failure.sum()==0
    # Matching, label accounting, and saved training/deployment disjointness.
    paired=b[b.method.isin(['filtered','matched_random'])].groupby(['dataset','model','seed','split']).n_calib.nunique()
    assert paired.eq(1).all()
    paired=p[p.method.isin(['filtered','matched_random'])].groupby(['dataset','model','seed','split','label_fraction']).n_calib.nunique()
    assert paired.eq(1).all()
    for path in D.glob('*_hgb_*_scores_*.npz'):
        z=np.load(path);assert not np.intersect1d(z['train'],z['deploy']).size
        assert np.isfinite(z['scores']).all()
        if path.name.startswith('amazon'):assert np.all(z['train']>=3305) and np.all(z['deploy']>=3305)
    am=['discoveries','true_discoveries','fdp','power','abstain','reviewed_anomalies','certificate_failure']
    sm=summarize(a,['dataset','model','budget','method'],am,'strong_certificate_summary.csv')
    bs=summarize(b,['dataset','model','method'],['fdp','power','discoveries','n_calib'],'strong_baseline_summary.csv')
    ps=summarize(p,['dataset','model','label_fraction','method'],['fdp','power','discoveries','n_calib','reference_labels'],'partial_label_summary.csv')
    ds=summarize(c.drop_duplicates(['dataset','model','seed']),['dataset','model'],['auroc','average_precision','training_labels'],'strong_detector_summary.csv')
    # All ten seeds remain visible; audit repetitions are not graph replications.
    conditional=a[a.discoveries>0].groupby(['dataset','model','budget','method']).fdp.agg(['mean','max','count'])
    conditional.to_csv(D/'conditional_certificate_fdp.csv')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    colors=['#B84330','#236DA0','#48865C']
    methods=['filtered','matched_random','full_reference']
    for j,method in enumerate(methods):
        cells=bs[(bs.dataset=='amazon')&(bs.method==method)].set_index('model').loc[['hgb_attributes','hgb_graph_features']]
        axes[0].bar(np.arange(2)+(j-1)*.24,cells.fdp,.23,color=colors[j],label=method.replace('_',' '))
        axes[0].errorbar(np.arange(2)+(j-1)*.24,cells.fdp,yerr=cells.fdp_seed_sd,fmt='none',color='#333333',capsize=3,lw=1)
    axes[0].axhline(.1,color='black',ls='--',lw=1)
    axes[0].set(xticks=[0,1],xticklabels=['Attributes','Attributes + graph'],ylabel='Mean FDP (SD across 10 seeds)',title='Amazon: accurate supervised controls')
    axes[0].legend(fontsize=8)
    for method,color in zip(['uniform_candidates','score_strata','score_degree_strata','uniform_graph'],['#236DA0','#48865C','#B84330','#777777']):
        cells=sm[(sm.dataset=='amazon')&(sm.model=='hgb_attributes')&(sm.method==method)].sort_values('budget')
        axes[1].plot(cells.budget,cells.true_discoveries,marker='o',color=color,label=method.replace('_',' '))
    axes[1].set(xlabel='Additional audit labels',ylabel='Mean unreviewed true discoveries',title='Amazon: fixed-batch certification',xticks=[100,250,500])
    axes[1].legend(fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(D/'feasibility_results.png',dpi=180);fig.savefig(D/'feasibility_results.pdf');plt.close(fig)
    files=[f for f in D.iterdir() if f.suffix in ('.py','.md','.csv','.npz') and 'partial.csv' not in f.name]
    manifest=dict(status='passed',strong_audit_trials=len(a),strong_baseline_rows=len(b),partial_label_rows=len(p),
        training_score_caches=len(list(D.glob('*_hgb_*_scores_*.npz'))),
        python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (D/'RESULTS_VALIDATION.json').write_text(json.dumps(manifest,indent=2))
    print(ps.to_string(index=False));print('Validated all matrices and wrote figure/summaries.')
if __name__=='__main__':main()
