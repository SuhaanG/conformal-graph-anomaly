"""Validate completed GPU outputs and generate manuscript tables from all runs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
F=D.parent/'aistats_followup'
G=F/'h200_results/paper/aistats_followup'
METHOD={'filtered':'Filtered','random':'Random','mixture_half':'Half mixture'}
MODEL={'dominant_pygod':'DOMINANT','gae':'GAE','isolation_forest':'Isolation Forest','degree':'Degree'}

def table(name,header,columns,rows):
    (D/name).write_text('\\begin{tabular}{'+columns+'}\\toprule\n'+header+r'\\\midrule'+'\n'+
                       '\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n'+r'\bottomrule\end{tabular}'+'\n')

def main():
    completion=json.loads((G/'gpu_completion.json').read_text())
    assert completion['model_seed_evaluations']==60
    for marker in G.glob('gpu_job_*.json'):
        for name,digest in json.loads(marker.read_text()).items():
            assert hashlib.sha256((G/name).read_bytes()).hexdigest()==digest,name
    dfs=[];total=0
    keys=['dataset','detector','seed','split','design','label_fraction','trim','budget','method']
    for p in sorted(G.glob('evaluation_*_summary.csv')):
        s=pd.read_csv(p);t=pd.read_csv(p.with_name(p.name.replace('_summary','_trials')))
        assert not s.duplicated(keys).any()
        assert np.isfinite(t[['fdp','power','discoveries']]).all().all()
        assert t[['fdp','power']].ge(0).all().all() and t[['fdp','power']].le(1).all().all()
        a=t.groupby(keys)[['fdp','power','discoveries']].mean().add_suffix('_mean')
        b=s.set_index(keys).loc[a.index]
        assert np.allclose(a,b[a.columns],rtol=0,atol=1e-11),p.name
        assert t.groupby(keys).size().eq(200).all()
        if 'degree' not in p.name:
            assert set(s.seed)=={int(p.stem.split('_')[-2])}
        dfs.append(s);total+=len(t)
    df=pd.concat(dfs,ignore_index=True)
    group=['dataset','detector','design','label_fraction','trim','budget','method']
    metrics=['fdp_mean','power_mean','discoveries_mean','n_calib','auroc','auprc','null_tail001_exact']
    rows=[]
    for key,c in df.groupby(group):
        z=dict(zip(group,key));z.update(seeds=c.seed.nunique(),splits=c.split.nunique())
        assert z['seeds']==(1 if z['detector']=='degree' else 10)
        assert z['splits']==5
        for m in metrics:
            byseed=c.groupby('seed')[m].mean();bysplit=c.groupby('split')[m].mean()
            z[m]=byseed.mean();z[m+'_seed_sd']=byseed.std();z[m+'_split_min']=bysplit.min();z[m+'_split_max']=bysplit.max()
        rows.append(z)
    agg=pd.DataFrame(rows);agg.to_csv(D/'followup_aggregate.csv',index=False)
    def get(ds,model,method,design='primary',fraction=1.,trim=0.,budget=1000):
        q=agg[(agg.dataset==ds)&(agg.detector==model)&(agg.method==method)&(agg.design==design)&
              (agg.label_fraction==fraction)&(agg.trim==trim)&(agg.budget==budget)]
        assert len(q)==1,(ds,model,method,design,fraction,trim,budget)
        return q.iloc[0]
    def pm(r,m): return f'${r[m]:.4f}\\pm{r[m+"_seed_sd"]:.4f}$'
    rows=[]
    for ds in ('amazon','tolokers'):
        for model in ('dominant_pygod','gae','isolation_forest'):
            for method in ('filtered','random'):
                r=get(ds,model,method)
                rows.append([ds.capitalize(),MODEL[model],METHOD[method],f'{r.n_calib:.0f}',pm(r,'fdp_mean'),pm(r,'power_mean'),f'{r.auroc:.3f}'])
    table('untrimmed_table.tex','Graph & Detector & Rule & Mean $n$ & FDP & Power & AUROC','lllrrrr',rows)
    rows=[]
    for ds in ('amazon','tolokers'):
        for fraction in (.25,.5,1.):
            v=[get(ds,'dominant_pygod',m,fraction=fraction) for m in METHOD]
            rows.append([ds.capitalize(),str(int(fraction*100))]+[f'{r.fdp_mean:.4f} / {r.power_mean:.4f}' for r in v])
    table('partial_table.tex','Graph & Labels (\\%) & Filtered & Random & Half mixture','llrrr',rows)
    rows=[]
    for ds in ('amazon','tolokers'):
        for design,trim in [('primary',0.),('calibration_only_trim',.01),('calibration_only_trim',.05),
                            ('enriched_both_trim_verified',0.),('enriched_both_trim_verified',.01),('enriched_both_trim_verified',.05)]:
            r=get(ds,'dominant_pygod','filtered',design=design,trim=trim)
            q=get(ds,'dominant_pygod','random',design=design,trim=trim)
            label={'primary':'Natural, no trim','calibration_only_trim':'Natural, reference trim','enriched_both_trim_verified':'Enriched, both pools'}[design]
            rows.append([ds.capitalize(),label,str(int(trim*100)),f'{r.fdp_mean:.4f}',f'{r.power_mean:.4f}',f'{q.fdp_mean:.4f}',f'{q.power_mean:.4f}'])
    table('trimming_table.tex','Graph & Test / trimming design & Trim (\\%) & Filter FDP & Power & Random FDP & Power','llrrrrr',rows)
    w=pd.read_csv(F/'wcs_summary.csv')
    names={'selected_bh':'Ordinary BH','weighted_bh':'Weighted BH','wcs_deterministic':'WCS deterministic','wcs_homogeneous':'WCS homogeneous','weighted_by':'Weighted BY'}
    table('wcs_full_table.tex',r'$\theta$ & $q$ & Method & FDP (MCSE) & Power','rrlrr',[
        [f'{r.theta:g}',f'{r.q:g}',names[r.method],f'${r.fdp_mean:.4f}\\;( {r.fdp_mcse:.4f})$',f'{r.power_mean:.4f}'] for r in w.itertuples()])
    ig=pd.read_csv(F/'independent_graph_summary.csv')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7,3.7),sharey=True,sharex=True)
    levels=[(0.,1.),(0.,4.),(1.2,1.),(1.2,4.)]
    for row,eta in enumerate((0.,.5)):
        for col,theta in enumerate((0.,2.)):
            ax=axes[row,col]
            for method,label,color,offset in [('oracle_filtered','Filtered','#b83835',-.12),('matched_random','Random','#28648c',0),('mixture_half','Half mixture','#997329',.12)]:
                c=ig[(ig.all_null==0)&(ig.theta==theta)&(ig.neighbor_noise_fraction==eta)&(ig.method==method)].set_index(['sigma','cross_multiplier']).loc[levels]
                ax.errorbar(np.arange(4)+offset,c.fdp_mean,yerr=1.96*c.fdp_mcse,fmt='o',ms=3,capsize=2,label=label,color=color)
            ax.axhline(.1,color='.4',ls='--',lw=.8);ax.set_title(f'Score degree effect {theta:g}; neighbor noise {eta:g}')
            ax.set_xticks(range(4),['0 / 1','0 / 4','1.2 / 1','1.2 / 4']);ax.set_ylim(-.015,.83)
            ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel('Mean FDP')
            if row==1:ax.set_xlabel('Degree heterogeneity / cross-class multiplier')
    axes[0,0].legend(frameon=False,fontsize=7,loc='upper left')
    fig.tight_layout();fig.savefig(D/'independent_graphs.pdf');fig.savefig(D/'independent_graphs.png',dpi=180);plt.close(fig)
    rows=[]
    for key,c in ig.groupby(['all_null','sigma','cross_multiplier','theta','neighbor_noise_fraction']):
        idx=c.set_index('method')
        vals=[f'{idx.loc[m,"fdp_mean"]:.3f} / {idx.loc[m,"power_mean"]:.3f}' for m in ('oracle_filtered','matched_random','mixture_half','partial_filtered','partial_random')]
        rows.append([str(key[0])]+[f'{v:g}' for v in key[1:]]+vals)
    table('independent_full_table.tex',r'All-null & $\sigma$ & $c$ & $\theta$ & $\eta$ & Filter & Random & Mixture & Partial & Partial random','rrrrrrrrrr',rows)
    manifest=dict(gpu_model_seed_jobs=60,total_trials=total,summary_rows=len(df),
                  result_checksums_verified=True,summary_means_recomputed=True,
                  uncertainty='SD across 10 training-seed means after averaging 5 crossed splits and 200 draws; conditional on fixed graph and these splits.')
    (D/'followup_validation.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
