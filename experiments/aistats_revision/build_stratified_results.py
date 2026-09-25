"""Render all role-randomized comparisons and paired fixed-graph intervals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parent;A=D.parent/'audit_method'
NAMES={'amazon':'Amazon','tolokers':'Tolokers','weibo':'Weibo (PyGOD)','tfinance':'T-Finance','weibo_gadbench':'Weibo (GADBench)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def main():
    frames=[pd.read_csv(p) for p in sorted(A.glob('stratified_seeds_*.csv'))]
    z=pd.concat(frames,ignore_index=True)
    keys=['dataset','model','label_fraction','design','method','seed']
    assert not z.duplicated(keys).any()
    metrics=['fdp','power','discoveries','nonempty','n_reference','cal_low','cal_high']
    x=z.groupby(keys[:-1])[metrics].mean().reset_index()
    x.to_csv(A/'stratified_all_summary.csv',index=False)
    def cell(ds,model,design,method,rho=.5):
        v=x[(x.dataset==ds)&(x.model==model)&(x.label_fraction==rho)&(x.design==design)&(x.method==method)]
        assert len(v)==1;return v.iloc[0]
    a=cell('amazon','hgb_attributes','zero_exposure','stratified')
    b=cell('amazon','hgb_attributes','zero_exposure','full_random')
    ag=cell('amazon','hgb_graph_features','zero_exposure','stratified')
    bg=cell('amazon','hgb_graph_features','zero_exposure','full_random')
    main=rf'''\paragraph{{Stratification restores a guarantee, with a power cost.}}
The random-role remedy uses both exposure strata and divides $\alpha$ equally between them. With half of non-test labels revealed, the Amazon attribute scorer gives mean FDP ${a.fdp:.3f}$ and power ${a.power:.3f}$ for zero/positive-exposure strata, versus ${b.fdp:.3f}$ and ${b.power:.3f}$ for the pooled full random reference. The graph-feature scorer gives ${ag.fdp:.3f}/{ag.power:.3f}$ versus ${bg.fdp:.3f}/{bg.power:.3f}$. Thus the remedy supplies a design-based guarantee but does not outperform representative pooled calibration here. The 80/20 exposure partition and all other conditions are retained in Appendix~\ref{{app:stratified}}. These cutoffs are fixed under normal-role changes, so their filtered comparisons differ from the reference-quantile dose intervention.
'''
    (D/'stratified_main_results.tex').write_text(main)
    sections=[]
    for ds in NAMES:
        if ds not in set(x.dataset):continue
        rows=[]
        for model,name in [('hgb_attributes','Attributes'),('hgb_graph_features','Graph')]:
            for rho in (.25,.5,1.):
                for design,label in [('zero_exposure','Zero/positive'),('exposure_80','80/20 ranks')]:
                    values=[cell(ds,model,design,m,rho) for m in ('filtered_low','full_random','stratified')]
                    rows.append(f'{name} & {rho:.2f} & {label} & '+' & '.join(f'${v.fdp:.3f}/{v.power:.3f}$' for v in values)+r' \\')
        sections.append(r'\begin{table*}[t]\centering\small'+'\n'+rf'\caption{{{NAMES[ds]}: complete stratified comparison, mean FDP / power over ten training-seed averages. Both partitions and all revelation fractions are shown; no partition is selected using these outcomes.}}'+'\n'+r'\begin{tabular}{lllrrr}\toprule Score & $\rho$ & Partition & Low reference & Full random & Stratified\\\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\end{table*}')
    (D/'stratified_results.tex').write_text('\n\n'.join(sections))
    # Pointwise paired t intervals over independent training-seed averages,
    # conditional on each fixed graph and this declared evaluation protocol.
    seeds=[pd.read_csv(A/'selection_dose_seeds.csv')]
    for ds in ('tfinance','weibo_gadbench'):
        p=A/f'{ds}_dose_seeds.csv'
        if p.exists():seeds.append(pd.read_csv(p))
    seeds=pd.concat(seeds,ignore_index=True)
    idx=['dataset','model','label_fraction','removed','seed']
    pairs=[]
    for metric in ('fdp','power'):
        q=seeds.pivot(index=idx,columns='method',values=metric)
        d=(q.exposure-q.random).rename('difference').reset_index()
        v=d.groupby(idx[:-1]).difference.agg(['mean','std','count']).reset_index()
        assert v['count'].eq(10).all()
        half=t.ppf(.975,9)*v['std']/np.sqrt(10)
        v['lower']=v['mean']-half;v['upper']=v['mean']+half;v['metric']=metric
        pairs.append(v)
    pairs=pd.concat(pairs,ignore_index=True);pairs.to_csv(A/'dose_paired_intervals.csv',index=False)
    figs=[]
    for ds in NAMES:
        if ds not in set(pairs.dataset):continue
        fig,axs=plt.subplots(2,3,figsize=(7.05,4.1),layout='constrained',sharex=True)
        for i,metric in enumerate(('fdp','power')):
            for j,rho in enumerate((.25,.5,1.)):
                ax=axs[i,j]
                for model,color,name in [('hgb_attributes','#b65132','Attributes'),('hgb_graph_features','#246f99','Graph features')]:
                    v=pairs[(pairs.dataset==ds)&(pairs.model==model)&(pairs.label_fraction==rho)&(pairs.metric==metric)].sort_values('removed')
                    xx=v.removed.to_numpy()*100
                    ax.plot(xx,v['mean'],color=color,marker='o',ms=2,label=name)
                    ax.fill_between(xx,v.lower.to_numpy(),v.upper.to_numpy(),color=color,alpha=.15,lw=0)
                ax.axhline(0,color='#555',ls=':',lw=.8);ax.grid(alpha=.15)
                if i==0:ax.set_title(f'{rho:.0%} revealed')
                if j==0:ax.set_ylabel('Change in '+('FDP' if metric=='fdp' else 'power'))
                if i==1:ax.set_xlabel('Removed (%)')
                ax.set_xticks([0,20,40,60,80,100])
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
        fig.savefig(D/f'dose_paired_{ds}.pdf');plt.close(fig)
        figs.append(r'\begin{figure*}[t]\centering'+f'\n\\includegraphics[width=\\textwidth]{{dose_paired_{ds}.pdf}}\n'+rf'\caption{{{NAMES[ds]}: exposure removal minus size-matched random removal over the complete dose grid. Shading gives pointwise 95\% paired $t$ intervals across ten training-seed averages after averaging splits and calibration repetitions. Intervals describe repeated evaluation conditional on this graph; they are neither simultaneous across the sweep nor uncertainty over new graphs.}}'+'\n'+r'\end{figure*}')
    text=r'''\section{PAIRED UNCERTAINTY OVER THE COMPLETE DOSE GRID}
For each training seed, subtract random-removal FDP (or power) from the corresponding exposure-removal value after averaging its five test splits and fifty calibration repetitions. Report the mean of ten paired differences and the pointwise interval $\bar d\pm t_{9,.975}s_d/\sqrt{10}$. This estimates repeated-training/evaluation uncertainty on the given graph, not variation across graphs. The intervals are pointwise and do not adjust for searching across conditions. The complete grid is shown rather than retaining only positive intervals.
'''+ '\n\n'.join(figs)
    (D/'paired_uncertainty_appendix.tex').write_text(text)
    am=pairs[(pairs.dataset=='amazon')&(pairs.label_fraction==.5)&(pairs.removed==.2)]
    (D/'STRATIFIED_RESULTS.json').write_text(json.dumps(dict(datasets=list(x.dataset.unique()),amazon_half_labels_zero_exposure=dict(stratified_fdp=a.fdp,stratified_power=a.power,full_fdp=b.fdp,full_power=b.power),amazon_paired20=am.to_dict('records')),indent=2))
    print(am[['model','metric','mean','lower','upper']].to_string(index=False))


if __name__=='__main__':main()
