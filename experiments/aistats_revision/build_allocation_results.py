"""Report every frozen allocation/acquisition variant, including power losses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;A=D.parent/'audit_method'
NAMES={'amazon':'Amazon','tfinance':'T-Finance','weibo_gadbench':'Weibo (GADBench)'}
METHODS={'pooled_reference':('Pooled reference','#a84432'), 'equal_alpha':('Equal alpha','#7859a6'),
         'size_alpha':('Size alpha','#ad7e20'),'pooled_stratum':('Pooled stratified','#167a86'),
         'weighted_bh':('Weighted BH*','#399142'),'weighted_by':('Weighted BY','#555555'),
         'uniform_counterfactual':('Uniform counterfactual','#4477bb')}
def main():
    trials=pd.read_csv(A/'allocation_acquisition_trials.csv.gz')
    group=['phase','dataset','model','label_fraction','design','scenario','method']
    metrics=['fdp','power','reference_labels','n_reference']
    seed=trials.groupby(group+['seed'])[metrics].mean().reset_index()
    stored=pd.read_csv(A/'allocation_acquisition_seeds.csv')
    v=seed.merge(stored,on=group+['seed'],validate='one_to_one',suffixes=('_a','_b'))
    assert len(v)==len(seed)
    for metric in metrics:assert np.allclose(v[metric+'_a'],v[metric+'_b'],rtol=1e-12,atol=1e-12)
    x=seed.groupby(group)[metrics].mean().reset_index()
    def cell(phase,ds,model,method,design='exposure_80',rho=.5,scenario='uniform'):
        v=x[(x.phase==phase)&(x.dataset==ds)&(x.model==model)&(x.method==method)&(x.design==design)&(x.label_fraction==rho)&(x.scenario==scenario)]
        assert len(v)==1;return v.iloc[0]
    # Full appendix tables, all conditions; no selection by performance.
    tables=[]
    for phase in ('allocation','acquisition'):
        methods=['pooled_reference','equal_alpha','size_alpha','pooled_stratum']
        if phase=='allocation':methods+=['filtered_low']
        else:methods+=['weighted_bh','weighted_by','uniform_counterfactual']
        headers=['Pooled','Equal','Size','Joint']+(['Low only'] if phase=='allocation' else ['W-BH*','W-BY','Uniform'])
        for ds in NAMES:
            rows=[]
            for model,label in [('hgb_attributes','Attr.'),('hgb_graph_features','Graph')]:
                for design,dl in [('zero_exposure','Zero'),('exposure_80','80/20')]:
                    settings=[(r,'uniform',f'{r:.2f}') for r in (.25,.5,1.)] if phase=='allocation' else [(0.,s,s[:3]) for s in ('neutral','moderate','severe')]
                    for rho,scenario,sl in settings:
                        vv=[cell(phase,ds,model,m,design,rho,scenario) for m in methods]
                        rows.append(f'{label} & {dl} & {sl} & '+' & '.join(f'${v.fdp:.3f}/{v.power:.3f}$' for v in vv)+r' \\')
            tables.append(r'\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}'+'\n'+rf'\caption{{{NAMES[ds]}, {phase}: mean FDP / power. Every declared partition and setting is shown. For acquisition, neu/mod/sev denote $(\rho_L,\rho_H)=(.5,.5),(.5,.1),(.5,.025)$. W-BH* is descriptive; its marginal-rank argument is not a BH guarantee. Uniform is a counterfactual acquisition design.}}'+'\n'+r'\begin{tabular}{lll'+ 'r'*len(methods)+r'}\toprule Score & Strata & Setting & '+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\end{table*}')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,3,figsize=(7.1,6.6),layout='constrained')
    for j,ds in enumerate(NAMES):
        ax=axes[0,j]
        for method,(label,color) in METHODS.items():
            if method not in ('pooled_reference','equal_alpha','size_alpha','pooled_stratum'):continue
            for model,marker in [('hgb_attributes','o'),('hgb_graph_features','s')]:
                v=cell('allocation',ds,model,method)
                ax.scatter(v.power,v.fdp,c=color,marker=marker,s=24,label=label if model=='hgb_attributes' else None)
        ax.set_title(NAMES[ds]);ax.text(.02,.98,'A: allocation',transform=ax.transAxes,va='top',fontsize=7)
        for i,model in enumerate(('hgb_attributes','hgb_graph_features'),1):
            ax=axes[i,j]
            for method,(label,color) in METHODS.items():
                vv=[cell('acquisition',ds,model,method,rho=0.,scenario=s) for s in ('neutral','moderate','severe')]
                ax.plot([v.power for v in vv],[v.fdp for v in vv],color=color,lw=.8,alpha=.8)
                for v,marker in zip(vv,('o','s','^')):ax.scatter(v.power,v.fdp,c=color,marker=marker,s=18)
            ax.text(.02,.98,'B: '+('attributes' if i==1 else 'graph features'),transform=ax.transAxes,va='top',fontsize=7)
        for i in range(3):
            ax=axes[i,j];ax.axhline(.1,color='#333',ls=':',lw=.9);ax.set_xlim(-.03,1.02);ax.set_ylim(-.005,.205);ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Mean FDP')
            if i==2:ax.set_xlabel('Power')
    handles=[plt.Line2D([0],[0],color=c,label=label,lw=2) for label,c in METHODS.values()]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=7)
    fig.savefig(D/'allocation_acquisition.pdf');plt.close(fig)
    text=r'''The follow-up protocol was frozen after the equal-alpha results, before these variants were evaluated. It is an exploratory follow-up. Phase A reproduces 5,400 previously stored outcomes before comparing allocations. Phase B fixes exposure using only training labels and graph degree, then independently acquires each non-test node's label with the stated stratum probability. Acquired anomalies count toward label cost but are not normal references. Strata are never updated with newly acquired labels. The uniform counterfactual uses the deployment-universe mean acquisition probability, fixed before normal-role assignment; costs are close in expectation, not matched exactly per realization. Each phase retains all ten training seeds and five splits; phase B adds ten acquisition repetitions. The full ledger has 135,000 procedure evaluations.

All means include abstentions. Pointwise paired intervals over ten seed averages are supplied in the result CSVs; they do not quantify cross-graph uncertainty. Exact small-design checks include 6,246 conditional block assignments with nonzero false-discovery events, and 11,264 unequal-propensity assignments for marginal weighted ranks. These checks support implementation verification and do not replace the argument above.

'''+ '\n\n'.join(tables)
    (D/'allocation_results.tex').write_text(text)
    a=[cell('allocation','amazon','hgb_attributes',m) for m in ('equal_alpha','size_alpha','pooled_stratum')]
    b=[cell('acquisition','tfinance','hgb_graph_features',m,rho=0.,scenario='moderate') for m in ('pooled_reference','pooled_stratum','weighted_bh')]
    main=rf'''\paragraph{{Allocation alone does not remove the power cost.}}
We test two cheaper alternatives to equal splitting: $\alpha_h=\alpha m_h/m$, and one BH over stratum-specific p-values. For the stated random-role design, conditioning on all stratum counts makes the blocks independent; within-block PRDS then implies pooled PRDS (Appendix~\ref{{app:allocation}}). Both alternatives control FDR under this design. Neither dominates in power. In the Amazon 80/20 partition with half revelation, attribute-score power is ${a[0].power:.3f}$ with equal splitting, ${a[1].power:.3f}$ with size allocation and ${a[2].power:.3f}$ with pooled stratified ranks. The full random reference gives $0.779$. Thus low FDP under equal splitting does not establish that its error allocation is the principal cause of lost power.

\paragraph{{When the acquired reference is already biased.}}
We next acquire labels with fixed stratum probabilities, using training labels alone to define exposure. At probabilities $(.5,.1)$ for the lower/upper exposure strata, T-Finance graph-score FDP is ${b[0].fdp:.3f}$ with pooled biased-reference BH, versus ${b[1].fdp:.3f}$ with pooled stratified ranks; powers are ${b[0].power:.3f}$ and ${b[1].power:.3f}$. Known-propensity weighted BH has FDP ${b[2].fdp:.3f}$ and power ${b[2].power:.3f}$, but its marginal-rank justification does not supply a joint BH guarantee. Weighted BY supplies a conservative alternative. Figure~\ref{{fig:allocation}} retains all three acquisition severities. Sparse strata can lose nearly all power, even with pooled testing; a validity remedy is not automatically an efficient one.
'''
    (D/'allocation_main.tex').write_text(main)
    figure=r'''\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{allocation_acquisition.pdf}
\caption{FDP versus power for the prespecified 80/20 partition. Top: uniform acquisition at half revelation, comparing allocation rules; circles/squares denote attribute/graph scorers. Middle and bottom: biased acquisition with attribute and graph scorers. Circles, squares and triangles follow neutral, moderate and severe acquisition bias; lines connect those settings, not a tuned frontier. Dotted lines mark nominal $0.10$. Weighted BH* has no joint guarantee established here. Uniform acquisition is a counterfactual, not an available reference within the biased-design scenario. All methods share scores and test identities; means include abstentions.}
\label{fig:allocation}
\end{figure*}
'''
    (D/'allocation_figure.tex').write_text(figure)
    pairs=[]
    for metric in ('fdp','power'):
        ix=[k for k in group if k!='method']+['seed']
        wide=seed.pivot(index=ix,columns='method',values=metric)
        for method in ('size_alpha','pooled_stratum'):
            diff=(wide[method]-wide.equal_alpha).rename('difference').reset_index()
            v=diff.groupby(ix[:-1]).difference.agg(['mean','std','count']).reset_index()
            v['lower']=v['mean']-t.ppf(.975,9)*v['std']/np.sqrt(10);v['upper']=v['mean']+t.ppf(.975,9)*v['std']/np.sqrt(10)
            v['method']=method;v['metric']=metric;pairs.append(v)
    pd.concat(pairs).to_csv(A/'allocation_paired_intervals.csv',index=False)
    (D/'ALLOCATION_RESULTS.json').write_text(json.dumps({'rows_reaggregated':len(trials),'amazon_attribute_power':{m:float(v.power) for m,v in zip(('equal','size','pooled'),a)},'tfinance_moderate_graph':{m:{'fdp':float(v.fdp),'power':float(v.power)} for m,v in zip(('biased','stratified','weighted_bh'),b)}},indent=2))
    print('Reaggregated',len(trials),'rows; all tables and figure generated')
if __name__=='__main__':main()
