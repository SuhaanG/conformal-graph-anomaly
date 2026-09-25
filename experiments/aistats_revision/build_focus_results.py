"""Generate submission figures and prose directly from complete sensitivity outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;A=D.parent/'audit_method'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
COLORS={'filtered':'#c85c36','matched_random':'#287ca0','full_reference':'#555a64'}

def main():
    x=pd.read_csv(A/'selection_dose_summary.csv')
    assert len(x)==3*2*3*9*2 and set(x.dataset)=={'amazon','tolokers','weibo'}
    def cell(ds,model,rho,drop,method):
        z=x[(x.dataset==ds)&(x.model==model)&(x.label_fraction==rho)&(x.removed==drop)&(x.method==method)]
        assert len(z)==1;return z.iloc[0]
    a=cell('amazon','hgb_attributes',.5,.2,'exposure');b=cell('amazon','hgb_attributes',.5,.2,'random')
    ag=cell('amazon','hgb_graph_features',.5,.2,'exposure');bg=cell('amazon','hgb_graph_features',.5,.2,'random')
    (D/'dose_abstract.tex').write_text(f'With only half of non-test labels available, removing the highest-exposure 20\\% of reference normals gives FDP ${a.fdp_mean:.3f}$ versus ${b.fdp_mean:.3f}$ under random removal, while both retain power above $0.77$.')
    # Existing primary contrasts, showing useful and weak-score regimes together.
    sup=pd.read_csv(A/'strong_baseline_summary.csv');unsup=pd.read_csv(D/'followup_aggregate.csv')
    unsup=unsup[(unsup.detector=='dominant_pygod')&(unsup.design=='primary')&(unsup.label_fraction==1)&(unsup.trim==0)&(unsup.budget==1000)]
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.5),layout='constrained')
    labels=['Amazon\nAttributes','Amazon\nGraph features','Amazon\nDOMINANT','Tolokers\nDOMINANT'];pos=np.arange(4);width=.23
    for ax,metric,title in zip(axs,('fdp','power'),('Mean FDP','Mean power')):
        for j,method in enumerate(('filtered','matched_random','full_reference')):
            means=[];sds=[]
            for model in ('hgb_attributes','hgb_graph_features'):
                r=sup[(sup.dataset=='amazon')&(sup.model==model)&(sup.method==method)].iloc[0];means.append(r[metric]);sds.append(r[metric+'_seed_sd'])
            for ds in ('amazon','tolokers'):
                if method=='full_reference':means.append(0.);sds.append(0.)
                else:
                    r=unsup[(unsup.dataset==ds)&(unsup.method==('random' if method=='matched_random' else method))]
                    assert len(r)==1,(ds,method,r)
                    r=r.iloc[0];means.append(r[metric+'_mean']);sds.append(r[metric+'_mean_seed_sd'])
            ax.bar(pos+(j-1)*width,means,width,color=COLORS[method],label={'filtered':'Zero exposure','matched_random':'Matched random','full_reference':'Full reference'}[method],yerr=sds,capsize=2,error_kw={'linewidth':.8})
        ax.set_xticks(pos,labels,fontsize=7.8);ax.set_ylim(0,1.04);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if metric=='fdp':ax.axhline(.1,color='#555',ls='--',lw=1)
    handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='outside upper center',ncol=3,frameon=False,fontsize=8)
    fig.savefig(D/'headline_results.pdf');plt.close(fig)
    # Main figure: fixed half-revelation setting, all removal fractions.
    fig,axs=plt.subplots(1,2,figsize=(7.05,2.45),layout='constrained')
    for ax,metric,title in zip(axs,('fdp','power'),('Amazon: mean FDP','Amazon: mean power')):
        for model,color,name in [('hgb_attributes','#b65132','Attributes'),('hgb_graph_features','#246f99','Graph features')]:
            for method,style in [('exposure','-'),('random','--')]:
                z=x[(x.dataset=='amazon')&(x.model==model)&(x.label_fraction==.5)&(x.method==method)].sort_values('removed')
                xx=100*z.removed.to_numpy();yy=z[metric+'_mean'].to_numpy();sd=z[metric+'_std'].to_numpy()
                ax.plot(xx,yy,style,marker='o' if method=='exposure' else '.',ms=3,color=color,label=name+(' / exposure' if method=='exposure' else ' / random'))
                if method=='exposure':ax.fill_between(xx,np.maximum(0,yy-sd),np.minimum(1,yy+sd),color=color,alpha=.12,lw=0)
        ax.set_title(title);ax.set_xlabel('Reference normals removed (%)');ax.set_xlim(-2,100);ax.set_xticks([0,20,40,60,80,100]);ax.set_ylim(0,.24 if metric=='fdp' else .92);ax.grid(alpha=.16)
        if metric=='fdp':ax.axhline(.1,color='#555',ls=':',lw=1)
    handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='outside upper center',ncol=2,frameon=False,fontsize=8)
    fig.savefig(D/'selection_dose.pdf');plt.close(fig)
    # Complete grid, including unfavorable score regimes and every label budget.
    for metric in ('fdp','power'):
        fig,axs=plt.subplots(3,3,figsize=(7.1,6.5),layout='constrained',sharex=True)
        for i,ds in enumerate(('amazon','tolokers','weibo')):
            for j,rho in enumerate((.25,.5,1.)):
                ax=axs[i,j]
                for model,color in [('hgb_attributes','#b65132'),('hgb_graph_features','#246f99')]:
                    for method,style in [('exposure','-'),('random','--')]:
                        z=x[(x.dataset==ds)&(x.model==model)&(x.label_fraction==rho)&(x.method==method)].sort_values('removed')
                        ax.plot(100*z.removed,z[metric+'_mean'],style,color=color,marker='o',ms=2,lw=1.1)
                ax.set_title(f'{ds.title()}, {rho:.0%} revealed',fontsize=9);ax.grid(alpha=.16);ax.set_ylim(bottom=0)
                if metric=='fdp':ax.axhline(.1,color='#555',ls=':',lw=.8);ax.set_ylim(0,max(.22,ax.get_ylim()[1]))
                if i==2:ax.set_xlabel('Removed (%)')
                if j==0:ax.set_ylabel('Mean '+metric.upper())
        fig.savefig(D/f'selection_dose_all_{metric}.pdf');plt.close(fig)
    # Main results generated from stored summaries, with negative mild-selection results.
    five=cell('amazon','hgb_attributes',.5,.05,'exposure');ten=cell('amazon','hgb_attributes',.5,.1,'exposure')
    w=x[x.dataset=='weibo'];t=x[x.dataset=='tolokers']
    we=cell('weibo','hgb_attributes',.5,.05,'exposure');wr=cell('weibo','hgb_attributes',.5,.05,'random')
    text=rf'''\paragraph{{Moderate observed-exposure removal changes useful discoveries.}}
The graded sweep addresses the severity of zero-exposure filtering (Figure~\ref{{fig:dose}}). With 50\% of non-test labels revealed, removing the highest-exposure 20\% of Amazon reference normals gives attribute-score FDP ${a.fdp_mean:.3f}$ versus ${b.fdp_mean:.3f}$ for matched random removal; powers are ${a.power_mean:.3f}$ and ${b.power_mean:.3f}$. The graph-feature scorer gives FDP ${ag.fdp_mean:.3f}$ versus ${bg.fdp_mean:.3f}$, with powers ${ag.power_mean:.3f}$ and ${bg.power_mean:.3f}$. The paired FDP increase is positive in all ten training seeds for both scorers. Mean reference size remains {a.n_calib_mean:.0f}, so these contrasts do not require discarding nearly all verified normals or forcing a tiny calibration budget.

The dose response is not uniformly harmful or monotone. At 5\% removal, the Amazon attribute model's mean FDP remains ${five.fdp_mean:.3f}$; at 10\%, it is ${ten.fdp_mean:.3f}$. At 97\%, power falls sharply as reference resolution deteriorates. Tolokers power is at most ${t.power_mean.max():.3f}$ across the complete sweep.

Weibo supplies a distinct selection-sensitive regime. Its attribute scorer has AUROC $0.929$ but average precision only $0.292$. With half of non-test labels revealed, removing just 5\% by exposure gives FDP ${we.fdp_mean:.3f}$ versus ${wr.fdp_mean:.3f}$ for random removal, with powers ${we.power_mean:.3f}$ and ${wr.power_mean:.3f}$. This replicates selection sensitivity under mild removal, but its nearly powerless random baseline does not replicate Amazon's useful calibrated regime. Appendix~\ref{{app:dose}} reports every graph, scorer, label budget, and removal fraction.

\begin{{figure*}}[t]
\centering\includegraphics[width=\textwidth]{{selection_dose.pdf}}
\caption{{Observed-exposure removal with half of non-test labels revealed. Scores and test identities are fixed within paired comparisons, and both rules use identical remaining reference sizes. Solid lines remove high-exposure normals; dashed lines remove normals uniformly. Shading is one SD of ten training-seed averages for the exposure arm; it does not measure new-graph uncertainty. Means include every zero-discovery outcome.}}
\label{{fig:dose}}
\end{{figure*}}
'''
    (D/'selection_dose_main.tex').write_text(text)
    diagnostics=pd.concat([pd.read_csv(A/'strong_diagnostics.csv').drop_duplicates(['dataset','model','seed']),pd.read_csv(A/'weibo_control_diagnostics.csv')])
    diagnostic=diagnostics.groupby(['dataset','model'])[['auroc','average_precision']].agg(['mean','std'])
    rows=[]
    for (ds,model),r in diagnostic.iterrows():
        rows.append(f"{ds.title()} & {'Attributes' if model=='hgb_attributes' else 'Attributes + graph'} & ${r['auroc','mean']:.3f}\\pm{r['auroc','std']:.3f}$ & ${r['average_precision','mean']:.3f}\\pm{r['average_precision','std']:.3f}$ \\\\")
    appendix=r'''\section{OBSERVED-EXPOSURE REMOVAL SENSITIVITY}
\label{app:dose}
This exploratory follow-up was specified after external feedback on the manuscript. The Amazon/Tolokers grid was fixed before its outcomes were inspected; the Weibo extension was specified after those two-graph results, before fitting its supervised models. We retain both scorers and every condition. Weibo uses the official file described in Appendix~\ref{app:fresh}, the same fixed classifier settings as Appendix~\ref{app:supervised}, and a uniform 10\% training panel (840 labels), with ten training seeds and five test splits per seed. Its graph features use the same unweighted neighbor means and log degree. Training runs use one CPU thread; no Weibo outcome was used to tune model settings.

\begin{table}[ht]\centering\small
\caption{Ranking quality on all nontraining nodes: mean $\pm$ SD across ten training seeds. Average precision is reported alongside AUROC because discovery uses the upper score tail.}
\begin{tabular}{llrr}\toprule Graph & Score & AUROC & Average precision\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{table}

For each test split, independently reveal each non-test deployment label with probability $\rho\in\{0.25,0.50,1\}$, using nested draws. Exposure is the number of neighbors known anomalous from training or revealed non-test labels, divided by graph degree (zero for isolated nodes). Only revealed normal deployment nodes are eligible calibration nodes. At removal fraction $d$, retain $\lceil(1-d)M\rceil$ of these $M$ nodes. The exposure rule retains the smallest exposures, using a fresh uniform tie ordering each repetition; the comparator retains a uniform subset of identical size. Both orderings are shared across scorers and removal fractions within a repetition. Independent node-level uniform marks resolve score ties lexicographically once per scorer before the evaluation splits; they never reverse a strict score order.

The nine removal fractions are $0,.05,.10,.20,.40,.60,.80,.90,.97$. Each graph--scorer--seed--split--revelation--fraction--rule has 50 repetitions, giving 810,000 evaluations across three graphs. Repetitions and splits are averaged within training seed before computing the reported means and seed SDs. At zero removal the two rules are identical. Unlike the original zero-exposure comparison, no fixed 1,000-node cap restricts the remaining reference. The 100\% revelation condition still excludes every test label and is not the oracle zero-exposure intervention.

All 450 graph--seed--split--revelation checks leave exposure unchanged when every unknown label is flipped. Disjointness, normal-only references, matched sample sizes, and identical zero-removal outcomes are asserted in the run. One hundred random tied-score examples check the optimized score-ordered BH implementation against the independent reference implementation after applying the same tie convention. Saved protocols, scores, complete compressed trials, split summaries, seed summaries, and hashes support reproduction without GPU training. Figures~\ref{fig:doseallfdp} and~\ref{fig:doseallpower} report the full grid. The tables below isolate the prespecified mild-removal fractions; remaining numerical cells, including their SDs and discovery counts, are in the complete summary CSV.
'''
    for ds in ('amazon','tolokers','weibo'):
        appendix+=rf'''\begin{{table}}[ht]\centering\small
\caption{{{ds.title()}: all mild-removal outcomes, mean FDP / power. Each pair uses the same remaining reference size; $\rho$ is the revealed non-test label fraction.}}
\begin{{tabular}}{{llrrr}}\toprule Score & $\rho$ & Removed & Exposure & Random\\\midrule
'''
        for model in ('hgb_attributes','hgb_graph_features'):
            for rho in (.25,.5,1.):
                for drop in (.05,.1,.2):
                    e=cell(ds,model,rho,drop,'exposure');r=cell(ds,model,rho,drop,'random')
                    appendix+=f"{'Attributes' if model=='hgb_attributes' else 'Attributes + graph'} & {rho:.2f} & {drop:.2f} & ${e.fdp_mean:.3f}/{e.power_mean:.3f}$ & ${r.fdp_mean:.3f}/{r.power_mean:.3f}$ \\\\\n"
        appendix+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    for metric in ('fdp','power'):
        appendix+=rf'''\clearpage\begin{{figure}}[p]\centering
\includegraphics[width=.96\textwidth]{{selection_dose_all_{metric}.pdf}}
\caption{{Complete removal grid: mean {metric.upper()} over training-seed averages, including abstention. Orange is attributes; blue adds graph features. Solid lines remove by observed exposure; dashed lines remove uniformly. Every cell uses ten training seeds, five splits, and 50 paired repetitions. Different graphs may have different vertical scales.}}
\label{{fig:doseall{metric}}}\end{{figure}}
'''
    (D/'selection_dose_appendix.tex').write_text(appendix)
    result=dict(amazon_half_labels_drop20_attributes={k:float(a[k]) for k in ('fdp_mean','power_mean','n_calib_mean')},matched_random={k:float(b[k]) for k in ('fdp_mean','power_mean')},graph_features={k:float(ag[k]) for k in ('fdp_mean','power_mean')},graph_random={k:float(bg[k]) for k in ('fdp_mean','power_mean')},weibo_max_power=float(w.power_mean.max()),tolokers_max_power=float(t.power_mean.max()))
    (D/'FOCUS_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
