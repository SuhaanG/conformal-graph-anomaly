"""Simplify the display of completed results without changing experiments."""
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

def main():
    seeds=pd.concat([pd.read_csv(A/n) for n in ('selection_dose_seeds.csv','tfinance_dose_seeds.csv','weibo_gadbench_dose_seeds.csv')])
    z=seeds[(seeds.dataset.isin(NAMES))&(seeds.label_fraction==.5)&(seeds.removed==.2)]
    rows=[];report=[]
    for ds in NAMES:
        for model,label in [('hgb_attributes','Attributes'),('hgb_graph_features','Graph features')]:
            v=z[(z.dataset==ds)&(z.model==model)]
            assert len(v)==20 and not v.duplicated(['method','seed']).any()
            means=v.groupby('method')[['fdp','power']].mean();e=means.loc['exposure'];r=means.loc['random']
            paired=v.pivot(index='seed',columns='method',values='fdp');delta=paired.exposure-paired.random
            half=t.ppf(.975,9)*delta.std(ddof=1)/np.sqrt(10)
            rows.append(f'{NAMES[ds]} & {label} & ${e.fdp:.3f}/{e.power:.3f}$ & ${r.fdp:.3f}/{r.power:.3f}$ & $+{100*delta.mean():.1f}\\ [{100*(delta.mean()-half):.1f},{100*(delta.mean()+half):.1f}]$'+r' \\')
            report.append(dict(dataset=ds,model=model,exposure=e.to_dict(),random=r.to_dict(),fdp_difference=delta.mean(),lower=delta.mean()-half,upper=delta.mean()+half))
    table=r'''\begin{table*}[t]
\centering\small\setlength{\tabcolsep}{5pt}
\caption{Moderate reference selection with useful scorers on three graphs. Half of non-test labels are revealed; the highest-exposure 20\% of reference normals are removed, or the same count is removed randomly. Scores, test identities and remaining reference sizes match within each pair. Entries give mean FDP / power at nominal $0.10$. The final column gives the paired FDP increase in percentage points with a pointwise 95\% interval over ten seed averages, conditional on each graph. All abstentions are included.}
\label{tab:threegraphs}
\begin{tabular}{llrrl}\toprule
Graph & Score & Exposure & Random & $\Delta$ FDP (pp) [95\% interval]\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\end{table*}
'''
    (D/'three_graph_results.tex').write_text(table)
    z=pd.read_csv(A/'allocation_acquisition_seeds.csv')
    z=z[(z.phase=='acquisition')&(z.design=='exposure_80')&(z.scenario=='moderate')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.8),layout='constrained',sharex=True,sharey=True)
    methods=[('pooled_reference','Pooled biased reference','#b34b39'),('pooled_stratum','Pooled stratified ranks','#167a86'),('uniform_counterfactual','Uniform acquisition (counterfactual)','#497abb')]
    for ax,ds in zip(axs,NAMES):
        for method,label,color in methods:
            for model,marker in [('hgb_attributes','o'),('hgb_graph_features','s')]:
                v=z[(z.dataset==ds)&(z.model==model)&(z.method==method)]
                assert len(v)==10
                ax.scatter(v.power.mean(),v.fdp.mean(),marker=marker,c=color,s=38,zorder=3)
        ax.axhline(.1,c='#444',ls=':',lw=1)
        ax.set_title(NAMES[ds],fontsize=10);ax.set_xlim(-.03,1.02);ax.set_ylim(-.005,.18)
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.05,.1,.15]);ax.grid(alpha=.16);ax.set_xlabel('Power')
    axs[0].set_ylabel('Mean FDP')
    handles=[plt.Line2D([0],[0],color=c,marker='o',ls='',label=label,ms=5) for _,label,c in methods]
    fig.legend(handles=handles,loc='outside lower center',ncol=1,frameon=False,fontsize=8)
    fig.savefig(D/'acquisition_main.pdf');plt.close(fig)
    figure=r'''\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{acquisition_main.pdf}
\caption{Correction can reduce FDP while sacrificing useful discoveries. Training labels define the 80/20 exposure strata; acquisition probabilities are $0.50$ and $0.10$ in the lower and upper strata. Circles denote attribute scorers and squares graph-feature scorers. All three graphs are shown, including Weibo's nearly powerless stratified result. The blue reference is a counterfactual uniform-acquisition design at comparable expected label cost, not an available reference in the biased scenario. Points average ten seed means; the dotted line marks nominal $0.10$. Figure~\ref{fig:allocationfull} and Appendix~\ref{app:allocation} retain every allocation, weighting and acquisition-severity comparison.}
\label{fig:allocation}
\end{figure*}
'''
    (D/'allocation_figure.tex').write_text(figure)
    (D/'EDITORIAL_DISPLAY_VALIDATION.json').write_text(json.dumps(dict(source='unchanged stored seed outcomes',table_rows=report,figure='All three declared graphs, both scorers, moderate biased acquisition; complete original figure retained in appendix'),indent=2))
    print('Verified six table rows from paired seeds; generated three-panel acquisition display')
if __name__=='__main__':main()
