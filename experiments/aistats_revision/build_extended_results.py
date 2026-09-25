"""Report every declared T-Finance and GADBench-Weibo dose outcome."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parent;A=D.parent/'audit_method'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def main():
    summaries={ds:pd.read_csv(A/f'{ds}_dose_summary.csv') for ds in ('tfinance','weibo_gadbench')}
    diagnostics=pd.concat([pd.read_csv(A/f'{ds}_diagnostics.csv') for ds in summaries])
    def cell(ds,model,rho,removed,method):
        x=summaries[ds];v=x[(x.model==model)&(x.label_fraction==rho)&(x.removed==removed)&(x.method==method)]
        assert len(v)==1;return v.iloc[0]
    rows=[];figs=[];record={}
    for ds,title in [('tfinance','T-Finance'),('weibo_gadbench','Weibo (GADBench)')]:
        assert len(summaries[ds])==2*3*9*2
        record[ds]={}
        for model,name in [('hgb_attributes','Attributes'),('hgb_graph_features','Graph features')]:
            diag=diagnostics[(diagnostics.dataset==ds)&(diagnostics.model==model)]
            assert len(diag)==10
            f=cell(ds,model,.5,.2,'exposure');r=cell(ds,model,.5,.2,'random')
            record[ds][model]=dict(auroc=float(diag.auroc.mean()),average_precision=float(diag.average_precision.mean()),exposure_fdp=float(f.fdp_mean),random_fdp=float(r.fdp_mean),exposure_power=float(f.power_mean),random_power=float(r.power_mean))
            rows.append(f'{title} & {name} & ${diag.auroc.mean():.3f}$ & ${diag.average_precision.mean():.3f}$ & ${f.fdp_mean:.3f}/{f.power_mean:.3f}$ & ${r.fdp_mean:.3f}/{r.power_mean:.3f}$ '+r'\\')
        fig,axs=plt.subplots(2,3,figsize=(7.05,4.1),layout='constrained',sharex=True)
        for i,metric in enumerate(('fdp','power')):
            for j,rho in enumerate((.25,.5,1.)):
                ax=axs[i,j]
                for model,color,name in [('hgb_attributes','#b65132','Attributes'),('hgb_graph_features','#246f99','Graph features')]:
                    for method,style in [('exposure','-'),('random','--')]:
                        z=summaries[ds];z=z[(z.model==model)&(z.label_fraction==rho)&(z.method==method)].sort_values('removed')
                        ax.plot(z.removed*100,z[metric+'_mean'],style,color=color,marker='o',ms=2,label=name+' / '+method)
                if metric=='fdp':ax.axhline(.1,color='#555',ls=':',lw=.8)
                ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.set_xticks([0,20,40,60,80,100])
                if i==0:ax.set_title(f'{rho:.0%} revealed')
                if j==0:ax.set_ylabel('Mean '+('FDP' if metric=='fdp' else 'power'))
                if i==1:ax.set_xlabel('Reference normals removed (%)')
        h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=2,frameon=False,fontsize=8)
        fig.savefig(D/f'extension_dose_{ds}.pdf');plt.close(fig)
        figs.append(r'\begin{figure*}[t]\centering'+f'\n\\includegraphics[width=\\textwidth]{{extension_dose_{ds}.pdf}}\n'+rf'\caption{{{title}: complete prespecified removal sweep. Both scorers, all nine removal fractions and all three revelation fractions are shown. Solid lines use exposure removal; dashed lines use equally sized random references. Means retain every abstention and average ten training-seed means.}}'+'\n'+r'\end{figure*}')
    fa=record['tfinance']['hgb_attributes'];fg=record['tfinance']['hgb_graph_features'];wa=record['weibo_gadbench']['hgb_attributes'];wg=record['weibo_gadbench']['hgb_graph_features']
    main=rf'''\paragraph{{An additional benchmark and a source sensitivity.}}
The frozen T-Finance extension uses 39,357 nodes and 1,803 anomalies. At the designated 20\%-removal, half-revelation setting, attribute-score FDP is ${fa['exposure_fdp']:.3f}$ versus ${fa['random_fdp']:.3f}$ for random removal (power ${fa['exposure_power']:.3f}$ versus ${fa['random_power']:.3f}$); graph-feature FDP is ${fg['exposure_fdp']:.3f}$ versus ${fg['random_fdp']:.3f}$. The separately sourced GADBench Weibo file contains 868 anomalies, compared with PyGOD's 347. Under the same designated intervention, GADBench attribute-score FDP is ${wa['exposure_fdp']:.3f}$ versus ${wa['random_fdp']:.3f}$ (power ${wa['exposure_power']:.3f}$ versus ${wa['random_power']:.3f}$). Appendix~\ref{{app:extensions}} reports all settings and both score families; the two Weibo releases are source sensitivities of one graph, not independent graph replications.
'''
    (D/'extended_benchmark_main.tex').write_text(main)
    text=r'''\section{T-FINANCE AND WEIBO SOURCE SENSITIVITY}
\label{app:extensions}
\paragraph{Protocol and acquisition.}
The T-Finance protocol was saved before loading model outcomes. We downloaded the T-Finance and Weibo members of the official GADBench archive~\citep{tang2023gadbench}, verified ZIP CRC and SHA-256, and retained their distributed binary labels and numerical features without relabeling. Raw adjacency is symmetrized, binarized, and stripped of self loops. T-Finance has 39,357 nodes, 10 features, 1,803 anomalous labels and 21,222,543 undirected edges. GADBench Weibo has 8,405 nodes, 400 features, 868 anomalous labels and 377,271 undirected edges. Original GADBench train/validation/test masks are not used; we apply the same uniform 10\% training-panel design as the other controls.

\paragraph{Weibo provenance.}
The actual official PyGOD file has 347 anomalous labels despite its README's stated 868. A fresh download is byte-identical to our raw input, and the cached labels equal those distributed labels. The separately distributed GADBench file has 868 anomalous labels. The two raw feature matrices agree entrywise and their symmetrized, loop-free adjacency matrices agree. Two nodes share identical features and symmetric neighborhoods, so external node identities are not established uniquely from those arrays alone. The existing PyGOD processed cache standardizes features; this source extension preserves GADBench's raw distributed features. We therefore report a source sensitivity, not a claim that every numerical difference isolates label changes alone. PyGOD and GADBench outcomes are never pooled.

The raw GADBench Weibo SHA-256 is
\texttt{111402a2cb3b6a10\allowbreak ec586e96122da809\allowbreak d62bfe3617add07cb\allowbreak 006de53fc109f05};
T-Finance is
\texttt{8f3b05272703220b\allowbreak b3248b0caac92e91\allowbreak 7c7524704094c02f\allowbreak 802b400026ed1d38}.
The machine-readable provenance includes full hashes, upstream commit, download ranges, actual field shapes and processed-graph hashes.

\paragraph{Training and evaluation.}
For each graph, fit ten attribute-only and ten graph-feature histogram-gradient-boosting scorers with the fixed settings in Appendix~\ref{app:supervised}. T-Finance has 3,936 training labels and GADBench Weibo has 840. Training labels never re-enter calibration or testing. All calculations use sparse adjacency and CPU training. Each extension retains five test splits, three label-revelation fractions, nine removal fractions, two matched rules and fifty calibration repetitions: 270,000 evaluations per source. The Weibo sensitivity uses the earlier Weibo RNG namespace, while T-Finance uses a new dataset index; class-stratified test identities can change when source labels differ. All results, including abstentions and unsuccessful contrasts, are retained. Neither score direction nor hyperparameters are selected using these outcomes.

\begin{table*}[t]\centering\small
\caption{Prespecified 20\%-removal comparison with half of non-test labels revealed. FDP/power pairs average all outcomes; ranking metrics average ten training seeds. Full curves follow.}
\begin{tabular}{llrrrr}\toprule Graph & Score & AUROC & AP & Exposure FDP/power & Random FDP/power\\\midrule
'''+ '\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\end{table*}'+'\n\n'+'\n\n'.join(figs)
    (D/'extended_benchmarks.tex').write_text(text)
    (D/'EXTENDED_RESULTS.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
