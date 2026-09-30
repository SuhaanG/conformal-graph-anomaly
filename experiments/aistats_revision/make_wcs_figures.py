"""Figures and tables for the WCS results, generated directly from audit_method CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;A=HERE.parent/'audit_method'
GRAPHS=[('amazon','Amazon'),('tfinance','T-Finance'),('weibo_gadbench','Weibo (GADBench)')]
SCORERS=[('hgb_graph_features','graph-feature scorer'),('hgb_attributes','attribute scorer')]
SCEN=[('moderate','Moderate'),('severe','Severe'),('smooth','Smooth')]
# Fixed categorical order (validated palette); marker shape is the secondary encoding.
METHODS=[('wcs_rand_homogeneous','Randomized WCS','#2a78d6','o'),
         ('pooled_reference','Pooled BH (biased labels)','#eb6834','s'),
         ('wcs_homogeneous','WCS (deterministic p)','#1baf7a','^'),
         ('weighted_bh','Weighted BH, det. p (no guarantee)','#eda100','D'),
         ('weighted_bh_rand','Weighted BH, rand. p (no guarantee)','#8a5cc2','v')]
INK,MUTED,GRID='#0b0b0b','#52514e','#e4e3df'
T9=2.262157  # t quantile 0.975 with 9 df: ten training seeds are the independent units

def figure(scorer,label,out):
    s=pd.read_csv(A/'wcs_endtoend_summary.csv');s=s[s.model==scorer]
    fig,axes=plt.subplots(2,3,figsize=(10,5.2),sharex=True,sharey='row')
    for c,(ds,name) in enumerate(GRAPHS):
        d=s[s.dataset==ds]
        for r,metric in enumerate(('fdp','power')):
            ax=axes[r,c]
            for k,(m,mlab,col,mk) in enumerate(METHODS):
                x=np.arange(3)+(k-2)*0.14
                sub=d[d.method==m].set_index('scenario').loc[[sc for sc,_ in SCEN]]
                yv=sub[f'{metric}_mean'].values
                err=T9*sub['fdp_std'].values/np.sqrt(10) if metric=='fdp' else None  # seed-clustered 95% t interval
                ax.errorbar(x,yv,yerr=err,fmt=mk,ms=6,color=col,mec='white',mew=.8,elinewidth=1.2,capsize=0,label=mlab,zorder=3)
            if metric=='fdp':
                b=d.bound_mean.mean();ax.axhline(b,color=MUTED,lw=1,ls='--',zorder=1)
                ax.set_ylim(0,.23)
            else:ax.set_ylim(0,1)
            ax.set_xticks(range(3));ax.set_xticklabels([t for _,t in SCEN])
            ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True)
            for sp in ('top','right'):ax.spines[sp].set_visible(False)
            for sp in ('left','bottom'):ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=MUTED,labelsize=8.5)
            if r==0:ax.set_title(name,fontsize=10,color=INK)
            if c==0:ax.set_ylabel('Mean FDP' if metric=='fdp' else 'Power',fontsize=9,color=INK)
    h,l=axes[0,0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([],[],color=MUTED,lw=1,ls="--"));l.append("FDR bound α·m₀/m")
    fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=8.5,bbox_to_anchor=(.5,-.02))
    fig.tight_layout(rect=(0,.1,1,1));fig.savefig(out,bbox_inches='tight');fig.savefig(out.with_suffix('.png'),dpi=160,bbox_inches='tight');plt.close(fig)

def fmt(r):return f"{r['fdp_mean']:.3f} / {r['power_mean']:.3f}"

def endtoend_table(out):
    s=pd.read_csv(A/'wcs_endtoend_summary.csv').set_index(['dataset','model','scenario','method'])
    lines=[r'\begin{tabular}{lllccccc}',r'\toprule',
           r'Graph & Scorer & Bias & Pooled BH & W.\ BH, det.\ $p^\dagger$ & W.\ BH, rand.\ $p^\dagger$ & WCS, det.\ $p$ & WCS, rand.\ $p$\\',r'\midrule']
    for ds,name in GRAPHS:
        for sc,slab in SCORERS:
            for scen,sl in SCEN:
                cells=[fmt(s.loc[(ds,sc,scen,m)]) for m in ('pooled_reference','weighted_bh','weighted_bh_rand','wcs_homogeneous','wcs_rand_homogeneous')]
                lines.append(f"{name} & {'Graph' if 'graph' in sc else 'Attr.'} & {sl} & "+' & '.join(cells)+r'\\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}');out.write_text('\n'.join(lines)+'\n')

def estimated_table(out):
    s=pd.read_csv(A/'wcs_estimated_summary.csv').set_index(['dataset','model','method'])
    lines=[r'\begin{tabular}{llcccc}',r'\toprule',
           r'Graph & Scorer & Pooled BH & WCS, true $\rho$ & WCS, estimated $\rho$ & WCS, 2-stratum $\hat\rho$\\',r'\midrule']
    for ds,name in GRAPHS:
        for sc,slab in SCORERS:
            cells=[fmt(s.loc[(ds,sc,m)]) for m in ('pooled_reference','wcs_rand_oracle','wcs_rand_logit','wcs_rand_strata2')]
            lines.append(f"{name} & {'Graph' if 'graph' in sc else 'Attr.'} & "+' & '.join(cells)+r'\\')
    lines+= [r'\bottomrule',r'\end{tabular}'];out.write_text('\n'.join(lines)+'\n')

def resolution_table(out):
    s=pd.read_csv(A/'wcs_randomized_summary.csv')
    s=s[(s.design=='exposure_80')&(s.model=='hgb_graph_features')].set_index(['dataset','scenario','method'])
    lines=[r'\begin{tabular}{llccccc}',r'\toprule',
           r'Graph & Bias & Kish ESS & $mf/\alpha$ & Anomaly share & WCS (det.\ $p$) & Randomized WCS\\',r'\midrule']
    for ds,name in GRAPHS:
        for scen,sl in [('neutral','None'),('moderate','Moderate'),('severe','Severe')]:
            r=s.loc[(ds,scen,'wcs_homogeneous')];q=s.loc[(ds,scen,'wcs_rand_homogeneous')]
            lines.append(f"{name} & {sl} & {r['kish_ess_mean']:.0f} & {r['need_anom_high_mean']:.0f} & {r['frac_anom_high_mean']:.2f} & {r['power_mean']:.3f} & {q['power_mean']:.3f}"+r'\\')
    lines+= [r'\bottomrule',r'\end{tabular}'];out.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    figure('hgb_graph_features','graph-feature scorer',HERE/'wcs_endtoend_figure.pdf')
    figure('hgb_attributes','attribute scorer',HERE/'wcs_endtoend_figure_attributes.pdf')
    endtoend_table(HERE/'wcs_endtoend_table.tex');estimated_table(HERE/'wcs_estimated_table.tex');resolution_table(HERE/'wcs_resolution_table.tex')
    print('written')
