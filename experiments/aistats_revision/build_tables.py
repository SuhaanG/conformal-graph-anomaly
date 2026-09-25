from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
G=list(csv.DictReader((D/'graph_summary.csv').open()))
S=list(csv.DictReader((D/'synthetic_summary.csv').open()))
order=['amazon','tolokers','weibo','reddit']; models=['dominant_pygod','gae']
def pm(r,k,d=3): return '$'+format(float(r[k+'_mean']),f'.{d}f')+r'\pm'+format(float(r[k+'_sd']),f'.{d}f')+'$'
def cell(g,d,m,s): return next(r for r in g if r['dataset']==d and r['detector']==m and r['strategy']==s)
lines=[]
for d in order:
    for m in models:
        for s in ['clean','random']:
            r=cell(G,d,m,s)
            lines.append(' & '.join([d.capitalize(), 'DOMINANT' if m=='dominant_pygod' else 'GAE', 'Filtered' if s=='clean' else 'Random',str(round(float(r['n_calib_mean']))),pm(r,'fdp'),pm(r,'power'),pm(r,'n_discoveries',1),format(float(r['gamma_t_lo_mean']),'.2f')])+r' \\')
    if d!=order[-1]: lines.append(r'\addlinespace')
(D/'graph_table.tex').write_text(r'\begin{tabular}{lllrrrrr}\toprule Graph & Detector & Rule & $n$ & FDP & Power & Discoveries & $g(t)$\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')
lines=[]
for d in order:
    for m in models:
        r=cell(G,d,m,'random_full')
        lines.append(' & '.join([d.capitalize(),'DOMINANT' if m=='dominant_pygod' else 'GAE',pm(r,'fdp'),pm(r,'power'),pm(r,'n_discoveries',1)])+r' \\')
(D/'larger_table.tex').write_text(r'\begin{tabular}{llrrr}\toprule Graph & Detector & FDP & Power & Discoveries\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')
lines=[]
for r in S:
    label={'random':'Random','selected':'Filtered','weighted_correct':'Correct weights','weighted_legacy':'Fixed test weight','independent_contamination':'Contaminated'}[r['method']]
    lines.append(' & '.join([r['theta'],r['q'],r['epsilon'],label,format(float(r['fdp_mean']),'.4f')+r' $\pm$ '+format(float(r['fdp_mcse']),'.4f'),format(float(r['power_mean']),'.4f'),format(float(r['null_tail_005_mean']),'.4f')])+r' \\')
(D/'simulation_table.tex').write_text(r'\begin{tabular}{rrrlrrr}\toprule $\theta$ & $q(1)$ & $\epsilon$ & Method & FDP $\pm$ MCSE & Power & $\Prb_0(p\le.05)$\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,3,figsize=(6.75,2.2))
qs=[1.,.25,.05,0.]; xx=np.arange(4)
for theta,color,marker in [('0.0','#2878A3','o'),('2.0','#BF5338','s')]:
    rr=[next(r for r in S if r['theta']==theta and float(r['q'])==q and r['method']==('random' if q==1 else 'selected')) for q in qs]
    ax[0].errorbar(xx,[float(r['fdp_mean']) for r in rr],yerr=[1.96*float(r['fdp_mcse']) for r in rr],label=r'$\theta='+theta+'$',color=color,marker=marker,capsize=2,linewidth=1.2)
ax[0].set_xticks(xx,['1','.25','.05','0']);ax[0].set_xlabel('Selection propensity q(1)');ax[0].set_ylabel('Mean FDP');ax[0].legend(frameon=False);ax[0].set_title('(a) Selection and score sensitivity')
methods=['selected','weighted_correct','weighted_legacy']; labs=['Filtered','Correct\nweights','Fixed test\nweight']; colors=['#BF5338','#2E8979','#8062A5']
rr=[next(r for r in S if r['theta']=='2.0' and float(r['q'])==.05 and r['method']==m) for m in methods]
for a,key,title in [(ax[1],'fdp','(b) Weighting: error'),(ax[2],'power','(c) Weighting: power')]:
    a.bar(range(3),[float(r[key+'_mean']) for r in rr],color=colors,width=.64,yerr=[1.96*float(r[key+'_mcse']) for r in rr],capsize=2)
    a.set_xticks(range(3),labs);a.set_title(title);a.set_ylim(0,.85 if key=='power' else .8)
for a in ax[:2]: a.axhline(.1,color='#444444',linestyle='--',linewidth=.8)
fig.tight_layout(pad=.5,w_pad=1.0);fig.savefig(D/'selection_validation.pdf');fig.savefig(D/'selection_validation.png',dpi=180)
print('Tables and figure generated from saved CSVs')
if (D/'graph_rank_aggregate.csv').exists():
    R=list(csv.DictReader((D/'graph_rank_aggregate.csv').open()))
    if all(int(r['training_seeds'])==5 for r in R):
        for filename,methods in [('graph_audit_table.tex',['clean','random']),('graph_audit_full_table.tex',['clean','random','exposed_only','random_full'])]:
            lines=[]
            for r in R:
                if r['method'] not in methods: continue
                label={'clean':'Filtered','random':'Random','exposed_only':'Positive exposure','random_full':'Larger random'}[r['method']]
                values=[]
                for k in ('fdp','power','discoveries'):
                    d=1 if k=='discoveries' else 3
                    values.append('$'+format(float(r[k+'_mean']),f'.{d}f')+r'\pm'+format(float(r[k+'_training_sd']),f'.{d}f')+'$')
                lines.append(' & '.join([str(round(100*float(r['trim'])))+r'\%',label]+values+[format(float(r['tail001_exact']),'.4f')])+r' \\')
            text=r'\begin{tabular}{llrrrr}\toprule Trim & Rule & FDP & Power & Discoveries & $a(0.01)$\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n'
            (D/filename).write_text(text)
        print('Five-seed conditional graph audit tables generated')
    else: print('Graph audit tables deferred: incomplete training seeds')
