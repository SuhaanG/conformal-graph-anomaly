"""Integrate the fixed CPU follow-up; generate all new tables from summaries."""
from pathlib import Path
import pandas as pd
D=Path(__file__).resolve().parent
A=D.parent/'audit_method'

def pm(row,key): return f'${row[key]:.3f}\\pm{row[key+"_seed_sd"]:.3f}$'
def modelname(s):return {'hgb_attributes':'Attributes','hgb_graph_features':'Attributes + graph'}[s]
def dsname(s):return {'amazon':'Amazon','tolokers':'Tolokers'}[s]
def methodname(s):return {'filtered':'Filtered','matched_random':'Matched random','full_reference':'Full reference','uniform_candidates':'Candidate uniform','score_strata':'Score strata','score_degree_strata':'Score + degree strata','uniform_graph':'Graph uniform'}[s]

def build():
    b=pd.read_csv(A/'strong_baseline_summary.csv');p=pd.read_csv(A/'partial_label_summary.csv')
    c=pd.read_csv(A/'strong_certificate_summary.csv');d=pd.read_csv(A/'strong_detector_summary.csv')
    out=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Graph & Supervised score & AUROC & Filtered & Matched random & Full reference \\',r'\midrule']
    for ds in ('amazon','tolokers'):
        for model in ('hgb_attributes','hgb_graph_features'):
            z=b[(b.dataset==ds)&(b.model==model)].set_index('method');r=d[(d.dataset==ds)&(d.model==model)].iloc[0]
            vals=[f'${z.loc[m,"fdp"]:.3f}/{z.loc[m,"power"]:.3f}$' for m in ('filtered','matched_random','full_reference')]
            out.append(' & '.join([dsname(ds),modelname(model),f'${r.auroc:.3f}$']+vals)+r' \\')
    out += [r'\bottomrule',r'\end{tabular}']
    (D/'supervised_table.tex').write_text('\n'.join(out)+'\n')
    out=[r'\begin{tabular}{lllrrrr}',r'\toprule',r'Graph & Score & Reference & $n$ & FDP & Power & Discoveries \\',r'\midrule']
    for _,r in b.iterrows():
        out.append(' & '.join([dsname(r.dataset),modelname(r.model),methodname(r.method),f'{r.n_calib:.1f}',pm(r,'fdp'),pm(r,'power'),f'{r.discoveries:.1f}'])+r' \\')
    out += [r'\bottomrule',r'\end{tabular}']
    (D/'supervised_full_table.tex').write_text('\n'.join(out)+'\n')
    out=[r'\begin{tabular}{llrlrrr}',r'\toprule',r'Graph & Score & $\rho$ & Reference & FDP & Power & Labels \\',r'\midrule']
    for _,r in p.iterrows():
        out.append(' & '.join([dsname(r.dataset),modelname(r.model),f'{r.label_fraction:.2f}',methodname(r.method),pm(r,'fdp'),pm(r,'power'),f'{r.reference_labels:.0f}'])+r' \\')
    out += [r'\bottomrule',r'\end{tabular}']
    (D/'supervised_partial_table.tex').write_text('\n'.join(out)+'\n')
    out=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Graph & Score & \multicolumn{3}{c}{Candidate uniform} & Score strata & Score + degree & Graph uniform \\',r' & & $B=100$ & $B=250$ & $B=500$ & $B=500$ & $B=500$ & $B=500$ \\',r'\midrule']
    for ds in ('amazon','tolokers'):
        for model in ('hgb_attributes','hgb_graph_features'):
            z=c[(c.dataset==ds)&(c.model==model)].set_index(['method','budget'])
            vals=[f'{z.loc[("uniform_candidates",n),"true_discoveries"]:.1f}' for n in (100,250,500)]
            vals += [f'{z.loc[(m,500),"true_discoveries"]:.1f}' for m in ('score_strata','score_degree_strata','uniform_graph')]
            out.append(' & '.join([dsname(ds),modelname(model)]+vals)+r' \\')
    out += [r'\bottomrule',r'\end{tabular}']
    (D/'certificate_table.tex').write_text('\n'.join(out)+'\n')

def integrate():
    build();path=D/'aistats.tex';s=path.read_text()
    if '% CPU AUDIT EXTENSION INTEGRATED' in s:return
    backup=D/'aistats_before_audit.tex'
    if not backup.exists():backup.write_text(s)
    start=s.index('\\begin{abstract}');end=s.index('\\end{abstract}',start)
    s=s[:start]+r'''\begin{abstract}
The sampling rule used to choose normal calibration nodes can determine whether conformal graph anomaly detection yields reliable discoveries. We isolate this choice by holding trained scores, test identities, and reference size fixed. Without score trimming, excluding calibration nodes with anomalous neighbors gives mean false discovery proportions of $0.936$ on Amazon and $0.740$ on Tolokers at nominal $0.10$ with DOMINANT. The failure also occurs with accurate scores: a supervised Amazon detector with AUROC $0.969$ gives mean FDP $0.192$ under filtering versus $0.067$ under matched random calibration. Its full unfiltered reference retains power $0.783$ with mean FDP $0.092$. Partial-label experiments show that the effect can persist without using test labels for filtering. An exact conditional rank audit quantifies the reference shift on fixed graphs, while 2,500 independently generated graphs separate score sensitivity, neighborhood dependence, and discovery resolution. Together, these results show why normal-only calibration is insufficient and identify a practical comparison: evaluate reference selection alongside detector accuracy, reporting both false discoveries and retained power.
'''+s[end:]
    start=s.index('Three findings organize the paper.');end=s.index('\\paragraph{Relation to prior work.}',start)
    s=s[:start]+r'''Our study makes three contributions. First, matched graph experiments show that neighborhood-based calibration selection can inflate false discoveries without score trimming or access to test labels in the filter. The effect occurs for DOMINANT and for an accurate supervised Amazon detector, so it cannot be dismissed solely as a consequence of poor anomaly ranking. Second, an exact finite-population calculation measures the change in null ranks conditional on the graph and scores, without assuming independent nodes. Third, experiments on 2,500 independent graphs isolate degree sensitivity as one sufficient mechanism and separate rank distortion from limited discovery resolution.

These findings also identify a useful operating regime. For the accurate Amazon detector, using the full unfiltered reference retains substantial power with mean FDP below nominal in the experiment. For weaker scores, increasing reference size alone does not recover discoveries. Calibration quality and detector quality therefore need to be assessed together. We additionally examine weighted conformal selection under its independent-observation assumptions and a direct fixed-batch certification procedure in the supplement.

'''+s[end:]
    anchor=r'\paragraph{Untrimmed test and reference populations.}'
    method=r'''\paragraph{Supervised controls.}
An exploratory CPU follow-up tests whether selection also matters for accurate scores. We train two fixed histogram-gradient-boosting classifiers on a uniform random 10\% panel of verified nodes: attributes alone, or attributes augmented with neighbor-average attributes and log degree. The other 90\% supplies disjoint calibration and test nodes. Ten training seeds each use five stratified 25\% test splits, shared between the two scorers. This study uses additional supervision and is reported separately from the unsupervised replication. Appendix~\ref{app:supervised} gives settings, label costs, and complete results.

'''
    s=s.replace(anchor,method+anchor)
    anchor=r'\paragraph{Trimming and prevalence change the magnitude.}'
    result=r'''\begin{table*}[t]
\centering\small
\caption{Supervised controls at nominal $0.10$: entries are mean FDP / power, retaining zero-discovery runs. Ten training panels and five test splits per panel; filtered and random references are matched in size. The full reference uses all eligible normals and is not matched in label cost to audit certification. Seed SDs, reference sizes, and partial-label results appear in Appendix~\ref{app:supervised}.}
\label{tab:supervised}\input{supervised_table.tex}
\end{table*}

\paragraph{Accurate scores also exhibit selection sensitivity.}
Amazon's attribute-only supervised control has AUROC $0.969$, yet filtered calibration gives mean FDP $0.192$ versus $0.067$ under matched random calibration (Table~\ref{tab:supervised}). Its full reference gives FDP $0.092$ and power $0.783$, demonstrating useful discoveries in this score regime. The graph-feature control shows the same ordering. With only 50\% of non-test deployment labels revealed, plus the already-known training labels, the attribute model gives filtered FDP $0.140$ versus matched $0.069$. At 25\% revelation, both means remain below $0.10$. Tolokers retains low power with these controls. Selection sensitivity thus extends beyond poorly ranking detectors, but its magnitude depends on the score and available labels.

\paragraph{A larger reference is not sufficient by itself.}
Using every eligible non-test normal gives no discoveries for the original DOMINANT, GAE, and Isolation Forest scores on either graph. Dividing DOMINANT scores by $\log(1+d+10^{-8})$ lowers Amazon's filtered mean FDP to $0.165$, but matched random power remains negligible. The successful full-reference supervised result therefore reflects a favorable score regime, not reference size alone (Appendix~\ref{app:supervised}).

'''
    s=s.replace(anchor,result+anchor)
    s=s.replace('Several detectors rank these benchmark labels poorly.', 'Several unsupervised detectors rank these benchmark labels poorly; the accurate supervised controls require a separate labeled training panel and were added as an exploratory follow-up. Their results on one favorable graph do not establish a general correction.')
    s=s.replace('Mean score gaps, random sampling alone, or an observed mean FDP below nominal cannot certify graph FDR control.', 'The full-reference supervised comparison shows that useful power can coexist with low empirical FDP. Mean score gaps, random sampling alone, or a mean FDP below nominal nevertheless do not certify graph FDR control. A separate fixed-batch label audit offers a high-probability FDP certificate when its random sampling assumptions hold (Appendix~\\ref{app:certificate}).')
    s=s.replace(r'\end{document}',r'\clearpage\input{audit_extension.tex}'+'\n'+r'\end{document}')
    path.write_text('% CPU AUDIT EXTENSION INTEGRATED\n'+s)

if __name__=='__main__':
    integrate()
    if (D/'finalize_2027.py').exists():
        import runpy
        runpy.run_path(str(D/'finalize_2027.py'),run_name='__main__')
