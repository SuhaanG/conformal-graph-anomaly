"""Generate the exploratory pilot comparison and operational manuscript text."""
from pathlib import Path
import pandas as pd
D=Path(__file__).resolve().parent;A=D.parent/'audit_method'
MARK='% PILOT AUDIT COMPARISON INTEGRATED'

def main():
    pilot=pd.read_csv(A/'pilot_trials.csv')
    uniform=pd.read_csv(A/'strong_certificate_trials.csv')
    uniform=uniform[uniform.method=='uniform_candidates']
    keys=['dataset','model','budget'];metrics=['true_discoveries','audit_labels','fdp','abstain']
    frames={}
    for name,df in [('Uniform',uniform),('Pilot',pilot)]:
        sm=df.groupby(keys+['seed'])[metrics].mean()
        frames[name]=(sm.groupby(keys).mean(),sm.groupby(keys).std())
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule Score & Audit & Cap & Labels & True discoveries & Nonempty \\',r'\midrule']
    for model,label in [('hgb_attributes','Attributes'),('hgb_graph_features','Attributes + graph')]:
        for b in (100,250,500):
            for name in ('Uniform','Pilot'):
                mean,sd=frames[name];r=mean.loc[('amazon',model,b)];v=sd.loc[('amazon',model,b)]
                lines.append(f'{label} & {name} & {b} & {r.audit_labels:.1f} & '+
                    f'${r.true_discoveries:.1f}\\pm {v.true_discoveries:.1f}$ & {100*(1-r.abstain):.1f}\\%'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (D/'pilot_table.tex').write_text('\n'.join(lines)+'\n')
    p=D/'aistats.tex';s=p.read_text()
    if MARK not in s:
        old='Together, these results show why normal-only calibration is insufficient and identify a practical comparison: evaluate reference selection alongside detector accuracy, reporting both false discoveries and retained power.'
        new='For a separate operational task, random label audits certify a proposed batch\'s FDP without independent-node assumptions; on the accurate Amazon scorer, a 500-label audit retains 212 true automatic discoveries on average. These results connect diagnosis of reference bias to an actionable choice between recalibration, direct certification, and abstention.'
        assert old in s;s=s.replace(old,new)
        old='We additionally examine weighted conformal selection under its independent-observation assumptions and a direct fixed-batch certification procedure in the supplement.'
        new='We also evaluate an established finite-population certification route: audit a random subset of proposed discoveries and retain an unreviewed set only when its precision is certified. This yields useful discoveries in the accurate Amazon regime without assuming independent graph nodes.'
        assert old in s;s=s.replace(old,new)
        start=s.index('The operational implication is to specify the target population')
        end=s.index('The benchmark evidence concerns',start)
        s=s[:start]+r'''\paragraph{From diagnosis to certified discoveries.}
Two tasks require different labels. To diagnose a calibration rule, freeze scores and test identities on a labeled development graph, compare matched references through Equation~\eqref{eq:exacttail}, and report FDP, power, and the full-reference comparator. A mean score gap or a successful benchmark comparison alone does not certify deployment FDR.

For a fixed deployment batch, the following procedure instead certifies proposed discoveries directly using classical finite-population bounds~\citep{angelopoulos2022ltt,anthony2026audits}:
\begin{enumerate}
\item Freeze the scorer, $K$ candidate score prefixes, audit budget, FDP target $q$, and failure probability $\delta$ before inspecting audit labels.
\item Audit a uniform sample from their union. For candidate $k$, obtain an exact hypergeometric upper bound $U_k$ on its total normal count at level $\delta/K$. If $h_k$ nodes were audited, including $X_k$ normals, its unreviewed FDP is bounded by $(U_k-X_k)/(R_k-h_k)$, where $R_k$ is its size.
\item Return the largest nonempty unreviewed candidate whose bound is at most $q$, or abstain. All audited nodes are excluded from automatic discoveries.
\end{enumerate}
Appendix~\ref{app:certificate} proves $\Prb_{\rm audit}(\FDP\le q)\ge1-\delta$ conditional on the entire graph and correct labels. This is a batch certificate, not a future-graph guarantee or an FDR-$q$ theorem. At $q=0.10,\delta=0.05$, Amazon's attribute scorer retains mean 211.5 true automatic discoveries using 500 audit labels, in addition to 864 training labels. Tolokers abstains. A pilot-selected variant spends fewer labels on average but returns fewer discoveries at the same budget cap (Appendix~\ref{app:pilot}). Neither construction is a new confidence-bound principle; their role here is to provide and evaluate an operational response to the diagnosed failure.

'''+s[end:]
        p.write_text(MARK+'\n'+s)
    p=D/'audit_extension.tex';s=p.read_text()
    if r'\label{app:pilot}' not in s:
        s+=r'''

\subsection{Pilot-selected independent certification}
\label{app:pilot}
A pilot can select a promising candidate before spending labels on its certificate. This is a sample-splitting construction related to Learn then Test~\citep{angelopoulos2022ltt}, not a new inference principle. We specified the following exploratory heuristic after inspecting the earlier uniform-audit results and then ran every cached scorer without tuning the heuristic to its outcomes.

\paragraph{Design.}
With total budget cap $B$, draw $b_0=\lfloor0.2B\rfloor$ pilot labels uniformly from the largest candidate prefix. In each prefix, let $h$ be its pilot count and $x$ its observed normal count, and use $(x+1)/(h+2)$ only to plan the next sample. Remove all pilot nodes. For each remaining candidate of size $N$, consider fresh audit sizes in $\{25,50,100,200,400,B-b_0\}$, retaining positive sizes below $N$ and at most $B-b_0$. Predict the audit normal count as $\lceil n(x+1)/(h+2)\rceil$ and invert the hypergeometric CDF at level $\delta$. Among pairs predicted to certify FDP at most $q$, choose the one maximizing $N-n$; ties follow increasing prefix size and then audit size. If none qualifies, abstain after the pilot.

Freeze the chosen candidate and $n$. Draw a fresh uniform sample of $n$ of its remaining nodes, observe its normal count $X$, and set
\[
 U(X)=\max\{v:H_{N,v,n}(X)\ge\delta\},\qquad
 B_{\rm FDP}=\frac{U(X)-X}{N-n}.
\]
Return the unreviewed remainder only when $B_{\rm FDP}\le q$; otherwise abstain. A failed certificate does not trigger another test or candidate choice. Every pilot and certification label counts toward cost, and every audited node is removed from automatic discoveries. The heuristic can be inaccurate without invalidating the fresh certificate.

\paragraph{Conditional guarantee.}
Condition on the graph, labels, scores, and the complete pilot. The selected population, its total normal count $V$, and the fresh sample size $n$ are now fixed. Uniform sampling gives $X\sim\operatorname{Hypergeom}(N,V,n)$, even though graph labels may be arbitrarily dependent. The inversion argument above yields $\Prb\{U(X)<V\mid\text{pilot}\}\le\delta$. On its complement, the unreviewed FDP is $(V-X)/(N-n)\le B_{\rm FDP}$. If certification fails or no plan is chosen, the procedure returns the empty set with FDP zero. Averaging over pilots therefore gives $\Prb_{\rm audit}\{\FDP>q\}\le\delta$. Only one candidate is tested after conditioning on the pilot, so no candidate multiplicity factor is required. Certificate labels must not be reused to change the plan, scorer, or sample size.

\paragraph{Complete evaluation.}
We retain the existing prefixes, $q=0.10$, $\delta=0.05$, budgets 100/250/500, ten seeds, and 200 audit repetitions per scorer and budget. Master seed 20260929 defines the new audits. Both graphs, all four original score conditions, and both supervised controls give 72,000 trials. The original score conditions and every Tolokers configuration abstain. Table~\ref{tab:pilot} reports all Amazon supervised cells, including zero-discovery outcomes. Training labels are unchanged and additional; the two procedures use independent audit randomizations. SDs summarize the ten training-seed averages, not independent graph uncertainty.

\begin{table}[t]
\centering\small
\caption{Pilot-selected versus candidate-uniform auditing on Amazon. Labels are actual mean audit use, distinct from the budget cap. True discoveries exclude all audited nodes and include abstentions; entries are mean $\pm$ seed SD. Nonempty is the percentage of audits returning discoveries. All Tolokers and original-score pilot cells abstain.}
\label{tab:pilot}\input{pilot_table.tex}
\end{table}

At the 500-label cap, the attribute pilot uses mean 173.9 labels and retains 134.1 true automatic discoveries, versus 500 labels and 211.5 true discoveries for uniform auditing. It returns a nonempty set in 55.6\% of runs, compared with 80.75\% for uniform auditing. The graph-feature pilot uses 171.2 labels and retains 125.3 true discoveries, versus 202.8 for uniform auditing. The pilot saves labels partly by abstaining earlier and also removes fewer candidate nodes through review. This is a label-use/yield trade-off, not evidence of superiority at a matched fixed cost or budget cap. We retain the simpler uniform procedure as the main operational example.

\paragraph{Validation and interpretation.}
No returned set violates the FDP target in the 72,000 trials; 22 total-normal-count bounds under-cover, but their returned sets still meet the FDP target. Zero FDP violations therefore must not be confused with universal count-bound coverage. Separate exact enumeration over all 256 binary labelings of an eight-node population and all pilot/certificate samples checks a label-adaptive candidate and sample-size rule. At $\delta=0.20$, its largest exact FDP violation probability is $0.0286$, with nonempty selections included. Brute-force inversion agrees with the implementation, and 2,000 additional full-procedure synthetic audits include nonvacuous high-precision controls. These checks support the implementation; the conditional proof supplies the guarantee. No claim of a new graph-specific correction follows from this exploratory comparison.
'''
        p.write_text(s)

if __name__=='__main__':main()
