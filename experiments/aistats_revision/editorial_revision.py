"""Bounded exposition revision; no new experiments or mathematical claims."""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
def replace_between(s,start,end,new):
    a=s.index(start);b=s.index(end,a);return s[:a]+new+s[b:]
def main():
    backup=D/'before_editorial_pass_20260924';backup.mkdir(exist_ok=True)
    for name in ('aistats.tex','aistats.pdf','allocation_figure.tex','allocation_main.tex','selection_dose_main.tex'):
        if not (backup/name).exists():shutil.copy2(D/name,backup/name)
    s=(D/'aistats.tex').read_text(encoding='utf-8')
    s=replace_between(s,r'\begin{abstract}',r'\end{abstract}',r'''\begin{abstract}
Graph-informed selection of verified normal nodes can make a useful anomaly detector produce unreliable discoveries. We study this effect by changing the calibration reference while holding scores, test nodes and reference size fixed. On Amazon, T-Finance and Weibo, removing the highest-exposure 20\% of reference normals raises mean false discovery proportion above the nominal 10\% level for two accurate supervised scorers; matched random references remain below that level. An exact finite-population rank audit measures the reference shift, and controlled graph experiments identify degree sensitivity as one sufficient mechanism. We then derive conditions under which randomized label acquisition supports valid stratum-specific calibration, including one pooled multiple-testing procedure. Experiments with exposure-biased acquisition show that these corrections can reduce false discoveries substantially, but sparse references can erase their discovery power. The practical lesson is to evaluate calibration acquisition and selection alongside the detector: a valid correction must also leave enough reference information for useful decisions.
''')
    s=replace_between(s,r'\section{INTRODUCTION}',r'\paragraph{Relation to prior work.}',r'''\section{INTRODUCTION}
A fraud analyst may collect verified normal accounts as a reference for an anomaly detector. Excluding accounts linked to known fraud can appear prudent: the remaining reference looks less suspicious. Yet the exclusion changes which normal neighborhoods are represented. A score that is unusual relative to this selected reference may be ordinary among the accounts being tested.

We study exposure-based selection as an interpretable model of graph-informed reference curation. The question is how much this selection changes actual discoveries, whether the effect persists with useful detectors, and which sampling designs permit a valid correction. Our experiments use known training or revealed non-test labels to construct exposure; hidden test labels are reserved for benchmark evaluation. We isolate selection among verified normals, separately from contamination by anomalous reference points.

The main evidence comes from accurate supervised scorers on Amazon, T-Finance and GADBench Weibo (Table~\ref{tab:threegraphs}). Removing a moderate fraction of high-exposure reference normals increases false discoveries even when scores, test identities and reference size are held fixed. Random references already support useful detection on all three graphs. The effect therefore extends beyond a weak detector or an extremely small calibration set.

We connect this observation to a practical choice. Representative random acquisition supports pooled calibration. When acquisition probabilities vary across fixed exposure groups, calibrating within those groups can restore a guarantee under the stated randomized-role design. It can also reduce power sharply. Comparing reference selection and its corrections requires both error and discovery measurements.

The paper makes three contributions:
\begin{enumerate}
\item \textbf{Matched empirical evidence.} Measure the FDP--power tradeoff of observed-label reference selection across three useful-score benchmarks, retaining the complete removal curves and unfavorable controls.
\item \textbf{An exact diagnostic and a controlled mechanism.} Use the finite-population rank law to quantify a fixed reference shift, and isolate degree sensitivity in independent graph experiments.
\item \textbf{Sampling conditions and tested remedies.} Establish when randomized roles justify stratified ranks and pooled BH, then evaluate their power under representative and exposure-biased label acquisition.
\end{enumerate}
These are applications and evaluations of established conformal and multiple-testing principles. Their contribution is to make the consequences and sampling requirements of graph-reference curation explicit.

''')
    s=s.replace('The expectation averages over the random partition; it is not a guarantee conditional on each realized test set. A label-dependent filter, or a reference count chosen from its realized eligible pool, needs a separate argument. We do not assign that guarantee to every empirical comparator.',
        'The expectation averages over the random partition. Appendix~\\ref{app:randomdesign} distinguishes this guarantee from a fixed-test-set audit and from comparators whose reference count depends on the realized filter.')
    start=s.index('\\medskip\\noindent\\fbox{');end=s.index('\\section{EXACT CONDITIONAL RANK AUDIT}',start)
    s=s[:start]+s[end:]
    s=s.replace('All BH tests use $\\alpha=0.10$.','The family-level target is $\\alpha=0.10$; separate-stratum allocations are specified below.')
    # Preserve the superseded oracle opening and table as supporting evidence.
    start=s.index('\\paragraph{Accurate scores expose the practical cost.}')
    end=s.index('\\input{selection_dose_main.tex}',start)
    oracle=s[start:end]
    s=s[:start]+r'''\input{three_graph_results.tex}
\paragraph{Selection changes discoveries with useful scores.}
Table~\ref{tab:threegraphs} compares the designated moderate-removal setting across all three useful-score graphs. The reference is selected using half of non-test labels; no test label enters the exposure rule. Both scorers show higher mean FDP after exposure removal, with paired increases above zero in the pointwise intervals. The size of the effect varies: graph-feature FDP rises from $0.095$ to $0.191$ on T-Finance, and from $0.091$ to $0.286$ on GADBench Weibo. These are changes to an already useful discovery pipeline, with both random references retaining substantial power.

'''+s[end:]
    s=s.replace('\\input{extended_benchmark_main.tex}\n','')
    s=s.replace('\\label{app:unsupervised}','\\label{app:unsupervised}\n'+oracle)
    s=replace_between(s,r'\paragraph{Extreme contrasts need weak scores to be interpreted separately.}',r'\section{CONTROLLED MECHANISM AND BOUNDARIES}',r'''\paragraph{Distortion and useful detection are different questions.}
The unsupervised controls clarify why both measurements matter. DOMINANT has high filtered FDP but weak ranking and negligible power under random calibration. Positive degree alone reproduces that pattern. Conversely, the PyGOD Weibo GAE audit detects a null-rank shift with few discoveries. The exact audit measures reference distortion; Equation~\eqref{eq:crossing} supplies the additional rank-resolution condition for BH to reject. Full unsupervised results and comparisons of the exact audit with Monte Carlo appear in Appendices~\ref{app:unsupervised} and~\ref{app:fresh}.

''')
    s=replace_between(s,r'\section{FROM DIAGNOSIS TO A DEPLOYMENT DECISION}',r'\clearpage'+'\n'+r'\section*{AI Use Statement}',r'''\section{PRACTICAL IMPLICATIONS AND LIMITATIONS}
Calibration design determines which normal population a reference represents. When random acquisition is feasible and satisfies the conditions in Section 2, pooled calibration is a justified and effective baseline. With exposure-biased acquisition, the stratified procedures offer guarantees under a specific role-assignment law; Figure~\ref{fig:allocation} shows why power must still be checked. Known-propensity weighted BH performs well in several evaluated settings, but the marginal validity of its ranks alone is insufficient for a joint BH guarantee. Appendix~\ref{app:allocation} supplies the conservative weighted-BY comparison and the precise assumptions.

For development, compare references at matched size with frozen scores and test nodes, report FDP and power together, and retain a full-reference baseline where available. If the acquisition mechanism cannot support a calibration guarantee, a fresh random audit can instead certify the FDP of a fixed proposed discovery batch. That established alternative has a different label budget and target; details appear in Appendix~\ref{app:deploymentaudit}.

Our exposure rule is a controlled proxy for graph-informed curation; we have not measured its prevalence in operational fraud systems. The follow-ups were exploratory, with protocols fixed before their respective new outcomes and complete grids retained. Three useful-score benchmarks support the empirical comparison, while Tolokers remains an unfavorable control. The two Weibo releases are source sensitivities of one graph. Seed intervals describe repeated evaluation on each fixed graph, not uncertainty over a deployment population.

The formal results require correct labels, role-invariant scores and strata, the specified acquisition law, and independent tie marks. The controlled simulations identify degree sensitivity as a sufficient mechanism, rather than a unique explanation of trained detectors. Finally, the observed power losses characterize the tested corrections: they are not an impossibility theorem or an optimal validity--power frontier. The practical objective is a sampling design that supports both a defensible error guarantee and useful discoveries.

''')
    (D/'aistats.tex').write_text(s,encoding='utf-8')
    # Keep the complete original prespecified display in the appendix.
    full=(backup/'allocation_figure.tex').read_text().replace('\\label{fig:allocation}','\\label{fig:allocationfull}')
    (D/'allocation_full_figure.tex').write_text(full)
    p=D/'allocation_proof.tex';s=p.read_text()
    if 'allocation_full_figure.tex' not in s:s=s.replace('\\input{allocation_results.tex}','\\input{allocation_full_figure.tex}\n\\input{allocation_results.tex}')
    p.write_text(s)
    # Numerical generators and the final narrative are distinct: this pass edits
    # the prose inputs while retaining the complete numerical generators.
    p=D/'selection_dose_main.tex';s=p.read_text();fig=s[s.index('\\begin{figure*}'):]
    p.write_text(r'''\paragraph{The full removal curve matters.}
Figure~\ref{fig:dose} shows how Amazon moves from mild selection to severe loss of reference resolution. Small removals need not raise mean FDP above nominal; very severe removal can suppress power. The useful-score findings therefore describe a tradeoff over the evaluated range, not a monotone law. Appendices~\ref{app:dose} and~\ref{app:extensions} retain every removal level, label budget and scorer, including Tolokers' low-power outcomes and the separate PyGOD Weibo source. The paired uncertainty plots cover the complete grid.

'''+fig)
    p=D/'allocation_main.tex';s=p.read_text()
    s=s.replace('Thus low FDP under equal splitting does not establish that its error allocation is the principal cause of lost power.',
        'Changing the allocation alone leaves a substantial power gap in this setting.')
    s=s.replace('Figure~\\ref{fig:allocation} retains all three acquisition severities. Sparse strata can lose nearly all power, even with pooled testing; a validity remedy is not automatically an efficient one.',
        'Figure~\\ref{fig:allocation} compares the moderate-acquisition setting on all three graphs; Weibo shows that sparse strata can lose nearly all power. The complete allocation and severity grid is retained in Figure~\\ref{fig:allocationfull}.')
    p.write_text(s)
    p=D/'package_revision.py';s=p.read_text();anchor="source += ['allocation_main.tex'"
    if "source += ['three_graph_results.tex'" not in s:s=s.replace(anchor,"source += ['three_graph_results.tex','acquisition_main.pdf','allocation_full_figure.tex']\n"+anchor)
    s=s.replace("'build_allocation_results.py']:","'build_allocation_results.py','build_editorial_display.py']:")
    p.write_text(s)
    print('Editorial source revision complete; original source/PDF backed up')
if __name__=='__main__':main()
