"""Focused rewrite from an archived pre-feedback source; preserves research files."""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent

def main():
    backup=D/'before_focus_20260923';backup.mkdir(exist_ok=True)
    for name in ('aistats.tex','audit_extension.tex','aistats_refs.bib','aistats.pdf'):
        if not (backup/name).exists():shutil.copy2(D/name,backup/name)
    old=(backup/'aistats.tex').read_text()
    preamble=old[:old.index(r'\begin{abstract}')]
    ai=old[old.index(r'\clearpage'+'\n'+r'\section*{AI Use Statement}'):old.index(r'\section{PROOFS AND SCOPE}')]
    appendix=old[old.index(r'\section{PROOFS AND SCOPE}'):]
    # Keep valid current evidence; obsolete result tables remain in the archive.
    for start,end in [(r'\section{LARGER RANDOM CALIBRATION}',r'\section{COMPLETE SCORE SIMULATION}'),
                      (r'\section{INCLUSION AUDIT OF EXISTING RESULTS}',r'\begin{table*}[t]'+'\n'+r'\centering\small'+'\n'+r'\caption{Complete fresh conditional graph audit'),
                      (r'\section{HISTORICAL GRAPH RESULTS AND INCLUSION RULES}',r'\clearpage\input{audit_extension.tex}')]:
        i=appendix.index(start);j=appendix.index(end,i);appendix=appendix[:i]+appendix[j:]
    i=appendix.index('For a given seed, scores are reused across calibration strategies.')
    j=appendix.index(r'\section{COMPLETE SCORE SIMULATION}',i)
    appendix=appendix[:i]+'''Scores and test identities are shared within each matched comparison. The supplement supplies the current score caches, input manifests, trial-level outcomes, and executable table generators. Repeated calibration draws quantify sampling variation conditional on a graph; seed SDs summarize training variability on that same graph.\n\n'''+appendix[j:]
    # Add the missing mathematical bridge, including its conditioning scope.
    proof=r'''\subsection{Random Partition Baseline}
\label{app:randomdesign}
Fix the finite graph, labels, and scores, and condition on any separate supervised training panel and model randomness. Resolve equal scores by independent continuous node-level tie breakers drawn before the evaluation partition. Fix the anomalous test nodes and the counts $n$ and $m_0$. Assign the remaining normal nodes uniformly to $n$ calibration positions, $m_0$ null-test positions, and unused positions. Conditional on the unordered scores assigned to the first two groups, their assignment to those positions is a uniform permutation. Calibration and null-test scores are therefore jointly exchangeable conditional on the anomalous test scores. The distinct-score conditional-exchangeability result of \citet[Theorem 3.3]{marandon2024adaptive} makes the conformal p-values super-uniform and PRDS on the nulls. BH consequently satisfies $\E[\FDP]\le\alpha m_0/m$. This is an application of existing theory to the sampling design, not a new graph-specific FDR theorem.

A uniformly sampled test-normal subset followed by uniform calibration from the remaining normals implements this assignment. Uniform label revelation and additional uniform reference subsampling also preserve it after conditioning on their counts. Transductive use of the full graph's unlabeled features is compatible with this argument when the frozen scores do not depend on evaluation roles. In particular, the full-reference and matched random arms of the removal experiment use predetermined removal fractions, so their reference counts depend on label counts rather than which high-exposure normals remain. Their lexicographic score ordering is fixed before the split.

The expectation averages over random normal-node assignments; it is not conditional on the realized identities of null test nodes. The rank audit instead conditions on those identities and randomizes calibration only. Nor does the result automatically cover a matched size chosen as the number of nodes passing a label-dependent filter: that size can depend on the realized partition. The earlier zero-exposure comparisons use conservative nonrandomized ties and are reported empirically, without borrowing a guarantee from a different tie or sampling convention. The new removal experiment explicitly implements the distinct-score convention above.

'''
    appendix=appendix.replace(r'\subsection{BH Crossing and Resolution}',proof+r'\subsection{BH Crossing and Resolution}',1)
    # Retain weighting details as supporting analysis, not a second main story.
    weighting=old[old.index(r'\paragraph{A covariate mechanism.}'):old.index(r'\section{GRAPH EXPERIMENTAL DESIGN}')]
    simulation=old[old.index(r'\section{INDEPENDENT VALIDATION OF THE MECHANISM}'):old.index(r'\section{DISCUSSION AND LIMITATIONS}')]
    simulation=simulation.replace(r'\section{INDEPENDENT VALIDATION OF THE MECHANISM}'+ '\n'+r'\label{sec:simulation}',r'\subsection{Independent score and weighting comparisons}')
    appendix=appendix.replace(r'\section{COMPLETE SCORE SIMULATION}',r'\section{COMPLETE SCORE SIMULATION}'+'\n'+weighting+simulation,1)
    # Fix missing validation summary and source-file/statistics discrepancy.
    appendix=appendix.replace('and 347 anomalous labels.',r'''and 347 anomalous labels. The official distributed \texttt{weibo.pt} file contains 8,058 zeros and 347 ones, whereas the repository's descriptive table lists 868 outliers. A fresh download on September 23, 2026 was byte-identical to our cached raw file; we use its actual labels without recoding them to match the table~\citep{pygoddata}. The source-file checksum and label counts are retained in the supplement.''',1)
    appendix=appendix.replace('For seeds 0--4, the single-draw matched filtered and random metrics', 'For seeds 0--4, the single-draw matched filtered and random metrics',1)
    start=appendix.index('For seeds 0--4, the single-draw matched filtered and random metrics')
    end=appendix.index('For the conditional audit,',start)
    appendix=appendix[:start]+appendix[end:]
    marker='The SD in Table~\\ref{tab:audit} is instead calculated across the five score/test realizations after averaging those draws.'
    appendix=appendix.replace(marker,marker+r''' The positive-exposure rule restricts calibration to normal nodes with at least one anomalous neighbor. At thresholds $0.01$ and $0.05$, five seeds, three trimming levels, and four rules give 120 exact-versus-Monte-Carlo tail checks. The maximum absolute discrepancy is $2.9232$ Monte Carlo SEs; all individual checks are supplied in the reproducibility archive.''')
    appendix=appendix.replace(r'\clearpage\input{audit_extension.tex}',r'\clearpage\input{selection_dose_appendix.tex}'+'\n'+r'\clearpage\input{audit_extension.tex}')
    extension=(backup/'audit_extension.tex').read_text()
    extension=extension.replace(' Historical larger-reference results in Appendix~\\ref{app:larger} use a different score and test construction and cannot replace this corrected comparison.','')
    (D/'audit_extension.tex').write_text(extension)
    maintext=r'''\begin{abstract}
Selecting verified normal nodes by their graph neighborhoods can distort a calibration reference. We isolate this effect by holding trained scores, test identities, and reference size fixed. An accurate Amazon scorer (AUROC $0.969$) yields mean false discovery proportion (FDP) $0.192$ with zero-exposure calibration versus $0.067$ with matched random calibration at nominal $0.10$. \input{dose_abstract.tex} An exact conditional rank audit measures the reference shift, and independent graph experiments isolate degree sensitivity as one sufficient mechanism. We distinguish these diagnostics from the random-sampling conditions that justify conformal BH. When representative calibration is unavailable, an established random-label audit can instead certify a fixed batch's FDP; 500 audit labels retain 212 true automatic discoveries on average in the accurate Amazon setting. The resulting evaluation separates reference bias, limited rank resolution, and detector quality.
\end{abstract}

\section{INTRODUCTION}
A fraud analyst may collect verified normal accounts as a reference for an anomaly detector. Excluding accounts linked to known fraud can appear prudent: the remaining reference looks less suspicious. But this choice also changes which graph neighborhoods are represented. A high score may then indicate a normal account from an excluded neighborhood rather than an anomaly. The relevant question is whether a selection policy distorts decisions made with an otherwise useful scorer.

We compare a zero-exposure rule, which keeps reference normals with no anomalous neighbors, with graded removal of high-exposure normals. Observed-label versions use only training and revealed non-test labels. Scores, test nodes, calibration size, and the multiple-testing rule remain fixed within each comparison, isolating reference composition from training and sample-size changes.

The useful-score regime is central. On Amazon, an attribute classifier with AUROC $0.969$ and average precision $0.895$ gives mean FDP $0.192$ with zero-exposure calibration, compared with $0.067$ for matched random calibration. Using the full reference yields FDP $0.092$ and power $0.783$. Thus selection can damage a scorer that otherwise supports useful discoveries. Weak reconstruction scores produce more extreme contrasts, but their low power under random calibration is a separate limitation (Figure~\ref{fig:headline}).

Our contribution is an evaluation methodology with three components:
\begin{enumerate}
\item \textbf{Matched evidence.} Compare exposure-based and random calibration, including accurate scores, observed-label selection, and a graded removal experiment that tests whether extreme filtering is necessary.
\item \textbf{A conditional diagnostic.} Apply the exact finite-population rank law to measure reference-induced changes on a fixed labeled graph. Independent graph experiments isolate a sufficient score-level mechanism.
\item \textbf{An operational comparison.} Evaluate representative calibration against an established finite-batch label-audit certificate, reporting discoveries, power, label costs, and abstention.
\end{enumerate}
We use established statistical tools to identify when reference selection changes useful graph discoveries.

\paragraph{Relation to prior work.}
Conformal ranks require a sampling symmetry~\citep{vovk2005algorithmic}. Shared-reference outlier testing~\citep{bates2023testing} and AdaDetect~\citep{marandon2024adaptive} establish BH guarantees under suitable joint score conditions. \citet{gazin2024transductive} characterize transductive conformal ranks under exchangeable scores, including adaptive constructions. Graph dependence therefore does not by itself preclude validity: the labeling and evaluation design matters. Conformal link prediction explicitly aligns calibration with edge-observation mechanisms~\citep{marandon2023link}, while graph prediction-set methods target coverage~\citep{huang2023uncertainty,zargarbashi2023conformal}. We examine how selecting among verified normal nodes changes a discovery pipeline.

Weighted conformal inference and selection address specified distribution shifts~\citep{tibshirani2019conformal,jin2026weighted}; robust-reference methods address contamination~\citep{bashari2025robust}. Our principal intervention changes which normals are represented. Finite-population certification offers a different action when representative calibration cannot be justified~\citep{angelopoulos2022ltt,anthony2026audits}. We evaluate that established option rather than claim a new confidence-bound principle.

\section{SAMPLING DESIGN AND DISCOVERY}
Let $s(v)$ be a fixed graph-node score, with larger values more anomalous. A calibration set $\calset$ contains $n$ normal nodes, disjoint from $m$ test nodes $\testset$. The upper-tail conformal p-value is
\begin{equation}
p(v)=\frac{1+\sum_{u\in\calset}\ind\{s(u)\ge s(v)\}}{n+1}.
\label{eq:p}
\end{equation}
BH rejects through the largest index $k$ with $p_{(k)}\le\alpha k/m$, or rejects none. If $V$ of $R$ discoveries are normal, $\FDP=V/(R\vee1)$ and $\FDR=\E[\FDP]$. Power is the fraction of anomalous test nodes discovered. We report zero-discovery runs in every mean.

\paragraph{A valid random-sampling baseline.}
Condition on the graph, labels, trained scores, and any separate training panel. For prescribed counts, uniformly assign eligible normals to calibration and null-test positions, independently of their scores, and break score ties with independent continuous marks fixed before the assignment. Calibration and null-test scores are then exchangeable conditional on anomalous test scores. Existing PRDS theory gives BH FDR at most $\alpha m_0/m$, where $m_0$ is the null-test count~\citep{marandon2024adaptive}. Appendix~\ref{app:randomdesign} supplies the sampling argument. The expectation averages over the random partition; it is not a guarantee conditional on each realized test set. A label-dependent filter, or a reference count chosen from its realized eligible pool, needs a separate argument. We do not assign that guarantee to every empirical comparator.

\paragraph{Resolution.}
Every rank is at least $1/(n+1)$. Write $r(v)=(n+1)p(v)$ and $N(r)=|\{v:r(v)\le r\}|$. BH discovers something exactly when an integer $r$ satisfies
\begin{equation}
N(r)\ge\frac{mr}{\alpha(n+1)}.
\label{eq:crossing}
\end{equation}
Every nonempty rejection set therefore has at least $\lceil m/[\alpha(n+1)]\rceil$ nodes. At $n=162$, $m=2,159$, and $\alpha=0.10$, this is 133 nodes. A small reference can suppress discoveries despite good overall ranking. This arithmetic motivates both matched-size and full-reference comparisons.

\medskip\noindent\fbox{\begin{minipage}{0.94\columnwidth}\small
\textbf{Three distinct questions.} The \emph{rank audit} diagnoses a reference on a labeled, fixed graph. A \emph{BH guarantee} additionally needs a valid joint sampling argument. A \emph{batch certificate} uses fresh random labels to bound the FDP of a fixed deployment set. These statements condition on different information and are not interchangeable.
\end{minipage}}\medskip

\section{EXACT CONDITIONAL RANK AUDIT}
Condition on the graph, scores, test set, and an eligible normal reference pool $\mathcal P$ of size $M$. Draw $n\le M$ calibration nodes uniformly without replacement. For a fixed test node, let $K_v=|\{u\in\mathcal P:s(u)\ge s(v)\}|$. Then
\begin{equation}
(n+1)p(v)-1\sim\operatorname{Hypergeom}(M,K_v,n).
\label{eq:hypergeom}
\end{equation}
Writing $H_{M,K,n}$ for this CDF, the expected fraction of null p-values at or below $t$ is
\begin{equation}
\frac{1}{|\testset_0|}\sum_{v\in\testset_0}H_{M,K_v,n}\bigl(\lfloor t(n+1)\rfloor-1\bigr).
\label{eq:exacttail}
\end{equation}
This standard sampling identity permits arbitrary graph dependence and ties: the scores are conditioned on. Changing the reference pool changes its exceedance counts even when $n$ is held fixed. The diagnostic requires labeled development data and evaluates fixed thresholds; it is not an FDR bound at BH's adaptive threshold.

\paragraph{Direction under an independent score model.}
The following elementary comparison explains one route to reference distortion.
\begin{proposition}[Score-level selection comparison]
\label{prop:order}
Let $S_1,\ldots,S_n$ be independent calibration scores with continuous CDF $F_C$, independent of a null score $T$ with continuous CDF $F_0$. If $F_C(x)\ge F_0(x)$ for all $x$, then
\begin{equation}
\Prb\{p(T)\le t\}\ge\frac{\lfloor t(n+1)\rfloor}{n+1},\quad 0\le t\le1.
\label{eq:order}
\end{equation}
Reversing the stochastic order reverses the inequality; equality holds for identical score laws.
\end{proposition}
Common-uniform coupling makes the selected-reference p-value no larger than its exchangeable counterpart. This orders it against a discrete uniform rank; the displayed lower bound alone does not establish a strict exceedance of $t$. The exact independent-model probability follows from
\begin{equation}
(n+1)p(T)-1\mid T=t_0\sim\operatorname{Bin}\bigl(n,1-F_C(t_0)\bigr).
\label{eq:binomial}
\end{equation}
Appendix~\ref{app:proofs} gives the proof and covariate conditions. Selecting low-exposure nodes can also select degree, but degree correlation alone does not verify those conditions for a trained graph score.

\section{EXPERIMENTAL DESIGN}
\paragraph{Graphs and scores.}
Amazon has 11,944 nodes, of which 7,818 are verified normal and 821 anomalous; its first 3,305 nodes are unlabeled and excluded from calibration and evaluation~\citep{dou2020enhancing,dglfraud}. Tolokers has 11,758 nodes and 2,566 positive labels, repurposed here as anomalies~\citep{platonov2023critical}. Weibo supplies a third observed-exposure control with 8,405 nodes and 347 anomalous labels in the distributed file~\citep{pygoddata}. No experiment in the primary comparison trims high-score normals.

The useful-score controls are fixed histogram-gradient-boosting classifiers trained on a uniform 10\% labeled panel, using attributes alone or attributes plus neighbor-average attributes and log degree. The other 90\% supplies disjoint reference and test sets. There are ten training seeds and five stratified 25\% test splits per seed. These controls and the removal sweep were specified as exploratory follow-ups. We report all graphs and both scorers, including unsuccessful settings. Complete settings and label costs appear in Appendix~\ref{app:supervised}.

Unsupervised comparisons use DOMINANT~\citep{ding2019deep}, PyGOD attribute-reconstruction GAE~\citep{liu2024pygod}, and Isolation Forest~\citep{liu2008isolation}, plus positive degree as an untrained control. The neural models use width 64, four layers, no dropout, and 100 full-batch Adam steps at learning rate $0.01$; Isolation Forest uses 200 trees and subsample size 256. Model settings and score directions are fixed. Ten training seeds share five stratified test splits. Graph features are transductive; evaluation labels are not reconstruction targets.

\paragraph{Selection and matching.}
The zero-exposure oracle retains reference normals with no anomalous neighbors. Matched random calibration draws equally many reference normals, capped at 1,000; a full-reference arm uses every eligible normal. The partial-label filter instead uses known training labels and a random revealed fraction $\rho\in\{0.25,0.50\}$ of non-test deployment labels. Test labels enter metrics and benchmark stratification, never this filter. We use 200 calibration repetitions per matched comparison.

The removal sweep reveals fractions $\rho\in\{0.25,0.50,1\}$ of non-test labels and ranks eligible normals by the fraction of neighbors known anomalous. It removes the highest-exposure $d\in\{0,.05,.10,.20,.40,.60,.80,.90,.97\}$, breaking exposure ties randomly. The comparator removes the same count uniformly. Both arms retain their entire remaining reference, with 50 paired repetitions; no 1,000-node cap artificially limits the mild-selection cases. Independent lexicographic score tie breakers, fixed before the split, implement the random-baseline convention in Section 2. Appendix~\ref{app:dose} reports the full grid and validation.

\paragraph{Metrics.}
All BH tests use $\alpha=0.10$. Average repetitions and splits within each training seed, then report the mean and sample SD of ten seed averages. This SD measures variation on one graph, not uncertainty over new graphs. AUROC and average precision describe ranking; FDP, power, and nonempty-discovery frequency describe the actual decisions. Label budgets for training, reference acquisition, and certification are reported separately.

\section{RESULTS: COMPOSITION, POWER, AND RESOLUTION}
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{headline_results.pdf}
\caption{Reference composition changes FDP (left) and power (right) differently. Each panel compares accurate supervised Amazon scorers with DOMINANT on Amazon and Tolokers. Bars show seed means; error bars are one seed SD, not confidence intervals over new graphs. Filtered and random references are matched in size; full references are larger. The horizontal line marks nominal $0.10$ in the FDP panel. Low FDP with zero power is distinguished from useful discovery.}
\label{fig:headline}
\end{figure*}
\paragraph{Accurate scores expose the practical cost.}
Amazon's attribute scorer has AUROC $0.969$ and average precision $0.895$. Under the oracle zero-exposure filter, its mean FDP is $0.192$, compared with $0.067$ under matched random calibration. Their powers are $0.809$ and $0.495$. The full reference yields FDP $0.092$ and power $0.783$, showing useful discoveries without the severe reference-size restriction. The graph-feature scorer gives the same ordering (Table~\ref{tab:supervised}). With 50\% of non-test labels revealed, attribute-score FDP is $0.140$ for zero-exposure selection versus $0.069$ for matched random calibration. With 25\%, both means are below $0.10$.

\begin{table*}[t]
\centering\small
\caption{Supervised zero-exposure controls at nominal $0.10$: mean FDP / power including abstention. Filtered and random references are size matched; the full reference uses all eligible normals. Ten training panels and five test splits per panel. Seed SDs, reference sizes, and partial-label outcomes appear in Appendix~\ref{app:supervised}.}
\label{tab:supervised}\input{supervised_table.tex}
\end{table*}

\input{selection_dose_main.tex}

\paragraph{Extreme contrasts need weak scores to be interpreted separately.}
For DOMINANT, zero-exposure filtering gives FDP $0.936$ on Amazon and $0.740$ on Tolokers, with powers $0.449$ and $0.398$. Matched random calibration has FDP $0$ and $0.004$, with essentially zero power. DOMINANT AUROCs are $0.340$ and $0.530$, so those contrasts do not establish a useful calibrated detector. GAE and Isolation Forest make no oracle-filtered discoveries on either graph; full-reference calibration makes none for any of the three original scorers. The complete unsupervised comparison is in Appendix~\ref{app:unsupervised}. Positive degree alone reproduces high filtered FDP, demonstrating that learned message passing is unnecessary for this particular failure.

\paragraph{The rank audit separates distortion from discovery.}
On the untrimmed DOMINANT scores, Equation~\eqref{eq:exacttail} gives expected null fractions at or below $0.01$ of $0.536$ versus $0.00545$ for filtered and random Amazon calibration, and $0.243$ versus $0.01015$ on Tolokers. Filtered GAE and Isolation Forest fractions remain below $0.01$. A separate five-seed Weibo GAE audit gives $0.0233$ versus $0.0095$ without trimming, but filtered mean FDP is only $0.041$ with power $0.004$. Thus a detectable rank shift need not produce many false discoveries; Equation~\eqref{eq:crossing} explains the additional resolution requirement. Appendix~\ref{app:fresh} compares the exact calculation with repeated calibration draws.

\section{CONTROLLED MECHANISM AND BOUNDARIES}
To isolate one sufficient mechanism, generate 2,000 independent graphs across four degree-heterogeneity and cross-class-connectivity cells, plus 500 all-null controls. Each graph has 1,200 nodes and, except in the all-null control, 10\% anomalies. For standardized log degree $L_v$, set $S_v=4Y_v+\theta L_v+\varepsilon_v$, with $\theta\in\{0,2\}$ and either independent or neighbor-aggregated Gaussian noise. These are controlled analytic scores, not trained detectors; 500 graphs per cell provide the independent replication unit.

With degree-insensitive independent scores, oracle-filter mean FDP ranges from $0$ to $0.081$. With heterogeneous degree, enhanced cross-class connectivity, and $\theta=2$, mean FDP is $0.751$ for filtering versus $0.061$ for random calibration; powers are $0.979$ and $0.288$. Neighborhood noise gives similar FDPs, $0.753$ and $0.067$. A fixed half-filtered, half-random mixture lowers the first FDP to $0.119$, still above nominal. Figure~\ref{fig:indgraphs} and Appendix~\ref{app:indgraphs} retain every cell, including all-null and partial-label controls.

Independent score simulations additionally distinguish selection from contamination and compare correct weighting with a fixed test-weight heuristic. Correct density-ratio ranks require the ratio at both calibration and test points, overlap, and a common conditional score law. WCS provides a multiple-testing guarantee under its own assumptions~\citep{jin2026weighted}; its low FDP in our independent simulation comes with limited power. These supporting comparisons appear in Appendices~\ref{app:simulation} and~\ref{app:wcs}, rather than serving as a correction theorem for the exposure-selected graph pipeline.

\section{FROM DIAGNOSIS TO A DEPLOYMENT DECISION}
For a labeled development graph, freeze scores and test identities, compare size-matched references with Equation~\eqref{eq:exacttail}, and report FDP and power alongside a full-reference arm. Where the acquisition design satisfies Section 2's random-sampling conditions, representative calibration has a principled baseline guarantee. An empirical rank diagnostic alone cannot establish that those conditions hold in deployment.

If labels can instead be acquired for proposed discoveries, an established finite-population audit gives a direct option~\citep{angelopoulos2022ltt,anthony2026audits}:
\begin{enumerate}
\item Fix the scorer, $K$ candidate score prefixes, label budget, FDP target $q$, and failure probability $\delta$ before inspecting audit labels.
\item Audit a uniform sample from their union. For candidate $k$, compute an exact hypergeometric upper bound $U_k$ on its total normal count at level $\delta/K$.
\item If $h_k$ of its $R_k$ nodes were audited, including $X_k$ normals, retain the largest nonempty unreviewed candidate with $(U_k-X_k)/(R_k-h_k)\le q$; otherwise abstain.
\end{enumerate}
All audited nodes are excluded from automatic discoveries. Simultaneous coverage gives $\Prb_{\rm audit}\{\FDP\le q\}\ge1-\delta$ conditional on the graph and correct labels (Appendix~\ref{app:certificate}). It certifies this batch; it does not guarantee future-batch FDR $q$. Indeed its generic expectation bound is $q+(1-q)\delta$, or $0.145$ when $q=0.10,\delta=0.05$.

In the accurate Amazon attribute-score regime, 500 candidate-audit labels retain mean 211.5 true automatic discoveries, with nonempty output in 80.75\% of trials. The 864 training labels are additional. Tolokers and the original unsupervised scorers abstain. Whole-graph and degree-stratified audit allocations do not improve the evaluated candidate-uniform design. This is evidence for a useful established alternative in one score regime, not a new certification method or an equal-label-cost comparison with normal-reference acquisition.

\section{DISCUSSION AND LIMITATIONS}
The central empirical question is how much reference selection changes decisions made with useful scores. Matched comparisons, the graded removal experiment, and the conditional audit address different parts of that question. The controlled graph experiment identifies degree sensitivity as one sufficient route; it does not establish the unique cause of a trained detector's behavior. A low overall FDP can also reflect abstention, making power and discovery counts essential.

The evidence remains exploratory and benchmark-specific. Amazon provides the clearest useful-score regime; Tolokers is a repurposed classification task. The observed-exposure rule is a controlled sensitivity intervention, not an established acquisition policy whose prevalence we measured. The graph instances do not constitute independent samples from a deployment population, and seed SDs do not quantify cross-graph generalization. Simulations probe specified mechanisms, while random-design guarantees depend on actual label acquisition, role-independent scores, and the stated tie convention. Correct labels and a frozen candidate family are essential for direct certification. These limits motivate evaluation of realistic acquisition policies on additional independently collected graphs.

'''
    (D/'aistats.tex').write_text(preamble+maintext+ai+appendix)
    # Reinsert current unsupervised evidence; never the invalid historical tables.
    s=(D/'aistats.tex').read_text();mark=r'\section{IMPLEMENTATION AND REPRODUCIBILITY}'
    unsup=r'''\section{UNSUPERVISED COMPARISONS}
\label{app:unsupervised}
\begin{table*}[ht]\centering\small
\caption{Untrimmed, verified-label comparisons. FDP and power are means $\pm$ SD of ten seed averages, each averaged across five test splits and calibration repetitions. All zero-discovery outcomes are retained.}
\label{tab:untrimmed}\input{untrimmed_table.tex}
\end{table*}
Positive degree, fixed before evaluation, has AUROC $0.221$ on Amazon and $0.560$ on Tolokers; filtered mean FDP is $0.955$ and $0.758$. With 25\% of non-test labels revealed, DOMINANT filtered FDP is $0.973$ and $0.721$; with 50\%, it is $0.964$ and $0.726$. These scores have low power under random calibration. Table~\ref{tab:partial} reports the full FDP--power comparisons. Reference-only trimming and enriched test populations are separate sensitivity interventions in Table~\ref{tab:trim}, not definitions of the primary population.

'''
    s=s.replace(mark,unsup+mark,1)
    s=s.replace(r'Section~\ref{sec:simulation}',r'Appendix~\ref{app:simulation}')
    # Put the headline figure early; avoid excessive vertical paragraph stretching.
    s=s.replace(r'\begin{document}',r'\begin{document}'+'\n'+r'\raggedbottom',1)
    i=s.index(r'\begin{figure*}[t]'+'\n'+r'\centering\includegraphics[width=\textwidth]{headline_results.pdf}')
    j=s.index(r'\end{figure*}',i)+len(r'\end{figure*}')
    headline=s[i:j];s=s[:i]+s[j:]
    s=s.replace('Our contribution is an evaluation methodology with three components:',headline+'\n\n'+'Our contribution is an evaluation methodology with three components:',1)
    fig=old[old.index(r'\begin{figure*}[t]'+ '\n'+r'\centering\includegraphics[width=\textwidth]{independent_graphs.pdf}'):]
    fig=fig[:fig.index(r'\end{figure*}')+len(r'\end{figure*}')]
    s=s.replace(r'\label{app:indgraphs}',r'\label{app:indgraphs}'+'\n'+fig,1)
    (D/'aistats.tex').write_text(s)

if __name__=='__main__':main()
