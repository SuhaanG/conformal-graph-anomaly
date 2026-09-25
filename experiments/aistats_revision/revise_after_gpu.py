"""Integrate the completed GPU replication; preserve the pre-follow-up draft."""
from pathlib import Path
D=Path(__file__).resolve().parent
p=D/'aistats.tex'
backup=D/'aistats_before_h200.tex'
if not backup.exists():backup.write_bytes(p.read_bytes())
s=backup.read_text()
title='Calibration Selection Can Inflate False Discoveries on Graphs'
s=s.replace('Calibration Selection, Rank Distortion, and False Discoveries on Graphs',title)

def replace_between(start,end,text):
    global s
    i=s.index(start);j=s.index(end,i)
    s=s[:i]+text+'\n\n'+s[j:]

replace_between(r'\begin{abstract}',r'\section{INTRODUCTION}',r'''\begin{abstract}
Selecting normal calibration nodes by their neighborhoods can make a conformal reference sample unrepresentative of normal test nodes. We study this effect by changing calibration selection while holding trained scores, test identities, and reference size fixed. Without score trimming, restricting calibration to nodes with no anomalous neighbors yields mean false discovery proportions of $0.936$ on Amazon and $0.740$ on Tolokers at nominal level $0.10$ across ten DOMINANT training seeds. Matched random calibration gives $0$ and $0.004$, respectively, but has little power. The effect persists with partially observed non-test labels and differs substantially across detectors. An exact conditional rank audit quantifies the reference shift on fixed graphs; experiments on 2,500 independently generated graphs separate degree sensitivity, neighborhood dependence, and discovery resolution. A comparison with weighted conformal selection in an independent covariate-shift model shows that valid correction can carry a substantial power cost. These results establish calibration selection as an explicit design choice in graph anomaly evaluation: normal-only references are insufficient unless their sampling rule represents the intended target population.
\end{abstract}''')

replace_between(r'\section{INTRODUCTION}',r'\section{STATISTICAL SETTING}',r'''\section{INTRODUCTION}
Graph anomaly detectors rank accounts, transactions, or other connected entities for investigation. A useful threshold should limit false discoveries while retaining anomalous cases. Conformal outlier testing compares each test score with a reference sample of normal scores; under suitable sampling and dependence assumptions, the resulting p-values support false discovery rate (FDR) control with the Benjamini--Hochberg (BH) procedure~\citep{bates2023testing,benjamini1995controlling}.

The labels in the reference sample are only part of this argument. A sample can contain exclusively normal nodes and still give misleading ranks if its selection rule removes normal nodes that resemble the test population. On a graph, selecting a node by its neighbors' labels also selects its structural environment. High-degree normal nodes have more opportunities to neighbor an anomaly. Excluding them can lower reference scores for a degree-sensitive detector, making other normal nodes appear anomalous.

We examine a concrete intervention: restrict calibration to normal nodes with no anomalous neighbors. We compare this rule with random normal calibration at identical size and on the same trained scores and test nodes. An oracle version uses all neighbor labels; partial-label versions use only a randomly revealed subset of non-test labels. The intervention is a diagnostic of selection bias, not a claim that this particular oracle rule is standard deployment practice.

Three findings organize the paper. First, the effect survives evaluation without score trimming: mean FDP for DOMINANT is $0.936$ on Amazon and $0.740$ on Tolokers at nominal $0.10$. These results exclude Amazon's unlabeled nodes from calibration and testing. Second, the effect depends on the score and discovery regime. GAE and Isolation Forest differ from DOMINANT, and Weibo can exhibit distorted null ranks with few discoveries. Third, correction involves a power tradeoff: random references, a fixed reference mixture, and weighted conformal selection can substantially reduce discoveries, and the mixture does not uniformly meet the nominal target.

Our contribution is an empirical and statistical account of this failure mode. Matched experiments isolate the calibration intervention. A finite-population audit uses the hypergeometric sampling law to compute exact null-tail probabilities conditional on arbitrary fixed graph scores. Independent graph simulations then vary degree heterogeneity, anomalous connectivity, and score sensitivity separately. Together these components connect a specified reference-sampling rule to rank distortion and distinguish the conditions under which that distortion translates into false discoveries.

\paragraph{Relation to prior work.}
Conformal inference relies on symmetry in the observations used to calibrate a score~\citep{vovk2005algorithmic}. Conformal outlier testing establishes conditions under which shared-reference p-values support BH~\citep{bates2023testing}. Graph conformal prediction studies coverage under appropriate graph symmetry and sampling conditions~\citep{huang2023uncertainty,zargarbashi2023conformal}; prediction-set coverage does not by itself give BH FDR control. Weighted conformal inference addresses covariate shift~\citep{tibshirani2019conformal}, and weighted conformal selection (WCS) explicitly addresses the multiple-testing dependence of weighted ranks~\citep{jin2026weighted}. We include WCS as a comparator where its independent-observation assumptions apply. Broader departures from exchangeability require corresponding assumptions~\citep{barber2023conformal}. Contaminated-reference methods address anomalous observations within calibration~\citep{bashari2025robust}; our main intervention instead selects among normals. The study complements graph-detector evaluations~\citep{ding2019deep,liu2024pygod,tang2023gadbench} by treating calibration sampling as part of the evaluated procedure.''')

s=s.replace('We therefore evaluate BH with these weights empirically in the independent simulation, without claiming a graph correction theorem.',
r'''We compare weighted BH with the hypothesis-conditional WCS procedure of \citet{jin2026weighted} in an independent simulation. This separates a marginal weighting correction from a multiple-testing procedure with a guarantee under its stated assumptions; neither supplies a correction theorem for our transductive graph experiments.''')

replace_between(r'\section{GRAPH EXPERIMENTAL DESIGN}',r'\section{INDEPENDENT VALIDATION OF THE MECHANISM}',r'''\section{GRAPH EXPERIMENTAL DESIGN}
The primary replication uses Amazon and Tolokers, with 11,944 and 11,758 nodes. Amazon's first 3,305 nodes are unlabeled in the official release; stored zero values at these indices must not be treated as verified normals~\citep{dglfraud}. Calibration and evaluation therefore use its 7,818 labeled normal and 821 anomalous nodes. Unsupervised training retains the full graph. Tolokers supplies 9,192 normal and 2,566 positive labels; it is a crowdworker classification graph repurposed here as a binary anomaly task~\citep{dou2020enhancing,platonov2023critical}.

\paragraph{Scores and replication.}
We train PyGOD DOMINANT and attribute-reconstruction GAE with hidden width 64, four layers, zero dropout, and 100 full-batch Adam steps at learning rate $0.01$~\citep{liu2024pygod}. Feature columns are standardized over the full graph; labels are not reconstruction targets. Isolation Forest provides an attribute-only comparator with 200 trees and subsample size 256. Positive node degree is a fixed, untrained control. Score directions and hyperparameters are fixed independently of evaluation FDP. Each learned detector uses ten seeds. GPU results are reported separately from CPU replications rather than pooled. Appendix~\ref{app:implementation} records the execution details.

\paragraph{Untrimmed test and reference populations.}
For each graph, five independent label-stratified splits assign approximately 25\% of labeled nodes to testing, retaining the graph's labeled class prevalence to rounding. We remove no high-score normals. The test sets contain 2,159 Amazon and 2,940 Tolokers nodes. Test identities are shared across training seeds, giving a crossed seed--split design. Normal nodes outside the test set form the reference pool.

The oracle filter retains reference normals with no anomalous neighbors, using all anomaly labels. For a partial-label intervention, we reveal each non-test labeled node independently with probability $\rho\in\{0.25,0.50\}$. Both calibration rules use only revealed normal nodes; the filter counts only revealed anomalous neighbors. No test labels enter this partial filter. Ground-truth test labels remain available to calculate evaluation metrics. Thus partial labeling concerns a labeled reference panel, not fully unsupervised deployment.

\paragraph{Matched comparisons and sensitivity.}
Filtered and random calibration use the same size $n=\min(b,|\mathcal P_F|)$, where $\mathcal P_F$ is the filtered reference pool and $b\in\{200,1000\}$. We report $b=1000$ in the main comparison and both budgets in the supplement. We also fix a half-mixture rule: sample $\lfloor n/2\rfloor$ filtered nodes, then fill the remaining positions uniformly from the reference pool without duplicating nodes. This fraction is not tuned on test FDP. We make 200 calibration draws for each score vector, test split, and comparison, preserving zero-discovery outcomes.

Sensitivity experiments trim the top 1\% or 5\% of reference-normal scores while leaving the primary test set unchanged. Separate enriched-test experiments include all anomalies and 2,000 normals, with 0\%, 1\%, or 5\% trimming of both normal pools. These are distinct target populations, not interchangeable estimates of deployment performance. Historical Amazon rows that counted unlabeled nodes as normal are retained only in the provenance supplement.

\paragraph{Metrics and uncertainty.}
All tests use $\alpha=0.10$. We average FDP, power, and discoveries first across calibration draws and the five test splits, then report the mean and sample SD of the ten training-seed averages. This SD describes training variation conditional on the graph and these splits; it is not uncertainty over independent graphs. The supplement also reports the range of split averages. For the degree control there is one fixed score vector. The 2,325,000 calibration-method evaluations in the complete GPU output are repeated evaluations of two graphs, not that many independent experiments.

\section{GRAPH RESULTS}
\begin{table*}[t]
\centering\small
\caption{Untrimmed GPU replication at nominal $\alpha=0.10$. FDP and power are mean $\pm$ SD across ten training-seed averages, each averaging five test splits and 200 calibration draws. Matched rules share test identities and reference size within every comparison; $b=1000$. AUROC summarizes detector ranking over the labeled population and is shared by calibration rules.}
\label{tab:untrimmed}
\input{untrimmed_table.tex}
\end{table*}

\paragraph{The DOMINANT effect survives removal of trimming.}
Table~\ref{tab:untrimmed} shows mean FDP $0.936$ on Amazon and $0.740$ on Tolokers under filtered calibration, with powers $0.449$ and $0.398$. Matched random calibration gives FDP $0$ and $0.004$, with powers $0$ and $0.0004$. The difference therefore survives both untrimmed testing and the Amazon label correction. The small random-reference rejection sets also make the practical limitation clear: reducing false discoveries here does not recover a useful detector.

\paragraph{The conditional audit identifies the reference shift.}
For these same untrimmed DOMINANT scores, Equation~\eqref{eq:exacttail} gives mean null fractions below $p=0.01$ of $0.536$ under filtering versus $0.00545$ under random calibration on Amazon, and $0.243$ versus $0.01015$ on Tolokers. These are exact conditional probabilities averaged over seeds and splits, rather than estimates from independent test nodes. The shift is detector-specific: the filtered fractions for GAE and Isolation Forest are below $0.01$ on both graphs. This links the main false-discovery contrast to the change in reference ranks, rather than to anomaly ranking alone.

\paragraph{Partial labels do not eliminate the effect.}
With 25\% of non-test labels revealed, filtered DOMINANT gives FDP $0.973$ on Amazon and $0.721$ on Tolokers. With 50\%, the values are $0.964$ and $0.726$. The corresponding random-reference means remain below nominal in these experiments. Table~\ref{tab:partial} reports the complete FDP--power comparison. Thus selection bias persists when the filter uses only a partially labeled reference panel and has no access to test-neighbor labels.

\paragraph{Detector ranking matters.}
GAE and Isolation Forest make no discoveries under the oracle filtered rule on either graph, whereas positive degree alone produces high FDP (Amazon $0.955$, Tolokers $0.758$). These contrasts support sensitivity to the score used; they do not establish that every graph model fails. DOMINANT itself ranks anomalies poorly here: its AUROC is $0.340$ on Amazon and $0.530$ on Tolokers. A weak ranking model can limit power even with representative calibration, but ranking weakness alone does not justify anti-conservative reference ranks. On Amazon, the partial-label 25\% filter also gives GAE mean FDP $0.107$ with power $0.037$; partial-label behavior need not interpolate monotonically between oracle filtering and random calibration.

\paragraph{Trimming and prevalence change the magnitude.}
In the verified-label, enriched-test design without trimming, DOMINANT's mean filtered FDP is $0.792$ on Amazon and $0.394$ on Tolokers. These values are lower than the primary natural-prevalence results, confirming that the headline magnitude depends on the evaluation population. Holding the untrimmed Tolokers test set fixed while trimming only 1\% of reference normals raises random-calibration FDP from $0.004$ to $0.427$; 5\% raises it to $0.604$. Removing difficult normals from a reference can therefore matter even without a neighborhood filter. Table~\ref{tab:trim} reports both interventions separately.

\paragraph{Distorted ranks need not produce many discoveries.}
A complementary five-seed Weibo GAE experiment illustrates the role of discovery resolution. Without trimming, Equation~\eqref{eq:exacttail} gives expected null fractions below $0.01$ of $0.0233$ for filtering and $0.0095$ for random calibration. Across repeated calibration draws, filtered mean FDP is $0.041$ with power $0.004$, while random calibration makes no discoveries. At 1\% trimming, filtered mean FDP is $0.249$. The exact audit and 120 Monte Carlo tail checks agree within $2.93$ Monte Carlo SEs (Appendix~\ref{app:fresh}). Rank distortion and BH false discoveries are distinct measurements; the crossing condition in Equation~\eqref{eq:crossing} explains why one does not automatically imply the other.

\section{INDEPENDENT GRAPH EXPERIMENTS}
\label{sec:graphs}
To vary the proposed mechanism independently of a learned architecture, we generate 2,000 graphs across four graph-parameter cells, plus 500 all-null controls. Each graph has 1,200 nodes and, outside the all-null control, 10\% anomalies. Independent lognormal node propensities have heterogeneity $\sigma\in\{0,1.2\}$; cross-class edges receive multiplier $c\in\{1,4\}$. Edge probabilities are scaled to an approximate mean degree of 12 and capped at $0.9$. We use 500 independent graph draws per cell. Four score conditions share each graph, so they are paired conditions rather than additional graph replications.

Let $L_v$ be standardized log-degree. The score is $S_v=4Y_v+\theta L_v+\varepsilon_v$, where $\theta\in\{0,2\}$. Noise is either independent Gaussian or a normalized mixture of independent Gaussian noise and neighbor-aggregated noise, with mixing fraction $\eta=0.5$. These are specified analytic scores, not trained graph detectors. We use 25\% stratified test samples, up to 200 calibration nodes, and the same oracle, random, fixed-mixture, and partial-label interventions as above. The independent graph is the replication unit for Monte Carlo SEs.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{independent_graphs.pdf}
\caption{Independent graph experiment: mean FDP across 500 graphs per graph-parameter cell, with $1.96$ Monte Carlo SE bars. Each graph is reused across four paired score conditions. Dashed lines mark $0.10$. Selection has its largest effect when heterogeneous degree enters the score; a half-mixture reduces inflation but does not meet the target in every cell. Complete FDP--power results, including partial-label and all-null controls, appear in Appendix~\ref{app:indgraphs}.}
\label{fig:indgraphs}
\end{figure*}

With independent, degree-insensitive scores ($\theta=0,\eta=0$), oracle-filter means range from $0$ to $0.082$. With $\sigma=1.2$, $c=4$, and degree sensitivity $\theta=2$, mean FDP is $0.751$ under filtering and $0.061$ under random calibration; powers are $0.979$ and $0.288$. Adding neighborhood noise gives similar means, $0.753$ and $0.067$, respectively. Thus this failure mode appears across independent graphs and survives the specified local noise dependence.

The mixture is a useful comparison, not a guaranteed remedy. In those two heterogeneous, high-connectivity conditions, its mean FDP is $0.119$ and $0.120$, above nominal despite being much lower than filtering. When the filter leaves a small reference pool, the resulting rank resolution can instead suppress discoveries even with informative scores. The all-null control has mean FDP $0$ under the oracle filter and $0.004$ under its matched random comparator; with no anomalies, the oracle exposure restriction is vacuous. Full parameter definitions and all outcomes are retained in the supplement.''')

# Move the earlier simulation figure to the appendix to make room for new evidence.
start=s.index(r'\begin{figure*}[t]',s.index(r'\section{INDEPENDENT VALIDATION'))
end=s.index(r'\end{figure*}',start)+len(r'\end{figure*}')
simfigure=s[start:end];s=s[:start]+s[end:]
insert=s.index(r'\paragraph{Contamination is a separate intervention.}')
s=s[:insert]+r'''\paragraph{A multiple-testing comparator with a stated guarantee.}
We additionally implement hypothesis-conditional WCS~\citep{jin2026weighted}, using the outlier-detection Algorithm 2 in the authors' versioned manuscript~\citep{jin2023algorithm}. A separate 2,000-repetition experiment for each $(\theta,q)$ with $q>0$ compares ordinary BH, weighted BH, deterministic WCS, homogeneous-randomized WCS, and weighted BY. At $\theta=2,q=0.05$, ordinary BH has mean FDP $0.573$ and power $0.793$. Weighted BH gives $0.0071$ and $0.0610$; homogeneous WCS gives $0.0037$ and $0.0328$. Deterministic WCS and weighted BY make no discoveries. All five comparators and both score-sensitivity settings are reported in Appendix~\ref{app:wcs}.

These results distinguish correcting a rank from correcting its use in multiple testing. WCS has an FDR guarantee under independent observations, the specified shift model, and valid density-ratio weights. Its low empirical FDP here comes with limited power. The simulations neither refute WCS nor extend its theorem to transductively learned graph scores.

''' +s[insert:]

replace_between(r'\section{DISCUSSION AND LIMITATIONS}',r'\clearpage',r'''\section{DISCUSSION AND LIMITATIONS}
Calibration selection is part of the statistical method. Selecting only normal nodes does not ensure that their reference scores represent normal test nodes. Our matched comparisons show a large effect for DOMINANT without trimming, including when only non-test neighbor labels are available. The independent graph experiment identifies degree-sensitive scoring as one sufficient route to this behavior in a controlled model. It does not establish degree as the unique cause in a trained detector.

The operational implication is to specify the target population and label-acquisition rule before choosing a calibration design. With a labeled audit set, compare reference rules at fixed sample size and test identities, inspect null tails, and report FDP alongside power, including zero-discovery runs. The exact conditional audit makes the first comparison inexpensive once scores are available. Mean score gaps, random sampling alone, or an observed mean FDP below nominal cannot certify graph FDR control.

The benchmark evidence concerns two primary graph instances, transductive scores, fixed model settings, and a partially labeled reference panel; the reported seed SDs do not quantify new-graph uncertainty. Several detectors rank these benchmark labels poorly. Tolokers is a repurposed classification task, and the older Weibo results are sensitive to trimming. The synthetic graph conclusions depend on the specified generator and analytic scoring family. Finally, density-ratio weighting requires overlap and a defensible conditional score law; WCS supplies a multiple-testing solution under its own independent-observation assumptions.

The central design lesson is to evaluate how a calibration sample is acquired, alongside how a detector is trained. A matched comparison and a conditional rank audit distinguish reference-selection failures from weak ranking and limited discovery resolution. This provides a concrete starting point for developing graph calibration methods that preserve both useful power and a defensible error guarantee.''')

s=s.replace('designing the score simulation and conditional rank audit','designing the score and graph simulations and conditional rank audit')
s=s.replace('The accompanying code regenerates the synthetic trials, tables, and figure and audits the archived graph CSVs.',
'''The accompanying code regenerates the simulations and tables and audits completed graph outputs. The primary GPU batch contains 60 model--seed configurations and two fixed degree controls, producing 2,325,000 calibration-method evaluations. Every stored summary mean was recomputed from its per-draw CSV, and all 60 job checksum manifests were verified after download. Training used Python 3.13.14, PyTorch 2.13.0+cu126, PyG 2.8.0, PyGOD 1.1.0, and one NVIDIA H200 NVL, with TF32 disabled. The complete package records versions, hashes, scores, individual draws, and completion markers. CPU and GPU results are not pooled.''')
s=s.replace('Seeds per configuration & 5', 'Training seeds & 10')
s=s.replace('Normal-score trimming & Upper 1\\%', 'Score trimming & None')
s=s.replace('Normal test size & 2,000 (all graphs)', 'Test fraction & 25\\% stratified')
s=s.replace('Larger calibration size & 4,000 (all graphs)', 'Reference budget & 200 and 1,000')
s=s.replace('Settings shared by the main graph comparisons.', 'Neural-model settings for the primary graph comparisons.')
s=s.replace('This arm has a different resolution from Table~\\ref{tab:graph}; differences cannot be attributed only to selection.',
r'''These are historical trimmed, enriched-test runs with different resolution from Table~\ref{tab:graph}. Amazon includes unlabeled nodes treated as normal and is retained only as a provenance record, not verified-normal evidence.''')
s=s.replace('An existing one-configuration audit compares five detectors per graph with node degree and negative node degree.',
'''An existing one-configuration audit compares five detectors per graph with node degree and negative node degree. This twenty-cell matrix is not a matched calibration experiment, and its Amazon entries use the historical unlabeled-as-normal convention.''')
s=s.replace('In 16 of the 20 detector--graph combinations, detector AUROC exceeds the better of these two baselines by no more than $0.02$.',
'''We do not use its aggregate count as evidence for the corrected primary results.''')
s=s.replace('The better degree direction has AUROC $0.745$ on Amazon, $0.560$ on Tolokers, $0.778$ on Weibo, and $0.556$ on Reddit.',
'''The new positive-degree control instead fixes its direction before evaluation: its AUROC is $0.221$ on verified Amazon labels and $0.560$ on Tolokers. Its high filtered FDP establishes that learned message passing is not necessary for this particular selection effect.''')

addition=r'''
\section{COMPLETE FOLLOW-UP COMPARISONS}
\label{app:followup}
\begin{table*}[t]
\centering\small
\caption{Untrimmed DOMINANT comparisons, reported as mean FDP / power across all ten seeds, five splits, and 200 calibration draws. The 25\% and 50\% rows reveal only non-test labels. The 100\% row is the oracle intervention, which can use test-neighbor labels. Mixture composition is fixed in advance.}
\label{tab:partial}\input{partial_table.tex}
\end{table*}
\begin{table*}[t]
\centering\small
\caption{DOMINANT trimming and target-population sensitivity. Natural-prevalence reference-only trimming preserves test identities. Enriched-test rows include every anomaly and 2,000 normal test nodes; trimming both pools can also change test identities and reference size. Amazon uses verified labels throughout this table. All entries average ten seeds, five splits, and 200 calibration draws.}
\label{tab:trim}\input{trimming_table.tex}
\end{table*}
The supplement's \texttt{followup\_aggregate.csv} retains both reference budgets, all three learned detectors, the positive-degree control, all label fractions, both trimming designs, and the explicitly labeled historical Amazon sensitivity. It reports seed SDs and split-average ranges separately. A split is reused across training seeds, so these observations must not be treated as independent graph replicates. The ten-seed protocol, fixed mixture, and primary untrimmed comparison were written before inspecting their outcomes. The additional GPU GAE replication was specified after initial CPU DOMINANT results were available; no configuration was selected using the GPU FDP results.

An additional audit independently reconstructs all 600 primary learned-detector conditions (two graphs, three detectors, ten seeds, five splits, and two reference rules) at $b=1000$. For each condition, 500 fresh calibration draws estimate the null fraction below $0.01$, using master seed 20260921. The maximum absolute difference from Equation~\eqref{eq:exacttail} is $0.000721$. All 300 nonconstant conditions agree within $2.98$ Monte Carlo SEs; the other 300 conditions select the entire eligible filtered pool and agree exactly. SEs are computed across calibration draws, not across nodes. The code and all outcomes are supplied as \texttt{audit\_primary\_ranks.py} and \texttt{primary\_rank\_validation.csv}.

\section{INDEPENDENT GRAPH DETAILS}
\label{app:indgraphs}
\begin{table*}[t]
\centering\scriptsize
\caption{All independent graph conditions: mean FDP / power over 500 graph draws per row. Four score conditions share each non-null graph. Partial-label comparisons reveal 25\% of non-test labels. Power is recorded as zero by convention in the all-null control, where no alternatives exist.}
\input{independent_full_table.tex}
\end{table*}
For each graph, choose 120 anomalous nodes uniformly among 1,200. Generate independent $Z_i\sim N(0,1)$, set $a_i=\exp(\sigma Z_i-\sigma^2/2)$, and divide $a_i$ by its sample mean. For $i<j$, let $b_{ij}=c$ for different labels and 1 otherwise, and let $\bar b$ be the mean over unordered pairs. Edges are independent conditional on propensities and labels, with probability $\min\{0.9,12a_ia_jb_{ij}/(1199\bar b)\}$. The code records clipping fractions and realized edge counts. This normalization targets comparable density approximately, rather than conditioning all graphs on an identical edge count.

Let $z_i$ be new independent standard normals and $d_i$ the realized degree. Write $h_i=\sum_j A_{ij}z_j/\sqrt{d_i\vee1}$. Noise is
\[
\varepsilon_i=\frac{\sqrt{1-\eta}\,z_i+\sqrt{\eta}\,h_i}
{\sqrt{1-\eta+\eta\ind\{d_i>0\}}}.
\]
This preserves unit marginal variance conditional on the graph, including isolated nodes, while permitting shared-neighbor dependence when $\eta>0$. Standardized $\log(1+d_i)$ supplies $L_i$. The same graph, test set, and calibration assignments are reused across the four $(\theta,\eta)$ conditions. The all-null control uses $(\sigma,c,\theta,\eta)=(1.2,1,2,0.5)$ and zero anomalies. Master seed 20260920, graph index, and graph parameters determine separate NumPy seed streams. There are 2,500 unique graphs and 42,500 method evaluations; full Monte Carlo SEs and reference sizes are in the supplied CSVs.

\section{WEIGHTED CONFORMAL SELECTION COMPARATOR}
\label{app:wcs}
\begin{table*}[t]
\centering\small
\caption{Complete WCS comparison. FDP entries give mean (Monte Carlo SE) over 2,000 independent repetitions; power entries are means. At $q=1$, ordinary and weighted BH and both WCS variants coincide in this simulation. Zero-discovery runs are retained.}
\input{wcs_full_table.tex}
\end{table*}
This experiment uses master seed 20260916, 200 independent calibration observations, 400 test observations including 360 nulls, and the same two-stratum score model as Section~\ref{sec:simulation}. Values $q=1,0.25,0.05$ ensure positive support. The $q=1$ cell is the unselected ordinary-BH reference. Methods share data within a repetition; repetitions are independent. Homogeneous WCS pruning uses a separate uniform draw shared across its selected hypotheses. BY uses nominal level $0.10/\sum_{j=1}^{400}1/j$ on the weighted p-values~\citep{benjamini2001control}.

Our implementation follows the nonrandomized ranks and hypothesis-conditional Algorithm 2 in the versioned source~\citep{jin2023algorithm}. It was checked against a literal equation implementation on 80 random cases, including weight rescaling and test permutation, and against six constant-weight cases of the authors' reference implementation. The inspected reference file used a different test-weight index in an auxiliary numerator and randomized ranks; variable-weight agreement with that file is therefore not claimed. The source hash and precise discrepancy are recorded in \texttt{wcs\_validation.json}. This is implementation provenance, not an empirical comparison against running the authors' variable-weight software.

\section{HISTORICAL GRAPH RESULTS AND INCLUSION RULES}
\label{app:historical}
\begin{table*}[t]
\centering\small
\caption{Historical matched results, retained for provenance. These five-seed runs trim the top 1\% of normal scores and include all anomalies with 2,000 normal test nodes. Amazon additionally treats unlabeled nodes as normal, so those rows are not evidence about a verified-normal population. Primary conclusions instead use Table~\ref{tab:untrimmed}.}
\label{tab:graph}\input{graph_table.tex}
\end{table*}
The earlier test construction excluded the highest-scoring 1\% of normal nodes from both calibration and normal test eligibility. Its matched reference size was constrained by the zero-exposure, positive-exposure, and unrestricted pools, reflecting an additional historical arm. These choices differ from the primary replication. The broader historical calibration-distribution matrix also changes reference size and test composition between conditions; it cannot be restored as a matched test of the selection mechanism. We retain its source CSVs and distinguish it from both the corrected primary comparisons and the twenty-cell degree-ranking audit. Exploratory coverage does not substitute for holding test identities and calibration size fixed.

The archived tail ratio $g(t)$ is the fraction of null p-values below $t$, divided by $t$, using stored $t=\operatorname{round}(\max\{10/(n+1),10^{-4}\},5)$. It is a finite-threshold description, not a supremum inflation bound or an FDR test. Archived Amazon metrics are not relabeled as corrected results. Source and exclusion details are kept so every prior number can be traced without supporting the current claims with an invalid label convention.
'''
s=s.replace(r'\label{app:simulation}',r'\label{app:simulation}'+'\n'+simfigure)
s=s.replace('Averaging and uncertainty follow Table~\\ref{tab:audit}.',
            'Each cell averages 200 calibration draws, then reports mean and SD across five score/test realizations.')
s=s.replace(r'\input{graph_audit_full_table.tex}',r'\label{tab:audit}\input{graph_audit_full_table.tex}')
s=s.replace(r'\end{document}',addition+'\n'+r'\end{document}')
p.write_text(s)
b=D/'aistats_refs.bib';refs=b.read_text()
if 'jin2026weighted' not in refs:
    refs+=r'''
@article{jin2026weighted, author={Jin, Ying and Cand{\`e}s, Emmanuel J.}, title={Model-free selective inference under covariate shift via weighted conformal p-values}, journal={Biometrika}, volume={113}, number={1}, pages={asaf066}, year={2026}, doi={10.1093/biomet/asaf066}}
@misc{jin2023algorithm, author={Jin, Ying and Cand{\`e}s, Emmanuel J.}, title={Model-free selective inference under covariate shift via weighted conformal p-values}, year={2023}, note={arXiv:2307.09291v2, Algorithm 2}, url={https://arxiv.org/html/2307.09291v2}}
@misc{dglfraud, author={{DGL Contributors}}, title={FraudDataset source: Amazon label mask and dataset statistics}, year={2026}, url={https://www.dgl.ai/dgl_docs/_modules/dgl/data/fraud.html}, note={Accessed September 15, 2026}}
'''
    b.write_text(refs)
print('Revised',p)
if (D/'integrate_audit_results.py').exists():
    import runpy
    runpy.run_path(str(D/'integrate_audit_results.py'),run_name='__main__')
