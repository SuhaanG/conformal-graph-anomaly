"""Idempotent manuscript changes prompted by the workshop reviews."""
from pathlib import Path
D=Path(__file__).resolve().parent
MARK='% WORKSHOP FEEDBACK REVISION INTEGRATED'

def main():
    p=D/'aistats.tex';s=p.read_text()
    if MARK in s:return
    def replace(old,new):
        nonlocal s
        assert old in s,old[:100]
        s=s.replace(old,new,1)
    replace('Graph anomaly detectors rank accounts, transactions, or other connected entities for investigation. A useful threshold should limit false discoveries while retaining anomalous cases.',
        'Consider a fraud analyst ranking accounts linked by shared transactions. To set an investigation threshold, the analyst collects verified normal accounts as a reference. Excluding normals adjacent to known fraud may seem prudent, yet can remove precisely the structural environments encountered during deployment. A useful threshold must limit false discoveries while retaining anomalous cases.')
    replace('We train PyGOD DOMINANT and attribute-reconstruction GAE with hidden width 64, four layers, zero dropout, and 100 full-batch Adam steps at learning rate $0.01$~\\citep{liu2024pygod}.',
        'We train DOMINANT~\\citep{ding2019deep} and PyGOD\'s attribute-reconstruction GAE~\\citep{liu2024pygod} with hidden width 64, four layers, zero dropout, and 100 full-batch Adam steps at learning rate $0.01$.')
    replace('Isolation Forest provides an attribute-only comparator',
        'Isolation Forest~\\citep{liu2008isolation} provides an attribute-only comparator')
    replace('Appendix~\\ref{app:implementation} records the execution details.',
        'These scores contrast joint reconstruction, attribute reconstruction, and attribute-only isolation; they are a mechanism-oriented subset, not an exhaustive benchmark. Appendix~\\ref{app:implementation} records execution details and dataset choices.')
    replace('Rank distortion and BH false discoveries are distinct measurements; the crossing condition in Equation~\\eqref{eq:crossing} explains why one does not automatically imply the other.',
        'Neither a small mean-degree shift nor weak score--degree correlation rules out changes elsewhere in the null-score distribution. Degree is therefore a sufficient mechanism in our controlled model, not a complete explanation of Weibo. Rank distortion and BH discovery are distinct, linked by Equation~\\eqref{eq:crossing}.')
    replace('Shared calibration and graph training induce dependence among p-values.',
        'Shared calibration and graph training induce dependence among p-values: their joint distribution matters, not only each individual rank.')
    replace('The stored standardized gap has calibration-minus-test sign; the reported $\\Delta$ reverses that sign.',
        r'''The stored standardized gap has calibration-minus-test sign; the reported $\Delta$ reverses that sign. Specifically, $\Delta=(\bar s_{T_0}-\bar s_C)/s_{\rm pool}$, with
\[
 s_{\rm pool}^2=\frac{(n_C-1)\hat\sigma_C^2+(n_{T_0}-1)\hat\sigma_{T_0}^2}{n_C+n_{T_0}-2},
\]
where $\hat\sigma^2$ denotes the sample variance with denominator $n-1$. This is a descriptive standardized mean difference, not a test of exchangeability or a sufficient diagnostic of rank validity.''')
    replace('The GAE score is the sum of squared attribute reconstruction errors.',
        r'''The GAE score is the sum of squared attribute reconstruction errors. PyGOD traces its GAE family to \citet{kipf2016variational}; our deterministic attribute-decoder configuration is not the variational adjacency-reconstruction model of that paper.''')
    replace('Relation types in Amazon are collapsed into a common graph.',
        'Relation types in Amazon are collapsed into a common graph. Amazon supplies a fraud task with an explicit verified-label mask; Tolokers supplies a contrasting, less favorable score regime. Both allow full-graph reconstruction with the available compute. Auxiliary Weibo adds a separate rank-distortion example. This selection supports controlled within-graph comparisons rather than a claim of coverage across application domains.')
    anchor=r'\section{AUXILIARY DEGREE BASELINE}'
    replace(anchor,r'''\paragraph{Scope beyond reconstruction detectors.}
The statistical object is the conditional normal-score law after reference selection, regardless of how the scorer was trained. Semi-supervised, positive--unlabeled, cross-domain, and generalist graph detectors can therefore face the same calibration question, but this reasoning does not establish an empirical failure for all of them. The supervised controls in Appendix~\ref{app:supervised} directly extend the evidence beyond unsupervised reconstruction. In positive--unlabeled learning, an unlabeled pool cannot simply be treated as verified normal; contamination and selection must be distinguished. Cross-domain transfer adds a further target-shift problem. We do not evaluate generalist or cross-domain models here.

'''+anchor)
    old='For each configuration we run 2,000 independent repetitions, using a fixed master seed and a separate random stream per repetition. We report mean FDP, power, and null-tail probabilities, with Monte Carlo standard errors calculated over repetitions. The design and complete output, including unfavorable and zero-discovery cases, are given in Appendix~\\ref{app:simulation}. These simulations validate an independent score model and implementation; they do not reproduce graph dependence.'
    replace(old,'Each configuration uses 2,000 independent repetitions. Monte Carlo SEs and all FDP, power, and null-tail results appear in Appendix~\\ref{app:simulation}; these experiments validate the independent score model, not graph dependence.')
    p.write_text(MARK+'\n'+s)

if __name__=='__main__':main()
