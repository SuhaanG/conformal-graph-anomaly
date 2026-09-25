"""Foreground the existing batch certificate for the AISTATS audience."""
from pathlib import Path
D=Path(__file__).resolve().parent
MARK='% OPERATIONAL CERTIFICATE INTEGRATED'
def main():
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
Appendix~\ref{app:certificate} proves $\Prb_{\rm audit}(\FDP\le q)\ge1-\delta$ conditional on the entire graph and correct labels. This is a batch certificate, not a future-graph guarantee or an FDR-$q$ theorem. At $q=0.10,\delta=0.05$, Amazon's attribute scorer retains mean 211.5 true automatic discoveries using 500 audit labels, in addition to 864 training labels. Tolokers abstains. The construction uses established confidence bounds; its role here is to provide and evaluate an operational response to the diagnosed failure.

'''+s[end:]
        p.write_text(MARK+'\n'+s)

if __name__=='__main__':main()
