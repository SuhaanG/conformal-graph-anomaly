from pathlib import Path
import json,hashlib,zipfile
D=Path(__file__).resolve().parent
s=(D/'aistats.tex').read_text(encoding='utf-8');h=14695981039346656037
for c in s.replace('\r\n','\n'):h=((h^ord(c))*1099511628211)&((1<<64)-1)
assert format(h,'x')=='f1ef31bc64a3ad10','Cloud source fingerprint mismatch'
with zipfile.ZipFile(D/'aistats_overleaf_source.zip') as z:hashes=json.loads(z.read('FILE_HASHES.json'))
sync=dict(status='Synchronized 2026-09-24: 45 source/figure uploads completed; normalized main source fingerprint matches; cloud PDF 44 pages, Errors 0 and Warnings 0.',
    project='https://www.overleaf.com/project/6aa4b3a8249d9edf82646a2b',
    source_sha256=hashlib.sha256((D/'aistats.tex').read_bytes()).hexdigest(),
    source_files_sha256=hashes,cloud_pdf_pages=44,errors=0,warnings=0,
    source_text_fingerprint=dict(algorithm='FNV-1a 64 over Unicode code points, normalized line endings',value=format(h,'x')),
    verification_scope='Main source copied from visible editor and fingerprint compared; all source files uploaded in one completed batch; compiled page count and logs verified. Binary cloud files were not downloaded for hash comparison.')
(D/'OVERLEAF_SYNC.json').write_text(json.dumps(sync,indent=2))
p=D/'FINAL_VALIDATION.json';report=json.loads(p.read_text());report['overleaf']=sync['status']
report['allocation_acquisition']={'evaluations':135000,'previous_outcomes_reproduced':5400,'reaggregation':'passed','nonvacuous_finite_design_checks':'passed'}
report['pdf_sha256']=hashlib.sha256((D/'aistats.pdf').read_bytes()).hexdigest()
p.write_text(json.dumps(report,indent=2))
(D/'SUBMISSION_READINESS.md').write_text('''# AISTATS draft status — 24 September 2026

Completed: seven main-text pages, 44 pages total including references/checklist/appendices. Official unmodified 2027 style. Clean source-package build, no unresolved citations/references, visual layout reviewed. Source and 611-file reproducibility ZIP hashes and targeted identity scans pass. Overleaf received all 45 source/figure files; main source fingerprint matches local, cloud build has 44 pages, Errors 0 and Warnings 0. The official style emits one non-visible empty-box typesetting message.

Scientific additions: T-Finance and GADBench Weibo extension, explicit PyGOD/GADBench source discrepancy, stratified-role lemma, pooled-block PRDS argument, size-based allocation, biased label acquisition, complete FDP/power results, paired intervals and selection-conditional/graph-conformal citations. Allocation/acquisition adds 135,000 outcomes, reproduces 5,400 previous outcomes and passes finite-design checks with nonzero false discoveries. Pilot optimization is parked. Weighted BH is descriptive; its FDR guarantee is not claimed.

Authors still need to verify the proofs line by line, review all scientific claims and approve the final manuscript. A successful build is not scientific acceptance. No submission, withdrawal, exemption form or message to chairs has been sent.

Administrative items: correct OpenReview author profiles, frozen author list by abstract deadline, reciprocal reviewer or justified exemption. TAE acceptance is non-archival; confirm applicability of AISTATS wording if the workshop version exceeds the explicit four-page concurrent-workshop exception. The complete local reproducibility ZIP is about 177 MB; check the actual upload field's size limit before choosing that archive. No upload-limit value was stated in the public CFP/FAQ checked today.

Official sources: https://virtual.aistats.org/Conferences/2027/CallForPapers and https://virtual.aistats.org/Conferences/2027/SubmissionFAQ . Abstract deadline September 29; full paper and supplement October 6, 2026, 23:59 AoE.
''',encoding='utf-8')
(D/'ALLOCATION_REVISION_NOTES.md').write_text('''# Allocation/acquisition decision

The simpler follow-up was implemented instead of the proposed pilot. Both alternate stratified procedures have an argument under the specific random-role product law. Size-proportional alpha is fixed conditional on realized stratum counts; it is not fixed before all normal identities are assigned. Independent PRDS blocks permit one global BH only under the stated factorization.

Results contradict a universal power gain from removing equal alpha splitting. In Amazon's prespecified 80/20, half-revelation attribute setting, power is .581 equal, .277 size and .492 pooled-stratum, versus .779 full random. Error allocation and reference resolution both matter; low mean FDP does not establish the source of a power loss.

Exposure-biased acquisition creates a real use case, but the remedy's efficiency varies. Moderate bias on T-Finance graph features yields FDP/power .154/.818 for pooled biased reference, .092/.640 for pooled stratified ranks, and .095/.785 for descriptive weighted BH. BY is the conservative weighted method with the supplied marginal-validity argument. Complete tables retain adverse and neutral cases, both partitions, both scorers and all acquisition severities.

Next scientific priority is independent human verification of the finite-design proof and a focused first-two-pages read. Do not claim a novel general conformal principle or a numerical acceptance probability. Further method development should address the demonstrated power/resolution gap, not rename the standard remedy.
''')
print('Cloud fingerprint matches; readiness and validation records updated')
