"""Apply the official 2027 format without modifying the vendor styles."""
from pathlib import Path
import zipfile, hashlib, json
D=Path(__file__).resolve().parent

def main():
    z=zipfile.ZipFile(D/'AISTATS2027PaperPack.zip')
    for name in ('aistats2027.sty','fancyhdr.sty'):
        assert (D/name).read_bytes()==z.read('AISTATS2027PaperPack/'+name)
    sample=z.read('AISTATS2027PaperPack/sample_paper.tex').decode()
    start=sample.index('\\begin{enumerate}',sample.index('\\section*{Checklist}'))
    end=sample.index('\\clearpage',start)
    checklist=sample[start:end].strip()
    answers=[
      r'Yes. Sections 2--4 and Appendices A, L, and M specify the settings, assumptions, and procedures.',
      r'No. Reference sizes and audit budgets are reported, but a full time/space complexity analysis of every baseline is not provided.',
      r'Yes. The anonymized supplement includes source, version manifests, and reproduction instructions.',
      r'Yes. Section 3 and Appendices A and M state the assumptions.',
      r'Yes. Appendix A proves the score/rank statements; Appendix M proves the batch certificate.',
      r'Yes. Independence, support, fixed-batch conditioning, and sampling-design assumptions are distinguished.',
      r'Yes. The supplement contains scripts, inputs or source instructions, saved scores, and individual outcomes.',
      r'Yes. Section 4 and Appendices B and L give fixed settings and split construction; exploratory extensions are identified.',
      r'Yes. Section 4 and table captions distinguish seed SDs, Monte Carlo SEs, and repeated evaluations of a fixed graph.',
      r'Yes. Appendix B describes H200 training; Appendix L and the version manifests document the CPU extension.',
      r'Yes. Dataset, model, and statistical-method sources are cited.',
      r'No. Source provenance is recorded, but a consolidated inventory of all asset licenses is not included.',
      r'Yes. New code, saved scores, synthetic results, and checks are in the supplement.',
      r'No. We reuse public benchmark releases and did not collect new provider-consent information.',
      r'Not Applicable. The experiments use released numerical graph data and synthetic observations; no new personally identifying or offensive content is curated.',
      r'Not Applicable. We did not recruit participants or conduct a new human-subject study.',
      r'Not Applicable. We did not recruit participants or conduct a new human-subject study.',
      r'Not Applicable. We did not recruit participants or conduct a new human-subject study.',
    ]
    assert checklist.count('[Yes/No/Not Applicable]')==len(answers)
    for answer in answers:checklist=checklist.replace('[Yes/No/Not Applicable]','['+answer+']',1)
    (D/'checklist.tex').write_text('\\section*{Checklist}\n'+checklist+'\n')
    p=D/'aistats.tex';s=p.read_text()
    s=s.replace('% AISTATS 2027 revision; official 2026 style used provisionally.','% AISTATS 2027 revision; official unmodified 2027 style.')
    s=s.replace(r'\usepackage{aistats2026}',r'\usepackage{aistats2027}')
    if r'\input{checklist.tex}' not in s:s=s.replace(r'\clearpage\appendix',r'\clearpage\input{checklist.tex}'+'\n'+r'\clearpage\appendix\onecolumn')
    p.write_text(s)
    if (D/'revise_after_reviews.py').exists():
        import runpy
        runpy.run_path(str(D/'revise_after_reviews.py'),run_name='__main__')
    if (D/'integrate_operational.py').exists():
        import runpy
        runpy.run_path(str(D/'integrate_operational.py'),run_name='__main__')
    (D/'STYLE_PROVENANCE.json').write_text(json.dumps(dict(url='https://aistats.org/aistats2027/AISTATS2027PaperPack.zip',
        downloaded='2026-09-21',archive_sha256=hashlib.sha256((D/'AISTATS2027PaperPack.zip').read_bytes()).hexdigest(),
        styles={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('aistats2027.sty','fancyhdr.sty')},
        vendor_styles_unmodified=True),indent=2))
if __name__=='__main__':main()
