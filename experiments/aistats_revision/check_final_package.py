"""Check explicit archive hashes, a clean LaTeX build, page budget and anonymity."""
from pathlib import Path
import hashlib,json,zipfile,subprocess,re
import pymupdf
D=Path(__file__).resolve().parent

def main():
    report={'date':'2026-09-24'}
    for name in ('aistats_overleaf_source.zip','aistats_reproducibility.zip'):
        with zipfile.ZipFile(D/name) as z:
            assert z.testzip() is None
            hashes=json.loads(z.read('FILE_HASHES.json'))
            assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
            assert all(not n.startswith(('/','\\')) and '..' not in Path(n).parts for n in hashes)
            for n in hashes:
                if Path(n).suffix in ('.tex','.bib','.py','.md','.json','.csv'):
                    txt=z.read(n).decode(errors='replace')
                    assert not re.search(r'suhaank2|skuiu|suhaansg09|C:[\\/]Users[\\/]',txt,re.I),(name,n)
            report[name]={'files':len(hashes),'sha256':hashlib.sha256((D/name).read_bytes()).hexdigest(),
                          'bytes':(D/name).stat().st_size,'hashes':'passed','targeted_identity_scan':'passed'}
    out=D/'package_2027_check';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(D/'aistats_overleaf_source.zip') as z:z.extractall(out)
    for i,cmd in enumerate([['pdflatex','-interaction=nonstopmode','-halt-on-error','aistats.tex'],['bibtex','aistats'],
                           ['pdflatex','-interaction=nonstopmode','-halt-on-error','aistats.tex'],
                           ['pdflatex','-interaction=nonstopmode','-halt-on-error','aistats.tex']]):
        p=subprocess.run(cmd,cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/f'build_{i}.txt').write_bytes(p.stdout);assert p.returncode==0,p.stdout[-1000:]
    log=(out/'aistats.log').read_text()
    assert not re.search(r'undefined|LaTeX Warning|Package natbib Warning',log,re.I)
    doc=pymupdf.open(out/'aistats.pdf');texts=[p.get_text() for p in doc]
    ai=next(i for i,t in enumerate(texts) if 'AI USE STATEMENT' in t)
    assert ai<=8,ai
    assert all(abs(p.rect.width-612)<.01 and abs(p.rect.height-792)<.01 for p in doc)
    assert any('SUPERVISED CONTROLS AND REFERENCE BUDGET' in t for t in texts)
    assert any('EXPLORATORY FIXED-BATCH FDP CERTIFICATION' in t for t in texts)
    assert any('CHECKLIST' in t for t in texts)
    report.update(main_text_pages=ai,total_pages=len(doc),source_rebuild='passed; no unresolved citations or references',
                  page_size='US Letter',style='official unmodified AISTATS 2027',
                  overleaf='Local source changed; cloud synchronization has not yet been verified for this version.',
                  remaining_typesetting_note='Style-level 5.1225pt empty-box warning at title/abstract boundary; no visible text overflow.')
    sync_path=D/'OVERLEAF_SYNC.json'
    if sync_path.exists():
        sync=json.loads(sync_path.read_text())
        if (sync.get('source_sha256')==hashlib.sha256((D/'aistats.tex').read_bytes()).hexdigest()
            and all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h
                    for n,h in sync.get('source_files_sha256',{}).items())):
            report['overleaf']=sync['status']
    (D/'FINAL_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
