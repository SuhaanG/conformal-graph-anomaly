"""Create an allowlisted portable experiment bundle without existing outputs."""
from pathlib import Path
import hashlib, json, zipfile
ROOT=Path(__file__).resolve().parents[2]
FILES=['src/detector.py','src/detectors.py','paper/aistats_revision/validate_revision.py',
       'paper/aistats_followup/train_headlines.py','paper/aistats_followup/evaluate_headlines.py',
       'paper/aistats_followup/h200_runner.py','paper/aistats_followup/run_h200.sh',
       'paper/aistats_followup/H200_README.md','paper/aistats_followup/PROTOCOL.md',
       'paper/aistats_followup/Amazon.mat','tmp_tolokers/tolokers/raw/tolokers.npz']
dest=ROOT/'paper/aistats_followup/aistats_h200_bundle.zip'
hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in FILES}
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(ROOT/f,'aistats_h200_bundle/'+f)
    z.writestr('aistats_h200_bundle/H200_BUNDLE.txt','Isolated GPU replication. See paper/aistats_followup/H200_README.md.\n')
    z.writestr('aistats_h200_bundle/input_checksums.json',json.dumps(hashes,indent=2))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for f,digest in hashes.items():
        assert hashlib.sha256(z.read('aistats_h200_bundle/'+f)).hexdigest()==digest
print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))
