"""Optional reference-code verification dependency; do not silently use a changed version."""
from pathlib import Path
from urllib.request import urlopen
import hashlib,json
out=Path(__file__).resolve().parent
record=json.loads((out/'wcs_validation.json').read_text())
url='https://raw.githubusercontent.com/ying531/conformal-selection/main/confselect-python/ConfSelect/confselect.py'
data=urlopen(url,timeout=60).read()
assert hashlib.sha256(data).hexdigest()==record['source_sha256'],'Upstream reference changed: inspect before executing'
(out/'vendor').mkdir(exist_ok=True)
(out/'vendor/confselect_reference.py').write_bytes(data)
print('Reference source downloaded and checksum verified.')
