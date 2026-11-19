import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('matrix',Path(__file__).resolve().parents[1]/'tools/acceptance/matrix.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_production_pass_cannot_override_failed_prerequisite(tmp_path):
    for layer,status in ((1,'failed'),(2,'passed'),(3,'passed'),(4,'passed'),(5,'passed')):
        folder=tmp_path/f'layer{layer}'/'h2';folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps({'status':status}))
    row=module.matrix(tmp_path,['h2'])[0]
    assert row['layer2']=='passed'
    assert row['layer3']=='blocked_by_1'
    assert row['layer5'].startswith('blocked_by_1')


def test_missing_evidence_never_passes(tmp_path):
    row=module.matrix(tmp_path,['g1'])[0]
    assert row['layer1']=='not_run'
    assert row['layer2']=='not_run'
    assert row['layer3']=='blocked_by_1,2'
