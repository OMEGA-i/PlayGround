import shutil
from pathlib import Path

import pytest

from omega_playground.policy import package, verify
from omega_playground.policy.schedule import Schedule, physics_steps

SONIC = Path(__file__).resolve().parents[2] / "artifacts/policies/sonic-v1.1-g1"
needs_sonic = pytest.mark.skipif(not SONIC.exists(), reason="exported SONIC package not present in artifacts/")


def test_schedule_rejects_non_integer_periods():
    assert physics_steps(1 / 30, 120) == 4
    assert Schedule("event", poll_s=0.1).physics_steps(200) == 20
    with pytest.raises(ValueError):
        physics_steps(1 / 30, 200)
    with pytest.raises(ValueError):
        Schedule("periodic").physics_steps(120)




@needs_sonic
def test_sonic_fixture_parity():
    pytest.importorskip("onnxruntime")
    report = verify.verify(package.load(SONIC))
    assert report["passed"], {k: v for k, v in report["checks"].items() if not v["passed"]}


@needs_sonic
def test_tampered_package_is_rejected(tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(SONIC, copy)
    module = next((copy / "modules").glob("*.onnx"))
    module.chmod(0o600)
    with module.open("r+b") as f:
        f.seek(100)
        byte = f.read(1)
        f.seek(100)
        f.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(package.PackageError, match="sha256"):
        package.load(copy)


@needs_sonic
def test_unlisted_file_is_rejected(tmp_path):
    copy = tmp_path / "pkg"
    shutil.copytree(SONIC, copy)
    (copy / "modules/extra.pt").write_bytes(b"x")
    with pytest.raises(package.PackageError, match="SHA256SUMS"):
        package.load(copy)
