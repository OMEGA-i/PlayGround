"""Small JSON-safe execution provenance, captured in the owning interpreter."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys

PACKAGES = ('numpy','scipy','torch','mujoco','mujoco-warp','warp-lang','pillow')


def file_identity(path):
    path=Path(path)
    with path.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path.resolve()),sha256=digest,size_bytes=path.stat().st_size)


def repository(path):
    path=Path(path).resolve()
    try:
        def git(*args):
            return subprocess.check_output(['git','-C',str(path),*args],text=True,stderr=subprocess.DEVNULL).strip()
        return dict(path=str(path),commit=git('rev-parse','HEAD'),tracked_changes=git('status','--short','--untracked-files=no'))
    except subprocess.CalledProcessError:
        return dict(path=str(path),commit=None)


def runtime():
    packages={}
    for name in PACKAGES:
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    return dict(captured_at=datetime.now(timezone.utc).isoformat(),python=sys.version,
        executable=sys.executable,platform=platform.platform(),packages=packages)
