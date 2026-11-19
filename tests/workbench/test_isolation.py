import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import omega_playground

FORBIDDEN = ("newton", "isaacsim", "protomotions", "isaaclab", "isaaclab_physx", "isaacgym", "omni.physx")
SRC = Path(omega_playground.__file__).parent


def forbidden_import(name):
    return any(name == root or name.startswith(root + '.') for root in FORBIDDEN)


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_no_forbidden_imports_in_source():
    bad = [(str(p.relative_to(SRC)), name) for p in SRC.rglob("*.py") for name in _imports(p)
           if forbidden_import(name)]
    assert not bad, bad


def test_no_forbidden_modules_loaded_at_runtime():
    for mod in pkgutil.walk_packages([str(SRC)], prefix="omega_playground."):
        importlib.import_module(mod.name)
    loaded = [m for m in sys.modules if forbidden_import(m)]
    assert not loaded, loaded
