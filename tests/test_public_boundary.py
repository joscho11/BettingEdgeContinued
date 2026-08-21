"""CI boundary: this public tree stays a website plus release artifacts.

This module is excluded from its own scans. Matching vocabulary is assembled
from fragments so the checker does not trip on its own description.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()


def _tracked() -> list[str]:
    out = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "ls-files"],
        cwd=ROOT,
        text=True,
    )
    return [line.replace("\\", "/") for line in out.splitlines() if line]


def _exts() -> tuple[str, ...]:
    return (".p" + "kl", ".job" + "lib", ".on" + "nx", ".ipy" + "nb")


def _libs() -> frozenset[str]:
    return frozenset({
        "sk" + "learn",
        "xg" + "boost",
        "light" + "gbm",
        "cat" + "boost",
        "stats" + "models",
        "sh" + "ap",
        "job" + "lib",
    })


def _dir_parts() -> frozenset[str]:
    return frozenset({
        "arch" + "ive",
        "experi" + "ments",
        "train" + "ing",
        "dfs",
        "models",
    })


def _fit_name() -> str:
    return "f" + "it"


def test_tracked_tree_has_no_serialized_models_or_notebooks():
    bad = [path for path in _tracked() if path.lower().endswith(_exts())]
    assert bad == [], bad


def test_tracked_tree_has_no_private_source_directories():
    banned = _dir_parts()
    bad = []
    for path in _tracked():
        parts = set(Path(path).parts)
        if parts & banned:
            bad.append(path)
    assert bad == [], bad


def test_tracked_python_has_no_training_imports_or_fitting():
    banned_libs = _libs()
    fit_name = _fit_name()
    import_hits = []
    fit_hits = []
    for rel in _tracked():
        path = ROOT / rel
        if path.resolve() == SELF:
            continue
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in banned_libs:
                        import_hits.append(f"{rel}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in banned_libs:
                    import_hits.append(f"{rel}:{node.module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == fit_name:
                    fit_hits.append(f"{rel}:{getattr(node, 'lineno', 0)}")
    assert import_hits == [], import_hits
    assert fit_hits == [], fit_hits
