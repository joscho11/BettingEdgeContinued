"""Portable paths for public release code and immutable artifacts."""
from __future__ import annotations

from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]


def site_root(root: str | Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else SITE_ROOT


def releases_root(root: str | Path | None = None) -> Path:
    return site_root(root) / "data" / "releases"


def manifest_path(root: str | Path | None = None) -> Path:
    return releases_root(root) / "manifest.json"


def relative_to_site(path: str | Path, root: str | Path | None = None) -> str:
    base = site_root(root)
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError(f"release path is outside site root: {resolved}") from exc


def resolve_site_path(relative: str, root: str | Path | None = None) -> Path:
    base = site_root(root)
    resolved = (base / str(relative)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes site root: {relative!r}") from exc
    return resolved
