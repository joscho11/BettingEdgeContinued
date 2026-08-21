"""Release-contract primitives shared by candidates, validators, and the CLI."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
PRODUCTS = ("predictions", "fantasy")


class PublicationError(RuntimeError):
    """A release was refused before its active manifest pointer changed."""


@dataclass
class ValidationReport:
    product: str
    row_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> None:
        if not self.ok:
            raise PublicationError("release validation failed: " + "; ".join(self.errors))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "product": self.product,
            "row_count": int(self.row_count),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }


def utc_now_iso(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_aware_datetime(value: object) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("timestamp is empty")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_values_hash(values: Iterable[object]) -> str:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("utf-8")).hexdigest()


def candidate_sidecar_path(artifact: str | Path) -> Path:
    path = Path(artifact)
    return path.with_name(f"{path.stem}.metadata.json")
