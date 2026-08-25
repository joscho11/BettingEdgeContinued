"""Pixel compare for PNG screenshots. Pillow + numpy only, no extra test dep."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

COLOR_THRESHOLD = 28
MAX_DIFF_RATIO = 0.03


@dataclass(frozen=True)
class CompareResult:
    ok: bool
    ratio: float
    reason: str
    diff_png: bytes | None = None


def _to_rgb(payload: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def compare_png(
    actual: bytes,
    baseline: Path,
    *,
    color_threshold: int = COLOR_THRESHOLD,
    max_ratio: float = MAX_DIFF_RATIO,
) -> CompareResult:
    expected = _to_rgb(baseline.read_bytes())
    got = _to_rgb(actual)
    if got.shape != expected.shape:
        return CompareResult(
            ok=False,
            ratio=1.0,
            reason=f"size {got.shape[1]}x{got.shape[0]} vs baseline {expected.shape[1]}x{expected.shape[0]}",
            diff_png=actual,
        )
    delta = np.max(np.abs(got.astype(np.int16) - expected.astype(np.int16)), axis=2)
    changed = delta > color_threshold
    ratio = float(changed.mean())
    if ratio <= max_ratio:
        return CompareResult(ok=True, ratio=ratio, reason="match")
    overlay = got.copy()
    overlay[changed] = (220, 48, 48)
    return CompareResult(
        ok=False,
        ratio=ratio,
        reason=f"{ratio:.4%} pixels differ (limit {max_ratio:.4%})",
        diff_png=_png_bytes(overlay),
    )
