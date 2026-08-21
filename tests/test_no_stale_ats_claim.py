"""No superseded ATS result may remain the ACTIVE claim anywhere in the repo.

Three successive corrections produced three numbers:
    64.2% / 380-592   published, leaking sack + legacy identity   (RETRACTED)
    55.4167% / 133-240  dense sack, legacy All-Pro identity        (intermediate)
    54.2017% / 129-238  dense sack + identity incl. injuries       (FINAL)

Each intermediate is legitimate *inside a labelled audit bundle or an explicitly
superseded/retracted passage*. It is a defect anywhere else, because a reader taking the
first number they find would take a withdrawn one.
"""
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

FINAL_HIGH = "129/238"
SUPERSEDED = {
    "133/240": "intermediate — dense sack under the LEGACY All-Pro identity",
    "55.4167": "intermediate HIGH rate",
    "380/592": "the retracted published result",
    "380 of 592": "the retracted published result",
    "64.2%": "the retracted published headline",
}
_LABELLED = re.compile(
    r"retract|supersed|previously|intermediate|legacy|withdraw|published|bundle|audit|"
    r"control|historical|earlier|no longer|leaking|before the|used to|reproduc|"
    r"claimed|corrected|^\s*\|\s*\*?\*?[ABCDE]\b|arm ", re.I)
_LOOKBACK = 2
_SCAN_SUFFIXES = (".md", ".py", ".ipynb", ".yml")
_PUBLIC_DOCS = ("README.md", "AGENTS.md", "betting/GUIDE.md")


def _tracked():
    out = subprocess.check_output(
        ["git", "-c", "safe.directory=*", "ls-files"],
        cwd=_ROOT,
        text=True,
    )
    for rel in out.splitlines():
        rel = rel.replace("\\", "/")
        if not rel.lower().endswith(_SCAN_SUFFIXES):
            continue
        if rel == "tests/test_no_stale_ats_claim.py":
            continue
        yield _ROOT / rel, rel


def test_no_unlabelled_superseded_ats_number_remains():
    offenders = []
    for p, rel in _tracked():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, 1):
            window = " ".join(lines[max(0, i - 1 - _LOOKBACK):i])
            for token, why in SUPERSEDED.items():
                if token in line and not _LABELLED.search(window):
                    offenders.append(f"{rel}:{i}  [{token}: {why}]  {line.strip()[:110]}")
    assert offenders == [], (
        "superseded ATS result(s) presented as the active claim:\n  " +
        "\n  ".join(offenders))


def test_the_final_number_is_actually_present_in_the_public_docs():
    """Guard against the scan passing because every number was deleted."""
    for f in _PUBLIC_DOCS:
        text = (_ROOT / f).read_text(encoding="utf-8", errors="replace")
        assert FINAL_HIGH in text, f"{f} does not state the final result {FINAL_HIGH}"


def test_docs_label_the_retraction():
    for f in _PUBLIC_DOCS:
        text = (_ROOT / f).read_text(encoding="utf-8", errors="replace")
        assert "64.2%" in text and re.search(r"retract", text, re.I), (
            f"{f} must still name and retract the old 64.2% claim"
        )


def test_no_edge_is_claimed():
    """The Wilson lower bound is below break-even, so no doc may assert an edge."""
    banned = re.compile(r"is a REAL edge|the edge is real|edge is genuine", re.I)
    offenders = []
    for p, rel in _tracked():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if banned.search(line) and not re.search(r"no |never |not |without ", line, re.I):
                offenders.append(f"{rel}:{i}  {line.strip()[:110]}")
    assert offenders == [], "an edge is claimed while the lower bound is 47.86%:\n  " + \
        "\n  ".join(offenders)


def test_the_scanner_actually_bites():
    """Self-proof: a planted unlabelled stale claim must be caught."""
    planted = "The model hits 64.2% ATS on HIGH picks."
    assert any(t in planted for t in SUPERSEDED)
    assert not _LABELLED.search(planted), "the label regex would excuse a bare claim"
