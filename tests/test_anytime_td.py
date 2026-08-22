"""Anytime TDs demo page. Hermetic APP_OFFLINE=1."""
import os
import sys
from pathlib import Path

import pandas as pd

os.environ["APP_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest

_HERE = Path(__file__).resolve().parents[1]
_SITE_PAGES = _HERE / "site_pages"
sys.path.insert(0, str(_HERE))


def _render(tmp_path):
    harness = tmp_path / "h_anytime_td.py"
    harness.write_text(
        f"import sys; sys.path[:0] = [r'{_HERE}', r'{_SITE_PAGES}']\n"
        "import page_anytime_td as p\np.render()\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(harness), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def test_anytime_td_renders_and_owns_controls(tmp_path):
    at = _render(tmp_path)
    keys = {getattr(w, "key", None) for w in list(at.selectbox)}
    assert "atd_week" in keys, keys
    assert "atd_pos" not in keys
    controls = {w.key: w.value for w in at.selectbox}
    assert controls["atd_week"] == 10
    titles = " ".join(str(t.value) for t in at.title)
    assert "Anytime TDs" in titles
    captions = " ".join(str(c.value) for c in at.caption)
    blob = captions + " ".join(str(item.value) for item in at.info)
    assert "Passing TDs are out" in blob or "rushing or receiving" in blob
    assert "not even money" in blob
    assert "Bet responsibly" in blob
    assert "closer in 5" in blob
    assert "Eight players" not in blob
    assert any(getattr(w, "key", None) == "atd_search" for w in at.text_input)


def test_anytime_td_files_cover_weeks_10_17():
    import page_anytime_td as page

    weeks = page.available_weeks()
    assert list(weeks) == list(range(10, 18)), weeks


def test_priced_rows_drop_unpriced_and_keep_rb_fb():
    import page_anytime_td as page

    hit = pd.DataFrame({
        "player_display_name": ["A", "B", "C"],
        "position": ["RB", "FB", "WR"],
        "team": ["KC", "SF", "PHI"],
        "opponent_team": ["LV", "SEA", "DAL"],
        "p_ge1": [0.40, 0.20, 0.30],
        "p_ge2": [0.10, 0.02, 0.05],
        "p_book": [0.35, None, 0.28],
        "fair_amer": [150, 400, 250],
        "scored_anytime": [1, 0, 0],
    })
    priced = page.priced_rows(hit)
    assert list(priced.player_display_name) == ["A", "C"]
    backs = page.by_position(hit, "RB")
    assert list(backs.position) == ["RB", "FB"]
    summary = page.week_summary(priced)
    assert summary["n"] == 2
    assert summary["hits"] == 1


def test_week10_priced_board_is_larger_than_a_card():
    import page_anytime_td as page

    path = page.available_weeks()[10]
    priced = page.priced_rows(pd.read_csv(path))
    assert len(priced) > 8
    assert priced.p_book.notna().all()
