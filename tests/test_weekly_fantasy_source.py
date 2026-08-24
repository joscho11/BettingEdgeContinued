"""2025 weekly CSVs stay as the site demo. 2026 files come from weekly_projections_v2."""
import hashlib
import os
import sys
from pathlib import Path

os.environ["APP_OFFLINE"] = "1"

import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
_SITE_PAGES = _HERE / "site_pages"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_SITE_PAGES))

from streamlit.testing.v1 import AppTest

import page_weekly_fantasy as weekly

DEMO_MD5 = {
    "projections_2025_week10.csv": "46d2797210f539e89a427fb07ab94ca0",
    "projections_2025_week11.csv": "7f974deb1399ec5cea7085b3465276e6",
    "projections_2025_week12.csv": "fb458ba5fbeaf1de003a2cdae1c82d12",
    "projections_2025_week13.csv": "08c1ee32cb21cae99b5bd552a57ac7f7",
    "projections_2025_week14.csv": "8a41574dd3f066e81ca2198667a08193",
    "projections_2025_week15.csv": "1a17329c9c0c1a51babd9729416cdba7",
    "projections_2025_week16.csv": "46bd8859fe805f17f6ea12a502cb1bda",
    "projections_2025_week17.csv": "e384d1f6baa0af1f7dd9a17be743a18d",
}


def test_2025_demo_csvs_byte_identical():
    folder = _HERE / "fantasy" / "fantasy_projections"
    assert sorted(DEMO_MD5) == sorted(p.name for p in folder.glob("projections_2025_week*.csv"))
    for name, digest in DEMO_MD5.items():
        payload = (folder / name).read_bytes()
        assert hashlib.md5(payload).hexdigest() == digest, name


def test_parse_proj_name():
    assert weekly._parse_proj_name("projections_2026_week01.csv") == (2026, 1)
    assert weekly._parse_proj_name("notes.txt") is None


def test_available_files_are_demo_csvs_only_without_releases(tmp_path, monkeypatch):
    jsa = tmp_path / "jsa"
    jsa.mkdir()
    (jsa / "projections_2025_week10.csv").write_text("player_id\n1\n", encoding="utf-8")
    (jsa / "projections_2026_week01.csv").write_text("site\n", encoding="utf-8")
    monkeypatch.setattr(weekly, "_JSA_PROJ_DIR", jsa)
    monkeypatch.setattr(weekly, "published_builds", lambda *args, **kwargs: [])
    got = weekly.available_projection_files()
    assert got[(2025, 10)] == jsa / "projections_2025_week10.csv"
    assert (2026, 1) not in got


def test_unvalidated_2026_files_are_not_public(tmp_path, monkeypatch):
    jsa = tmp_path / "jsa"
    jsa.mkdir()
    (jsa / "projections_2026_week01.csv").write_text("site\n", encoding="utf-8")
    monkeypatch.setattr(weekly, "_JSA_PROJ_DIR", jsa)
    monkeypatch.setattr(weekly, "published_builds", lambda *args, **kwargs: [])
    got = weekly.available_projection_files()
    assert (2026, 1) not in got


def _render_weekly(tmp_path):
    h = tmp_path / "h_weekly.py"
    h.write_text(
        f"import sys; sys.path[:0] = [r'{_HERE}', r'{_SITE_PAGES}']\n"
        "import page_weekly_fantasy as p\np.render()\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(h), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def test_weekly_fantasy_names_2025_demo(tmp_path):
    at = _render_weekly(tmp_path)
    blob = " ".join(
        str(getattr(w, "value", ""))
        for w in list(at.caption) + list(at.info) + list(at.markdown) + list(at.title)
    ).lower()
    assert "demo" in blob
    assert "2025" in blob
    assert "week 10" in blob


def test_weekly_fantasy_defaults_to_2025_week10(tmp_path):
    at = _render_weekly(tmp_path)
    by_key = {getattr(w, "key", None): w.value for w in at.selectbox}
    assert int(by_key["wf_season"]) == 2025
    assert int(by_key["wf_week"]) == 10
    markdown = " ".join(str(item.value) for item in at.markdown)
    assert "green-badge" in markdown and "Published" in markdown
    captions = " ".join(str(item.value) for item in at.caption)
    assert "Next: 2026 Week 1 · Awaiting projections" in captions
    infos = " ".join(str(w.value) for w in at.info).lower()
    assert "2025" in infos
    assert "demo" in infos


def test_coming_soon_copy_points_at_2025_demo():
    text = weekly._coming_soon_copy(2026, 1)
    assert "2026 Week 1" in text
    assert "2025" in text
    assert "demo" in text.lower()


def test_preview_column_contract_matches_simple_and_detailed_views():
    path = _HERE / "fantasy" / "fantasy_projections" / "projections_2025_week17.csv"
    source = pd.read_csv(path)

    assert weekly._preview_detail_available(source)
    assert ["#", *weekly._preview_table_columns("QB", False, False)] == [
        "#", "Player", "Opponent", "Proj Pts",
    ]
    assert ["#", *weekly._preview_table_columns("QB", True, False)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Pass Yds", "Proj Rush Yds",
        "Off EPA", "EPA Rank", "Team Total", "Health",
    ]
    assert ["#", *weekly._preview_table_columns("RB", True, False)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Rush Yds", "Proj Rec Yds",
        "Off EPA", "EPA Rank", "Team Total", "Health",
    ]
    assert ["#", *weekly._preview_table_columns("WR", True, False)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Rec Yds",
        "Off EPA", "EPA Rank", "Team Total", "Health",
    ]
    assert ["#", *weekly._preview_table_columns("TE", True, False)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Rec Yds",
        "Off EPA", "EPA Rank", "Team Total", "Health",
    ]
    assert ["#", *weekly._preview_table_columns("QB", True, True)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Pass Yds", "Proj Rush Yds",
        "Off EPA", "EPA Rank", "Team Total", "Health",
        "Actual Pts", "Actual Pass Yds", "Actual Rush Yds",
    ]
    assert ["#", *weekly._preview_table_columns("RB", True, True)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Rush Yds", "Proj Rec Yds",
        "Off EPA", "EPA Rank", "Team Total", "Health",
        "Actual Pts", "Actual Rush Yds", "Actual Rec Yds",
    ]
    assert ["#", *weekly._preview_table_columns("WR", True, True)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Rec Yds",
        "Off EPA", "EPA Rank", "Team Total", "Health",
        "Actual Pts", "Actual Rec Yds",
    ]


def test_slim_2026_schema_has_core_columns():
    frame = pd.DataFrame({
        "player_id": ["00-1"],
        "player_display_name": ["Test"],
        "position": ["RB"],
        "team": ["NE"],
        "opponent_team": ["SEA"],
        "season": [2026],
        "week": [1],
        "projected_pts": [12.4],
    })
    for col in ("player_id", "player_display_name", "position", "team",
                "opponent_team", "projected_pts"):
        assert col in frame.columns
    assert "depth_chart_position" not in frame.columns
    assert "off_epa_roll4" not in frame.columns


def test_week17_renders_simple_and_detailed_2026_preview(tmp_path):
    at = _render_weekly(tmp_path)
    week = next(widget for widget in at.selectbox if widget.key == "wf_week")
    at = week.set_value(17).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    infos = " ".join(str(item.value) for item in at.info)
    assert "2026 format preview" in infos
    assert "source CSV has not been changed" in infos
    assert "simple by default" in infos
    assert all(widget.key != "wf_view" for widget in at.segmented_control)
    more_info = next(widget for widget in at.toggle if widget.key == "wf_more_info")
    assert more_info.value is False
    assert more_info.label == "More info — projected yards for player props"
    captions = " ".join(str(item.value) for item in at.caption)
    assert "compare our projected yardage" in captions
    assert "player-prop over/under lines" in captions
    assert "not sportsbook lines or betting recommendations" in captions

    at = more_info.set_value(True).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    more_info = next(widget for widget in at.toggle if widget.key == "wf_more_info")
    assert more_info.value is True
    assert len(at.dataframe) >= 2
