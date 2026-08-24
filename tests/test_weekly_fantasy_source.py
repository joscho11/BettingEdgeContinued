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
    assert ["#", *weekly._preview_table_columns(False, False)] == [
        "#", "Player", "Opponent", "Proj Pts",
    ]
    assert ["#", *weekly._preview_table_columns(True, False)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Pass Yds",
        "Proj Rush Yds", "Proj Rec Yds", "Off EPA", "EPA Rank",
        "Team Total", "Health",
    ]
    assert ["#", *weekly._preview_table_columns(True, True)] == [
        "#", "Player", "Opponent", "Proj Pts", "Proj Pass Yds",
        "Proj Rush Yds", "Proj Rec Yds", "Off EPA", "EPA Rank",
        "Team Total", "Health",
        "Actual Pts", "Actual Pass Yds", "Actual Rush Yds", "Actual Rec Yds",
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


def test_stat_reference_reshapes_available_component_estimates():
    frame = pd.DataFrame({
        "player_id": ["qb1", "rb1", "wr1", "te1"],
        "player_display_name": ["Quarter Back", "Running Back", "Wide Out", "Tight End"],
        "position": ["QB", "RB", "WR", "TE"],
        "team": ["NE", "SEA", "BUF", "MIA"],
        "opponent_team": ["SEA", "NE", "MIA", "BUF"],
        "is_home": [1, 0, 1, 0],
        "depth_chart_position": [1, 1, 1, 1],
        "projected_pts": [20.0, 14.0, 13.0, 9.0],
        "pred_qb_pass_yards": [250.5, None, None, None],
        "pred_qb_rush_yards": [31.0, None, None, None],
        "pred_rush_yards": [None, 72.5, None, None],
        "pred_rec_yards": [None, 21.0, None, None],
        "pred_wr_receptions": [None, None, 5.5, None],
        "pred_wr_rec_yards": [None, None, 68.0, None],
        "pred_te_receptions": [None, None, None, 4.0],
        "pred_te_rec_yards": [None, None, None, 44.0],
    })
    actuals = {
        "qb_pass_yds": {"qb1": 260},
        "rb_rush_yds": {"rb1": 81},
        "wr_recs": {"wr1": 6},
        "te_rec_yds": {"te1": 39},
    }

    got = weekly._build_stat_reference(frame, actuals)

    assert len(got) == 8
    assert set(got["Market"]) == {
        "Passing yards", "Rushing yards", "Receiving yards", "Receptions",
    }
    qb_pass = got[(got.player_id == "qb1") & (got.Market == "Passing yards")].iloc[0]
    assert qb_pass["Model estimate"] == 250.5
    assert qb_pass["Actual"] == 260
    assert qb_pass["Opponent"] == "vs SEA"
    te_rec = got[(got.player_id == "te1") & (got.Market == "Receiving yards")].iloc[0]
    assert te_rec["Actual"] == 39
    assert te_rec["Opponent"] == "@ BUF"


def test_stat_reference_does_not_infer_components_from_fantasy_points():
    slim = pd.DataFrame({
        "player_id": ["00-1"],
        "player_display_name": ["Test"],
        "position": ["RB"],
        "team": ["NE"],
        "opponent_team": ["SEA"],
        "projected_pts": [12.4],
    })

    got = weekly._build_stat_reference(slim)

    assert got.empty
    assert list(got.columns) == weekly._STAT_REFERENCE_COLUMNS


def test_weekly_fantasy_stat_reference_view_renders(tmp_path):
    at = _render_weekly(tmp_path)
    view = next(widget for widget in at.segmented_control if widget.key == "wf_view")
    at = view.set_value("Stat reference").run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    captions = " ".join(str(item.value) for item in at.caption)
    assert "Independent stat estimates" in captions
    assert "not a mathematical breakdown" in captions
    assert len(at.dataframe) >= 2


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
    more_info = next(widget for widget in at.toggle if widget.key == "wf_more_info")
    assert more_info.value is False

    at = more_info.set_value(True).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    more_info = next(widget for widget in at.toggle if widget.key == "wf_more_info")
    assert more_info.value is True
    assert len(at.dataframe) >= 2
