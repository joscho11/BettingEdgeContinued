from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from matchups.catalog import load_matchup_routes
from matchups.contract import MatchupContractError, matchup_slug, parse_game_id
from matchups.detail import load_matchup_detail
from matchups.social import render_social_card
from publishing.candidate import build_candidate_metadata
from publishing.publisher import publish_candidate
from publishing.validators import validate_candidate

_ROOT = Path(__file__).resolve().parents[1]
_SITE_PAGES = _ROOT / "site_pages"


def test_matchup_slug_is_stable_and_streamlit_safe():
    assert matchup_slug(2026, 1, "NE", "SEA") == "matchup-2026-week-1-ne-sea"
    assert parse_game_id("2026_01_NE_SEA") == (2026, 1, "NE", "SEA")
    try:
        matchup_slug(2026, 1, "NE/NY", "SEA")
    except MatchupContractError:
        pass
    else:
        raise AssertionError("route tokens containing a slash must fail closed")


def test_demo_catalog_has_exactly_weeks_10_through_16():
    routes = load_matchup_routes(_ROOT)
    demo = [route for route in routes if route.season == 2025]
    assert len(demo) == 105
    assert {route.week for route in demo} == set(range(10, 17))
    assert len({route.game_id for route in demo}) == len(demo)
    assert len({route.slug for route in demo}) == len(demo)
    assert all("/" not in route.slug for route in demo)
    assert not any(route.week == 17 for route in demo)


def test_historical_detail_is_explicit_about_provenance_gaps():
    detail = load_matchup_detail(_ROOT, "2025_10_ARI_SEA")
    assert detail["prediction"]["projected_margin"] == 11.300000190734863
    assert detail["prediction"]["market_spread"] == 7.0
    assert detail["prediction"]["model_edge"] == 4.3
    assert detail["status"]["label"] == "HIGH"
    assert detail["status"]["freeze_at"] is None
    assert "backfilled" in detail["status"]["freeze_note"]
    assert len(detail["model"]["inputs"]) == 35
    assert len(detail["model"]["drivers"]) == 8
    assert detail["model"]["drivers"][0]["feature"] == "spread_line"
    assert detail["model"]["reconstructed_margin"] == 11.256825
    assert "all 105 rounded margins reproduced" in detail["model"]["driver_note"]
    assert detail["history"]["line_history_available"] is False
    assert detail["result"]["ats_result"] == "WIN"
    assert detail["result"]["home_score"] == 44.0
    assert detail["result"]["away_score"] == 22.0


def test_historical_context_is_sourced_and_timing_labeled():
    detail = load_matchup_detail(_ROOT, "2025_10_ARI_SEA")
    weather = detail["context"]["weather"]
    assert weather["availability"] == "available"
    assert weather["temperature_f"] == 64.4
    assert weather["source"]["name"] == "Meteostat hourly observations"
    assert "not a frozen pregame forecast" in weather["source"]["timing"]
    injuries = detail["context"]["injuries"]
    assert injuries["source"]["name"] == "nflverse weekly injury reports"
    assert "snapshot time unavailable" in injuries["source"]["timing"]
    assert injuries["away"]["team"] == "ARI"
    assert injuries["home"]["team"] == "SEA"
    assert injuries["away"]["players"]
    assert injuries["home"]["players"]


def test_social_card_is_png_at_link_preview_dimensions():
    card = render_social_card(load_matchup_detail(_ROOT, "2025_10_ARI_SEA"))
    image = Image.open(BytesIO(card))
    assert image.format == "PNG"
    assert image.size == (1200, 630)


def test_matchup_page_renders_offline(tmp_path):
    harness = tmp_path / "matchup_harness.py"
    harness.write_text(
        f"import sys; sys.path[:0] = [r'{_ROOT}', r'{_SITE_PAGES}', r'{_ROOT / 'betting'}']\n"
        "import page_matchup\n"
        "page_matchup.render('2025_10_ARI_SEA')\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(harness), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [error.value for error in at.error]
    assert at.title[0].value == "ARI at SEA"
    metric_labels = {metric.label for metric in at.metric}
    assert {"Projected margin", "Market spread", "Model edge", "ATS result"} <= metric_labels
    markdown = " ".join(str(item.value) for item in at.markdown)
    assert "Most influential drivers" in markdown
    assert "Meteostat hourly observations" in markdown
    assert "nflverse weekly injury reports" in markdown
    assert at.expander[0].label == "All 35 model inputs"
    assert len(at.download_button) == 1


def test_app_registers_hidden_matchup_routes_and_weekly_links():
    app_source = (_ROOT / "app.py").read_text(encoding="utf-8")
    weekly_source = (_SITE_PAGES / "page_weekly_predictions.py").read_text(encoding="utf-8")
    assert 'visibility="hidden"' in app_source
    assert "*matchup_pages.values()" in app_source
    assert "nav_registry.MATCHUPS = matchup_pages" in app_source
    assert 'label="Matchup details"' in weekly_source


def test_hash_bound_2026_sidecar_validates_publishes_and_populates_detail(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    artifact = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "game_id": "2026_01_NE_SEA",
                "home_team": "SEA",
                "away_team": "NE",
                "season": 2026,
                "week": 1,
                "gameday": "2026-09-09",
                "gametime": "20:20",
                "predicted_margin": 4.0,
                "model_edge": 4.0,
                "recommendation": "HOME (SEA)",
                "logged_at": "2026-09-08T13:00:00Z",
                "tuesday_spread_line": 0.0,
                "mode": "tuesday",
            }
        ]
    ).to_csv(artifact, index=False)
    source = {
        "name": "Frozen test source",
        "url": "https://example.com/source",
        "timing": "2026-09-08T13:00:00Z",
    }
    matchups = tmp_path / "predictions.matchups.json"
    matchups.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "season": 2026,
                "week": 1,
                "games": {
                    "2026_01_NE_SEA": {
                        "model": {
                            "inputs": [{"feature": "mkt_spread_now", "value": 0.0}],
                            "drivers": [
                                {
                                    "feature": "qb_rating_diff",
                                    "value": 1.2,
                                    "contribution": 1.1,
                                    "direction": "SEA",
                                }
                            ],
                            "explanation_method": "ensemble SHAP plus Ridge contribution",
                        },
                        "context": {
                            "injuries": {
                                "source": source,
                                "away": {"team": "NE", "availability": "available", "players": []},
                                "home": {"team": "SEA", "availability": "available", "players": []},
                            },
                            "weather": {
                                "source": source,
                                "availability": "available",
                                "wind_mph": 8.0,
                            },
                        },
                        "social": {"title": "NE at SEA", "description": "Week 1 detail"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    metadata = build_candidate_metadata(
        "predictions",
        artifact,
        season=2026,
        week=1,
        model_version="spread-v3-test",
        produced_at="2026-09-08T13:00:00Z",
        matchup_artifact=matchups,
    )
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2026_01_NE_SEA",
                "home_team": "SEA",
                "away_team": "NE",
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "gameday": "2026-09-09",
                "gametime": "20:20",
            }
        ]
    )
    report = validate_candidate(artifact, metadata, schedule=schedule)
    assert report.ok, report.errors
    assert report.checks["matchup_games"] == 1
    assert not report.warnings
    entry = publish_candidate(artifact, metadata, schedule=schedule, root=site)
    assert (site / entry["matchups"]["artifact"]).is_file()
    detail = load_matchup_detail(site, "2026_01_NE_SEA")
    assert detail["model"]["driver_availability"] == "available"
    assert detail["model"]["drivers"][0]["feature"] == "qb_rating_diff"
    assert detail["context"]["weather"]["wind_mph"] == 8.0
    assert detail["social"]["title"] == "NE at SEA"
