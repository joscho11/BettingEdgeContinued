"""Hermetic Yahoo ADP overlay + board-market tests. No live Yahoo HTTP.

Covers: payload parse, id-map contract, overlay join (no Sleeper fill), coverage
abort leaving the prior overlay, apply_board_market repricing, and the committed
180-row overlay the site reads.
"""
import hashlib
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

os.environ["APP_OFFLINE"] = "1"

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "fantasy" / "seasonal_projections"))

import fetch_yahoo_adp as yahoo_fetch  # noqa: E402
import refresh_board_yahoo_adp as yahoo_rb  # noqa: E402
import refresh_board_adp as sleeper_rb  # noqa: E402


def _payload(*players):
    return {
        "fantasy_content": {
            "league": [
                {"league_key": "470.l.public"},
                {"settings": {}},
                {"players": {str(i): p for i, p in enumerate(players)} | {"count": len(players)}},
            ]
        }
    }


def _player(yahoo_id, name, pos, adp):
    analysis = [{"average_pick": adp}]
    return {
        "player": [
            [
                {"player_id": yahoo_id},
                {"name": {"full": name}},
                {"primary_position": pos},
                {"display_position": pos},
            ],
            {"draft_analysis": analysis},
        ]
    }


def test_parse_yahoo_payload_keeps_skill_adp_and_drops_the_rest():
    payload = _payload(
        _player(1, "Jahmyr Gibbs", "RB", "1.5"),
        _player(2, "Some Kicker", "K", "12.0"),
        _player(3, "Ja'Marr Chase", "WR", "4.1"),
        _player(4, "No Adp", "QB", "-"),
        _player(5, "Zero", "QB", "0"),
        _player(6, "Josh Allen", "QB", "22.9"),
        _player(1, "Duplicate Gibbs", "RB", "9.9"),
    )
    out = yahoo_fetch.parse_yahoo_payload(payload)
    assert set(out["position"]) <= {"QB", "RB", "WR", "TE"}
    assert "Some Kicker" not in set(out["player"])
    assert "No Adp" not in set(out["player"])
    assert "Zero" not in set(out["player"])
    assert list(out.loc[out["yahoo_id"].eq("1"), "player"]) == ["Jahmyr Gibbs"]
    assert float(out.loc[out["player"].eq("Jahmyr Gibbs"), "yahoo_adp"].iloc[0]) == 1.5


def test_committed_yahoo_id_map_covers_the_exact_180():
    ids = yahoo_fetch.load_yahoo_id_map()
    universe = sleeper_rb.load_board_universe()
    assert len(ids) == 180
    assert set(ids["player_id"]) == set(universe["player_id"].astype("string"))
    assert not ids["yahoo_id"].duplicated().any()
    assert ids["yahoo_id"].notna().all()
    washington = ids[ids["player_id"].eq("WAS797326")]
    assert len(washington) == 1
    assert washington.iloc[0]["yahoo_id"] == "42744"


def test_yahoo_overlay_does_not_fill_from_sleeper_and_ranks_within_180():
    universe = pd.DataFrame({
        "player_id": ["a", "b", "c", "d"],
        "player": ["Player A", "Player B", "Player C", "Player D"],
        "position": ["RB", "RB", "WR", "WR"],
        "adp_frozen": [10.0, 20.0, 5.0, 30.0],
    })
    id_map = pd.DataFrame({
        "player_id": ["a", "b", "c", "d"],
        "player": ["Player A", "Player B", "Player C", "Player D"],
        "position": ["RB", "RB", "WR", "WR"],
        "yahoo_id": ["1", "2", "3", "4"],
    })
    fresh = pd.DataFrame({
        "yahoo_id": ["1", "2", "3", "99"],
        "player": ["Player A", "Player B Jr.", "Player C", "Ghost"],
        "position": ["RB", "RB", "WR", "TE"],
        "yahoo_adp": [12.0, 8.0, 4.0, 99.0],
    })
    overlay, coverage = yahoo_rb.build_yahoo_overlay_full(
        universe, fresh, id_map, "2026-08-24"
    )
    assert len(overlay) == 4
    o = overlay.set_index("player_id")
    assert float(o.loc["a", "yahoo_adp"]) == 12.0
    assert float(o.loc["b", "yahoo_adp"]) == 8.0
    assert pd.isna(o.loc["d", "yahoo_adp"]), "unmatched Yahoo rows must stay blank"
    assert o.loc["d", "adp_source"] == "unmatched"
    assert int(o.loc["b", "yahoo_pos_rank"]) == 1
    assert int(o.loc["a", "yahoo_pos_rank"]) == 2
    assert pd.isna(o.loc["d", "yahoo_pos_rank"])
    assert coverage["matched"] == 3
    assert "adp_half_ppr" not in overlay.columns
    assert "sleeper" not in " ".join(overlay.columns)
    assert "espn" not in " ".join(overlay.columns)


def test_name_fallback_matches_when_yahoo_id_is_blank():
    universe = pd.DataFrame({
        "player_id": ["x"],
        "player": ["Mike Washington"],
        "position": ["RB"],
        "adp_frozen": [163.3],
    })
    id_map = pd.DataFrame({
        "player_id": ["x"],
        "player": ["Mike Washington"],
        "position": ["RB"],
        "yahoo_id": [pd.NA],
    })
    fresh = pd.DataFrame({
        "yahoo_id": ["42744"],
        "player": ["Mike Washington Jr."],
        "position": ["RB"],
        "yahoo_adp": [122.2],
    })
    overlay, coverage = yahoo_rb.build_yahoo_overlay_full(
        universe, fresh, id_map, "2026-08-24"
    )
    assert coverage["matched"] == 1
    assert float(overlay.iloc[0]["yahoo_adp"]) == 122.2


@pytest.fixture
def yahoo_sandbox(tmp_path, monkeypatch):
    overlay = tmp_path / "board_yahoo_adp_live_2026.csv"
    logs = tmp_path / "adp_logs"
    monkeypatch.setattr(yahoo_rb, "OVERLAY", overlay)
    monkeypatch.setattr(yahoo_rb, "LOGS_DIR", logs)
    monkeypatch.setattr(yahoo_rb, "LEDGER", logs / "yahoo_refresh_ledger.jsonl")
    monkeypatch.setattr(yahoo_rb, "_season_start", lambda: date(2099, 1, 1))
    monkeypatch.setattr(sys, "argv", ["refresh_board_yahoo_adp.py"])
    prior = pd.DataFrame({
        "player_id": ["QB000"], "yahoo_adp": [9.9], "yahoo_pos_rank": [1],
        "refreshed_at": ["2026-07-13"], "position": ["QB"],
        "adp_source": ["fresh"], "adp_matched": [True],
    })
    prior.to_csv(overlay, index=False)
    return {"overlay": overlay, "logs": logs, "prior_bytes": overlay.read_bytes()}


def test_yahoo_coverage_collapse_leaves_prior_overlay(yahoo_sandbox, monkeypatch):
    u = sleeper_rb.load_board_universe()
    ids = yahoo_fetch.load_yahoo_id_map()
    fresh = pd.DataFrame({
        "yahoo_id": [f"g{i}" for i in range(200)],
        "player": [f"Ghost {i}" for i in range(200)],
        "position": ["WR"] * 200,
        "yahoo_adp": [float(i + 1) for i in range(200)],
    })
    monkeypatch.setattr(yahoo_rb, "load_board_universe", lambda: u)
    monkeypatch.setattr(yahoo_rb, "load_yahoo_id_map", lambda path=None: ids)
    monkeypatch.setattr(yahoo_rb, "fetch_yahoo_adp", lambda: fresh)
    rc = yahoo_rb.main()
    assert rc == 1
    assert yahoo_sandbox["overlay"].read_bytes() == yahoo_sandbox["prior_bytes"]
    ledger = (yahoo_sandbox["logs"] / "yahoo_refresh_ledger.jsonl").read_text(encoding="utf-8")
    assert "coverage below floor" in ledger


def test_yahoo_healthy_run_writes_180(yahoo_sandbox, monkeypatch):
    u = sleeper_rb.load_board_universe()
    ids = yahoo_fetch.load_yahoo_id_map()
    fresh = ids[["yahoo_id", "player", "position"]].copy()
    fresh["yahoo_adp"] = range(1, len(fresh) + 1)
    monkeypatch.setattr(yahoo_rb, "load_board_universe", lambda: u)
    monkeypatch.setattr(yahoo_rb, "load_yahoo_id_map", lambda path=None: ids)
    monkeypatch.setattr(yahoo_rb, "fetch_yahoo_adp", lambda: fresh)
    rc = yahoo_rb.main()
    assert rc == 0
    out = pd.read_csv(yahoo_sandbox["overlay"])
    assert len(out) == 180
    assert out["adp_matched"].all()
    assert yahoo_sandbox["overlay"].read_bytes() != yahoo_sandbox["prior_bytes"]


def test_apply_board_market_reprices_and_recomputes_gaps():
    import draft_board_2026 as board

    loaded = board._load_board_2026()
    sleeper = board.apply_board_market(loaded, board.DEFAULT_ADP_MARKET)
    yahoo = board.apply_board_market(loaded, board.YAHOO_ADP_MARKET)
    assert len(sleeper) == len(yahoo) == 180
    assert sleeper["model_proj"].equals(yahoo["model_proj"])
    assert sleeper["model_proj_pos_rank"].astype("Int64").equals(
        yahoo["model_proj_pos_rank"].astype("Int64")
    )
    assert sleeper["model_draft_rank"].astype("Int64").equals(
        yahoo["model_draft_rank"].astype("Int64")
    )
    gadsden = "Oronde Gadsden"
    s = sleeper.set_index("player").loc[gadsden]
    y = yahoo.set_index("player").loc[gadsden]
    assert float(s["adp_half_ppr"]) != float(y["adp_half_ppr"])
    assert int(y["pos_rank"]) == int(y["yahoo_pos_rank"])
    assert int(y["model_gap"]) == int(y["pos_rank"] - y["model_proj_pos_rank"])
    assert int(y["sleeper_gap"]) == int(y["pos_rank"] - y["sleeper_proj_pos_rank"])
    assert sleeper["adp_half_ppr"].equals(loaded["adp_half_ppr"])
    unmatched = yahoo["yahoo_adp"].isna()
    assert unmatched.any(), "committed Yahoo overlay is expected to have unmatched rows"
    assert yahoo.loc[unmatched, "adp_half_ppr"].isna().all()


def test_sort_keys_swap_only_the_adp_label():
    import draft_board_2026 as board

    sleeper = board.sort_keys_for(board.DEFAULT_ADP_MARKET)
    espn = board.sort_keys_for(board.ESPN_ADP_MARKET)
    yahoo = board.sort_keys_for(board.YAHOO_ADP_MARKET)
    assert list(sleeper)[0] == "Sleeper ADP"
    assert list(espn)[0] == "ESPN ADP"
    assert list(yahoo)[0] == "Yahoo ADP"
    assert sleeper["Sleeper ADP"] == espn["ESPN ADP"] == yahoo["Yahoo ADP"] == "adp_half_ppr"
    assert "Yahoo ADP" not in sleeper
    assert "Yahoo ADP" not in espn
    assert "Sleeper ADP" not in yahoo
    assert "ESPN ADP" not in yahoo
    assert set(sleeper.values()) == set(espn.values()) == set(yahoo.values())
    draft = board.sort_keys_for(board.MODEL_DRAFT_MARKET)
    assert list(draft)[0] == board.MODEL_DRAFT_MARKET
    assert draft[board.MODEL_DRAFT_MARKET] == "adp_half_ppr"
    assert "Sleeper ADP" not in draft
    assert "model_draft_rank" not in draft.values()


def test_committed_yahoo_overlay_is_the_exact_180():
    overlay = pd.read_csv(
        yahoo_rb.OVERLAY, dtype={"player_id": "string", "yahoo_id": "string"}
    )
    universe = sleeper_rb.load_board_universe()
    assert len(overlay) == 180
    assert set(overlay["player_id"]) == set(universe["player_id"].astype("string"))
    matched = int(overlay["adp_matched"].sum())
    assert matched >= 150
    unmatched = overlay.loc[~overlay["adp_matched"].astype(bool)]
    assert unmatched["yahoo_adp"].isna().all()
    washington = overlay[overlay["player_id"].eq("WAS797326")].iloc[0]
    assert washington["yahoo_id"] == "42744"
    sha = hashlib.sha256(yahoo_rb.OVERLAY.read_bytes()).hexdigest()
    assert len(sha) == 64
