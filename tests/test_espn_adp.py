"""Hermetic ESPN ADP overlay + board-market tests. No live ESPN HTTP.

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

import fetch_espn_adp as espn_fetch  # noqa: E402
import refresh_board_espn_adp as espn_rb  # noqa: E402
import refresh_board_adp as sleeper_rb  # noqa: E402


def _payload(*players):
    return {"players": list(players)}


def _player(espn_id, name, pos_id, adp):
    return {
        "id": espn_id,
        "player": {
            "fullName": name,
            "defaultPositionId": pos_id,
            "ownership": {"averageDraftPosition": adp},
        },
    }


def test_parse_espn_payload_keeps_skill_adp_and_drops_the_rest():
    payload = _payload(
        _player(1, "Jahmyr Gibbs", 2, 1.5),
        _player(2, "Some Kicker", 5, 12.0),
        _player(3, "Ja'Marr Chase", 3, 4.1),
        _player(4, "No Adp", 1, None),
        {"id": 5, "player": {"fullName": "Zero", "defaultPositionId": 1,
                             "ownership": {"averageDraftPosition": 0}}},
        _player(6, "Josh Allen", 1, 22.9),
        _player(1, "Duplicate Gibbs", 2, 9.9),
    )
    out = espn_fetch.parse_espn_payload(payload)
    assert set(out["position"]) <= {"QB", "RB", "WR", "TE"}
    assert "Some Kicker" not in set(out["player"])
    assert "No Adp" not in set(out["player"])
    assert "Zero" not in set(out["player"])
    assert list(out.loc[out["espn_id"].eq("1"), "player"]) == ["Jahmyr Gibbs"]
    assert float(out.loc[out["player"].eq("Jahmyr Gibbs"), "espn_adp"].iloc[0]) == 1.5


def test_committed_espn_id_map_covers_the_exact_180():
    ids = espn_fetch.load_espn_id_map()
    universe = sleeper_rb.load_board_universe()
    assert len(ids) == 180
    assert set(ids["player_id"]) == set(universe["player_id"].astype("string"))
    assert not ids["espn_id"].duplicated().any()
    assert ids["espn_id"].notna().all()
    washington = ids[ids["player_id"].eq("WAS797326")]
    assert len(washington) == 1
    assert washington.iloc[0]["espn_id"] == "4686658"


def test_espn_overlay_does_not_fill_from_sleeper_and_ranks_within_180():
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
        "espn_id": ["1", "2", "3", "4"],
    })
    fresh = pd.DataFrame({
        "espn_id": ["1", "2", "3", "99"],
        "player": ["Player A", "Player B Jr.", "Player C", "Ghost"],
        "position": ["RB", "RB", "WR", "TE"],
        "espn_adp": [12.0, 8.0, 4.0, 99.0],
    })
    overlay, coverage = espn_rb.build_espn_overlay_full(
        universe, fresh, id_map, "2026-08-20"
    )
    assert len(overlay) == 4
    o = overlay.set_index("player_id")
    assert float(o.loc["a", "espn_adp"]) == 12.0
    assert float(o.loc["b", "espn_adp"]) == 8.0
    assert pd.isna(o.loc["d", "espn_adp"]), "unmatched ESPN rows must stay blank"
    assert o.loc["d", "adp_source"] == "unmatched"
    assert int(o.loc["b", "espn_pos_rank"]) == 1
    assert int(o.loc["a", "espn_pos_rank"]) == 2
    assert pd.isna(o.loc["d", "espn_pos_rank"])
    assert coverage["matched"] == 3
    assert "adp_half_ppr" not in overlay.columns
    assert "sleeper" not in " ".join(overlay.columns)


def test_name_fallback_matches_when_espn_id_is_blank():
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
        "espn_id": [pd.NA],
    })
    fresh = pd.DataFrame({
        "espn_id": ["4686658"],
        "player": ["Mike Washington Jr."],
        "position": ["RB"],
        "espn_adp": [169.72],
    })
    overlay, coverage = espn_rb.build_espn_overlay_full(
        universe, fresh, id_map, "2026-08-20"
    )
    assert coverage["matched"] == 1
    assert float(overlay.iloc[0]["espn_adp"]) == 169.72


@pytest.fixture
def espn_sandbox(tmp_path, monkeypatch):
    overlay = tmp_path / "board_espn_adp_live_2026.csv"
    logs = tmp_path / "adp_logs"
    monkeypatch.setattr(espn_rb, "OVERLAY", overlay)
    monkeypatch.setattr(espn_rb, "LOGS_DIR", logs)
    monkeypatch.setattr(espn_rb, "LEDGER", logs / "espn_refresh_ledger.jsonl")
    monkeypatch.setattr(espn_rb, "_season_start", lambda: date(2099, 1, 1))
    monkeypatch.setattr(sys, "argv", ["refresh_board_espn_adp.py"])
    prior = pd.DataFrame({
        "player_id": ["QB000"], "espn_adp": [9.9], "espn_pos_rank": [1],
        "refreshed_at": ["2026-07-13"], "position": ["QB"],
        "adp_source": ["fresh"], "adp_matched": [True],
    })
    prior.to_csv(overlay, index=False)
    return {"overlay": overlay, "logs": logs, "prior_bytes": overlay.read_bytes()}


def test_espn_coverage_collapse_leaves_prior_overlay(espn_sandbox, monkeypatch):
    u = sleeper_rb.load_board_universe()
    ids = espn_fetch.load_espn_id_map()
    fresh = pd.DataFrame({
        "espn_id": [f"g{i}" for i in range(200)],
        "player": [f"Ghost {i}" for i in range(200)],
        "position": ["WR"] * 200,
        "espn_adp": [float(i + 1) for i in range(200)],
    })
    monkeypatch.setattr(espn_rb, "load_board_universe", lambda: u)
    monkeypatch.setattr(espn_rb, "load_espn_id_map", lambda path=None: ids)
    monkeypatch.setattr(espn_rb, "fetch_espn_adp", lambda: fresh)
    rc = espn_rb.main()
    assert rc == 1
    assert espn_sandbox["overlay"].read_bytes() == espn_sandbox["prior_bytes"]
    ledger = (espn_sandbox["logs"] / "espn_refresh_ledger.jsonl").read_text(encoding="utf-8")
    assert "coverage below floor" in ledger


def test_espn_healthy_run_writes_180(espn_sandbox, monkeypatch):
    u = sleeper_rb.load_board_universe()
    ids = espn_fetch.load_espn_id_map()
    fresh = ids[["espn_id", "player", "position"]].copy()
    fresh["espn_adp"] = range(1, len(fresh) + 1)
    monkeypatch.setattr(espn_rb, "load_board_universe", lambda: u)
    monkeypatch.setattr(espn_rb, "load_espn_id_map", lambda path=None: ids)
    monkeypatch.setattr(espn_rb, "fetch_espn_adp", lambda: fresh)
    rc = espn_rb.main()
    assert rc == 0
    out = pd.read_csv(espn_sandbox["overlay"])
    assert len(out) == 180
    assert out["adp_matched"].all()
    assert espn_sandbox["overlay"].read_bytes() != espn_sandbox["prior_bytes"]


def test_apply_board_market_reprices_and_recomputes_gaps():
    import draft_board_2026 as board

    loaded = board._load_board_2026()
    sleeper = board.apply_board_market(loaded, board.DEFAULT_ADP_MARKET)
    espn = board.apply_board_market(loaded, board.ESPN_ADP_MARKET)
    assert len(sleeper) == len(espn) == 180
    assert sleeper["model_proj"].equals(espn["model_proj"])
    assert sleeper["model_proj_pos_rank"].astype("Int64").equals(
        espn["model_proj_pos_rank"].astype("Int64")
    )
    assert sleeper["model_draft_rank"].astype("Int64").equals(
        espn["model_draft_rank"].astype("Int64")
    )
    gadsden = "Oronde Gadsden"
    s = sleeper.set_index("player").loc[gadsden]
    e = espn.set_index("player").loc[gadsden]
    assert float(s["adp_half_ppr"]) != float(e["adp_half_ppr"])
    assert int(e["pos_rank"]) == int(e["espn_pos_rank"])
    assert int(e["model_gap"]) == int(e["pos_rank"] - e["model_proj_pos_rank"])
    assert int(e["sleeper_gap"]) == int(e["pos_rank"] - e["sleeper_proj_pos_rank"])
    assert sleeper["adp_half_ppr"].equals(loaded["adp_half_ppr"])


def test_sort_keys_swap_only_the_adp_label():
    import draft_board_2026 as board

    sleeper = board.sort_keys_for(board.DEFAULT_ADP_MARKET)
    espn = board.sort_keys_for(board.ESPN_ADP_MARKET)
    assert list(sleeper)[0] == "Sleeper ADP"
    assert list(espn)[0] == "ESPN ADP"
    assert sleeper["Sleeper ADP"] == espn["ESPN ADP"] == "adp_half_ppr"
    assert "Sleeper ADP" not in espn
    assert "ESPN ADP" not in sleeper
    assert set(sleeper.values()) == set(espn.values())
    draft = board.sort_keys_for(board.MODEL_DRAFT_MARKET)
    assert list(draft)[0] == board.MODEL_DRAFT_MARKET
    assert draft[board.MODEL_DRAFT_MARKET] == "adp_half_ppr"
    assert "Sleeper ADP" not in draft
    assert "model_draft_rank" not in draft.values()


def test_committed_espn_overlay_is_the_exact_180():
    overlay = pd.read_csv(
        espn_rb.OVERLAY, dtype={"player_id": "string", "espn_id": "string"}
    )
    universe = sleeper_rb.load_board_universe()
    assert len(overlay) == 180
    assert set(overlay["player_id"]) == set(universe["player_id"].astype("string"))
    unmatched = overlay.loc[~overlay["adp_matched"].astype(bool)]
    assert set(unmatched["player"]) <= {"MarShawn Lloyd"}
    assert overlay.loc[overlay["adp_matched"].astype(bool), "espn_adp"].notna().all()
    washington = overlay[overlay["player_id"].eq("WAS797326")].iloc[0]
    assert washington["espn_id"] == "4686658"
    assert float(washington["espn_adp"]) > 0
    sha = hashlib.sha256(espn_rb.OVERLAY.read_bytes()).hexdigest()
    assert len(sha) == 64
