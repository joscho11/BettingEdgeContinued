"""Batch-3c proof for the extracted Weekly Fantasy + League History pages. Each renders
offline-clean and owns its own controls (filter independence); League History lands on an
EMPTY league-ID box with the resting prompt (the earlier fix survives extraction).
Hermetic (APP_OFFLINE=1).
"""
import os
import sys
from pathlib import Path

import pandas as pd

os.environ["APP_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest

_HERE = Path(__file__).resolve().parents[1]
_SITE_PAGES = _HERE / "site_pages"
sys.path.insert(0, str(_HERE))

import page_league_history
from fantasy import league_intelligence as league_intel


def _render_page(tmp_path, module):
    h = tmp_path / f"h_{module}.py"
    h.write_text(f"import sys; sys.path[:0] = [r'{_HERE}', r'{_SITE_PAGES}']\n"
                 f"import {module} as p\np.render()\n", encoding="utf-8")
    at = AppTest.from_file(str(h), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def _control_keys(at):
    return {getattr(w, "key", None) for w in list(at.selectbox) + list(at.slider)}


def test_weekly_fantasy_renders_and_owns_controls(tmp_path):
    at = _render_page(tmp_path, "page_weekly_fantasy")
    keys = _control_keys(at)
    assert {"wf_season", "wf_week"} <= keys, f"Weekly Fantasy must own Season+Week; got {keys}"
    assert not any(str(k).startswith(("wp_", "tr_")) for k in keys), \
        "Weekly Fantasy must not carry another page's controls"


def test_weekly_actuals_cache_one_season_across_week_filters(monkeypatch):
    """Week changes reuse a bounded cached season pull without changing its API."""
    import nflreadpy as nfl
    import page_weekly_fantasy as weekly

    calls = []
    raw = pd.DataFrame([
        {
            "season_type": "REG", "week": 1, "position": "QB", "player_id": "qb1",
            "passing_yards": 250, "passing_tds": 2, "passing_interceptions": 0,
            "rushing_yards": 0, "rushing_tds": 0, "receptions": 0,
            "receiving_yards": 0, "receiving_tds": 0,
            "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0,
        },
        {
            "season_type": "REG", "week": 2, "position": "WR", "player_id": "wr1",
            "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
            "rushing_yards": 0, "rushing_tds": 0, "receptions": 5,
            "receiving_yards": 100, "receiving_tds": 1,
            "rushing_fumbles_lost": 0, "receiving_fumbles_lost": 0,
        },
    ])

    def _load_player_stats(seasons):
        calls.append(tuple(seasons))
        return raw.copy()

    monkeypatch.setattr(weekly, "_OFFLINE", False)
    monkeypatch.setattr(nfl, "load_player_stats", _load_player_stats)
    weekly._load_actual_stats_season.clear()
    try:
        week_one = weekly.load_actual_stats(2025, 1)
        week_two = weekly.load_actual_stats(2025, 2)

        assert calls == [(2025,)]
        assert set(week_one) == {
            "half_ppr", "qb_pass_yds", "qb_rush_yds", "rb_rush_yds", "rb_rec_yds",
            "wr_rec_yds", "wr_recs", "te_rec_yds", "te_recs",
        }
        assert week_one["half_ppr"] == {"qb1": 18.0}
        assert week_one["qb_pass_yds"] == {"qb1": 250}
        assert week_two["half_ppr"] == {"wr1": 18.5}
        assert week_two["wr_rec_yds"] == {"wr1": 100}
        assert week_two["wr_recs"] == {"wr1": 5}

        weekly._load_actual_stats_season.clear()
        missing_column = raw.drop(columns=["receiving_tds"])
        monkeypatch.setattr(nfl, "load_player_stats", lambda seasons: missing_column.copy())
        assert weekly.load_actual_stats(2024, 1) == {}

        monkeypatch.setattr(weekly, "_OFFLINE", True)
        assert weekly.load_actual_stats(2025, 3) == {}
        assert calls == [(2025,)]
    finally:
        weekly._load_actual_stats_season.clear()


def test_league_history_renders_and_lands_empty(tmp_path):
    at = _render_page(tmp_path, "page_league_history")
    ti = [t for t in at.text_input if getattr(t, "key", None) == "lh_league_id"]
    assert ti, "League History must render its league-ID input"
    assert ti[0].value == "", "League History must land on an EMPTY league-ID box"
    assert any(b.label == "Load league history" for b in at.button), \
        "League History must require an explicit Load action"
    info = " ".join(str(i.value) for i in at.info)
    assert "Enter your Sleeper league ID" in info, "resting-state prompt must be shown"


def test_league_history_rejects_implausible_ids_before_fetch():
    assert page_league_history._league_id_error("1255197436951932928") is None
    assert "digits only" in page_league_history._league_id_error("abc123")
    assert "does not look like" in page_league_history._league_id_error("123")


def test_league_history_estimate_counts_linked_seasons(monkeypatch):
    leagues = {
        "current": {"season": "2026", "previous_league_id": "prior-1"},
        "prior-1": {"season": "2025", "previous_league_id": "prior-2"},
        "prior-2": {"season": "2024", "previous_league_id": "0"},
    }

    def _fake_get(url):
        return leagues.get(url.rsplit("/", 1)[-1])

    monkeypatch.setattr(page_league_history, "_sleeper_get", _fake_get)
    page_league_history._league_history_season_count.clear()
    try:
        assert page_league_history._league_history_season_count("current") == 3
        assert page_league_history._history_load_estimate(3) == (6, 12)
        assert page_league_history._history_load_estimate(99) == (20, 40)
    finally:
        page_league_history._league_history_season_count.clear()


def test_rookie_board_excludes_direct_pff_fields_and_explains_availability(tmp_path):
    """The public Rookie Board excludes direct PFF data and explains blank projections."""
    at = _render_page(tmp_path, "page_rookie_board")
    assert any(w.label == "Draft class" for w in at.selectbox)
    assert any(w.label == "Position" for w in at.selectbox)
    # Three tables since 2026-07-27: the rookie board itself, plus the collapsed
    # "college QBs/RBs/WRs/TEs not in this rookie class" views. ALL must stay free of direct PFF fields.
    assert len(at.dataframe) == 5

    banned_pff = {"PFF Grade", "PFF Grade (Percentile)", "PFF Efficiency",
                  "PFF Efficiency (Percentile)", "grades_pass", "btt_rate", "twp_rate",
                  "pressure_grade", "accuracy_pct", "grades_run"}
    tables = []
    for element in at.dataframe:
        value = element.value
        tables.append(value.data if hasattr(value, "data") else value)
    for table in tables:
        assert not banned_pff & set(table.columns),             f"direct PFF fields leaked into a public table: {banned_pff & set(table.columns)}"

    shown = tables[0]
    assert {"Draft-Capital Hit-%", "College Hit-%", "Full Hit-%", "College Talent",
            "Athleticism (Percentile)", "Production (Percentile)"} <= set(shown.columns)

    for watch in tables[1:]:
        assert {"Player", "College", "College Talent"} <= set(watch.columns)

    copy = " ".join(str(x.value) for x in at.caption)
    assert "top-24 for RB/WR or top-12 for QB/TE" in copy
    assert "RB/WR/TE" in copy
    assert "Sleeper has not published one" in copy
    assert "Rookie QB model projections are intentionally withheld" in copy


def test_rookie_board_csvs_exclude_direct_pff_columns():
    banned = {"pff_grade", "pff_eff", "pct_pff_grade", "pct_pff_eff"}
    board_dir = _HERE / "fantasy" / "rookie" / "board_data"
    for cls in (2024, 2025, 2026):
        columns = set(pd.read_csv(board_dir / f"rookie_board_{cls}.csv", nrows=0).columns)
        assert not columns & banned


def test_league_history_caps_matchup_workers_without_network(monkeypatch):
    """A submitted history may fetch many weeks, but never unbounded concurrency."""
    seen_workers = []

    class _Executor:
        def __init__(self, max_workers):
            seen_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def map(self, fn, values):
            return map(fn, values)

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return []

    league_id = "1255197436951932928"

    def _sleeper_get(url):
        if url.endswith(f"/league/{league_id}"):
            return {
                "name": "Test league", "season": "2025", "status": "complete",
                "settings": {"playoff_week_start": 15}, "previous_league_id": "0",
            }
        return []

    monkeypatch.setattr(page_league_history, "_sleeper_get", _sleeper_get)
    monkeypatch.setattr(page_league_history._cf, "ThreadPoolExecutor", _Executor)
    monkeypatch.setattr(page_league_history.req, "get", lambda *_args, **_kwargs: _Response())
    page_league_history._fetch_sleeper_history.clear()
    try:
        history = page_league_history._fetch_sleeper_history(league_id)
        assert history["seasons"]
        assert seen_workers == [page_league_history._MATCHUP_FETCH_WORKERS]
        assert seen_workers[0] <= 6
    finally:
        page_league_history._fetch_sleeper_history.clear()


def test_league_intelligence_normalizes_drafts_and_explains_room_tendencies():
    seasons = {}
    for season in ("2023", "2024", "2025"):
        seasons[season] = {
            "draft_id": f"draft-{season}",
            "draft_picks": [
                {"pick_no": 1, "round": 1, "pick_in_round": 1, "draft_slot": 1,
                 "picked_by": "u1", "player_id": f"rb-{season}",
                 "metadata": {"first_name": "Early", "last_name": "Runner", "position": "RB"}},
                {"pick_no": 2, "round": 1, "pick_in_round": 2, "draft_slot": 2,
                 "picked_by": "u2", "player_id": f"wr-{season}",
                 "metadata": {"first_name": "Early", "last_name": "Receiver", "position": "WR"}},
                {"pick_no": 13, "round": 2, "pick_in_round": 1, "draft_slot": 2,
                 "picked_by": "u2", "player_id": f"te-{season}",
                 "metadata": {"first_name": "First", "last_name": "Tight End", "position": "TE"}},
                {"pick_no": 25, "round": 3, "pick_in_round": 1, "draft_slot": 1,
                 "picked_by": "u1", "player_id": f"qb-{season}",
                 "metadata": {"first_name": "First", "last_name": "Quarterback", "position": "QB"}},
                {"pick_no": 37, "round": 4, "pick_in_round": 1, "draft_slot": 1,
                 "picked_by": "u1", "player_id": f"qb2-{season}",
                 "metadata": {"first_name": "Backup", "last_name": "Quarterback", "position": "QB"}},
                {"pick_no": 38, "round": 4, "pick_in_round": 2, "draft_slot": 2,
                 "picked_by": "u2", "player_id": f"te2-{season}",
                 "metadata": {"first_name": "Backup", "last_name": "Tight End", "position": "TE"}},
            ],
        }

    picks = league_intel.draft_pick_frame(seasons)
    managers = league_intel.manager_season_frame(picks)
    construction = league_intel.roster_construction_frame(managers)
    timing = league_intel.first_pick_timing_frame(picks)

    assert len(picks) == 18
    assert set(timing["position"]) == {"QB", "TE"}
    assert set(timing[timing["position"] == "QB"]["round"]) == {3}
    assert set(timing[timing["position"] == "TE"]["round"]) == {2}
    assert construction["avg_qb"].eq(1).all()
    assert construction["avg_te"].eq(1).all()
    insights = league_intel.draft_insights(picks, managers, "u1")
    assert any(item["title"] == "Your early-round fingerprint" for item in insights)
    assert all({"finding", "meaning", "evidence", "confidence"} <= set(item) for item in insights)


def test_league_history_defaults_to_draft_and_roster_insights():
    assert page_league_history._LEAGUE_HISTORY_TABS[0] == "🧠 Draft & Roster Insights"
    assert page_league_history._LEAGUE_HISTORY_TABS[1] == "🏆 All-Time Leaderboard"


def test_manager_leaderboard_adjusts_scores_within_each_league_week():
    seasons = {
        "2024": {
            "standings": [
                {"username": "Alice", "wins": 2, "losses": 0, "playoff_finish": 1},
                {"username": "Bob", "wins": 0, "losses": 2, "playoff_finish": 2},
            ],
            "champion": {"username": "Alice"},
            "runner_up": {"username": "Bob"},
        },
        "2025": {
            "standings": [
                {"username": "Alice", "wins": 1, "losses": 1, "playoff_finish": 2},
                {"username": "Bob", "wins": 1, "losses": 1, "playoff_finish": 1},
            ],
            "champion": {"username": "Bob"},
            "runner_up": {"username": "Alice"},
        },
    }
    games = [
        {"season": "2024", "week": 1, "username": "Alice", "score": 110, "is_playoff": False},
        {"season": "2024", "week": 1, "username": "Bob", "score": 90, "is_playoff": False},
        {"season": "2024", "week": 2, "username": "Alice", "score": 80, "is_playoff": False},
        {"season": "2024", "week": 2, "username": "Bob", "score": 120, "is_playoff": False},
        {"season": "2024", "week": 15, "username": "Alice", "score": 200, "is_playoff": True},
    ]

    leaders = league_intel.manager_leaderboard_frame(seasons, games).set_index("manager")
    assert leaders.loc["Alice", "win_pct"] == 75.0
    assert leaders.loc["Alice", "titles"] == 1
    assert leaders.loc["Alice", "finals"] == 2
    assert leaders.loc["Alice", "seasons"] == 2
    assert leaders.loc["Alice", "avg_score"] == 95.0
    assert leaders.loc["Alice", "avg_above_league"] == -5.0
    assert leaders.loc["Bob", "avg_above_league"] == 5.0
    assert leaders.loc["Alice", "games"] == 2


def test_matchup_record_book_uses_weekly_all_play_for_luck():
    games = [
        {"season": "2025", "week": 1, "username": "Alice", "score": 110,
         "opp": "Bob", "opp_score": 100, "won": True, "is_playoff": False},
        {"season": "2025", "week": 1, "username": "Bob", "score": 100,
         "opp": "Alice", "opp_score": 110, "won": False, "is_playoff": False},
        {"season": "2025", "week": 1, "username": "Carol", "score": 150,
         "opp": "Dan", "opp_score": 140, "won": True, "is_playoff": False},
        {"season": "2025", "week": 1, "username": "Dan", "score": 140,
         "opp": "Carol", "opp_score": 150, "won": False, "is_playoff": False},
        {"season": "2025", "week": 2, "username": "Alice", "score": 4,
         "opp": "Bob", "opp_score": 3, "won": True, "is_playoff": False},
    ]

    matchups = league_intel.matchup_record_frame(games)
    assert len(matchups) == 2
    alice = matchups.set_index("winner").loc["Alice"]
    carol = matchups.set_index("winner").loc["Carol"]
    assert alice["combined"] == 210
    assert alice["margin"] == 10
    assert alice["all_play_wins"] == 1
    assert alice["all_play_opponents"] == 3
    assert alice["all_play_win_pct"] == 33.3
    assert carol["all_play_win_pct"] == 100.0


def test_rivalry_summary_tracks_series_scoring_playoffs_and_streaks():
    games = []
    for season, week, alice_score, bob_score, playoff in (
        ("2024", 1, 100, 90, False),
        ("2024", 2, 100, 110, False),
        ("2025", 15, 100, 120, True),
    ):
        games.extend([
            {"season": season, "week": week, "username": "Alice",
             "score": alice_score, "opp": "Bob", "opp_score": bob_score,
             "won": alice_score > bob_score, "is_playoff": playoff},
            {"season": season, "week": week, "username": "Bob",
             "score": bob_score, "opp": "Alice", "opp_score": alice_score,
             "won": bob_score > alice_score, "is_playoff": playoff},
        ])

    matchups = league_intel.matchup_record_frame(games)
    rivalry = league_intel.rivalry_summary_frame(matchups).iloc[0]
    assert rivalry["manager_a"] == "Alice"
    assert rivalry["manager_b"] == "Bob"
    assert rivalry["games"] == 3
    assert rivalry["manager_a_wins"] == 1
    assert rivalry["manager_b_wins"] == 2
    assert rivalry["manager_a_avg_score"] == 100.0
    assert rivalry["manager_b_avg_score"] == 106.67
    assert rivalry["avg_point_diff"] == -6.67
    assert rivalry["playoff_meetings"] == 1
    assert rivalry["current_streak_manager"] == "Bob"
    assert rivalry["current_streak"] == 2
    assert rivalry["closest_margin"] == 10
    assert rivalry["largest_margin"] == 20


def test_rivalry_week_scores_modes_and_uses_stable_unique_labels():
    identities = {"u1": "Alex", "u2": "Alex", "u3": "Casey", "u4": "Drew"}
    labels = league_intel.manager_display_labels(identities)
    assert labels["u1"] != labels["u2"]
    assert labels["u3"] == "Casey"

    games = []
    for season, week, alex_score, casey_score, playoff in (
        ("2023", 1, 101, 99, False),
        ("2023", 15, 112, 115, True),
        ("2024", 2, 120, 116, False),
        ("2024", 16, 104, 108, True),
        ("2025", 4, 130, 126, False),
        ("2025", 8, 99, 101, False),
    ):
        games.extend([
            {"season": season, "week": week, "username": labels["u1"],
             "score": alex_score, "opp": labels["u3"], "opp_score": casey_score,
             "won": alex_score > casey_score, "is_playoff": playoff},
            {"season": season, "week": week, "username": labels["u3"],
             "score": casey_score, "opp": labels["u1"], "opp_score": alex_score,
             "won": casey_score > alex_score, "is_playoff": playoff},
        ])

    matchups = league_intel.matchup_record_frame(games)
    managers = list(labels.values())
    classic = league_intel.rivalry_pair_score_frame(
        matchups, managers, "Classic Rivalries"
    )
    fresh = league_intel.rivalry_pair_score_frame(matchups, managers, "Fresh Blood")

    established_pair = frozenset((labels["u1"], labels["u3"]))
    classic_by_pair = {
        frozenset((row["manager_a"], row["manager_b"])): row
        for _, row in classic.iterrows()
    }
    fresh_by_pair = {
        frozenset((row["manager_a"], row["manager_b"])): row
        for _, row in fresh.iterrows()
    }
    assert len(classic) == 6
    assert classic_by_pair[established_pair]["games"] == 6
    assert classic_by_pair[established_pair]["playoff_meetings"] == 2
    assert classic_by_pair[established_pair]["rivalry_score"] > classic[
        classic["games"].eq(0)
    ]["rivalry_score"].max()
    assert fresh_by_pair[established_pair]["rivalry_score"] < fresh[
        fresh["games"].eq(0)
    ]["rivalry_score"].min()
    assert "First recorded meeting" in fresh[fresh["games"].eq(0)].iloc[0]["reason"]


def test_rivalry_week_slate_optimizes_globally_and_honors_locks():
    # Greedy AB first would total only 101. The global optimum is AC + BD = 198.
    scores = pd.DataFrame([
        {"manager_a": "A", "manager_b": "B", "rivalry_score": 100.0, "reason": "AB"},
        {"manager_a": "A", "manager_b": "C", "rivalry_score": 99.0, "reason": "AC"},
        {"manager_a": "A", "manager_b": "D", "rivalry_score": 2.0, "reason": "AD"},
        {"manager_a": "B", "manager_b": "C", "rivalry_score": 2.0, "reason": "BC"},
        {"manager_a": "B", "manager_b": "D", "rivalry_score": 99.0, "reason": "BD"},
        {"manager_a": "C", "manager_b": "D", "rivalry_score": 1.0, "reason": "CD"},
    ])

    slate = league_intel.rivalry_week_slate_frame(scores)
    pairs = {
        frozenset((row["manager_a"], row["manager_b"]))
        for _, row in slate.iterrows()
    }
    assert pairs == {frozenset(("A", "C")), frozenset(("B", "D"))}
    assert slate["rivalry_score"].sum() == 198.0

    locked = league_intel.rivalry_week_slate_frame(
        scores,
        locked_pairs=[("A", "B")],
        avoided_pairs=[("C", "D")],
    )
    locked_pairs = {
        frozenset((row["manager_a"], row["manager_b"]))
        for _, row in locked.iterrows()
    }
    assert locked_pairs == {frozenset(("A", "B")), frozenset(("C", "D"))}
    assert bool(locked.loc[locked["reason"].eq("AB"), "locked"].iloc[0])


def test_manager_performance_uses_peer_week_context_and_excludes_playoffs():
    games = []
    weekly = {
        1: {"Alice": (95, "Bob", 90), "Bob": (90, "Alice", 95),
            "Carol": (130, "Dan", 85), "Dan": (85, "Carol", 130)},
        2: {"Alice": (110, "Bob", 120), "Bob": (120, "Alice", 110),
            "Carol": (90, "Dan", 80), "Dan": (80, "Carol", 90)},
    }
    for week, managers in weekly.items():
        for manager, (score, opponent, opponent_score) in managers.items():
            games.append({
                "season": "2025", "week": week, "username": manager,
                "score": score, "opp": opponent, "opp_score": opponent_score,
                "won": score > opponent_score, "is_playoff": False,
            })
    games.extend([
        {"season": "2025", "week": 15, "username": "Alice", "score": 200,
         "opp": "Bob", "opp_score": 100, "won": True, "is_playoff": True},
        {"season": "2025", "week": 15, "username": "Bob", "score": 100,
         "opp": "Alice", "opp_score": 200, "won": False, "is_playoff": True},
    ])

    context = league_intel.weekly_score_context_frame(games)
    assert len(context) == 8
    alice_week_one = context[
        context["manager"].eq("Alice") & context["week"].eq(1)
    ].iloc[0]
    alice_week_two = context[
        context["manager"].eq("Alice") & context["week"].eq(2)
    ].iloc[0]
    assert alice_week_one["league_average"] == 100.0
    assert alice_week_one["league_median"] == 92.5
    assert alice_week_one["adjusted_score"] == -5.0
    assert alice_week_two["league_average"] == 100.0
    assert alice_week_two["league_median"] == 100.0
    assert alice_week_two["adjusted_score"] == 10.0

    performance = league_intel.manager_performance_frame(games).set_index("manager")
    alice = performance.loc["Alice"]
    assert alice["games"] == 2
    assert alice["wins"] == 1
    assert alice["losses"] == 1
    assert alice["win_pct"] == 50.0
    assert alice["avg_score"] == 102.5
    assert alice["avg_above_league"] == 2.5
    assert alice["std_dev"] == 10.61
    assert alice["lucky_wins"] == 1
    assert alice["unlucky_losses"] == 1

    consistency = league_intel.consistency_luck_frame(games).set_index("manager")
    alice_luck = consistency.loc["Alice"]
    carol_luck = consistency.loc["Carol"]
    assert alice_luck["games"] == 2
    assert alice_luck["avg_above_league"] == 2.5
    assert alice_luck["volatility"] == 10.61
    assert alice_luck["actual_wins"] == 1.0
    assert alice_luck["expected_wins"] == 1.33
    assert alice_luck["luck_delta"] == -0.33
    assert alice_luck["below_avg_wins"] == 1
    assert alice_luck["above_avg_losses"] == 1
    assert carol_luck["luck_delta"] == 0.67


def test_loaded_league_history_renders_insights_first_and_chart_first_leaderboard(tmp_path):
    fixture = {
        "league_name": "Test League",
        "seasons": {
            "2025": {
                "league_id": "1255197436951932928",
                "draft_id": "draft-2025",
                "status": "complete",
                "champion": {"username": "Alice", "team_name": "A Team"},
                "runner_up": {"username": "Bob", "team_name": "B Team"},
                "standings": [
                    {
                        "rank": 1, "roster_id": 1, "owner_id": "u1",
                        "username": "Alice", "team_name": "A Team", "wins": 1,
                        "losses": 0, "ties": 0, "fpts": 120.0,
                        "fpts_against": 80.0, "playoff_finish": 1,
                    },
                    {
                        "rank": 2, "roster_id": 2, "owner_id": "u2",
                        "username": "Bob", "team_name": "B Team", "wins": 0,
                        "losses": 1, "ties": 0, "fpts": 80.0,
                        "fpts_against": 120.0, "playoff_finish": 2,
                    },
                ],
                "matchups": [{
                    "season": "2025", "week": 1, "is_playoff": False,
                    "rid_a": "1", "rid_b": "2", "score_a": 120.0, "score_b": 80.0,
                }],
                "draft_picks": [],
                "roster_entries": [],
                "league_settings": {
                    "total_rosters": 2, "roster_positions": [], "scoring_settings": {},
                },
            },
        },
    }
    harness = tmp_path / "league_history_loaded.py"
    harness.write_text(
        f"import sys\nsys.path[:0] = [r'{_HERE}', r'{_SITE_PAGES}']\n"
        "import page_league_history as p\n"
        "p._OFFLINE = False\n"
        f"p._fetch_sleeper_history = lambda _league_id: {fixture!r}\n"
        "p._league_history_season_count = lambda _league_id: 1\n"
        "p._fetch_player_directory = lambda: {}\n"
        "p.render()\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(harness), default_timeout=180).run()
    league_input = next(widget for widget in at.text_input if widget.key == "lh_league_id")
    league_input.set_value("1255197436951932928")
    next(button for button in at.button if button.label == "Load league history").click()
    at.run()

    assert not at.exception, at.exception
    assert [tab.label for tab in at.tabs][:2] == [
        "🧠 Draft & Roster Insights", "🏆 All-Time Leaderboard",
    ]
    assert any(tab.label == "⚔️ Rivalries" for tab in at.tabs)
    assert len(at.get("plotly_chart")) >= 4
    assert any(metric.label == "Best Adjusted Scorer" for metric in at.metric)
    assert any(metric.label == "🍀 Luckiest Win (All-Play)" for metric in at.metric)
    assert any(metric.label == "Scoring vs League" for metric in at.metric)
    assert any(metric.label == "Most Consistent" for metric in at.metric)
    assert any(expander.label == "View complete manager records" for expander in at.expander)
    rivalry_view = next(widget for widget in at.radio if widget.key == "lh_rivalry_view")
    assert rivalry_view.value == "Build a Week"
    assert {"lh_rivalry_mode", "lh_rivalry_history"} <= {
        widget.key for widget in at.selectbox
    }
    assert not {"lh_h2h_manager_a", "lh_h2h_manager_b"} & {
        widget.key for widget in at.selectbox
    }
    assert any(
        button.label == "Generate another slate" for button in at.button
    )
    assert not any(
        expander.label == "View complete head-to-head record matrix"
        for expander in at.expander
    )

    rivalry_view.set_value("Explore a Matchup")
    at.run()
    assert not at.exception, at.exception
    assert {"lh_h2h_manager_a", "lh_h2h_manager_b"} <= {
        widget.key for widget in at.selectbox
    }
    assert not {"lh_rivalry_mode", "lh_rivalry_history"} & {
        widget.key for widget in at.selectbox
    }
    alice_average = next(
        metric for metric in at.metric if metric.label == "Alice Avg Score"
    )
    bob_average = next(
        metric for metric in at.metric if metric.label == "Bob Avg Score"
    )
    assert alice_average.value == "120.0 pts"
    assert bob_average.value == "80.0 pts"

    next(
        widget for widget in at.radio if widget.key == "lh_rivalry_view"
    ).set_value("League Matrix")
    at.run()
    assert not at.exception, at.exception
    assert any(
        expander.label == "View complete head-to-head record matrix"
        for expander in at.expander
    )
    assert not {"lh_h2h_manager_a", "lh_h2h_manager_b"} & {
        widget.key for widget in at.selectbox
    }
    assert any(
        expander.label == "View complete career season history"
        for expander in at.expander
    )
    assert any(
        expander.label == "View complete opponent breakdown"
        for expander in at.expander
    )
    assert any(
        expander.label == "View complete consistency and luck metrics"
        for expander in at.expander
    )
    assert any(metric.label == "League Average" for metric in at.metric)
    assert any(
        expander.label == "View complete score trend data"
        for expander in at.expander
    )

    season_filter = next(
        widget for widget in at.selectbox if widget.key == "lh_season_filter"
    )
    season_filter.set_value("2025")
    at.run()
    assert not at.exception, at.exception
    assert any(metric.label == "Scoring vs League" for metric in at.metric)
    assert any(metric.label == "League Average" for metric in at.metric)


def test_player_history_uses_four_week_filter_and_excludes_null_matchup_points():
    entries = []
    for week, matchup_id in ((1, 1), (2, 2), (3, 3), (18, None)):
        entries.append({
            "season": "2025", "week": week, "roster_id": 4, "matchup_id": matchup_id,
            "players": ["p1", "p2", "p3"] if week <= 3 else ["p1", "p3"],
            "starters": ["p1", "p3"] if week <= 2 else ["p3"],
            "players_points": {
                "p1": {1: 10, 2: 20, 3: 30, 18: 99}[week],
                "p2": 4,
                "p3": 5,
            },
        })
    seasons = {
        "2025": {
            "draft_id": "draft-2025",
            "standings": [{"roster_id": 4, "owner_id": "u1", "username": "Manager"}],
            "draft_picks": [{
                "pick_no": 88, "round": 8, "pick_in_round": 4, "draft_slot": 4,
                "picked_by": "u1", "player_id": "p1",
                "metadata": {"first_name": "Draft", "last_name": "Hit", "position": "QB"},
            }],
            "roster_entries": entries,
        }
    }
    directory = {
        "p1": {"full_name": "Draft Hit", "position": "QB"},
        "p2": {"full_name": "Short Stay", "position": "WR"},
        "p3": {"full_name": "Added Player", "position": "RB"},
    }

    weeks = league_intel.player_week_frame(seasons, directory)
    summary = league_intel.player_season_summary(weeks, min_roster_weeks=4)
    assert set(summary["player_id"]) == {"p1", "p3"}

    p1 = summary.set_index("player_id").loc["p1"]
    assert p1["roster_weeks"] == 4
    assert p1["starts"] == 2
    assert p1["lineup_points"] == 30
    assert p1["roster_points"] == 60
    assert p1["bench_points"] == 30

    values = league_intel.value_frame(summary, league_intel.draft_pick_frame(seasons))
    sources = values.set_index("player_id")["source"].to_dict()
    assert sources == {"p1": "Drafted", "p3": "In-season addition"}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        test_weekly_fantasy_renders_and_owns_controls(p)
        test_league_history_renders_and_lands_empty(p)
    print("OK  WF owns wf_* controls; LH lands empty with the resting prompt")
