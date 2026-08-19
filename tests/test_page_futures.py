"""Season Totals page: leftover mix, honest backtest, language fence.

Hermetic (APP_OFFLINE=1). READ-ONLY over futures/published/season_totals_2026.csv
and futures/published/evidence.json. No fit, no simulation, no network.

What is asserted:
* the page renders, and boots inside app.py
* table numbers match the published CSV; MAE figures match evidence.json
* 32 projections sum to 272
* language fence against futures/language_fence.py
* no Line/Book/Over/Under columns
* runtime imports stay json/pathlib/pandas/streamlit
"""
import ast
import json
import os
import sys
from pathlib import Path

os.environ["APP_OFFLINE"] = "1"

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "site_pages"))
sys.path.insert(0, str(_HERE / "futures"))

_CSV = _HERE / "futures" / "published" / "season_totals_2026.csv"
_EVIDENCE = _HERE / "futures" / "published" / "evidence.json"
_PAGE = _HERE / "site_pages" / "page_futures.py"

pytestmark = pytest.mark.skipif(
    not _CSV.exists(),
    reason="published season totals missing (run seasonal_totals_v2_beta: python src/publish_site.py)",
)


def _entry():
    import page_futures
    page_futures.render()


def _run(fn=None):
    at = AppTest.from_function(fn or _entry, default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def _text(at):
    parts = []
    for group in (at.title, at.markdown, at.caption, at.warning, at.info, at.error,
                  at.subheader, at.header):
        parts += [str(e.value) for e in group]
    for m in at.metric:
        parts += [str(getattr(m, a, "") or "") for a in ("label", "value", "delta", "help")]
    for el in at.dataframe:
        v = el.value
        d = v.data if hasattr(v, "data") else v
        try:
            parts += [str(c) for c in d.columns]
        except Exception:
            pass
    return " ".join(parts)


def _table(at):
    for el in at.dataframe:
        v = el.value
        d = v.data if hasattr(v, "data") else v
        try:
            if "Proj Wins" in list(d.columns) and "Posted" in list(d.columns):
                return d
        except Exception:
            pass
    return None


def test_proj_wins_display_is_one_decimal():
    src = _PAGE.read_text(encoding="utf-8")
    assert '"Proj Wins": st.column_config.NumberColumn(format="%.1f"' in src, \
        "Proj Wins must round on screen to one decimal; internals stay full precision"


def test_page_renders_and_shows_the_projection_table():
    at = _run()
    df = _table(at)
    assert df is not None, "the projection table must render"
    assert len(df) == 32, f"expected 32 teams, got {len(df)}"
    for c in ("Team", "Proj Wins", "Posted", "vs posted"):
        assert c in list(df.columns), f"expected column {c} missing"


def test_display_matches_the_artifact():
    at = _run()
    df = _table(at)
    csv = pd.read_csv(_CSV)
    assert set(df["Team"]) == set(csv["team"])
    merged = df.merge(csv, left_on="Team", right_on="team")
    assert len(merged) == 32
    assert (merged["Proj Wins"] - merged["proj_wins"]).abs().max() < 1e-9
    assert (merged["Posted"] - merged["posted"]).abs().max() < 1e-9
    assert (merged["vs posted"] - merged["vs_posted"]).abs().max() < 1e-9


def test_league_conservation_survives_to_the_display():
    at = _run()
    df = _table(at)
    assert abs(float(df["Proj Wins"].sum()) - 272.0) < 0.01, \
        "the displayed projections no longer sum to the scheduled game count"
    assert df["Proj Wins"].is_monotonic_decreasing


def test_the_backtest_result_is_stated_plainly():
    at = _run()
    text = _text(at).lower()
    assert "does not beat" in text
    assert "posted win total" in text
    assert "backtested" in text and "not live-validated" in text.replace(
        "live validated", "live-validated"
    )
    for forbidden_name in ("vegas", "the sportsbook line", "archived market consensus"):
        assert forbidden_name not in text, f"benchmark must never be called {forbidden_name!r}"


def test_claim_label_is_read_from_the_artifact_not_retyped():
    at = _run()
    label = str(pd.read_csv(_CSV)["claim"].iloc[0])
    assert label in _text(at), "the artifact's claim label must appear verbatim on the page"


def test_displayed_evidence_numbers_come_from_the_artifacts():
    at = _run()
    text = _text(at)
    ev = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    mae = float(ev["mae_model"])
    bench = float(ev["mae_market"])
    assert f"{mae:.4f} wins" in text, f"backtest MAE {mae:.4f} must be shown"
    assert f"{bench:.4f}" in text, f"posted MAE {bench:.4f} must be shown from the artifact"
    assert mae > bench, "the artifacts no longer support the 'does not beat' claim"
    assert f"{mae - bench:+.4f}" in text, "the gap must be shown with its sign"


def test_accuracy_ladder_is_anchored_not_a_bare_number():
    at = _run()
    ladder = None
    for el in at.dataframe:
        v = el.value
        dd = v.data if hasattr(v, "data") else v
        try:
            if "Approach" in list(dd.columns):
                ladder = dd
        except Exception:
            pass
    assert ladder is not None, "the accuracy ladder must render"
    assert len(ladder) == 4
    m = ladder["Average miss (wins)"]
    assert m.is_monotonic_decreasing, "the ladder must run worst to best"
    ev = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    assert abs(float(m.iloc[-1]) - ev["ladder"][-1]["mae"]) < 1e-9
    names = list(ladder["Approach"])
    assert names[-1] == "Posted win total"
    assert "This model" in names
    assert "Retired Monte Carlo" in names


def test_evidence_section_degrades_without_its_artifact(tmp_path, monkeypatch):
    import page_futures

    monkeypatch.setattr(page_futures, "_EVIDENCE", tmp_path / "absent.json")
    at = AppTest.from_function(_entry, default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    txt = " ".join(str(e.value) for g in (at.markdown, at.caption, at.subheader) for e in g)
    assert "How good is this" not in txt, "the section must vanish, not half-render"
    assert any("Proj Wins" in str(list((e.value.data if hasattr(e.value, "data") else e.value)
                                       .columns)) for e in at.dataframe)


def test_no_hardcoded_backtest_number_in_the_source():
    ev = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    src = _PAGE.read_text(encoding="utf-8")
    for n in (ev["mae_model"], ev["mae_market"], ev["mae_persist"], ev["mae_retired_m4c"]):
        for literal in (f"{n:.2f}", f"{n:.3f}", f"{n:.4f}"):
            assert literal not in src, \
                f"{literal!r} is typed into page_futures.py - read it from the artifact instead"


def test_language_fence_holds_against_the_guards_own_vocabulary():
    from language_fence import BANNED, tokens

    at = _run()
    text = _text(at)
    hits = sorted(tokens(text) & BANNED)
    assert not hits, f"fenced vocabulary on the Season Totals page: {hits}"


def test_no_market_columns_reach_the_display():
    from language_fence import BANNED, tokens

    market = ("win_total_line", "book", "line_as_of", "p_over", "p_under", "p_push",
              "Line", "Book", "Over", "Under", "Push")
    at = _run()
    cols = {str(c) for el in at.dataframe
            for c in list((el.value.data if hasattr(el.value, "data") else el.value).columns)}
    assert not (cols & set(market)), f"market columns on display: {sorted(cols & set(market))}"
    assert "Posted" in cols
    assert not (tokens("p_over") & BANNED), \
        "language_fence now covers p_over - re-derive which mechanism excludes market columns"


def test_runtime_separation_no_training_dependency():
    tree = ast.parse(_PAGE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"sklearn", "lightgbm", "xgboost", "scipy", "joblib", "numpy", "nflreadpy",
              "m4_engine", "papermill", "torch", "seasonal_totals_v2_beta"}
    assert not (imported & banned), f"page_futures imports training dependencies: {imported & banned}"
    assert imported <= {"json", "pathlib", "pandas", "streamlit"}, \
        f"unexpected page imports: {sorted(imported - {'json', 'pathlib', 'pandas', 'streamlit'})}"


def test_no_dash_pause_glyphs_in_page_copy():
    em, en = chr(0x2014), chr(0x2013)
    src = _PAGE.read_text(encoding="utf-8")
    assert em not in src, "em dash in page_futures.py"
    assert en not in src, "en dash in page_futures.py"
    at = _run()
    rendered = _text(at)
    assert em not in rendered, "em dash in the copy this page puts on screen"
    assert en not in rendered, "en dash in the copy this page puts on screen"
    label = str(pd.read_csv(_CSV)["claim"].iloc[0])
    assert em not in label and en not in label, "the artifact's claim label carries a dash glyph"


def test_page_uses_only_shared_layout_primitives():
    src = _PAGE.read_text(encoding="utf-8")
    assert "st.markdown(" in src
    assert "unsafe_allow_html" not in src, \
        "raw HTML would escape the shared mobile layer - use Streamlit primitives"
    assert "width=\"stretch\"" in src, "the table must stretch rather than carry a fixed pixel width"


def test_registered_in_the_multipage_entrypoint():
    src = (_HERE / "app.py").read_text(encoding="utf-8")
    assert 'url_path="season-totals"' in src, "the page needs a stable url_path"
    assert '"season-totals": fut_pg' in src, "the page must be in the cross-link registry"
    assert 'title="Season Totals (Beta)"' in src, "the nav label must carry the beta flag"
    tree = ast.parse(src)
    eager = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
             for a in n.names if a.name.startswith("page_")}
    assert eager == set(), f"pages must stay lazily imported; eager: {eager}"


def test_app_boots_with_the_page_wired_in():
    at = AppTest.from_file(str(_HERE / "app.py"), default_timeout=240).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]


if __name__ == "__main__":
    test_page_renders_and_shows_the_projection_table()
    test_display_matches_the_artifact()
    test_league_conservation_survives_to_the_display()
    test_the_backtest_result_is_stated_plainly()
    test_claim_label_is_read_from_the_artifact_not_retyped()
    test_displayed_evidence_numbers_come_from_the_artifacts()
    test_no_hardcoded_backtest_number_in_the_source()
    test_language_fence_holds_against_the_guards_own_vocabulary()
    test_no_market_columns_reach_the_display()
    test_runtime_separation_no_training_dependency()
    test_no_dash_pause_glyphs_in_page_copy()
    test_page_uses_only_shared_layout_primitives()
    test_registered_in_the_multipage_entrypoint()
    test_app_boots_with_the_page_wired_in()
    print("OK  Season Totals page: leftover mix, artifact-traceable numbers, fence holds, app boots")
