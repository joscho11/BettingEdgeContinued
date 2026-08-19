"""Batch-1 proof for the multipage nav skeleton (app.py).

Asserts: the default landing page is the 2026 Draft Board, Fantasy sits left of
Betting in the top nav, the sidebar renders EMPTY (nav is top, footer is in page
flow), the shared footer is present, and the shared modules are import-safe.
Hermetic: APP_OFFLINE=1 so no network. Run: pytest test_site_nav.py
"""
import ast
import os
import sys
from pathlib import Path

os.environ["APP_OFFLINE"] = "1"   # set before importing streamlit-touching modules

from streamlit.testing.v1 import AppTest

_HERE = Path(__file__).resolve().parents[1]
_SITE_PAGES = _HERE / "site_pages"
sys.path.insert(0, str(_HERE))
ENTRY = str(_HERE / "app.py")   # the multipage entrypoint (post-3e swap)
PAGE_MODULES = (
    "page_weekly_predictions",
    "page_track_record",
    "page_draft_board",
    "page_rookie_board",
    "page_weekly_fantasy",
    "page_dfs",
    "page_film_room",
    "page_league_history",
    "page_help",
    "page_futures",
)


def _run():
    at = AppTest.from_file(ENTRY, default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def _titles(at):
    return " ".join(str(t.value) for t in at.title)


def test_default_is_draft_board():
    at = _run()
    titles = _titles(at)
    assert "Draft Board" in titles, \
        f"default landing page should be the 2026 Draft Board; titles={titles!r}"
    assert "Predictions" not in titles, \
        f"Weekly Predictions must no longer be the default; titles={titles!r}"


def test_nav_groups_fantasy_then_betting():
    src = Path(ENTRY).read_text(encoding="utf-8")
    assert src.index('"Fantasy"') < src.index('"Betting"'), \
        "Fantasy must sit left of Betting in the top nav"
    assert "url_path=\"draft-board\", default=True" in src
    assert "url_path=\"weekly-predictions\", default=True" not in src


def test_sidebar_is_empty_and_footer_present():
    at = _run()
    # nav is position="top"; nothing writes to the sidebar -> empty
    assert len(list(at.sidebar.markdown)) == 0, "sidebar must carry no markdown"
    # the tip jar moved UP into the header, so it is no longer a button anywhere
    assert not any(getattr(b, "key", None) == "tip_jar_btn" for b in at.button), \
        "the footer tip-jar button must be gone (tip jar moved to the header)"
    caps = " ".join(str(c.value) for c in at.caption)
    assert "buy me a coffee" not in caps, "the coffee caption must not remain in the footer"
    # footer now carries only the centered public-repo line (an st.markdown, not a caption)
    md = " ".join(str(m.value) for m in at.markdown)
    assert "github.com/joscho11/JoSchoAnalytics" in md, "footer repo link missing"


def test_header_has_brand_and_tip_jar():
    at = _run()
    # the persistent header strip carries the brand (left) and the tip jar (right),
    # moved byte-identical from the old footer — both live in one markdown div.
    hdr = [str(m.value) for m in at.markdown
           if "JoScho Analytics" in str(m.value) and "Tip Jar — Venmo @JoScho" in str(m.value)]
    assert hdr, "header strip (brand + tip jar) must render on the page"
    assert "https://venmo.com/u/JoScho" in hdr[0], "tip jar must keep the byte-identical Venmo URL"
    assert "buy me a coffee ☕" in hdr[0], "the coffee line travels byte-identical as the tip-jar title"
    # and it must NOT be duplicated as a footer button
    assert not any(getattr(b, "key", None) == "tip_jar_btn" for b in at.button), \
        "tip jar must not remain in the footer"


def test_phone_nav_button_is_three_bars():
    src = (_HERE / "dashboard_chrome.py").read_text(encoding="utf-8")
    assert "[data-testid=\"stExpandSidebarButton\"]::after" in src
    assert "[data-testid=\"stSidebarCollapseButton\"]::after" in src
    assert "box-shadow:0 -.35rem 0 #fafafa, 0 .35rem 0 #fafafa" in src


def test_shared_modules_import_safe():
    # importing the shared modules must not fire network/data work at import time
    import dashboard_data
    import dashboard_chrome
    for fn in ("load_predictions", "load_totals", "load_calibration"):
        assert hasattr(dashboard_data, fn)
    for fn in ("send_ga_event", "inject_css", "render_header", "render_footer", "site_pageview_once"):
        assert hasattr(dashboard_chrome, fn)


def test_dfs_stays_built_but_off_the_nav():
    src = Path(ENTRY).read_text(encoding="utf-8")
    assert '_lazy_render("page_dfs")' in src
    assert (_HERE / "site_pages" / "page_dfs.py").is_file()
    fantasy = src.split('"Fantasy":', 1)[1].split("]", 1)[0]
    assert "dfs_pg" not in fantasy
    assert "wf_pg" in fantasy


def test_nonselected_pages_are_lazy_imported():
    tree = ast.parse(Path(ENTRY).read_text(encoding="utf-8"))
    eager_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("page_")
    }
    assert eager_imports == set()


def test_every_page_renders_offline_clean(tmp_path):
    for module in PAGE_MODULES:
        harness = tmp_path / f"h_{module}.py"
        harness.write_text(
            f"import sys; sys.path[:0] = [r'{_HERE}', r'{_SITE_PAGES}']\n"
            f"import {module} as page\npage.render()\n",
            encoding="utf-8",
        )
        at = AppTest.from_file(str(harness), default_timeout=180).run()
        assert not at.exception, f"{module}: {at.exception}"
        assert not at.error, f"{module}: {[error.value for error in at.error]}"


if __name__ == "__main__":
    test_default_is_draft_board()
    test_nav_groups_fantasy_then_betting()
    test_sidebar_is_empty_and_footer_present()
    test_header_has_brand_and_tip_jar()
    test_phone_nav_button_is_three_bars()
    test_shared_modules_import_safe()
    test_nonselected_pages_are_lazy_imported()
    print("OK  nav skeleton: Draft Board default, Fantasy then Betting, empty sidebar, header brand+tip jar, footer repo link")
