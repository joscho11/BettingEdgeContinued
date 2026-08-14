"""Film Room: one player, title list, a single TikTok embed.

Hermetic: APP_OFFLINE=1. Renders page_film_room directly, same pattern as
test_site_nav.page harnesses.
"""
import os
import sys
from pathlib import Path

os.environ["APP_OFFLINE"] = "1"

from streamlit.testing.v1 import AppTest

_HERE = Path(__file__).resolve().parents[1]
_SITE_PAGES = _HERE / "site_pages"
sys.path.insert(0, str(_HERE))

from video_content import INTRO_VIDEO, VIDEOS  # noqa: E402


def _render(tmp_path):
    harness = tmp_path / "h_film_room.py"
    harness.write_text(
        f"import sys; sys.path[:0] = [r'{_HERE}', r'{_SITE_PAGES}']\n"
        "import page_film_room as page\npage.render()\n",
        encoding="utf-8",
    )
    at = AppTest.from_file(str(harness), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


def _newest():
    return sorted(VIDEOS, key=lambda v: v.get("date") or "", reverse=True)[0]


def _md(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_default_is_newest_episode(tmp_path):
    at = _render(tmp_path)
    newest = _newest()
    md = _md(at)
    assert newest["title"] in md
    assert INTRO_VIDEO["title"] not in md
    watch = [m for m in at.markdown if "Watch on TikTok" in str(m.value)]
    assert len(watch) == 1, "only the selected video should render a Watch link"
    assert newest["tiktok_url"] in str(watch[0].value)


def test_picker_lists_intro_and_every_episode(tmp_path):
    at = _render(tmp_path)
    labels = [str(b.label) for b in at.button]
    assert any(lbl.startswith("Start here") for lbl in labels)
    for item in VIDEOS:
        if item.get("archived"):
            assert any(item["title"] in lbl for lbl in labels)
        else:
            assert item["title"] in labels
    breakdowns = [b for b in at.button if "Full breakdown" in str(b.label)
                  or "What is this?" in str(b.label)]
    assert len(breakdowns) == 1


def test_start_here_loads_intro_and_keeps_one_embed(tmp_path):
    at = _render(tmp_path)
    at.button("fr_pick___intro__").click()
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    md = _md(at)
    assert INTRO_VIDEO["title"] in md
    watch = [m for m in at.markdown if "Watch on TikTok" in str(m.value)]
    assert len(watch) == 1
    assert INTRO_VIDEO["tiktok_url"] in str(watch[0].value)
    assert any("What is this?" in str(b.label) for b in at.button)


def test_switching_episode_swaps_the_embed(tmp_path):
    at = _render(tmp_path)
    newest = _newest()
    other = next(v for v in VIDEOS if v["slug"] != newest["slug"])
    at.button(f"fr_pick_{other['slug']}").click()
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    watch = [m for m in at.markdown if "Watch on TikTok" in str(m.value)]
    assert len(watch) == 1
    assert other["tiktok_url"] in str(watch[0].value)
    assert newest["tiktok_url"] not in str(watch[0].value)
