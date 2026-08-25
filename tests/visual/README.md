# Visual regression suite

Page-level screenshots for every public Streamlit route at phone (390), tablet
(768), and desktop (1440). Extends the header geometry suite in
`tests/test_responsive_layout.py`; it does not replace it.

Streamlit scrolls inside `stApp`, not the document. `screenshot_png` expands
those containers so `full_page=True` is the whole page, not the first viewport.
Dataframe widgets still virtualize their own rows. Column overflow is caught
by the layout probe on the visible grid.

## What it catches

- Publishing badges: Published, Scheduled, Awaiting projections
- Matchup cards: SCORE / SPREAD / PREDICTED wrapping, MATCHUP-only 2026 rows
- Credibility: Home, Track Record empty/live, Help expanders, Season Totals,
  Anytime TDs, Film Room
- Fantasy: Draft Board ADP sources, Rookie Board, DFS empty and optimized
- League History: Sleeper/ESPN/Yahoo empty and private cookie forms, offline
  fail-closed, loaded intelligence tabs from a checked-in fixture

## Commands

Hermetic catalog and compare tests (no browser):

```bash
set APP_OFFLINE=1
python -m pytest tests/test_visual_regression.py -m "not visual" -v
```

Screenshots (Playwright Chromium). Copy and overflow checks run on every OS.
Pixel compare against committed PNGs runs on Linux unless `VISUAL_PIXELS=1`:

```bash
python -m playwright install chromium
set APP_OFFLINE=1
python -m pytest tests/test_visual_regression.py -v
```

Refresh committed baselines only for an intentional visual change, and only
from the same Linux Chromium CI uses (`ubuntu-latest`, Playwright 1.62):

```bash
python -m pytest tests/test_visual_regression.py --update-visual
```

From this Windows checkout, WSL with the Linux venv:

```bash
wsl -e bash tests/visual/_gen_linux_baselines.sh
```

Windows font rasterization will not match those PNGs. Local Windows runs still
fail on missing copy, leaked copy, and horizontal overflow.

## Frozen inputs

The Streamlit server started by `tests/conftest.py` sets:

- `APP_OFFLINE=1`
- `APP_TODAY=2026-08-24` so the preseason banner does not flip on 2026-09-09
- `JSA_VISUAL_LH_FIXTURE` to `tests/visual/fixtures/league_history.json`

Production never sets the fixture env. The payload applies only when the typed
league ID matches `fixture_league_id` (`1255197436951932928`).

## Files

| Path | Role |
|---|---|
| `catalog.py` | Scene list. `cases()` expands scene x viewport |
| `actions.py` | Clicks that are not a plain GET |
| `compare.py` | Pillow+numpy. Per-channel threshold 28, max 3% pixels |
| `fixtures/league_history.json` | Two-manager 2025 league plus waivers/trades |
| `baselines/` | Committed Linux PNGs |
| `artifacts/` | Gitignored actual/diff dumps |
