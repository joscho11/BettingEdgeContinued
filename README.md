# JoScho Analytics

JoScho Analytics is a source-available Streamlit site for NFL betting analysis, fantasy football projections, and league-history research. I publish the assumptions, results, and limits beside the products. This repository is the website plus checked release artifacts. Training code and serialized models live in private producer repositories.

**Live site:** [joschoanalytics.streamlit.app](https://joschoanalytics.streamlit.app)

## Website coverage

| Page | What it covers |
|---|---|
| **Home** | Site status, release timing, and links to the main products. |
| **Draft Board** | A 180-player 2026 board with independent season projections, Sleeper or ESPN ADP, positional ranks, rank gaps, and talent context. |
| **Rookie Board** | Rookie hit probabilities, season projections, college production, and athletic context. |
| **Weekly Fantasy** | Half-PPR and stat projections for QB, RB, WR, and TE. Actual results appear after games finish. |
| **Weekly Predictions** | NFL spread predictions, market comparisons, model confidence, and release status. |
| **Track Record** | Graded against-the-spread results, confidence tiers, model comparisons, streaks, and betting simulations. |
| **Season Totals** | Win projections for all 32 teams compared with posted season totals. |
| **League History** | Sleeper and ESPN league records, rivalries, manager profiles, draft habits, roster production, and league-wide trends. |
| **Film Room** | Short video breakdowns with written analysis and links to the related product. |
| **Help & Guide** | Betting definitions, model summaries, feature explanations, confidence rules, and product limitations. |

## Models and publication

The site covers four published product families:

- **NFL spreads:** an independent margin estimate compared with the sportsbook line. The site separates the live 2026 record from the 2025 demo.
- **Game and season totals:** a 2025 experimental game-total demo plus a separate 32-team season-win product.
- **Weekly fantasy:** player-level half-PPR and stat projections with postgame grading.
- **Draft and rookie analysis:** season projections, market ranks, rookie hit probabilities, and descriptive NFL and college talent scores.

Live spread and weekly-fantasy producers submit candidate files to this repository. The code in `publishing/` validates schemas, coverage, timestamps, hashes, and model versions before it creates an immutable release. The site reads those releases. The grading workflow writes results without changing the original prediction.

## Codebase map

| Path | Contents |
|---|---|
| `app.py` | Streamlit entrypoint and navigation. |
| `site_pages/` | One module per visible page. |
| `dashboard_*.py`, `mobile.py`, `theme_redesign.py` | Shared data loading, UI components, responsive styles, and site chrome. |
| `publishing/` | Candidate validation, immutable releases, manifests, grading, rollback, and CLI commands. |
| `data/releases/` | Published build artifacts and the active release manifest. |
| `betting/` | Display rules, graded trackers, and Help-page calibration. |
| `fantasy/` | Published weekly, draft, rookie, and talent CSVs. |
| `futures/` | Published season-win projections and their evidence file. |
| `film_room.py`, `video_content.py`, `video_breakdowns/` | Video registry and written breakdowns. |
| `tests/` | Unit, contract, offline page-render, responsive, publication, and public-boundary tests. |
| `.github/workflows/` | CI, release grading, and board market refreshes. |

Read [AGENTS.md](AGENTS.md) before changing production data paths. It documents the active sources, frozen files, and test contracts.

## Evidence and limits

Sportsbooks charge enough margin that a bettor needs about a 52.4% win rate at standard `-110` odds to break even. Backtests can overstate performance, so the site labels backtested, demo, experimental, and live results as separate evidence.

I retracted the old 64.2% spread-model claim after finding a pregame feature leak. I measured the corrected historical HIGH tier at 129/238, or 54.2017%, with a 47.8551% Wilson lower bound. That result does not demonstrate an edge over break-even. The live 2026 claim is the Tuesday HIGH book: 192/336 = 57.14% ATS, Wilson lower bound 52.66%. The Track Record page shows the graded evidence used for current decisions.

The site provides research and published outputs, not paid picks or financial advice. Sports betting involves risk.

## Run locally

```bash
git clone https://github.com/joscho11/JoSchoAnalytics.git
cd JoSchoAnalytics
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
```

Streamlit serves the app at `http://localhost:8501`.

Use the test dependency set for development:

```bash
pip install -r requirements-test.txt
python -m pytest tests/test_public_boundary.py tests/test_site_nav.py -q
```

Set `APP_OFFLINE=1` when running page tests without network access. CI uses the same offline path for dashboard coverage.

## Guides

- [Betting models](betting/GUIDE.md)
- [Weekly fantasy](fantasy/GUIDE.md)
- [Talent scores](fantasy/talent/GUIDE.md)
- [Rookie Board](fantasy/rookie/GUIDE.md)
- [Draft Board](fantasy/seasonal_projections/GUIDE.md)
- [Publication system](publishing/README.md)

## License

This repository uses the [PolyForm Noncommercial License 1.0.0](LICENSE). You may read, modify, and use the code for noncommercial work. Commercial use requires a separate license.

Copyright © 2026 Joseph Schoenbaum.
