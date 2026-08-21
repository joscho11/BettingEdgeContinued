# AGENTS.md

Guidance for coding agents in this repository.

## What this repo is

JoSchoAnalytics is the public Streamlit website plus immutable release artifacts.
Training, notebooks, serialized models, and DFS live in private producer repos.
Do not add them back.

**Live site:** https://joschoanalytics.streamlit.app

Private producers:

- `spread_v3_prod` writes weekly spread candidates
- `weekly_projections_v2_prod` writes weekly fantasy candidates
- Seasonal totals come from `seasonal_totals_v2_prod` into `futures/published/`

This repo validates candidates in `publishing/`, stores them under `data/releases/`,
and renders pages from those files plus frozen demo CSVs.

## Commands

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
pip install -r requirements-test.txt
set APP_OFFLINE=1
python -m pytest tests/test_public_boundary.py tests/test_site_nav.py tests/test_publishing_pipeline.py -q
python -m publishing.cli status
```

## Layout

| Path | Role |
|---|---|
| `app.py` | Streamlit entrypoint |
| `site_pages/` | One module per page |
| `publishing/` | Candidate validation, immutable releases, grading |
| `data/releases/` | Published builds and the active manifest |
| `betting/live_2026.py` | 2026 HIGH display rules |
| `betting/calibration.py` | Cover rates from the graded tracker |
| `draft_board_2026.py` | Draft Board, CSV-only |
| `tests/test_public_boundary.py` | Fails if modeling files or ML imports return |

## Rules

- Site pages read CSV/JSON artifacts. They do not load `.pkl` files.
- Do not restore `archive/`, notebooks, training scripts, or DFS.
- Keep HIGH at a 3-point Tuesday leftover. Do not claim CLV. Do not cite the retracted 64.2% as current.
- Retracted in-repo HIGH was 129/238 (54.20%). Live 2026 HIGH is 192/336.
- Draft Board copy must not name the 75/25 Sleeper mix.
- After any page change, run AppTest on the affected pages with `APP_OFFLINE=1`.
