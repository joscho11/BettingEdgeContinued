Baselines are Linux Chromium PNGs from the `responsive` CI job.

Do not commit Windows `--update-visual` output. Fonts will not match
`ubuntu-latest`. Copy and overflow checks still run locally.

Refresh:

```bash
pytest tests/test_visual_regression.py --update-visual
```
