"""Put the repo root, `site_pages/`, and `tests/` on sys.path so suites can import
app modules and the Playwright helpers regardless of the working directory.
"""
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "site_pages", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def pytest_addoption(parser):
    parser.addoption("--update-visual", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "visual: page-level screenshot regression")


@pytest.fixture(scope="session")
def update_visual(pytestconfig):
    return bool(
        pytestconfig.getoption("--update-visual")
        or os.environ.get("UPDATE_VISUAL") == "1"
    )


@pytest.fixture(scope="session")
def browser():
    from playwright_support import chromium_kwargs

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        pytest.skip(f"playwright not installed ({exc})")
    with sync_playwright() as p:
        try:
            launched = p.chromium.launch(**chromium_kwargs())
        except Exception as exc:
            pytest.skip(f"no chromium available: {str(exc).splitlines()[0][:120]}")
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture(scope="session")
def app_url(browser):
    from playwright_support import free_port, start_streamlit, stop_process, wait_for_port

    port = free_port()
    proc = start_streamlit(port)
    url = f"http://127.0.0.1:{port}"
    try:
        try:
            wait_for_port(proc, port)
        except RuntimeError as exc:
            pytest.fail(f"{exc}. The app itself is broken, not the test environment")
        yield url
    finally:
        stop_process(proc)
