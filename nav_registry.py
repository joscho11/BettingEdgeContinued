"""Cross-link page registry (design 4g). The entrypoint (app.py) populates
PAGES with the st.Page objects after building them; page modules read PAGES at render
time to build st.page_link cross-links without importing the entrypoint (no circular
import). Empty until the entrypoint runs — page modules must handle a missing key.
"""
PAGES = {}

# Hidden, release-backed matchup pages keyed by game_id. Populated by app.py on
# every session before st.navigation runs so direct shared URLs resolve cleanly.
MATCHUPS = {}
