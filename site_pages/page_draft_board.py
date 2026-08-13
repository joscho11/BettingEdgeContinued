"""Draft Board page for the multipage site (site revamp Batch 2).

The flagship page: site orientation + page-purpose, the pre-season banner, then the board
itself rendered via draft_board_2026.render(). The board is the exact 180-player independent
v6 publication universe, with live Sleeper market data and frozen Model Proj (75% v6, 25%
Sleeper published projection). All board copy/logic lives in draft_board_2026; this module
adds only the flagship strings.
"""
import os
from datetime import date

import streamlit as st

import dashboard_chrome as chrome
import draft_board_2026 as board
from seasonal_config import board_refresh_season_start

# Ratified 4d copy (verbatim).
ORIENTATION = ("I build machine-learning models for NFL betting and fantasy, run them "
               "live, and show my work — the numbers, the honest track record, and the "
               "code on my GitHub.")
PURPOSE = ("My pre-season draft board: the independent model's exact 180-player 2026 "
           "projection universe. Model Proj is 75% that model and 25% Sleeper's published "
           "projection. Sleeper ADP, Sleeper projections, and both rank gaps refresh "
           "daily; Model Proj points and ranks stay frozen until the planned "
           "early-September snapshot.")


def render():
    # Lead with the title, then the byline + purpose, then (pre-season) the notice.
    st.title("📋 2026 Draft Board")
    st.caption(ORIENTATION)
    st.markdown(f"**{PURPOSE}**")
    _ss = board_refresh_season_start()
    if date.today() < _ss:
        # No page_link here — this IS the Draft Board page, so the link would be circular.
        chrome.render_preseason_banner(None, _ss.year)
    board.render()
