"""Film Room page wrapper for the shared analysis and site-walkthrough library."""
import streamlit as st

import page_common
import film_room as _film_room

# Analysis shorts and product walkthroughs share one canonical video library.
HEADER = "Short analysis and site walkthroughs, each with the context behind it."


def render():
    st.title("Film room")
    st.caption(HEADER)
    film_room = page_common.reload_if_stale(_film_room)
    film_room.render_film_room(show_header=False)
