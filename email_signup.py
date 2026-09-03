"""Owned email-list signup CTA - link-out only.

No third-party script or iframe touches the site. Paste your list's PUBLIC signup
URL into SIGNUP_URL to turn the invitation on everywhere (Home card + global
footer). Empty string keeps it hidden, so nothing half-built ever ships.

Import-safe: constants and functions only, no st.* at import time.
"""
from __future__ import annotations

import streamlit as st

# >>> Paste your email-list signup URL here (Buttondown / Substack / Mailchimp / ...).
# Leave "" to hide the signup CTA across the whole site.
SIGNUP_URL = "https://substack.com/@joschoanalytics"

_LABEL = "Email me the weekly card"
_HELP = (
    "Free. My Tuesday HIGH card and weekly fantasy numbers when they publish. "
    "No picks for sale, no spam."
)


def is_configured() -> bool:
    """True once a real signup URL is set."""
    return bool(SIGNUP_URL.strip())


def render_button(*, on_click=None, **kwargs) -> None:
    """Render the signup link-button. No-op until SIGNUP_URL is set."""
    if not is_configured():
        return
    if on_click is not None:
        kwargs.setdefault("on_click", on_click)
        kwargs.setdefault("args", ("outbound_newsletter",))
    st.link_button(_LABEL, SIGNUP_URL, icon=":material/mail:", help=_HELP, **kwargs)


def render_card() -> None:
    """A bordered Home-page invitation. No-op until SIGNUP_URL is set."""
    if not is_configured():
        return
    with st.container(border=True):
        st.markdown("**Get the weekly numbers by email**")
        st.caption(_HELP)
        render_button(type="primary")
