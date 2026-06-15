"""Streamlit entry point for Language Ecosystem Evolution."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lang_ecosystem.streamlit_helpers import APP_CSS
from app_pages import about, community, concentration, embeddings, popularity, sampling, winners

st.set_page_config(
    page_title="Language Ecosystem Evolution",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)

pages = [
    st.Page(about.render, title="About the Data", icon=":material/info:", url_path="", default=True),
    st.Page(sampling.render, title="Sampling & Comparability", icon=":material/science:", url_path="sampling"),
    st.Page(popularity.render, title="Popularity & Dominance", icon=":material/trending_up:", url_path="popularity"),
    st.Page(winners.render, title="Winners & Decliners", icon=":material/leaderboard:", url_path="winners"),
    st.Page(community.render, title="Community Activity", icon=":material/groups:", url_path="community"),
    st.Page(concentration.render, title="Concentration & Diversity", icon=":material/pie_chart:", url_path="concentration"),
    st.Page(embeddings.render, title="Embeddings", icon=":material/hub:", url_path="embeddings"),
]

st.navigation(pages).run()
