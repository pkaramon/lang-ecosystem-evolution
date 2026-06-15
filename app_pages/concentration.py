"""Concentration and diversity page."""

import streamlit as st

from lang_ecosystem.streamlit_helpers import get_prepared_data, render_plotly
from lang_ecosystem.visuals import concentration_figure, diversity_figure


def render() -> None:
    prepared = get_prepared_data()

    st.header("Concentration & Diversity")
    st.markdown(
        "Top-k shares show directly how much activity belongs to the leading ecosystems. "
        "Effective counts complement that view by translating concentration and entropy "
        "into the number of equally sized ecosystems that would produce the same diversity."
    )

    render_plotly(concentration_figure(prepared.activity))
    render_plotly(diversity_figure(prepared.activity))
