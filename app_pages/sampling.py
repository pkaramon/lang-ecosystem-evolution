"""Sampling and comparability page."""

import streamlit as st

from lang_ecosystem.streamlit_helpers import get_prepared_data, render_plotly
from lang_ecosystem.visuals import sampling_bias_figure


def render() -> None:
    prepared = get_prepared_data()

    st.header("Sampling & Comparability")
    st.markdown(
        "The top panel sums event rows observed on each sampled day. Weekend samples "
        "are visibly lower. The bottom panel uses within-month proportions instead, "
        "which removes the shared calendar-day volume shock and preserves the relative "
        "ecosystem composition."
    )

    render_plotly(
        sampling_bias_figure(prepared.activity, prepared.top_12[:5], prepared.colors)
    )
