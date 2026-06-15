"""Popularity and dominance page."""

import streamlit as st

from lang_ecosystem.streamlit_helpers import get_prepared_data, render_plotly
from lang_ecosystem.visuals import (
    category_composition_figure,
    composite_trend_figure,
    dominance_turnover_figure,
    metric_trend_figure,
    rank_stability_figure,
    ranking_explorer_figure,
    stacked_area_figure,
)


def render() -> None:
    prepared = get_prepared_data()

    st.header("Popularity & Dominance")
    st.markdown(
        "These views move from the composite score to its individual ingredients. "
        "Hover for exact values, zoom into a period, and use the controls above each chart."
    )

    render_plotly(composite_trend_figure(prepared.activity, prepared.top_12, prepared.colors))
    render_plotly(metric_trend_figure(prepared.activity, prepared.top_12, prepared.colors))
    render_plotly(stacked_area_figure(prepared.activity, prepared.top_12, prepared.colors))
    render_plotly(category_composition_figure(prepared.activity))

    st.markdown(
        "The stacked view answers a market-share question; the ranking explorer answers a "
        "position question. It follows the **union of the top 15 at the first and last period** "
        "(switch signal and scope with the dropdowns above the chart)."
    )
    render_plotly(ranking_explorer_figure(prepared.activity, prepared.colors))
    render_plotly(dominance_turnover_figure(prepared.activity))
    render_plotly(rank_stability_figure(prepared.activity, prepared.colors))
