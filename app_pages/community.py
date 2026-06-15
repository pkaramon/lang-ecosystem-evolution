"""Community activity page."""

import streamlit as st

from lang_ecosystem.streamlit_helpers import get_prepared_data, render_plotly
from lang_ecosystem.visuals import (
    activity_specialization_figure,
    animated_activity_bubble,
    ecosystem_momentum_figure,
)


def render() -> None:
    prepared = get_prepared_data()

    st.header("Community Activity")
    st.markdown(
        "Contribution activity and reach/community activity are related but not identical. "
        "Stars and forks can indicate attention and adoption; contributors and active "
        "repositories indicate participation breadth."
    )

    render_plotly(
        animated_activity_bubble(prepared.activity, prepared.top_40, prepared.colors)
    )
    render_plotly(activity_specialization_figure(prepared.activity))
    render_plotly(ecosystem_momentum_figure(prepared.activity, prepared.colors))
