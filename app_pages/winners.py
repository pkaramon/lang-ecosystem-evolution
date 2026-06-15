"""Winners and decliners page."""

import streamlit as st

from lang_ecosystem.streamlit_helpers import (
    get_prepared_data,
    long_run_change_summary,
    render_plotly,
)
from lang_ecosystem.visuals import leaders_decliners_figure


def render() -> None:
    prepared = get_prepared_data()

    st.header("Winners & Decliners")
    st.markdown(
        "Comparing twelve-month windows is less sensitive to a single noisy sampled day. "
        "The chart reports changes in percentage points of composite activity share, not "
        "changes in raw event counts."
    )

    render_plotly(leaders_decliners_figure(prepared.activity, count=10))

    risers, decliners = long_run_change_summary(prepared.activity)
    st.markdown(
        "**Largest gains:** "
        + ", ".join(f"{name} ({value:+.2%})" for name, value in risers.items())
    )
    st.markdown(
        "**Largest declines:** "
        + ", ".join(f"{name} ({value:+.2%})" for name, value in decliners.items())
    )
