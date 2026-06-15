"""About the Data landing page."""

import streamlit as st

from lang_ecosystem.streamlit_helpers import (
    get_prepared_data,
    render_dataset_markdown,
)


def render() -> None:
    prepared = get_prepared_data()

    st.title("Language Ecosystem Evolution")
    st.caption("Interactive analysis of monthly programming-language activity, January 2016 to June 2025")

    st.info(
        "This dataset samples only the **15th day of each month**. "
        "Compare **shares, percentages, and trends** — not raw counts across months."
    )

    st.divider()
    render_dataset_markdown()

    with st.expander("Preview raw data (5 rows)"):
        st.dataframe(prepared.raw.head(), use_container_width=True)
