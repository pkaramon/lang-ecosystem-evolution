"""Embeddings page."""

import streamlit as st

from lang_ecosystem.streamlit_helpers import (
    embedding_score_winners,
    get_embeddings,
    get_prepared_data,
    render_plotly,
)
from lang_ecosystem.visuals import embedding_quality_figure, projection_method_figure


def render() -> None:
    st.header("Embeddings")
    st.markdown(
        "Each point represents **one language across the whole 114-month history**. "
        "Vectors concatenate chronological monthly-share time series per metric. "
        "Coordinates and axis directions have no intrinsic meaning — interpret "
        "neighborhoods and relative structure, not absolute positions."
    )

    with st.spinner("Computing embeddings (UMAP, TriMAP, PaCMAP)…"):
        data = get_embeddings()

    prepared = get_prepared_data()

    st.subheader("Activity vectors")
    st.dataframe(data.vector_summary, use_container_width=True)

    st.subheader("Embedding quality scores")
    formatted_scores = data.embedding_scores.copy()
    for column in ("trustworthiness", "knn_preservation", "distance_spearman"):
        formatted_scores[column] = formatted_scores[column].map("{:.3f}".format)
    st.dataframe(formatted_scores, use_container_width=True)

    for method in ("UMAP", "TriMAP", "PaCMAP"):
        st.subheader(method)
        render_plotly(
            projection_method_figure(data.embeddings, prepared.colors, method)
        )

    render_plotly(embedding_quality_figure(data.embedding_scores))

    st.markdown("**Best method by diagnostic (this dataset and parameterization):**")
    st.markdown("\n".join(embedding_score_winners(data.embedding_scores)))
