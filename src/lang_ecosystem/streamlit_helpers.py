"""Shared Streamlit helpers: cached data loading and UI utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .analysis import (
    METRICS,
    add_monthly_shares,
    add_trailing_shares,
    build_profile_vectors,
    complete_month_grid,
    evaluate_embeddings,
    load_activity_data,
    rank_languages,
    run_all_embeddings,
)
from .visuals import CHART_LIMITS, language_color_map

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "github_language_activity_monthly.csv"
DATASET_MD_PATH = ROOT / "data" / "DATASET.md"

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


@dataclass(frozen=True)
class PreparedData:
    raw: pd.DataFrame
    activity: pd.DataFrame
    ranking: pd.DataFrame
    top_12: list[str]
    top_40: list[str]
    top_150: list[str]
    colors: dict[str, str]


@dataclass(frozen=True)
class EmbeddingData:
    profile_vectors: dict[str, pd.DataFrame]
    embeddings: dict
    embedding_scores: pd.DataFrame
    vector_summary: pd.DataFrame


@st.cache_data(show_spinner=False)
def get_prepared_data() -> PreparedData:
    raw = load_activity_data(DATA_PATH)
    dense = complete_month_grid(raw)
    activity = add_monthly_shares(dense)
    activity = add_trailing_shares(activity, windows=(3, 12))
    ranking = rank_languages(activity)
    activity = activity.merge(
        ranking[["language", "rank", "mean_composite_share"]],
        on="language",
        how="left",
        validate="many_to_one",
    )

    top_150 = ranking.head(CHART_LIMITS["embeddings"])["language"].tolist()
    top_40 = top_150[: CHART_LIMITS["explorers"]]
    top_12 = top_150[: CHART_LIMITS["trajectories"]]
    colors = language_color_map(top_150)
    colors["Other"] = "#D7D1C8"

    return PreparedData(
        raw=raw,
        activity=activity,
        ranking=ranking,
        top_12=top_12,
        top_40=top_40,
        top_150=top_150,
        colors=colors,
    )


@st.cache_data(show_spinner=False)
def get_embeddings() -> EmbeddingData:
    prepared = get_prepared_data()
    profile_vectors = build_profile_vectors(prepared.activity, prepared.top_150)
    embeddings = run_all_embeddings(profile_vectors, prepared.ranking, seed=42)
    embedding_scores = evaluate_embeddings(
        profile_vectors,
        embeddings,
        n_neighbors=10,
    )
    vector_summary = vector_summary_table(profile_vectors)
    return EmbeddingData(
        profile_vectors=profile_vectors,
        embeddings=embeddings,
        embedding_scores=embedding_scores,
        vector_summary=vector_summary,
    )


def dataset_summary_metrics(activity: pd.DataFrame, top_150_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Value": [
                f"{activity['date'].min():%B %Y} - {activity['date'].max():%B %Y}",
                activity["date"].nunique(),
                activity["language"].nunique(),
                len(activity),
                len(METRICS),
                top_150_count,
            ]
        },
        index=[
            "Coverage",
            "Monthly samples",
            "Canonical labels",
            "Dense language-month rows",
            "Activity metrics",
            "Labels used in embeddings",
        ],
    )


def format_vector_columns(matrix: pd.DataFrame) -> list[str]:
    """Return human-readable labels for every feature column in a profile matrix."""
    return [
        f"{metric} / {date:%Y-%m}"
        for metric, date in matrix.columns
    ]


def vector_summary_table(profile_vectors: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize embedding input vectors with an explicit metrics list per profile."""
    return pd.DataFrame(
        {
            "Languages": [matrix.shape[0] for matrix in profile_vectors.values()],
            "Features": [matrix.shape[1] for matrix in profile_vectors.values()],
            "Metrics": [
                ", ".join(matrix.columns.get_level_values("metric").unique())
                for matrix in profile_vectors.values()
            ],
        },
        index=profile_vectors.keys(),
    )


def render_profile_vector_columns(profile_vectors: Mapping[str, pd.DataFrame]) -> None:
    with st.expander("All feature columns", expanded=False):
        st.markdown(
            "Each feature is one metric's within-month share for one calendar month. "
            "Columns are ordered metric-major: all months for the first metric, then "
            "all months for the next metric, and so on."
        )
        for profile, matrix in profile_vectors.items():
            st.markdown(f"**{profile}** — {matrix.shape[1]} features")
            st.dataframe(
                pd.DataFrame({"Feature": format_vector_columns(matrix)}),
                use_container_width=True,
            )


def render_plotly(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def render_dataset_markdown() -> None:
    st.markdown(DATASET_MD_PATH.read_text(encoding="utf-8"))


def render_metrics_glossary() -> None:
    with st.expander("How we score activity", expanded=False):
        st.markdown(
            """
**Within-month share** — For each metric (pushes, pull requests, issues,
issue comments, stars, forks, creates, contributors, active repos), a
language's count divided by the total for that metric in the same month.
Always compare shares, not raw counts — the dataset samples only the 15th
of each month.

**Composite share** — Equal-weight average of all nine metric shares. This
is the default popularity score used in most charts.

**Contribution share** — Average of five contribution signals: pushes, pull
requests, issues, issue comments, and creates.

**Community share** — Average of four reach signals: stars, forks,
contributors, and active repos.

**Trailing 3 / 12 months** — Rolling mean of monthly shares per language,
used to smooth single-day sampling noise.

**Top-k dominance** — Sum of composite shares held by the top 1, 5, or 10
languages in a month.
"""
        )


def render_diversity_metrics_explainer() -> None:
    with st.expander(
        "What do inverse HHI and exponential Shannon mean?", expanded=False
    ):
        st.markdown(
            """
**HHI (Herfindahl–Hirschman Index)** — Sum of squared shares; measures
concentration. One dominant language gives HHI near 1; many equal languages
give HHI near 1/N.

**Inverse HHI (1/HHI)** — Reframes concentration as an *effective number
of equal-sized ecosystems*. If 10 languages each held 10% share, inverse
HHI is about 10. Higher means more diverse.

**Exponential Shannon (exp(entropy))** — Shannon entropy is
−Σ p·log(p); exponentiating gives another effective-count measure with the
same equal-split intuition. It is typically slightly higher than inverse
HHI for the same distribution.

**How the chart uses them** — Each month, language shares within the
selected scope are renormalized to probabilities, both effective counts are
computed, then a trailing 3-month average is applied (matching the
chart's default smoothing).
"""
        )


def long_run_change_summary(activity: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dates = np.sort(activity["date"].unique())
    first_mean = (
        activity.loc[activity["date"].isin(dates[:12])]
        .groupby("language")["composite_share"]
        .mean()
    )
    last_mean = (
        activity.loc[activity["date"].isin(dates[-12:])]
        .groupby("language")["composite_share"]
        .mean()
    )
    long_run_change = (last_mean - first_mean).sort_values()
    risers = long_run_change.tail(5).sort_values(ascending=False)
    decliners = long_run_change.head(5)
    return risers, decliners


def embedding_score_winners(embedding_scores: pd.DataFrame) -> list[str]:
    score_winners = (
        embedding_scores.set_index(["profile", "method"])[
            ["trustworthiness", "knn_preservation", "distance_spearman"]
        ]
        .groupby(level="profile")
        .idxmax()
    )
    lines = []
    for profile in score_winners.index:
        local = score_winners.loc[profile, "trustworthiness"][1]
        neighbors = score_winners.loc[profile, "knn_preservation"][1]
        global_method = score_winners.loc[profile, "distance_spearman"][1]
        lines.append(
            f"- **{profile}:** trustworthiness `{local}`, "
            f"k-NN preservation `{neighbors}`, global distance `{global_method}`"
        )
    return lines


APP_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    div[data-testid="stPlotlyChart"] {
        background-color: #F7F4EE;
        border: 1px solid #DED8CF;
        border-radius: 0.5rem;
        padding: 0.25rem 0.5rem 0.5rem;
        margin-bottom: 0.5rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stAppDeployButton"] {visibility: hidden;}
    header[data-testid="stHeader"] {
        background-color: #E5E1D9;
    }
</style>
"""
