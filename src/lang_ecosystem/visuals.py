"""Plotly figures with a shared light editorial visual language."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import COMMUNITY_METRICS, EVENT_METRICS, METRICS, monthly_concentration

BACKGROUND = "#F7F4EE"
PANEL = "#FFFCF7"
TEXT = "#24221F"
MUTED = "#756F67"
GRID = "#DED8CF"
NEUTRAL = "#B8B2A9"
POSITIVE = "#16856B"
NEGATIVE = "#C44E52"

FIXED_LANGUAGE_COLORS = {
    "JavaScript": "#E0A800",
    "Python": "#3776AB",
    "TypeScript": "#3178C6",
    "Java": "#D95F45",
    "Go": "#00A7C7",
    "HTML": "#E34F26",
    "C++": "#00599C",
    "C#": "#68217A",
    "PHP": "#777BB4",
    "Ruby": "#CC342D",
    "C": "#64748B",
    "Shell": "#4EAA25",
    "Rust": "#B75C14",
    "CSS": "#1572B6",
    "Jupyter Notebook": "#F37626",
    "Kotlin": "#7F52FF",
    "Vue": "#2F9E73",
    "Swift": "#F05138",
    "Scala": "#DC322F",
    "Dart": "#0175C2",
    "PowerShell": "#3973B9",
    "Lua": "#2C2D72",
    "Objective-C": "#438EFF",
    "R": "#276DC3",
    "HCL": "#844FBA",
    "Dockerfile": "#2496ED",
    "Makefile": "#64706C",
    "SCSS": "#CC6699",
    "Julia": "#9558B2",
    "Haskell": "#5D4F85",
}

FALLBACK_COLORS = [
    "#2A9D8F",
    "#E76F51",
    "#264653",
    "#F4A261",
    "#6D597A",
    "#355070",
    "#B56576",
    "#588157",
    "#BC6C25",
    "#457B9D",
    "#9C6644",
    "#7B2CBF",
]


def language_color_map(
    ranked_languages: Sequence[str], highlight_count: int = 30
) -> dict[str, str]:
    """Return stable colors for leaders and a shared neutral for the long tail."""
    used = set(FIXED_LANGUAGE_COLORS.values())
    fallback = (color for color in FALLBACK_COLORS if color not in used)
    colors = {}
    for index, language in enumerate(ranked_languages):
        if index >= highlight_count:
            colors[language] = NEUTRAL
        elif language in FIXED_LANGUAGE_COLORS:
            colors[language] = FIXED_LANGUAGE_COLORS[language]
        else:
            colors[language] = next(fallback, "#8D6E63")
    return colors


def _title_html(title: str, subtitle: str | None = None) -> str:
    if not subtitle:
        return f"<b>{title}</b>"
    return (
        f"<b>{title}</b><br>"
        f"<span style='font-size:13px;color:{MUTED}'>{subtitle}</span>"
    )


def apply_editorial_theme(
    figure: go.Figure,
    title: str,
    subtitle: str | None = None,
    height: int = 580,
    legend_title: str | None = None,
) -> go.Figure:
    """Apply the shared notebook/Streamlit-ready visual theme."""
    figure.update_layout(
        title={"text": _title_html(title, subtitle), "x": 0.015, "xanchor": "left"},
        height=height,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        font={"family": "Inter, Arial, sans-serif", "color": TEXT, "size": 12},
        margin={"l": 65, "r": 35, "t": 105, "b": 55},
        hoverlabel={"bgcolor": PANEL, "font_color": TEXT},
        legend={
            "title": {"text": legend_title or ""},
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": GRID,
            "borderwidth": 1,
        },
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    figure.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return figure


def sampling_bias_figure(
    data: pd.DataFrame,
    leaders: Sequence[str],
    colors: Mapping[str, str],
) -> go.Figure:
    """Contrast weekday-sensitive raw totals with proportional trends."""
    totals = (
        data.groupby("date", as_index=False)[EVENT_METRICS]
        .sum()
        .assign(raw_events=lambda frame: frame[EVENT_METRICS].sum(axis=1))
    )
    totals["sample_date"] = totals["date"] + pd.Timedelta(days=14)
    totals["day_type"] = np.where(
        totals["sample_date"].dt.dayofweek >= 5, "Weekend", "Weekday"
    )
    marker_colors = totals["day_type"].map(
        {"Weekend": "#D07A59", "Weekday": "#7B8794"}
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Raw event count on the sampled day",
            "Three-month composite activity share",
        ),
    )
    figure.add_trace(
        go.Bar(
            x=totals["date"],
            y=totals["raw_events"],
            marker_color=marker_colors,
            customdata=np.column_stack(
                [totals["sample_date"].dt.strftime("%A, %d %B %Y"), totals["day_type"]]
            ),
            hovertemplate=(
                "%{customdata[0]}<br>%{customdata[1]} sample"
                "<br>%{y:,.0f} events<extra></extra>"
            ),
            name="Sampled events",
        ),
        row=1,
        col=1,
    )
    for language in leaders:
        subset = data.loc[data["language"] == language]
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["composite_share_smooth"],
                mode="lines",
                line={"color": colors[language], "width": 2.2},
                name=language,
                hovertemplate=f"<b>{language}</b><br>%{{x|%b %Y}}"
                "<br>%{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="Events", tickformat="~s", row=1, col=1)
    figure.update_yaxes(title_text="Monthly share", tickformat=".1%", row=2, col=1)
    return apply_editorial_theme(
        figure,
        "Why proportions, not raw monthly counts?",
        "The 15th falls on weekends in some months; within-month shares are the comparable signal.",
        height=720,
    )


def composite_trend_figure(
    data: pd.DataFrame,
    languages: Sequence[str],
    colors: Mapping[str, str],
) -> go.Figure:
    """Interactive composite popularity trends with a smoothing toggle."""
    figure = go.Figure()
    for language in languages:
        subset = data.loc[data["language"] == language]
        for column, visible, suffix in [
            ("composite_share_smooth", True, "3-month mean"),
            ("composite_share", False, "monthly"),
        ]:
            figure.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset[column],
                    mode="lines",
                    visible=visible,
                    name=language,
                    legendgroup=language,
                    showlegend=visible,
                    line={
                        "color": colors[language],
                        "width": 2.7 if visible else 1.7,
                    },
                    customdata=np.column_stack(
                        [
                            subset["rank"] if "rank" in subset else np.zeros(len(subset)),
                            subset["category"],
                        ]
                    ),
                    hovertemplate=(
                        f"<b>{language}</b> ({suffix})<br>%{{x|%B %Y}}"
                        "<br>Composite share: %{y:.2%}"
                        "<br>%{customdata[1]}<extra></extra>"
                    ),
                )
            )

    count = len(languages)
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.13,
                "buttons": [
                    {
                        "label": "3-month mean",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
                                    index % 2 == 0 for index in range(count * 2)
                                ],
                                "showlegend": [
                                    index % 2 == 0 for index in range(count * 2)
                                ],
                            }
                        ],
                    },
                    {
                        "label": "Monthly",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
                                    index % 2 == 1 for index in range(count * 2)
                                ],
                                "showlegend": [
                                    index % 2 == 1 for index in range(count * 2)
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
    )
    figure.update_yaxes(title="Equal-weight activity share", tickformat=".1%")
    figure.update_xaxes(title=None)
    return apply_editorial_theme(
        figure,
        "Language popularity is not a single event count",
        "Composite share averages nine within-month activity proportions.",
        height=620,
        legend_title="Language",
    )


def metric_trend_figure(
    data: pd.DataFrame,
    languages: Sequence[str],
    colors: Mapping[str, str],
) -> go.Figure:
    """Per-metric trend explorer controlled by a Plotly dropdown."""
    figure = go.Figure()
    for metric_index, metric in enumerate(METRICS):
        column = f"{metric}_share_smooth"
        for language in languages:
            subset = data.loc[data["language"] == language]
            figure.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset[column],
                    mode="lines",
                    visible=metric_index == 0,
                    name=language,
                    legendgroup=language,
                    showlegend=metric_index == 0,
                    line={"color": colors[language], "width": 2.2},
                    hovertemplate=(
                        f"<b>{language}</b><br>%{{x|%B %Y}}"
                        f"<br>{metric.replace('_', ' ').title()} share: %{{y:.2%}}"
                        "<extra></extra>"
                    ),
                )
            )

    traces_per_metric = len(languages)
    buttons = []
    for metric_index, metric in enumerate(METRICS):
        visible = [False] * (len(METRICS) * traces_per_metric)
        start = metric_index * traces_per_metric
        visible[start : start + traces_per_metric] = [True] * traces_per_metric
        buttons.append(
            {
                "label": metric.replace("_", " ").title(),
                "method": "update",
                "args": [
                    {"visible": visible, "showlegend": visible},
                    {"yaxis": {"title": "Monthly share", "tickformat": ".1%"}},
                ],
            }
        )
    figure.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "y": 1.14,
                "showactive": True,
            }
        ]
    )
    figure.update_yaxes(title="Monthly share", tickformat=".1%")
    return apply_editorial_theme(
        figure,
        "Explore each activity signal",
        "Three-month means; choose a metric from the dropdown.",
        height=610,
        legend_title="Language",
    )


def stacked_area_figure(
    data: pd.DataFrame,
    leaders: Sequence[str],
    colors: Mapping[str, str],
) -> go.Figure:
    """Stacked composite shares for leaders and the aggregated remainder."""
    order = [*leaders, "Other"]
    area = data[["date", "language", "composite_share_smooth"]].copy()
    area["display_language"] = np.where(
        area["language"].isin(leaders), area["language"], "Other"
    )
    area = (
        area.groupby(["date", "display_language"], as_index=False)[
            "composite_share_smooth"
        ]
        .sum()
        .sort_values("date")
    )
    figure = go.Figure()
    for language in order:
        subset = area.loc[area["display_language"] == language]
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["composite_share_smooth"],
                stackgroup="one",
                mode="lines",
                name=language,
                line={
                    "width": 0.7,
                    "color": colors.get(language, "#D7D1C8"),
                },
                hovertemplate=(
                    f"<b>{language}</b><br>%{{x|%B %Y}}"
                    "<br>%{y:.2%}<extra></extra>"
                ),
            )
        )
    figure.update_yaxes(title="Composite share", tickformat=".0%", range=[0, 1])
    return apply_editorial_theme(
        figure,
        "How the ecosystem's center of gravity changed",
        "Top languages by full-period popularity; all remaining labels are grouped as Other.",
        height=630,
        legend_title="Language",
    )


def annual_bump_figure(
    data: pd.DataFrame,
    languages: Sequence[str],
    colors: Mapping[str, str],
) -> go.Figure:
    """Annual rank trajectories for the leading languages."""
    annual = (
        data.groupby(["language", "year"], as_index=False)["composite_share"]
        .mean()
        .assign(
            annual_rank=lambda frame: frame.groupby("year")[
                "composite_share"
            ].rank(method="min", ascending=False)
        )
    )
    selected_max_rank = int(
        annual.loc[annual["language"].isin(languages), "annual_rank"].max()
    )
    figure = go.Figure()
    for language in languages:
        subset = annual.loc[annual["language"] == language]
        figure.add_trace(
            go.Scatter(
                x=subset["year"],
                y=subset["annual_rank"],
                mode="lines+markers",
                name=language,
                line={"color": colors[language], "width": 2.4},
                marker={"size": 7},
                hovertemplate=(
                    f"<b>{language}</b><br>%{{x}} rank: %{{y:.0f}}"
                    "<br>Share: %{customdata:.2%}<extra></extra>"
                ),
                customdata=subset["composite_share"],
            )
        )
    figure.update_yaxes(
        title="Annual rank",
        autorange="reversed",
        dtick=1 if selected_max_rank <= 16 else 2,
        range=[selected_max_rank + 0.7, 0.5],
    )
    figure.update_xaxes(title=None, dtick=1)
    return apply_editorial_theme(
        figure,
        "Annual ranking trajectories",
        "Rank 1 is the highest average composite share in that calendar year.",
        height=650,
        legend_title="Language",
    )


def share_rank_heatmap(
    data: pd.DataFrame,
    languages: Sequence[str],
) -> go.Figure:
    """Monthly heatmap with a share/rank dropdown."""
    subset = data.loc[data["language"].isin(languages)].copy()
    subset["monthly_rank"] = data.groupby("date")["composite_share"].rank(
        method="min", ascending=False
    )
    share = (
        subset.pivot(index="language", columns="date", values="composite_share")
        .reindex(languages)
        * 100
    )
    rank = subset.pivot(
        index="language", columns="date", values="monthly_rank"
    ).reindex(languages)

    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=share,
            x=share.columns,
            y=share.index,
            colorscale="YlOrRd",
            colorbar={"title": "Share (%)"},
            hovertemplate="<b>%{y}</b><br>%{x|%b %Y}<br>%{z:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=rank,
            x=rank.columns,
            y=rank.index,
            colorscale="Viridis_r",
            colorbar={"title": "Rank"},
            visible=False,
            hovertemplate="<b>%{y}</b><br>%{x|%b %Y}<br>Rank %{z:.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Share",
                        "method": "update",
                        "args": [{"visible": [True, False]}],
                    },
                    {
                        "label": "Rank",
                        "method": "update",
                        "args": [{"visible": [False, True]}],
                    },
                ],
            }
        ]
    )
    return apply_editorial_theme(
        figure,
        "A decade of monthly share and rank",
        "Use the toggle to separate absolute prominence from position in the ecosystem.",
        height=820,
    )


def leaders_decliners_figure(data: pd.DataFrame, count: int = 10) -> go.Figure:
    """Largest changes between the first and final twelve-month windows."""
    dates = np.sort(data["date"].unique())
    first_dates = dates[:12]
    last_dates = dates[-12:]
    first = (
        data.loc[data["date"].isin(first_dates)]
        .groupby("language")["composite_share"]
        .mean()
    )
    last = (
        data.loc[data["date"].isin(last_dates)]
        .groupby("language")["composite_share"]
        .mean()
    )
    change = ((last - first) * 100).dropna().sort_values()
    movers = pd.concat([change.head(count), change.tail(count)]).sort_values()
    colors = np.where(movers >= 0, POSITIVE, NEGATIVE)
    figure = go.Figure(
        go.Bar(
            x=movers.values,
            y=movers.index,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>%{x:+.2f} percentage points<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_color=MUTED, line_width=1)
    figure.update_xaxes(title="Change in average composite share (percentage points)")
    return apply_editorial_theme(
        figure,
        "The largest long-run gains and declines",
        "Final 12 months compared with the first 12 months in the dataset.",
        height=690,
    )


def animated_activity_bubble(
    data: pd.DataFrame,
    languages: Sequence[str],
    colors: Mapping[str, str],
) -> go.Figure:
    """Animate contribution versus reach/community activity."""
    subset = data.loc[data["language"].isin(languages)].copy()
    subset["period"] = subset["date"].dt.strftime("%Y-%m")
    epsilon = 1e-6
    subset["contribution_plot"] = subset["contribution_share"].clip(lower=epsilon)
    subset["community_plot"] = subset["community_share"].clip(lower=epsilon)
    figure = px.scatter(
        subset,
        x="contribution_plot",
        y="community_plot",
        animation_frame="period",
        animation_group="language",
        size="composite_share",
        size_max=54,
        color="language",
        color_discrete_map=dict(colors),
        hover_name="language",
        hover_data={
            "contribution_plot": ":.2%",
            "community_plot": ":.2%",
            "composite_share": ":.2%",
            "category": True,
            "period": False,
        },
        log_x=True,
        log_y=True,
        range_x=[
            subset["contribution_plot"].min() * 0.7,
            subset["contribution_plot"].max() * 1.4,
        ],
        range_y=[
            subset["community_plot"].min() * 0.7,
            subset["community_plot"].max() * 1.4,
        ],
        labels={
            "contribution_plot": "Contribution activity share",
            "community_plot": "Reach / community share",
        },
    )
    figure.update_traces(marker={"line": {"color": PANEL, "width": 0.8}})
    return apply_editorial_theme(
        figure,
        "Community activity changes month by month",
        "Bubble size is composite share; both axes use logarithmic scales.",
        height=680,
        legend_title="Language",
    )


def community_profile_heatmap(
    data: pd.DataFrame,
    languages: Sequence[str],
) -> go.Figure:
    """Average monthly community-share profiles for leading languages."""
    columns = [f"{metric}_share" for metric in COMMUNITY_METRICS]
    profile = (
        data.loc[data["language"].isin(languages)]
        .groupby("language")[columns]
        .mean()
        .reindex(languages)
        * 100
    )
    profile.columns = [metric.replace("_", " ").title() for metric in COMMUNITY_METRICS]
    figure = go.Figure(
        go.Heatmap(
            z=profile,
            x=profile.columns,
            y=profile.index,
            colorscale="Tealgrn",
            colorbar={"title": "Average share (%)"},
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
        )
    )
    return apply_editorial_theme(
        figure,
        "Different communities leave different activity signatures",
        "Average within-month share for reach and participation metrics.",
        height=690,
    )


def concentration_figure(data: pd.DataFrame) -> go.Figure:
    """Top-k share and effective-number trends."""
    concentration = monthly_concentration(data)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Share held by leaders", "Effective number of languages"),
    )
    for column, label, color in [
        ("top_1_share", "Top 1", "#D95F45"),
        ("top_5_share", "Top 5", "#D89B35"),
        ("top_10_share", "Top 10", "#2A9D8F"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=concentration["date"],
                y=concentration[column],
                mode="lines",
                name=label,
                line={"color": color, "width": 2.5},
                hovertemplate=f"{label}<br>%{{x|%b %Y}}: %{{y:.1%}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=concentration["date"],
            y=concentration["effective_languages"],
            mode="lines",
            name="Effective languages",
            line={"color": "#5D4F85", "width": 2.8},
            hovertemplate="%{x|%b %Y}<br>%{y:.1f} effective languages<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title="Composite share", tickformat=".0%", row=1, col=1)
    figure.update_yaxes(title="1 / HHI", row=2, col=1)
    return apply_editorial_theme(
        figure,
        "Is GitHub language activity becoming more concentrated?",
        "The effective count is the inverse Herfindahl-Hirschman index.",
        height=720,
    )


def embedding_comparison_figure(
    embeddings: pd.DataFrame,
    colors: Mapping[str, str],
    profile: str = "All activity",
    label_count: int = 15,
) -> go.Figure:
    """Three-panel comparison of UMAP, TriMAP, and PaCMAP."""
    methods = ["UMAP", "TriMAP", "PaCMAP"]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=methods,
        horizontal_spacing=0.055,
    )
    for column, method in enumerate(methods, start=1):
        subset = embeddings.loc[
            (embeddings["profile"] == profile) & (embeddings["method"] == method)
        ].sort_values("rank")
        marker_colors = [colors.get(language, NEUTRAL) for language in subset["language"]]
        figure.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                marker={
                    "color": marker_colors,
                    "size": np.where(subset["rank"] <= 30, 10, 6),
                    "opacity": np.where(subset["rank"] <= 30, 0.9, 0.42),
                    "line": {"color": PANEL, "width": 0.7},
                },
                customdata=np.column_stack(
                    [
                        subset["language"],
                        subset["rank"],
                        subset["category"],
                        subset["mean_composite_share"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Overall rank: %{customdata[1]:.0f}"
                    "<br>%{customdata[2]}"
                    "<br>Mean composite share: %{customdata[3]:.2%}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        labels = subset.nsmallest(label_count, "rank")
        figure.add_trace(
            go.Scatter(
                x=labels["x"],
                y=labels["y"],
                mode="text",
                text=labels["language"],
                textposition="top center",
                textfont={"size": 10, "color": TEXT},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(visible=False, row=1, col=column)
        figure.update_yaxes(visible=False, row=1, col=column)
    return apply_editorial_theme(
        figure,
        f"{profile}: three views of activity-vector similarity",
        "Each point is one language represented by its complete normalized time series.",
        height=610,
    )


def profile_embedding_figure(
    embeddings: pd.DataFrame,
    colors: Mapping[str, str],
) -> go.Figure:
    """Use a dropdown to compare profile-specific embeddings across reducers."""
    profiles = list(embeddings["profile"].drop_duplicates())
    methods = ["UMAP", "TriMAP", "PaCMAP"]
    figure = make_subplots(rows=1, cols=3, subplot_titles=methods)
    trace_profiles = []
    for profile_index, profile in enumerate(profiles):
        for column, method in enumerate(methods, start=1):
            subset = embeddings.loc[
                (embeddings["profile"] == profile)
                & (embeddings["method"] == method)
            ].sort_values("rank")
            figure.add_trace(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="markers",
                    visible=profile_index == 0,
                    marker={
                        "color": [
                            colors.get(language, NEUTRAL)
                            for language in subset["language"]
                        ],
                        "size": np.where(subset["rank"] <= 30, 9, 5),
                        "opacity": np.where(subset["rank"] <= 30, 0.9, 0.38),
                        "line": {"color": PANEL, "width": 0.6},
                    },
                    customdata=np.column_stack(
                        [subset["language"], subset["rank"], subset["category"]]
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Rank %{customdata[1]:.0f}"
                        "<br>%{customdata[2]}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            trace_profiles.append(profile)
            figure.update_xaxes(visible=False, row=1, col=column)
            figure.update_yaxes(visible=False, row=1, col=column)
    buttons = []
    for profile in profiles:
        buttons.append(
            {
                "label": profile,
                "method": "update",
                "args": [
                    {"visible": [item == profile for item in trace_profiles]},
                    {
                        "title": {
                            "text": _title_html(
                                f"{profile}: profile-specific similarity",
                                "The same 100 languages, with only the selected metric family in each vector.",
                            )
                        }
                    },
                ],
            }
        )
    figure.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "y": 1.12,
            }
        ]
    )
    return apply_editorial_theme(
        figure,
        f"{profiles[0]}: profile-specific similarity",
        "The same 100 languages, with only the selected metric family in each vector.",
        height=590,
    )


def embedding_quality_figure(scores: pd.DataFrame) -> go.Figure:
    """Compact heatmaps for local and global embedding diagnostics."""
    score_columns = [
        ("trustworthiness", "Trustworthiness"),
        ("knn_preservation", "k-NN preservation"),
        ("distance_spearman", "Distance correlation"),
    ]
    profiles = list(scores["profile"].drop_duplicates())
    methods = ["UMAP", "TriMAP", "PaCMAP"]
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[label for _, label in score_columns],
        horizontal_spacing=0.12,
    )
    for column, (score_column, _) in enumerate(score_columns, start=1):
        matrix = (
            scores.pivot(index="method", columns="profile", values=score_column)
            .reindex(index=methods, columns=profiles)
        )
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=matrix.columns,
                y=matrix.index,
                zmin=0,
                zmax=1,
                colorscale="Tealgrn",
                showscale=column == 3,
                colorbar={"title": "Score"} if column == 3 else None,
                text=np.round(matrix.to_numpy(), 3),
                texttemplate="%{text:.3f}",
                hovertemplate="%{y}<br>%{x}<br>%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    return apply_editorial_theme(
        figure,
        "No single projection preserves every kind of structure",
        "Higher is better; local-neighborhood and global-distance scores answer different questions.",
        height=510,
    )
