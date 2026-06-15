"""Plotly figures with a shared light editorial visual language."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import (
    EVENT_METRICS,
    LABEL_SCOPES,
    METRICS,
    SCORE_COLUMNS,
    activity_specialization,
    dominance_turnover,
    ecosystem_momentum,
    filter_label_scope,
    ranking_trajectories,
    top_k_dominance,
)

BACKGROUND = "#F7F4EE"
PANEL = "#FFFCF7"
TEXT = "#24221F"
MUTED = "#756F67"
GRID = "#DED8CF"
NEUTRAL = "#B8B2A9"
POSITIVE = "#16856B"
NEGATIVE = "#C44E52"

CHART_LIMITS = {
    "trajectories": 20,
    "explorers": 40,
    "embeddings": 150,
    "embedding_labels": 25,
}

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
    control_band: bool = False,
) -> go.Figure:
    """Apply the shared notebook/Streamlit-ready visual theme."""
    figure.update_layout(
        title={
            "text": _title_html(title, subtitle),
            "x": 0.015,
            "xanchor": "left",
            "xref": "container",
            "y": 0.955,
            "yanchor": "top",
            "yref": "container",
            "pad": {"t": 0},
        },
        height=height,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        font={"family": "Inter, Arial, sans-serif", "color": TEXT, "size": 12},
        margin={"l": 65, "r": 35, "t": 175 if control_band else 110, "b": 55},
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


def _menu(
    buttons: list[dict],
    x: float,
    label: str,
    active: int = 0,
) -> tuple[dict, dict]:
    menu = {
        "buttons": buttons,
        "direction": "down",
        "x": x,
        "xanchor": "left",
        "y": 1.16,
        "yanchor": "top",
        "active": active,
        "showactive": True,
        "pad": {"r": 12, "t": 4},
    }
    annotation = {
        "text": label,
        "x": x,
        "xref": "paper",
        "xanchor": "left",
        "y": 1.205,
        "yref": "paper",
        "showarrow": False,
        "font": {"size": 11, "color": MUTED},
    }
    return menu, annotation


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
    """Interactive composite popularity trends with trailing-window controls."""
    figure = go.Figure()
    subsets = {}
    for language in languages:
        subset = data.loc[data["language"] == language]
        subsets[language] = subset
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["composite_share_3m"],
                mode="lines",
                name=language,
                line={"color": colors[language], "width": 2.2},
                customdata=np.column_stack([subset["category"]]),
                hovertemplate=(
                    f"<b>{language}</b><br>%{{x|%B %Y}}"
                    "<br>Composite share: %{y:.2%}"
                    "<br>%{customdata[0]}<extra></extra>"
                ),
            )
        )

    smoothing = [
        ("Monthly", "composite_share"),
        ("Trailing 3 months", "composite_share_3m"),
        ("Trailing 12 months", "composite_share_12m"),
    ]
    buttons = []
    for label, column in smoothing:
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"y": [subsets[language][column] for language in languages]},
                    {
                        "yaxis": {
                            "title": "Equal-weight activity share",
                            "tickformat": ".1%",
                        }
                    },
                ],
            }
        )
    menu, annotation = _menu(buttons, 0.01, "Smoothing", active=1)
    figure.update_layout(
        updatemenus=[menu],
        annotations=[annotation],
    )
    figure.update_yaxes(title="Equal-weight activity share", tickformat=".1%")
    figure.update_xaxes(title=None)
    return apply_editorial_theme(
        figure,
        "Language popularity is not a single event count",
        "Composite share averages nine within-month activity proportions.",
        height=620,
        legend_title="Language",
        control_band=True,
    )


def metric_trend_figure(
    data: pd.DataFrame,
    languages: Sequence[str],
    colors: Mapping[str, str],
) -> go.Figure:
    """Per-metric trend explorer with independent metric and smoothing controls."""
    figure = go.Figure()
    trace_frames = []
    for metric_index, metric in enumerate(METRICS):
        for language in languages:
            subset = data.loc[data["language"] == language]
            trace_frames.append((metric, language, subset))
            figure.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset[f"{metric}_share_3m"],
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
    metric_buttons = []
    for metric_index, metric in enumerate(METRICS):
        visible = [False] * (len(METRICS) * traces_per_metric)
        start = metric_index * traces_per_metric
        visible[start : start + traces_per_metric] = [True] * traces_per_metric
        metric_buttons.append(
            {
                "label": metric.replace("_", " ").title(),
                "method": "update",
                "args": [
                    {"visible": visible, "showlegend": visible},
                    {"yaxis": {"title": "Monthly share", "tickformat": ".1%"}},
                ],
            }
        )
    smoothing_buttons = []
    for label, suffix in [
        ("Monthly", ""),
        ("Trailing 3 months", "_3m"),
        ("Trailing 12 months", "_12m"),
    ]:
        smoothing_buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {
                        "y": [
                            subset[f"{metric}_share{suffix}"]
                            for metric, _, subset in trace_frames
                        ]
                    }
                ],
            }
        )
    metric_menu, metric_label = _menu(metric_buttons, 0.01, "Activity signal")
    smooth_menu, smooth_label = _menu(
        smoothing_buttons, 0.29, "Smoothing", active=1
    )
    figure.update_layout(
        updatemenus=[metric_menu, smooth_menu],
        annotations=[metric_label, smooth_label],
    )
    figure.update_yaxes(title="Monthly share", tickformat=".1%")
    return apply_editorial_theme(
        figure,
        "Explore each activity signal",
        "Choose an activity signal and a trailing smoothing window.",
        height=610,
        legend_title="Language",
        control_band=True,
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


def _trajectory_payload(
    data: pd.DataFrame,
    score: str,
    granularity: str,
    scope: str,
    count: int,
) -> list[dict]:
    frame = ranking_trajectories(data, score, granularity, scope, count)
    leaders = (
        frame.groupby("language")["share"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    payload = []
    for slot in range(count):
        if slot >= len(leaders):
            payload.append(
                {"x": [], "y": [], "name": "", "customdata": np.empty((0, 2))}
            )
            continue
        language = leaders[slot]
        subset = frame.loc[frame["language"] == language]
        payload.append(
            {
                "x": subset["period"],
                "y": subset["rank"],
                "name": language,
                "customdata": np.column_stack([subset["share"], subset["category"]]),
            }
        )
    return payload


def ranking_explorer_figure(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    count: int = CHART_LIMITS["trajectories"],
) -> go.Figure:
    """Explore dynamic leaders by signal, period granularity, and label scope."""
    scores = list(SCORE_COLUMNS)
    granularities = ["Month", "Quarter", "Year"]
    scopes = list(LABEL_SCOPES)
    payloads = {
        (score, granularity, scope): _trajectory_payload(
            data, score, granularity, scope, count
        )
        for score in scores
        for granularity in granularities
        for scope in scopes
    }
    figure = go.Figure()
    trace_keys = []
    initial_score = "Composite"
    for scope in scopes:
        for granularity in granularities:
            payload = payloads[(initial_score, granularity, scope)]
            for item in payload:
                language = item["name"]
                figure.add_trace(
                    go.Scatter(
                        x=item["x"],
                        y=item["y"],
                        mode="lines+markers",
                        visible=granularity == "Year",
                        opacity=1 if scope == "All labels" else 0,
                        name=language,
                        showlegend=scope == "All labels",
                        line={
                            "color": colors.get(language, NEUTRAL),
                            "width": 2.2,
                        },
                        marker={"size": 6},
                        customdata=item["customdata"],
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>%{x|%b %Y}"
                            "<br>Rank %{y:.0f}<br>Share %{customdata[0]:.2%}"
                            "<br>%{customdata[1]}<extra></extra>"
                        ),
                    )
                )
                trace_keys.append((scope, granularity))

    metric_buttons = []
    for score in scores:
        flattened = [
            item
            for scope in scopes
            for granularity in granularities
            for item in payloads[(score, granularity, scope)]
        ]
        metric_buttons.append(
            {
                "label": score,
                "method": "update",
                "args": [
                    {
                        "x": [item["x"] for item in flattened],
                        "y": [item["y"] for item in flattened],
                        "name": [item["name"] for item in flattened],
                        "customdata": [item["customdata"] for item in flattened],
                        "line.color": [
                            colors.get(item["name"], NEUTRAL) for item in flattened
                        ],
                    }
                ],
            }
        )
    granularity_buttons = [
        {
            "label": granularity,
            "method": "update",
            "args": [
                {
                    "visible": [
                        trace_granularity == granularity
                        for _, trace_granularity in trace_keys
                    ]
                }
            ],
        }
        for granularity in granularities
    ]
    scope_buttons = [
        {
            "label": scope,
            "method": "update",
            "args": [
                {
                    "opacity": [
                        1 if trace_scope == scope else 0
                        for trace_scope, _ in trace_keys
                    ],
                    "showlegend": [
                        trace_scope == scope for trace_scope, _ in trace_keys
                    ],
                }
            ],
        }
        for scope in scopes
    ]
    metric_menu, metric_label = _menu(metric_buttons, 0.01, "Ranking signal")
    granularity_menu, granularity_label = _menu(
        granularity_buttons, 0.31, "Period", active=2
    )
    scope_menu, scope_label = _menu(scope_buttons, 0.49, "Label scope")
    figure.update_layout(
        updatemenus=[metric_menu, granularity_menu, scope_menu],
        annotations=[metric_label, granularity_label, scope_label],
    )
    figure.update_yaxes(
        title="Rank",
        autorange=False,
        range=[count + 0.5, 0.5],
        dtick=1,
        fixedrange=True,
    )
    figure.update_xaxes(title=None)
    return apply_editorial_theme(
        figure,
        "Ranking trajectories across signals and timeframes",
        "Leaders are recalculated for the selected signal and scope; rank 1 is highest.",
        height=720,
        legend_title="Dynamic leaders",
        control_band=True,
    )


def dominance_turnover_figure(
    data: pd.DataFrame,
    top_k: int = 10,
) -> go.Figure:
    """Show which labels enter and leave the leading tier."""
    scores = list(SCORE_COLUMNS)
    granularities = ["Month", "Quarter", "Year"]
    scopes = list(LABEL_SCOPES)

    def matrix(score: str, granularity: str, scope: str):
        frame = dominance_turnover(data, score, granularity, scope, top_k)
        languages = frame["language"].drop_duplicates().tolist()
        pivot = frame.pivot(index="language", columns="period", values="rank").reindex(
            languages
        )
        return {
            "x": pivot.columns,
            "y": pivot.index.astype(str),
            "z": pivot.to_numpy(),
            "customdata": pivot.to_numpy(),
        }

    matrices = {
        (score, granularity, scope): matrix(score, granularity, scope)
        for score in scores
        for granularity in granularities
        for scope in scopes
    }
    figure = go.Figure()
    trace_keys = []
    for scope in scopes:
        for granularity in granularities:
            item = matrices[("Composite", granularity, scope)]
            figure.add_trace(
                go.Heatmap(
                    x=item["x"],
                    y=item["y"],
                    z=item["z"],
                    customdata=item["customdata"],
                    visible=granularity == "Year",
                    opacity=1 if scope == "All labels" else 0,
                    zmin=1,
                    zmax=top_k,
                    colorscale="Viridis_r",
                    colorbar={"title": "Rank"},
                    hovertemplate=(
                        "<b>%{y}</b><br>%{x|%b %Y}"
                        "<br>Top-tier rank %{customdata:.0f}<extra></extra>"
                    ),
                )
            )
            trace_keys.append((scope, granularity))
    metric_buttons = []
    for score in scores:
        items = [
            matrices[(score, granularity, scope)]
            for scope in scopes
            for granularity in granularities
        ]
        metric_buttons.append(
            {
                "label": score,
                "method": "update",
                "args": [
                    {
                        "x": [item["x"] for item in items],
                        "y": [item["y"] for item in items],
                        "z": [item["z"] for item in items],
                        "customdata": [item["customdata"] for item in items],
                    }
                ],
            }
        )
    period_buttons = [
        {
            "label": granularity,
            "method": "update",
            "args": [
                {
                    "visible": [
                        item_granularity == granularity
                        for _, item_granularity in trace_keys
                    ]
                }
            ],
        }
        for granularity in granularities
    ]
    scope_buttons = [
        {
            "label": scope,
            "method": "update",
            "args": [
                {
                    "opacity": [
                        1 if item_scope == scope else 0
                        for item_scope, _ in trace_keys
                    ]
                }
            ],
        }
        for scope in scopes
    ]
    menus = [
        _menu(metric_buttons, 0.01, "Ranking signal"),
        _menu(period_buttons, 0.31, "Period", active=2),
        _menu(scope_buttons, 0.49, "Label scope"),
    ]
    figure.update_layout(
        updatemenus=[menu for menu, _ in menus],
        annotations=[label for _, label in menus],
    )
    return apply_editorial_theme(
        figure,
        "Dominance turnover: who enters and leaves the top tier?",
        f"Only top-{top_k} appearances are colored; gaps indicate time outside the leading tier.",
        height=760,
        control_band=True,
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


def ecosystem_momentum_figure(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    count: int = CHART_LIMITS["explorers"],
) -> go.Figure:
    """Compare current prominence with change from the preceding year."""
    scores = list(SCORE_COLUMNS)
    scopes = list(LABEL_SCOPES)
    frames = {
        (score, scope): ecosystem_momentum(data, score, scope, count=count)
        for score in scores
        for scope in scopes
    }
    figure = go.Figure()
    for scope in scopes:
        frame = frames[("Composite", scope)]
        figure.add_trace(
            go.Scatter(
                x=frame["current_share"],
                y=frame["change"],
                mode="markers+text",
                text=np.where(
                    frame["current_share"].rank(ascending=False) <= 12,
                    frame["language"],
                    "",
                ),
                textposition="top center",
                opacity=1 if scope == "All labels" else 0,
                marker={
                    "size": 12,
                    "color": [colors.get(name, NEUTRAL) for name in frame["language"]],
                    "line": {"color": PANEL, "width": 0.7},
                },
                customdata=np.column_stack(
                    [frame["language"], frame["previous_share"], frame["category"]]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Latest 12m: %{x:.2%}"
                    "<br>Change: %{y:+.2%}<br>Prior 12m: %{customdata[1]:.2%}"
                    "<br>%{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    score_buttons = []
    for score in scores:
        selected = [frames[(score, scope)] for scope in scopes]
        score_buttons.append(
            {
                "label": score,
                "method": "update",
                "args": [
                    {
                        "x": [frame["current_share"] for frame in selected],
                        "y": [frame["change"] for frame in selected],
                        "text": [
                            np.where(
                                frame["current_share"].rank(ascending=False) <= 12,
                                frame["language"],
                                "",
                            )
                            for frame in selected
                        ],
                        "customdata": [
                            np.column_stack(
                                [
                                    frame["language"],
                                    frame["previous_share"],
                                    frame["category"],
                                ]
                            )
                            for frame in selected
                        ],
                        "marker.color": [
                            [colors.get(name, NEUTRAL) for name in frame["language"]]
                            for frame in selected
                        ],
                    }
                ],
            }
        )
    scope_buttons = [
        {
            "label": scope,
            "method": "update",
            "args": [
                {
                    "opacity": [
                        1 if item_scope == scope else 0 for item_scope in scopes
                    ]
                }
            ],
        }
        for scope in scopes
    ]
    score_menu, score_label = _menu(score_buttons, 0.01, "Activity signal")
    scope_menu, scope_label = _menu(scope_buttons, 0.31, "Label scope")
    figure.update_layout(
        updatemenus=[score_menu, scope_menu],
        annotations=[score_label, scope_label],
    )
    figure.add_hline(y=0, line_color=MUTED, line_width=1)
    figure.update_xaxes(title="Average share in latest 12 months", tickformat=".1%")
    figure.update_yaxes(
        title="Change from preceding 12 months", tickformat="+.1%"
    )
    return apply_editorial_theme(
        figure,
        "Ecosystem momentum: prominence versus recent change",
        "Upper-right labels are both prominent and gaining share; lower-right labels are prominent but declining.",
        height=690,
        control_band=True,
    )


def activity_specialization_figure(
    data: pd.DataFrame,
    count: int = CHART_LIMITS["explorers"],
) -> go.Figure:
    """Compare absolute activity shares with each label's metric specialization."""
    scopes = list(LABEL_SCOPES)
    frames = {}
    for scope in scopes:
        scoped = filter_label_scope(data, scope)
        leaders = (
            scoped.groupby("language")["composite_share"]
            .mean()
            .sort_values(ascending=False)
            .head(count)
            .index.tolist()
        )
        long = activity_specialization(scoped, leaders)
        absolute = (
            long.pivot(index="language", columns="metric", values="absolute_share")
            .reindex(index=leaders, columns=METRICS)
            * 100
        )
        relative = long.pivot(
            index="language", columns="metric", values="over_index"
        ).reindex(index=leaders, columns=METRICS)
        frames[scope] = (absolute, relative)

    figure = go.Figure()
    for scope in scopes:
        absolute, _ = frames[scope]
        figure.add_trace(
            go.Heatmap(
                z=absolute,
                x=[metric.replace("_", " ").title() for metric in absolute.columns],
                y=absolute.index,
                opacity=1 if scope == "All labels" else 0,
                colorscale="Tealgrn",
                colorbar={"title": "Share (%)"},
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
            )
        )
    mode_buttons = []
    for mode, index, colorscale, title, template in [
        (
            "Absolute share",
            0,
            "Tealgrn",
            "Share (%)",
            "<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
        ),
        (
            "Relative over-index",
            1,
            "RdBu",
            "Index (1 = neutral)",
            "<b>%{y}</b><br>%{x}: %{z:.2f}x average<extra></extra>",
        ),
    ]:
        mode_buttons.append(
            {
                "label": mode,
                "method": "update",
                "args": [
                    {
                        "z": [
                            frames[scope][index].to_numpy() for scope in scopes
                        ],
                        "colorscale": [colorscale] * len(scopes),
                        "zmid": [1 if index else None] * len(scopes),
                        "hovertemplate": [template] * len(scopes),
                        "colorbar.title": [title] * len(scopes),
                    }
                ],
            }
        )
    scope_buttons = [
        {
            "label": scope,
            "method": "update",
            "args": [
                {
                    "opacity": [
                        1 if item_scope == scope else 0 for item_scope in scopes
                    ]
                }
            ],
        }
        for scope in scopes
    ]
    mode_menu, mode_label = _menu(mode_buttons, 0.01, "Heatmap mode")
    scope_menu, scope_label = _menu(scope_buttons, 0.27, "Label scope")
    figure.update_layout(
        updatemenus=[mode_menu, scope_menu],
        annotations=[mode_label, scope_label],
    )
    return apply_editorial_theme(
        figure,
        "Activity specialization across ecosystem signals",
        "Absolute share measures prominence; over-index shows which activities define each ecosystem.",
        height=980,
        control_band=True,
    )


def concentration_figure(data: pd.DataFrame) -> go.Figure:
    """Show the monthly composite share held by leading labels."""
    concentration = top_k_dominance(data)
    figure = go.Figure()
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
            )
        )
    figure.update_yaxes(title="Composite share held by leaders", tickformat=".0%")
    return apply_editorial_theme(
        figure,
        "How much activity is held by the leading ecosystems?",
        "Top-k shares provide a direct, interpretable view of dominance over time.",
        height=560,
    )


def projection_method_figure(
    embeddings: pd.DataFrame,
    colors: Mapping[str, str],
    method: str,
    label_count: int = CHART_LIMITS["embedding_labels"],
) -> go.Figure:
    """Render one full-width projection method with an activity-profile selector."""
    if method not in {"UMAP", "TriMAP", "PaCMAP"}:
        raise ValueError(f"Unknown projection method: {method}")
    profiles = list(embeddings["profile"].drop_duplicates())
    figure = go.Figure()
    for profile_index, profile in enumerate(profiles):
        subset = embeddings.loc[
            (embeddings["profile"] == profile)
            & (embeddings["method"] == method)
        ].sort_values("rank")
        labels = np.where(subset["rank"] <= label_count, subset["language"], "")
        figure.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont={"size": 10, "color": TEXT},
                visible=profile_index == 0,
                marker={
                    "color": [
                        colors.get(language, NEUTRAL)
                        for language in subset["language"]
                    ],
                    "size": np.where(subset["rank"] <= 30, 11, 6),
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
                    "<b>%{customdata[0]}</b><br>Overall rank %{customdata[1]:.0f}"
                    "<br>%{customdata[2]}"
                    "<br>Mean composite share %{customdata[3]:.2%}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    buttons = []
    for profile_index, profile in enumerate(profiles):
        buttons.append(
            {
                "label": profile,
                "method": "update",
                "args": [
                    {
                        "visible": [
                            index == profile_index for index in range(len(profiles))
                        ]
                    },
                    {
                        "title": {
                            "text": _title_html(
                                f"{method}: {profile} activity similarity",
                                "Each point is one label represented by its complete normalized time series.",
                            )
                        }
                    },
                ],
            }
        )
    menu, annotation = _menu(buttons, 0.01, "Activity profile")
    figure.update_layout(updatemenus=[menu], annotations=[annotation])
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return apply_editorial_theme(
        figure,
        f"{method}: {profiles[0]} activity similarity",
        "Each point is one label represented by its complete normalized time series.",
        height=760,
        control_band=True,
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
        "Projection Quality Diagnostics",
        "Higher is better; local-neighborhood and global-distance scores answer different questions.",
        height=510,
    )
