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
    category_composition,
    dominance_turnover,
    ecosystem_diversity,
    ecosystem_momentum,
    filter_label_scope,
    rank_stability,
    ranking_trajectories,
    signal_rank_agreement,
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

TITLE_Y = 0.96
CONTROL_TOP_MARGIN = 145
BOTTOM_MARGIN = 55
CONTROL_MENU_GAP = 12
CONTROL_LABEL_OFFSET = 60

CHART_LIMITS = {
    "trajectories": 12,
    "explorers": 40,
    "heatmap_rows": 30,
    "embeddings": 150,
    "embedding_labels": 25,
}

ROLLING_WINDOWS = {
    "Monthly": 1,
    "Trailing 3 months": 3,
    "Trailing 12 months": 12,
}

FIXED_LANGUAGE_COLORS = {
    "JavaScript": "#E0A800",
    "Python": "#2E7D4A",
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
    top_margin = CONTROL_TOP_MARGIN if control_band else 110
    figure.update_layout(
        title={
            "text": _title_html(title, subtitle),
            "x": 0.015,
            "xanchor": "left",
            "xref": "container",
            "y": TITLE_Y,
            "yanchor": "top",
            "yref": "container",
            "pad": {"t": 0},
        },
        height=height,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        font={"family": "Inter, Arial, sans-serif", "color": TEXT, "size": 12},
        margin={"l": 65, "r": 35, "t": top_margin, "b": BOTTOM_MARGIN},
        hoverlabel={"bgcolor": PANEL, "font_color": TEXT},
        legend={
            "title": {"text": legend_title or ""},
            "bgcolor": "rgba(255,255,255,0.75)",
            "bordercolor": GRID,
            "borderwidth": 1,
        },
    )
    if control_band:
        plot_height = height - top_margin - BOTTOM_MARGIN
        menu_y = 1 + CONTROL_MENU_GAP / plot_height
        label_y = 1 + CONTROL_LABEL_OFFSET / plot_height
        for menu in figure.layout.updatemenus:
            menu.y = menu_y
            menu.yanchor = "bottom"
            menu.pad.t = 0
        for annotation in figure.layout.annotations:
            annotation.y = label_y
            annotation.yanchor = "middle"
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
        "y": 1,
        "yanchor": "bottom",
        "active": active,
        "showactive": True,
        "pad": {"r": 12, "t": 0},
    }
    annotation = {
        "text": label,
        "x": x,
        "xref": "paper",
        "xanchor": "left",
        "y": 1,
        "yref": "paper",
        "yanchor": "middle",
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
                {
                    "x": [],
                    "y": [],
                    "name": "",
                    "text": [],
                    "customdata": np.empty((0, 3)),
                }
            )
            continue
        language = leaders[slot]
        subset = frame.loc[frame["language"] == language]
        endpoint_labels = np.full(len(subset), "", dtype=object)
        if len(endpoint_labels):
            endpoint_labels[-1] = language
        payload.append(
            {
                "x": subset["period"],
                "y": subset["rank"],
                "name": language,
                "text": endpoint_labels,
                "customdata": np.column_stack(
                    [
                        subset["share"],
                        subset["category"],
                        np.repeat(score, len(subset)),
                    ]
                ),
            }
        )
    return payload


def ranking_trajectory_figure(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    score: str = "Composite",
    granularity: str = "Year",
    scope: str = "All labels",
    count: int = CHART_LIMITS["trajectories"],
) -> go.Figure:
    """Render one complete rank-history view for app-level controls."""
    payload = _trajectory_payload(data, score, granularity, scope, count)
    figure = go.Figure()
    for item in payload:
        figure.add_trace(
            go.Scatter(
                x=item["x"],
                y=item["y"],
                mode="lines+markers+text",
                name=item["name"],
                text=item["text"],
                textposition="middle right",
                cliponaxis=False,
                line={
                    "color": colors.get(item["name"], NEUTRAL),
                    "width": 2.2,
                },
                marker={"size": 5},
                customdata=item["customdata"],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>%{x|%b %Y}"
                    "<br>%{customdata[2]} rank: %{y:.0f}"
                    "<br>Share: %{customdata[0]:.2%}"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_yaxes(title="Rank", autorange="reversed")
    figure.update_xaxes(title=None)
    apply_editorial_theme(
        figure,
        "Ranking trajectories across signals and timeframes",
        "A stable leader cohort is followed through every period; rank 1 is highest.",
        height=720,
    )
    figure.update_layout(margin_r=150)
    return figure


def ranking_explorer_figure(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    count: int = CHART_LIMITS["trajectories"],
) -> go.Figure:
    """Explore complete leader histories by signal and period/scope view."""
    scores = list(SCORE_COLUMNS)
    granularities = ["Month", "Quarter", "Year"]
    scopes = list(LABEL_SCOPES)
    views = [
        (granularity, scope)
        for granularity in granularities
        for scope in scopes
    ]
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
    for granularity, scope in views:
        payload = payloads[(initial_score, granularity, scope)]
        for item in payload:
            language = item["name"]
            figure.add_trace(
                go.Scatter(
                    x=item["x"],
                    y=item["y"],
                    mode="lines+markers+text",
                    visible=granularity == "Year" and scope == "All labels",
                    name=language,
                    text=item["text"],
                    textposition="middle right",
                    cliponaxis=False,
                    showlegend=False,
                    line={
                        "color": colors.get(language, NEUTRAL),
                        "width": 2.2,
                    },
                    marker={"size": 5},
                    customdata=item["customdata"],
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>%{x|%b %Y}"
                        "<br>%{customdata[2]} rank: %{y:.0f}"
                        "<br>Share: %{customdata[0]:.2%}"
                        "<br>%{customdata[1]}<extra></extra>"
                    ),
                )
            )
            trace_keys.append((granularity, scope))

    metric_buttons = []
    for score in scores:
        flattened = [
            item
            for granularity, scope in views
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
                        "text": [item["text"] for item in flattened],
                        "customdata": [item["customdata"] for item in flattened],
                        "line.color": [
                            colors.get(item["name"], NEUTRAL) for item in flattened
                        ],
                    },
                    {"yaxis.autorange": "reversed"},
                ],
            }
        )
    view_buttons = [
        {
            "label": f"{granularity} · {scope}",
            "method": "update",
            "args": [
                {
                    "visible": [
                        trace_granularity == granularity
                        and trace_scope == scope
                        for trace_granularity, trace_scope in trace_keys
                    ],
                },
                {"yaxis.autorange": "reversed"},
            ],
        }
        for granularity, scope in views
    ]
    metric_menu, metric_label = _menu(metric_buttons, 0.01, "Ranking signal")
    view_menu, view_label = _menu(
        view_buttons,
        0.31,
        "Period and label scope",
        active=views.index(("Year", "All labels")),
    )
    figure.update_layout(
        updatemenus=[metric_menu, view_menu],
        annotations=[metric_label, view_label],
    )
    figure.update_yaxes(
        title="Rank",
        autorange="reversed",
    )
    figure.update_xaxes(title=None)
    apply_editorial_theme(
        figure,
        "Ranking trajectories across signals and timeframes",
        "A stable top-12 cohort is followed through every period; rank 1 is highest.",
        height=720,
        control_band=True,
    )
    figure.update_layout(margin_r=150)
    return figure


def _turnover_matrix(
    data: pd.DataFrame,
    score: str,
    granularity: str,
    scope: str,
    top_k: int,
) -> dict:
    frame = dominance_turnover(data, score, granularity, scope, top_k)
    languages = frame["language"].drop_duplicates().astype(str).tolist()
    pivot = frame.pivot(index="language", columns="period", values="rank").reindex(
        languages
    )
    return {
        "x": pivot.columns,
        "y": pivot.index.astype(str),
        "z": pivot.to_numpy(),
        "customdata": pivot.to_numpy(),
    }


def dominance_turnover_view_figure(
    data: pd.DataFrame,
    score: str = "Composite",
    granularity: str = "Year",
    scope: str = "All labels",
    top_k: int = 10,
) -> go.Figure:
    """Render one top-tier membership timeline for app-level controls."""
    item = _turnover_matrix(data, score, granularity, scope, top_k)
    figure = go.Figure(
        go.Heatmap(
            x=item["x"],
            y=item["y"],
            z=item["z"],
            customdata=item["customdata"],
            zmin=1,
            zmax=top_k,
            colorscale="Viridis_r",
            colorbar={"title": "Rank", "dtick": 1},
            hoverongaps=False,
            hovertemplate=(
                "<b>%{y}</b><br>%{x|%b %Y}"
                "<br>Top-tier rank %{customdata:.0f}<extra></extra>"
            ),
        )
    )
    figure.update_yaxes(autorange="reversed")
    return apply_editorial_theme(
        figure,
        "Top-tier membership over time",
        f"Colored cells show top-{top_k} rank; blank cells mean outside the tier.",
        height=760,
    )


def dominance_turnover_figure(
    data: pd.DataFrame,
    top_k: int = 10,
) -> go.Figure:
    """Explore top-tier membership by signal and period/scope view."""
    scores = list(SCORE_COLUMNS)
    granularities = ["Month", "Quarter", "Year"]
    scopes = list(LABEL_SCOPES)
    views = [
        (granularity, scope)
        for granularity in granularities
        for scope in scopes
    ]
    matrices = {
        (score, granularity, scope): _turnover_matrix(
            data, score, granularity, scope, top_k
        )
        for score in scores
        for granularity in granularities
        for scope in scopes
    }
    figure = go.Figure()
    trace_keys = []
    for granularity, scope in views:
        item = matrices[("Composite", granularity, scope)]
        figure.add_trace(
            go.Heatmap(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                customdata=item["customdata"],
                visible=granularity == "Year" and scope == "All labels",
                zmin=1,
                zmax=top_k,
                colorscale="Viridis_r",
                colorbar={"title": "Rank", "dtick": 1},
                hoverongaps=False,
                hovertemplate=(
                    "<b>%{y}</b><br>%{x|%b %Y}"
                    "<br>Top-tier rank %{customdata:.0f}<extra></extra>"
                ),
            )
        )
        trace_keys.append((granularity, scope))
    metric_buttons = []
    for score in scores:
        items = [
            matrices[(score, granularity, scope)]
            for granularity, scope in views
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
    view_buttons = [
        {
            "label": f"{granularity} · {scope}",
            "method": "update",
            "args": [
                {
                    "visible": [
                        item_granularity == granularity and item_scope == scope
                        for item_granularity, item_scope in trace_keys
                    ]
                }
            ],
        }
        for granularity, scope in views
    ]
    menus = [
        _menu(metric_buttons, 0.01, "Ranking signal"),
        _menu(
            view_buttons,
            0.31,
            "Period and label scope",
            active=views.index(("Year", "All labels")),
        ),
    ]
    figure.update_layout(
        updatemenus=[menu for menu, _ in menus],
        annotations=[label for _, label in menus],
    )
    figure.update_yaxes(autorange="reversed")
    return apply_editorial_theme(
        figure,
        "Top-tier membership over time",
        f"Colored cells show top-{top_k} rank; blank cells mean outside the tier.",
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


def ecosystem_momentum_view_figure(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    score: str = "Composite",
    scope: str = "All labels",
    count: int = CHART_LIMITS["explorers"],
) -> go.Figure:
    """Render one momentum view for app-level controls."""
    frame = ecosystem_momentum(data, score, scope, count=count)
    figure = go.Figure(
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
            marker={
                "size": 12,
                "color": [
                    colors.get(name, NEUTRAL) for name in frame["language"]
                ],
                "line": {"color": PANEL, "width": 0.7},
            },
            customdata=np.column_stack(
                [
                    frame["language"],
                    frame["previous_share"],
                    frame["category"],
                    np.repeat(score, len(frame)),
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[3]}"
                "<br>Latest 12m: %{x:.2%}"
                "<br>Change: %{y:+.2%}<br>Prior 12m: %{customdata[1]:.2%}"
                "<br>%{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
        )
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
        height=650,
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
    for scope_index, scope in enumerate(scopes):
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
                visible=scope_index == 0,
                marker={
                    "size": 12,
                    "color": [colors.get(name, NEUTRAL) for name in frame["language"]],
                    "line": {"color": PANEL, "width": 0.7},
                },
                customdata=np.column_stack(
                    [
                        frame["language"],
                        frame["previous_share"],
                        frame["category"],
                        np.repeat("Composite", len(frame)),
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[3]}"
                    "<br>Latest 12m: %{x:.2%}"
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
                                    np.repeat(score, len(frame)),
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
                    "visible": [
                        item_scope == scope for item_scope in scopes
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


def _specialization_matrices(
    data: pd.DataFrame,
    scope: str,
    count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return absolute, relative


def activity_specialization_view_figure(
    data: pd.DataFrame,
    scope: str = "All labels",
    mode: str = "Relative over-index",
    count: int = CHART_LIMITS["heatmap_rows"],
) -> go.Figure:
    """Render one specialization heatmap for app-level controls."""
    if mode not in {"Relative over-index", "Absolute share"}:
        raise ValueError(f"Unknown specialization mode: {mode}")
    absolute, relative = _specialization_matrices(data, scope, count)
    is_relative = mode == "Relative over-index"
    matrix = relative if is_relative else absolute
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[metric.replace("_", " ").title() for metric in matrix.columns],
            y=matrix.index,
            colorscale="RdBu" if is_relative else "Tealgrn",
            zmid=1 if is_relative else None,
            colorbar={
                "title": "Index (1 = neutral)" if is_relative else "Share (%)"
            },
            hovertemplate=(
                "<b>%{y}</b><br>%{x}: %{z:.2f}x own average<extra></extra>"
                if is_relative
                else "<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>"
            ),
        )
    )
    figure.update_yaxes(autorange="reversed")
    return apply_editorial_theme(
        figure,
        "Activity specialization across ecosystem signals",
        "Over-index shows emphasis relative to each ecosystem's own average signal share.",
        height=max(650, 250 + 21 * len(matrix)),
    )


def activity_specialization_figure(
    data: pd.DataFrame,
    count: int = CHART_LIMITS["heatmap_rows"],
) -> go.Figure:
    """Explore prominence and specialization without retaining hidden rows."""
    scopes = list(LABEL_SCOPES)
    frames = {
        scope: _specialization_matrices(data, scope, count) for scope in scopes
    }

    figure = go.Figure()
    for scope_index, scope in enumerate(scopes):
        _, relative = frames[scope]
        figure.add_trace(
            go.Heatmap(
                z=relative,
                x=[metric.replace("_", " ").title() for metric in relative.columns],
                y=relative.index,
                visible=scope_index == 0,
                colorscale="RdBu",
                zmid=1,
                colorbar={"title": "Index (1 = neutral)"},
                hovertemplate=(
                    "<b>%{y}</b><br>%{x}: %{z:.2f}x own average<extra></extra>"
                ),
            )
        )
    mode_buttons = []
    for mode, index, colorscale, title, template in [
        (
            "Relative over-index",
            1,
            "RdBu",
            "Index (1 = neutral)",
            "<b>%{y}</b><br>%{x}: %{z:.2f}x own average<extra></extra>",
        ),
        (
            "Absolute share",
            0,
            "Tealgrn",
            "Share (%)",
            "<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>",
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
                    "visible": [
                        item_scope == scope for item_scope in scopes
                    ]
                },
                {"yaxis.autorange": "reversed"},
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
    figure.update_yaxes(autorange="reversed")
    return apply_editorial_theme(
        figure,
        "Activity specialization across ecosystem signals",
        "Over-index shows emphasis relative to each ecosystem's own average signal share.",
        height=880,
        control_band=True,
    )


def category_composition_view_figure(
    data: pd.DataFrame,
    score: str = "Composite",
    window: int = 3,
) -> go.Figure:
    """Render language-versus-artifact composition for app-level controls."""
    frame = category_composition(data, score, window)
    figure = go.Figure()
    for category, color in [
        ("Programming language", "#2A9D8F"),
        ("Technology / artifact", "#E76F51"),
    ]:
        subset = frame.loc[frame["category"] == category]
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["share"],
                stackgroup="categories",
                mode="lines",
                name=category,
                line={"color": color, "width": 1.5},
                customdata=np.column_stack(
                    [subset["raw_share"], np.repeat(score, len(subset))]
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>%{x|%B %Y}"
                    "<br>%{customdata[1]} share: %{y:.2%}"
                    "<br>Raw month: %{customdata[0]:.2%}<extra></extra>"
                ),
            )
        )
    figure.update_yaxes(title="Share of selected signal", tickformat=".0%", range=[0, 1])
    return apply_editorial_theme(
        figure,
        "Programming languages versus technology artifacts",
        "The selected activity signal is partitioned between the two label categories.",
        height=570,
        legend_title="Label category",
    )


def category_composition_figure(data: pd.DataFrame) -> go.Figure:
    """Explore category composition by signal and smoothing window."""
    scores = list(SCORE_COLUMNS)
    windows = list(ROLLING_WINDOWS.items())
    categories = ["Programming language", "Technology / artifact"]
    payloads = {
        (score, window): category_composition(data, score, window)
        for score in scores
        for _, window in windows
    }
    figure = go.Figure()
    trace_keys = []
    for window_label, window in windows:
        frame = payloads[("Composite", window)]
        for category, color in [
            ("Programming language", "#2A9D8F"),
            ("Technology / artifact", "#E76F51"),
        ]:
            subset = frame.loc[frame["category"] == category]
            figure.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset["share"],
                    stackgroup=f"categories-{window}",
                    mode="lines",
                    visible=window == 3,
                    name=category,
                    showlegend=window == 3,
                    line={"color": color, "width": 1.5},
                    customdata=np.column_stack(
                        [
                            subset["raw_share"],
                            np.repeat("Composite", len(subset)),
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>%{x|%B %Y}"
                        "<br>%{customdata[1]} share: %{y:.2%}"
                        "<br>Raw month: %{customdata[0]:.2%}<extra></extra>"
                    ),
                )
            )
            trace_keys.append((window_label, category))
    score_buttons = []
    for score in scores:
        selected = []
        for _, window in windows:
            frame = payloads[(score, window)]
            selected.extend(
                frame.loc[frame["category"] == category] for category in categories
            )
        score_buttons.append(
            {
                "label": score,
                "method": "update",
                "args": [
                    {
                        "x": [frame["date"] for frame in selected],
                        "y": [frame["share"] for frame in selected],
                        "customdata": [
                            np.column_stack(
                                [
                                    frame["raw_share"],
                                    np.repeat(score, len(frame)),
                                ]
                            )
                            for frame in selected
                        ],
                    }
                ],
            }
        )
    window_buttons = [
        {
            "label": window_label,
            "method": "update",
            "args": [
                {
                    "visible": [
                        trace_window == window_label
                        for trace_window, _ in trace_keys
                    ],
                    "showlegend": [
                        trace_window == window_label
                        for trace_window, _ in trace_keys
                    ],
                }
            ],
        }
        for window_label, _ in windows
    ]
    score_menu, score_label = _menu(score_buttons, 0.01, "Activity signal")
    window_menu, window_label = _menu(
        window_buttons, 0.31, "Smoothing", active=1
    )
    figure.update_layout(
        updatemenus=[score_menu, window_menu],
        annotations=[score_label, window_label],
    )
    figure.update_yaxes(title="Share of selected signal", tickformat=".0%", range=[0, 1])
    return apply_editorial_theme(
        figure,
        "Programming languages versus technology artifacts",
        "The selected activity signal is partitioned between the two label categories.",
        height=610,
        legend_title="Label category",
        control_band=True,
    )


def _agreement_matrix(
    data: pd.DataFrame,
    reference_score: str,
    scope: str,
    window: int,
) -> dict:
    frame = signal_rank_agreement(data, reference_score, scope, window)
    signals = [score for score in SCORE_COLUMNS if score != reference_score]
    agreement = frame.pivot(index="signal", columns="date", values="agreement").reindex(
        signals
    )
    raw = frame.pivot(
        index="signal", columns="date", values="raw_agreement"
    ).reindex(signals)
    reference = np.full(agreement.shape, reference_score, dtype=object)
    return {
        "x": agreement.columns,
        "y": agreement.index,
        "z": agreement.to_numpy(),
        "customdata": np.dstack([raw.to_numpy(), reference]),
    }


def signal_agreement_view_figure(
    data: pd.DataFrame,
    reference_score: str = "Composite",
    scope: str = "All labels",
    window: int = 3,
) -> go.Figure:
    """Render one signal-agreement heatmap for app-level controls."""
    item = _agreement_matrix(data, reference_score, scope, window)
    figure = go.Figure(
        go.Heatmap(
            x=item["x"],
            y=item["y"],
            z=item["z"],
            customdata=item["customdata"],
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            colorbar={"title": "Rank correlation"},
            hovertemplate=(
                "<b>%{y} vs %{customdata[1]}</b><br>%{x|%B %Y}"
                "<br>Smoothed agreement: %{z:.3f}"
                "<br>Raw month: %{customdata[0]:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_yaxes(autorange="reversed")
    return apply_editorial_theme(
        figure,
        "How similarly do activity signals rank ecosystems?",
        "Spearman correlation compares one reference ranking with every other signal.",
        height=700,
    )


def signal_agreement_figure(data: pd.DataFrame) -> go.Figure:
    """Explore ranking agreement by reference signal and scope/window view."""
    scores = list(SCORE_COLUMNS)
    views = [
        (scope, window_label, window)
        for window_label, window in ROLLING_WINDOWS.items()
        for scope in LABEL_SCOPES
    ]
    matrices = {
        (score, scope, window): _agreement_matrix(data, score, scope, window)
        for score in scores
        for scope, _, window in views
    }
    figure = go.Figure()
    trace_keys = []
    for scope, window_label, window in views:
        item = matrices[("Composite", scope, window)]
        figure.add_trace(
            go.Heatmap(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                customdata=item["customdata"],
                visible=scope == "All labels" and window == 3,
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale="RdBu",
                colorbar={"title": "Rank correlation"},
                hovertemplate=(
                    "<b>%{y} vs %{customdata[1]}</b><br>%{x|%B %Y}"
                    "<br>Smoothed agreement: %{z:.3f}"
                    "<br>Raw month: %{customdata[0]:.3f}<extra></extra>"
                ),
            )
        )
        trace_keys.append((scope, window_label))
    reference_buttons = []
    for score in scores:
        items = [
            matrices[(score, scope, window)] for scope, _, window in views
        ]
        reference_buttons.append(
            {
                "label": score,
                "method": "update",
                "args": [
                    {
                        "x": [item["x"] for item in items],
                        "y": [item["y"] for item in items],
                        "z": [item["z"] for item in items],
                        "customdata": [item["customdata"] for item in items],
                    },
                    {"yaxis.autorange": "reversed"},
                ],
            }
        )
    view_buttons = [
        {
            "label": f"{window_label} · {scope}",
            "method": "update",
            "args": [
                {
                    "visible": [
                        trace_scope == scope and trace_window == window_label
                        for trace_scope, trace_window in trace_keys
                    ]
                },
                {"yaxis.autorange": "reversed"},
            ],
        }
        for scope, window_label, _ in views
    ]
    reference_menu, reference_label = _menu(
        reference_buttons, 0.01, "Reference signal"
    )
    view_menu, view_label = _menu(
        view_buttons,
        0.31,
        "Smoothing and label scope",
        active=views.index(("All labels", "Trailing 3 months", 3)),
    )
    figure.update_layout(
        updatemenus=[reference_menu, view_menu],
        annotations=[reference_label, view_label],
    )
    figure.update_yaxes(autorange="reversed")
    return apply_editorial_theme(
        figure,
        "How similarly do activity signals rank ecosystems?",
        "Spearman correlation compares one reference ranking with every other signal.",
        height=760,
        control_band=True,
    )


def _stability_payload(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    score: str,
    granularity: str,
    scope: str,
    count: int,
) -> dict:
    frame = rank_stability(data, score, granularity, scope, count)
    labels = np.where(
        frame["mean_share"].rank(ascending=False) <= 12,
        frame["language"],
        "",
    )
    return {
        "x": frame["mean_share"],
        "y": frame["rank_volatility"],
        "text": labels,
        "marker.color": [
            colors.get(language, NEUTRAL) for language in frame["language"]
        ],
        "customdata": np.column_stack(
            [
                frame["language"],
                frame["mean_rank"],
                frame["best_rank"],
                frame["worst_rank"],
                frame["category"],
                np.repeat(score, len(frame)),
                frame["periods"],
            ]
        ),
    }


def rank_stability_view_figure(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    score: str = "Composite",
    granularity: str = "Quarter",
    scope: str = "All labels",
    count: int = CHART_LIMITS["explorers"],
) -> go.Figure:
    """Render one prominence-versus-volatility view for app-level controls."""
    item = _stability_payload(data, colors, score, granularity, scope, count)
    figure = go.Figure(
        go.Scatter(
            x=item["x"],
            y=item["y"],
            mode="markers+text",
            text=item["text"],
            textposition="top center",
            marker={
                "size": 12,
                "color": item["marker.color"],
                "line": {"color": PANEL, "width": 0.7},
            },
            customdata=item["customdata"],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[5]}"
                "<br>Mean share: %{x:.2%}<br>Rank volatility: %{y:.1f}"
                "<br>Mean rank: %{customdata[1]:.1f}"
                "<br>Best–worst: %{customdata[2]:.0f}–%{customdata[3]:.0f}"
                "<br>%{customdata[4]} · %{customdata[6]:.0f} periods"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_xaxes(title="Mean activity share", type="log", tickformat=".2%")
    figure.update_yaxes(title="Rank volatility (standard deviation)")
    return apply_editorial_theme(
        figure,
        "Prominence versus ranking stability",
        "Higher points move more in the ranking; farther-right points hold more activity share.",
        height=660,
    )


def rank_stability_figure(
    data: pd.DataFrame,
    colors: Mapping[str, str],
    count: int = CHART_LIMITS["explorers"],
) -> go.Figure:
    """Explore prominence and rank volatility by signal and period/scope."""
    scores = list(SCORE_COLUMNS)
    views = [
        (granularity, scope)
        for granularity in ["Month", "Quarter", "Year"]
        for scope in LABEL_SCOPES
    ]
    payloads = {
        (score, granularity, scope): _stability_payload(
            data, colors, score, granularity, scope, count
        )
        for score in scores
        for granularity, scope in views
    }
    figure = go.Figure()
    trace_keys = []
    for granularity, scope in views:
        item = payloads[("Composite", granularity, scope)]
        figure.add_trace(
            go.Scatter(
                x=item["x"],
                y=item["y"],
                mode="markers+text",
                text=item["text"],
                textposition="top center",
                visible=granularity == "Quarter" and scope == "All labels",
                marker={
                    "size": 12,
                    "color": item["marker.color"],
                    "line": {"color": PANEL, "width": 0.7},
                },
                customdata=item["customdata"],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[5]}"
                    "<br>Mean share: %{x:.2%}<br>Rank volatility: %{y:.1f}"
                    "<br>Mean rank: %{customdata[1]:.1f}"
                    "<br>Best–worst: %{customdata[2]:.0f}–%{customdata[3]:.0f}"
                    "<br>%{customdata[4]} · %{customdata[6]:.0f} periods"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
        trace_keys.append((granularity, scope))
    score_buttons = []
    for score in scores:
        items = [
            payloads[(score, granularity, scope)]
            for granularity, scope in views
        ]
        score_buttons.append(
            {
                "label": score,
                "method": "update",
                "args": [
                    {
                        "x": [item["x"] for item in items],
                        "y": [item["y"] for item in items],
                        "text": [item["text"] for item in items],
                        "marker.color": [
                            item["marker.color"] for item in items
                        ],
                        "customdata": [item["customdata"] for item in items],
                    }
                ],
            }
        )
    view_buttons = [
        {
            "label": f"{granularity} · {scope}",
            "method": "update",
            "args": [
                {
                    "visible": [
                        trace_granularity == granularity
                        and trace_scope == scope
                        for trace_granularity, trace_scope in trace_keys
                    ]
                }
            ],
        }
        for granularity, scope in views
    ]
    score_menu, score_label = _menu(score_buttons, 0.01, "Ranking signal")
    view_menu, view_label = _menu(
        view_buttons,
        0.31,
        "Period and label scope",
        active=views.index(("Quarter", "All labels")),
    )
    figure.update_layout(
        updatemenus=[score_menu, view_menu],
        annotations=[score_label, view_label],
    )
    figure.update_xaxes(title="Mean activity share", type="log", tickformat=".2%")
    figure.update_yaxes(title="Rank volatility (standard deviation)")
    return apply_editorial_theme(
        figure,
        "Prominence versus ranking stability",
        "Higher points move more in the ranking; farther-right points hold more activity share.",
        height=700,
        control_band=True,
    )


def diversity_view_figure(
    data: pd.DataFrame,
    score: str = "Composite",
    scope: str = "All labels",
    window: int = 3,
) -> go.Figure:
    """Render one effective-diversity time series for app-level controls."""
    frame = ecosystem_diversity(data, score, scope, window)
    figure = go.Figure()
    for column, raw_column, label, color in [
        ("effective_hhi", "raw_effective_hhi", "Inverse HHI", "#D95F45"),
        (
            "effective_entropy",
            "raw_effective_entropy",
            "Exponential Shannon",
            "#2A9D8F",
        ),
    ]:
        figure.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[column],
                mode="lines",
                name=label,
                line={"color": color, "width": 2.5},
                customdata=np.column_stack(
                    [frame[raw_column], np.repeat(score, len(frame))]
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>%{{x|%B %Y}}"
                    "<br>%{customdata[1]} effective count: %{y:.1f}"
                    "<br>Raw month: %{customdata[0]:.1f}<extra></extra>"
                ),
            )
        )
    figure.update_yaxes(title="Effective number of ecosystems", rangemode="tozero")
    return apply_editorial_theme(
        figure,
        "How diverse is the observed ecosystem?",
        "Effective counts translate concentration and entropy into an intuitive number of equal-sized ecosystems.",
        height=590,
        legend_title="Diversity measure",
    )


def diversity_figure(data: pd.DataFrame) -> go.Figure:
    """Explore effective diversity by signal and scope/smoothing view."""
    scores = list(SCORE_COLUMNS)
    views = [
        (scope, window_label, window)
        for window_label, window in ROLLING_WINDOWS.items()
        for scope in LABEL_SCOPES
    ]
    frames = {
        (score, scope, window): ecosystem_diversity(
            data, score, scope, window
        )
        for score in scores
        for scope, _, window in views
    }
    definitions = [
        ("effective_hhi", "raw_effective_hhi", "Inverse HHI", "#D95F45"),
        (
            "effective_entropy",
            "raw_effective_entropy",
            "Exponential Shannon",
            "#2A9D8F",
        ),
    ]
    figure = go.Figure()
    trace_keys = []
    for scope, window_label, window in views:
        frame = frames[("Composite", scope, window)]
        for column, raw_column, label, color in definitions:
            figure.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[column],
                    mode="lines",
                    visible=scope == "All labels" and window == 3,
                    name=label,
                    showlegend=scope == "All labels" and window == 3,
                    line={"color": color, "width": 2.5},
                    customdata=np.column_stack(
                        [frame[raw_column], np.repeat("Composite", len(frame))]
                    ),
                    hovertemplate=(
                        f"<b>{label}</b><br>%{{x|%B %Y}}"
                        "<br>%{customdata[1]} effective count: %{y:.1f}"
                        "<br>Raw month: %{customdata[0]:.1f}<extra></extra>"
                    ),
                )
            )
            trace_keys.append((scope, window_label))
    score_buttons = []
    for score in scores:
        selected = []
        for scope, _, window in views:
            frame = frames[(score, scope, window)]
            selected.extend(
                (frame, column, raw_column)
                for column, raw_column, _, _ in definitions
            )
        score_buttons.append(
            {
                "label": score,
                "method": "update",
                "args": [
                    {
                        "x": [frame["date"] for frame, _, _ in selected],
                        "y": [frame[column] for frame, column, _ in selected],
                        "customdata": [
                            np.column_stack(
                                [
                                    frame[raw_column],
                                    np.repeat(score, len(frame)),
                                ]
                            )
                            for frame, _, raw_column in selected
                        ],
                    }
                ],
            }
        )
    view_buttons = [
        {
            "label": f"{window_label} · {scope}",
            "method": "update",
            "args": [
                {
                    "visible": [
                        trace_scope == scope and trace_window == window_label
                        for trace_scope, trace_window in trace_keys
                    ],
                    "showlegend": [
                        trace_scope == scope and trace_window == window_label
                        for trace_scope, trace_window in trace_keys
                    ],
                }
            ],
        }
        for scope, window_label, _ in views
    ]
    score_menu, score_label = _menu(score_buttons, 0.01, "Activity signal")
    view_menu, view_label = _menu(
        view_buttons,
        0.31,
        "Smoothing and label scope",
        active=views.index(("All labels", "Trailing 3 months", 3)),
    )
    figure.update_layout(
        updatemenus=[score_menu, view_menu],
        annotations=[score_label, view_label],
    )
    figure.update_yaxes(title="Effective number of ecosystems", rangemode="tozero")
    return apply_editorial_theme(
        figure,
        "How diverse is the observed ecosystem?",
        "Effective counts translate concentration and entropy into an intuitive number of equal-sized ecosystems.",
        height=630,
        legend_title="Diversity measure",
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
                    }
                ],
            }
        )
    menu, annotation = _menu(buttons, 0.01, "Activity profile")
    figure.update_layout(updatemenus=[menu], annotations=[annotation])
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return apply_editorial_theme(
        figure,
        f"{method}: activity-profile similarity",
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
