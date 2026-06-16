"""Export static, report-ready figures for the LaTeX project report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lang_ecosystem.analysis import (  # noqa: E402
    EVENT_METRICS,
    METRICS,
    activity_specialization,
    add_monthly_shares,
    add_trailing_shares,
    build_profile_vectors,
    complete_month_grid,
    ecosystem_diversity,
    ecosystem_momentum,
    evaluate_embeddings,
    load_activity_data,
    language_group,
    rank_languages,
    run_all_embeddings,
    top_k_dominance,
)
from lang_ecosystem.visuals import (  # noqa: E402
    FIXED_LANGUAGE_COLORS,
    LANGUAGE_GROUP_COLORS,
    NEUTRAL,
    language_color_map,
    sampling_bias_figure,
)

DATA_PATH = ROOT / "data" / "github_language_activity_monthly.csv"
REPORT_DIR = ROOT / "report"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

BACKGROUND = "#FFFFFF"
PANEL = "#FBF8F2"
GRID = "#DED8CF"
TEXT = "#24221F"
MUTED = "#6F6558"
POSITIVE = "#16856B"
NEGATIVE = "#C44E52"
ACCENT = "#3178C6"

WIDTH = 1100
HEIGHT = 640


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    raw = load_activity_data(DATA_PATH)
    activity = add_monthly_shares(complete_month_grid(raw))
    activity = add_trailing_shares(activity, windows=(3, 12))
    ranking = rank_languages(activity)
    top_150 = ranking.head(150)["language"].tolist()
    colors = language_color_map(top_150, use_semantic_groups=True)
    colors["Other"] = NEUTRAL
    return raw, activity, ranking, colors


def apply_report_theme(
    fig: go.Figure,
    title: str,
    subtitle: str | None = None,
    height: int = HEIGHT,
    legend_title: str | None = None,
) -> go.Figure:
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:16px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.015,
            "xanchor": "left",
            "y": 0.94,
            "yanchor": "top",
        },
        width=WIDTH,
        height=height,
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        font={"family": "Arial, sans-serif", "color": TEXT, "size": 15},
        margin={"l": 72, "r": 92, "t": 125, "b": 70},
        legend={
            "title": {"text": legend_title or ""},
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "left",
            "x": 0,
        },
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return fig


def export_figure(fig: go.Figure, name: str, *, width: int = WIDTH, height: int = HEIGHT) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.update_layout(width=width, height=height)
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    png_path = FIGURE_DIR / f"{name}.png"
    try:
        fig.write_image(pdf_path)
        fig.write_image(png_path, scale=2)
    except RuntimeError as exc:
        raise RuntimeError(
            "Plotly static export failed. Kaleido v1 requires Chrome or Chromium. "
            "Install it with `uv run plotly_get_chrome` or run "
            "`python -c \"import plotly.io as pio; pio.get_chrome()\"`."
        ) from exc
    print(f"wrote {pdf_path.relative_to(ROOT)} and {png_path.relative_to(ROOT)}")


def add_endpoint_labels(
    fig: go.Figure,
    frame: pd.DataFrame,
    y_column: str,
    colors: dict[str, str],
    x_offset_months: int = 4,
) -> None:
    last_date = frame["date"].max()
    label_date = (last_date + pd.DateOffset(months=x_offset_months)).to_pydatetime()
    for language, group in frame.groupby("language", sort=False):
        last = group.sort_values("date").iloc[-1]
        fig.add_annotation(
            x=label_date,
            y=last[y_column],
            text=language,
            showarrow=False,
            xanchor="left",
            font={"size": 13, "color": colors.get(language, TEXT)},
        )


def sampling_figure(activity: pd.DataFrame, ranking: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    leaders = ranking.head(5)["language"].tolist()
    return sampling_bias_figure(activity, leaders, colors)


def composite_trend_figure(activity: pd.DataFrame, ranking: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    top_languages = ranking.head(8)["language"].tolist()
    frame = activity.loc[activity["language"].isin(top_languages)].copy()
    fig = go.Figure()
    for language in top_languages:
        subset = frame.loc[frame["language"] == language]
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["composite_share_12m"],
                mode="lines",
                name=language,
                line={"width": 3, "color": colors.get(language)},
                showlegend=False,
            )
        )
    add_endpoint_labels(fig, frame, "composite_share_12m", colors)
    fig.update_xaxes(
        range=[
            frame["date"].min().to_pydatetime(),
            (frame["date"].max() + pd.DateOffset(months=15)).to_pydatetime(),
        ]
    )
    fig.update_yaxes(title="Trailing 12-month composite share", tickformat=".0%")
    return apply_report_theme(
        fig,
        "Long-run popularity trends for leading ecosystems",
        "Equal-weight composite share, smoothed over trailing 12-month windows.",
    )


def window_rank_table(activity: pd.DataFrame) -> pd.DataFrame:
    dates = np.sort(activity["date"].unique())
    windows = {
        "2016": dates[:12],
        "2025": dates[-12:],
    }
    rows = []
    for label, window_dates in windows.items():
        share = (
            activity.loc[activity["date"].isin(window_dates)]
            .groupby("language")["composite_share"]
            .mean()
            .sort_values(ascending=False)
        )
        ranks = share.rank(method="min", ascending=False).astype(int)
        rows.append(
            pd.DataFrame(
                {
                    "language": share.index,
                    f"share_{label}": share.values,
                    f"rank_{label}": ranks.values,
                }
            )
        )
    return rows[0].merge(rows[1], on="language", how="outer").fillna(0)


def rank_change_figure(activity: pd.DataFrame, ranking: pd.DataFrame) -> go.Figure:
    ranks = window_rank_table(activity)
    latest_top = ranks.nsmallest(5, "rank_2025")["language"].tolist()
    important = list(
        dict.fromkeys(
            [
                *latest_top,
                "Rust",
                "Jupyter Notebook",
                "Ruby",
                "PHP",
                "CSS",
            ]
        )
    )
    frame = ranks.loc[ranks["language"].isin(important)].copy()
    frame["change_pp"] = (frame["share_2025"] - frame["share_2016"]) * 100
    frame = frame.sort_values(["rank_2025", "rank_2016"])
    colors = language_color_map(ranking.head(150)["language"].tolist())

    fig = go.Figure()
    for _, row in frame.iterrows():
        language = row["language"]
        color = colors.get(language, FIXED_LANGUAGE_COLORS.get(language, MUTED))
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[row["rank_2016"], row["rank_2025"]],
                mode="lines+markers",
                line={"width": 3, "color": color},
                marker={"size": 9, "color": color},
                showlegend=False,
                name=language,
            )
        )
        fig.add_annotation(
            x=-0.04,
            y=row["rank_2016"],
            text=f"{language} ({int(row['rank_2016'])})",
            xanchor="right",
            showarrow=False,
            font={"size": 13, "color": color},
        )
        fig.add_annotation(
            x=1.04,
            y=row["rank_2025"],
            text=f"{language} ({int(row['rank_2025'])})",
            xanchor="left",
            showarrow=False,
            font={"size": 13, "color": color},
        )
    max_rank = int(max(frame["rank_2016"].max(), frame["rank_2025"].max()) + 2)
    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=["First 12 months", "Latest 12 months"],
        range=[-0.32, 1.32],
        showgrid=False,
    )
    fig.update_yaxes(
        title="Rank by average composite share",
        autorange="reversed",
        range=[max_rank, 0],
        dtick=5,
    )
    return apply_report_theme(
        fig,
        "Rank change",
        "Slopegraph of selected leaders, major gainers, and major decliners; lower rank is better.",
    )


def winners_decliners_figure(activity: pd.DataFrame) -> go.Figure:
    dates = np.sort(activity["date"].unique())
    first = (
        activity.loc[activity["date"].isin(dates[:12])]
        .groupby("language")["composite_share"]
        .mean()
    )
    last = (
        activity.loc[activity["date"].isin(dates[-12:])]
        .groupby("language")["composite_share"]
        .mean()
    )
    change = (last - first).sort_values()
    selected = pd.concat([change.head(5), change.tail(5)]).sort_values()
    fig = go.Figure(
        go.Bar(
            x=selected.values * 100,
            y=selected.index,
            orientation="h",
            marker_color=[NEGATIVE if value < 0 else POSITIVE for value in selected],
            text=[f"{value:+.1f} pp" for value in selected.values * 100],
            textposition="outside",
        )
    )
    fig.update_xaxes(title="Change in average composite share, percentage points")
    fig.update_yaxes(title=None)
    return apply_report_theme(
        fig,
        "Largest long-run gainers and decliners",
        "Difference between the first and latest 12-month average composite shares.",
        height=620,
    )


def community_momentum_figure(activity: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    momentum = ecosystem_momentum(activity, score="Composite", window=12, count=22)
    labels = set(momentum.nlargest(5, "current_share")["language"])
    labels.update(momentum.nlargest(4, "change")["language"])
    labels.update(momentum.nsmallest(3, "change")["language"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=momentum["current_share"],
            y=momentum["change"],
            mode="markers+text",
            text=[language if language in labels else "" for language in momentum["language"]],
            textposition="top center",
            marker={
                "size": 13,
                "color": [colors.get(language, MUTED) for language in momentum["language"]],
                "line": {"width": 0.6, "color": "#FFFFFF"},
            },
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_width=1.2, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(title="Average composite share in latest 12 months", tickformat=".0%")
    fig.update_yaxes(title="Change from preceding 12 months", tickformat="+.1%")
    return apply_report_theme(
        fig,
        "Recent ecosystem momentum",
        "Latest 12 months compared with the previous 12 months among prominent ecosystems.",
    )


def diversity_concentration_figure(activity: pd.DataFrame) -> go.Figure:
    dominance = top_k_dominance(activity).sort_values("date")
    for column in ["top_1_share", "top_5_share", "top_10_share"]:
        dominance[f"{column}_3m"] = dominance[column].rolling(3, min_periods=1).mean()
    diversity = ecosystem_diversity(activity, score="Composite", window=3)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        row_heights=[0.5, 0.5],
    )
    for column, label, color in [
        ("top_1_share_3m", "Top 1", ACCENT),
        ("top_5_share_3m", "Top 5", POSITIVE),
        ("top_10_share_3m", "Top 10", NEGATIVE),
    ]:
        fig.add_trace(
            go.Scatter(
                x=dominance["date"],
                y=dominance[column],
                mode="lines",
                name=label,
                line={"width": 2.8, "color": color},
            ),
            row=1,
            col=1,
        )
    for column, label, color in [
        ("effective_hhi", "Inverse HHI", "#6D597A"),
        ("effective_entropy", "Exp. Shannon", "#E76F51"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=diversity["date"],
                y=diversity[column],
                mode="lines",
                name=label,
                line={"width": 2.8, "color": color},
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title="Leader share", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title="Effective ecosystems", row=2, col=1)
    fig.update_xaxes(title=None)
    return apply_report_theme(
        fig,
        "Concentration and effective ecosystem diversity",
        "Three-month smoothed dominance shares and diversity indices.",
        height=720,
    )


def specialization_figure(activity: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    languages = ["JavaScript", "Python", "TypeScript", "Java", "Go", "Rust", "Jupyter Notebook", "Ruby"]
    frame = activity_specialization(activity, languages)
    matrix = frame.pivot(index="language", columns="metric", values="over_index").reindex(languages)
    metric_order = METRICS
    fig = go.Figure(
        go.Heatmap(
            z=matrix[metric_order].to_numpy(),
            x=[metric.replace("_", " ") for metric in metric_order],
            y=matrix.index,
            colorscale=[
                [0, "#F5EFE6"],
                [0.5, "#F7C873"],
                [1, "#B2472D"],
            ],
            zmid=1,
            colorbar={"title": "Over-index"},
            text=np.round(matrix[metric_order].to_numpy(), 1),
            texttemplate="%{text}",
        )
    )
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(title=None)
    return apply_report_theme(
        fig,
        "Activity specialization by ecosystem",
        "Values above 1 mean the metric is stronger than that language's average profile.",
        height=620,
    )


def _embedding_axis_range(values: pd.Series, pad_fraction: float = 0.16) -> list[float]:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    if span <= 0:
        span = 1.0
    pad = span * pad_fraction
    return [minimum - pad, maximum + pad]


def embedding_method_figure(
    embeddings: pd.DataFrame,
    profile: str,
    method: str,
    colors: dict[str, str],
    label_count: int = 10,
) -> go.Figure:
    subset = embeddings.loc[
        (embeddings["profile"] == profile) & (embeddings["method"] == method)
    ].copy()
    subset["functional_group"] = subset["language"].map(language_group)
    subset["point_color"] = np.where(
        subset["rank"] <= 30,
        subset["language"].map(colors).fillna(NEUTRAL),
        subset["functional_group"].map(LANGUAGE_GROUP_COLORS),
    )
    size_values = np.log1p(subset["mean_composite_share"])
    subset["size"] = np.interp(size_values, (size_values.min(), size_values.max()), (6, 22))

    fig = go.Figure()
    tail = subset.loc[subset["rank"] > 30]
    named_leaders = subset.loc[subset["rank"] <= label_count].sort_values("rank")
    other_leaders = subset.loc[(subset["rank"] > label_count) & (subset["rank"] <= 30)]
    group_order = [
        "Infra / config",
        "Markup & templates",
        "ML / scientific",
        "Document / typesetting",
        "Web styling",
        "General-purpose",
    ]
    for group in group_order:
        group_tail = tail.loc[tail["functional_group"] == group]
        if group_tail.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group_tail["x"],
                y=group_tail["y"],
                mode="markers",
                marker={
                    "size": group_tail["size"],
                    "color": LANGUAGE_GROUP_COLORS[group],
                    "opacity": 0.82,
                    "line": {"width": 0},
                },
                name=group,
                showlegend=True,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=other_leaders["x"],
            y=other_leaders["y"],
            mode="markers",
            marker={
                "size": other_leaders["size"],
                "color": other_leaders["point_color"],
                "opacity": 0.78,
                "line": {"width": 0.6, "color": "#FFFFFF"},
            },
            name="Ranks 11-30",
            showlegend=False,
            )
        )
    label_offsets = [
        (40, -28),
        (-48, -26),
        (-56, 4),
        (-44, 34),
        (42, 2),
        (-16, 50),
        (44, 34),
        (2, 66),
        (62, -22),
        (64, 52),
    ]
    for _, row in named_leaders.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["x"]],
                y=[row["y"]],
                mode="markers",
                marker={
                    "size": max(float(row["size"]), 16),
                    "color": colors.get(row["language"], TEXT),
                    "opacity": 0.96,
                    "line": {"width": 1.2, "color": "#FFFFFF"},
                },
                name=f"{int(row['rank'])}. {row['language']}",
                showlegend=False,
            )
        )
        offset = label_offsets[int(row["rank"]) - 1] if int(row["rank"]) <= len(label_offsets) else (0, 24)
        fig.add_annotation(
            x=row["x"],
            y=row["y"],
            text=row["language"],
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.7,
            arrowcolor=colors.get(row["language"], TEXT),
            ax=offset[0],
            ay=offset[1],
            font={"size": 12, "color": colors.get(row["language"], TEXT)},
        )

    fig.update_xaxes(
        visible=False,
        range=_embedding_axis_range(subset["x"], pad_fraction=0.12),
    )
    fig.update_yaxes(
        visible=False,
        range=_embedding_axis_range(subset["y"], pad_fraction=0.12),
    )
    fig = apply_report_theme(
        fig,
        f"{method} projection for {profile.lower()} vectors",
        "Top-ranked labels keep individual colors; the long tail is tinted by functional group.",
        height=760,
        legend_title="Functional group",
    )
    fig.update_layout(
        margin={"l": 55, "r": 230, "t": 125, "b": 55},
        legend={
            "title": {"text": "Functional group"},
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.82)",
            "bordercolor": GRID,
            "borderwidth": 1,
            "font": {"size": 13},
        },
    )
    return fig


def embedding_score_table(scores: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / "embedding_scores.tex"
    rows = []
    for _, row in scores.sort_values(["profile", "method"]).iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["profile"]),
                    str(row["method"]),
                    f"{row['trustworthiness']:.3f}",
                    f"{row['knn_preservation']:.3f}",
                    f"{row['distance_spearman']:.3f}",
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{llrrr}",
                r"\toprule",
                r"Profile & Method & Trustworthiness & k-NN preservation & Distance Spearman \\",
                r"\midrule",
                body,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scores.to_csv(TABLE_DIR / "embedding_scores.csv", index=False)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    _raw, activity, ranking, colors = prepare_data()

    export_figure(sampling_figure(activity, ranking, colors), "sampling_comparability", height=720)
    export_figure(composite_trend_figure(activity, ranking, colors), "composite_trends")
    export_figure(rank_change_figure(activity, ranking), "rank_slopegraph")
    export_figure(winners_decliners_figure(activity), "winners_decliners", height=620)
    export_figure(community_momentum_figure(activity, colors), "community_momentum")
    export_figure(specialization_figure(activity, colors), "activity_specialization", height=620)
    export_figure(diversity_concentration_figure(activity), "diversity_concentration", height=720)

    top_150 = ranking.head(150)["language"].tolist()
    profile_vectors = build_profile_vectors(activity, top_150)
    embeddings = run_all_embeddings(profile_vectors, ranking, seed=42)
    scores = evaluate_embeddings(profile_vectors, embeddings, n_neighbors=10)
    embedding_score_table(scores)
    for profile, profile_slug in [
        ("All activity", "all_activity"),
        ("Contribution", "contribution"),
        ("Reach / community", "reach_community"),
    ]:
        for method, method_slug in [
            ("UMAP", "umap"),
            ("TriMAP", "trimap"),
            ("PaCMAP", "pacmap"),
        ]:
            export_figure(
                embedding_method_figure(embeddings, profile, method, colors),
                f"embedding_{profile_slug}_{method_slug}",
                width=1300,
                height=760,
            )


if __name__ == "__main__":
    main()
