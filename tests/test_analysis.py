from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from lang_ecosystem.analysis import (
    ACTIVITY_PROFILES,
    METRICS,
    add_monthly_shares,
    add_trailing_shares,
    activity_specialization,
    aggregate_period_shares,
    build_activity_vectors,
    build_profile_vectors,
    category_composition,
    complete_month_grid,
    dominance_turnover,
    ecosystem_diversity,
    embed_activity_vectors,
    ecosystem_momentum,
    evaluate_embeddings,
    filter_label_scope,
    load_activity_data,
    rank_stability,
    rank_languages,
    ranking_trajectories,
    run_all_embeddings,
    signal_rank_agreement,
    top_k_dominance,
)
from lang_ecosystem.visuals import (
    CHART_LIMITS,
    FIXED_LANGUAGE_COLORS,
    LANGUAGE_GROUP_COLORS,
    NEUTRAL,
    activity_specialization_figure,
    category_composition_figure,
    composite_trend_figure,
    diversity_figure,
    dominance_turnover_figure,
    ecosystem_momentum_figure,
    language_color_map,
    metric_trend_figure,
    projection_method_figure,
    rank_stability_figure,
    ranking_explorer_figure,
    signal_agreement_figure,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "github_language_activity_monthly.csv"


@pytest.fixture(scope="module")
def prepared_data():
    raw = load_activity_data(DATA_PATH)
    dense = complete_month_grid(raw)
    shares = add_monthly_shares(dense)
    shares = add_trailing_shares(shares)
    ranking = rank_languages(shares)
    return raw, dense, shares, ranking


def test_monthly_metric_shares_sum_to_one(prepared_data):
    _, _, shares, _ = prepared_data
    share_columns = [f"{metric}_share" for metric in METRICS]
    totals = shares.groupby("date")[share_columns].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-12)


def test_aliases_merge_without_duplicate_language_months(prepared_data):
    raw, _, _, _ = prepared_data
    assert not raw.duplicated(["language", "date"]).any()
    assert "Vim script" not in set(raw["language"])
    assert "Vim Script" in set(raw["language"])
    assert "Matlab" not in set(raw["language"])
    assert "MATLAB" in set(raw["language"])


def test_complete_grid_fills_absent_activity_with_zero(prepared_data):
    raw, dense, _, _ = prepared_data
    expected_rows = raw["language"].nunique() * raw["date"].nunique()
    assert len(dense) == expected_rows

    observed_dates = set(raw.loc[raw["language"] == "Rocq Prover", "date"])
    absent_row = dense.loc[
        (dense["language"] == "Rocq Prover")
        & (~dense["date"].isin(observed_dates))
    ].iloc[0]
    assert absent_row[METRICS].sum() == 0


def test_top_100_vector_dimensions_and_order(prepared_data):
    _, _, shares, ranking = prepared_data
    languages = ranking.head(100)["language"].tolist()
    vectors = build_profile_vectors(shares, languages)

    assert vectors["All activity"].shape == (100, 9 * 114)
    assert vectors["Contribution"].shape == (100, 5 * 114)
    assert vectors["Reach / community"].shape == (100, 4 * 114)
    assert vectors["All activity"].index.tolist() == languages

    columns = vectors["All activity"].columns
    assert columns.names == ["metric", "date"]
    assert columns[:114].get_level_values("metric").unique().tolist() == ["pushes"]
    assert columns[114:228].get_level_values("metric").unique().tolist() == [
        "pull_requests"
    ]
    dates = columns[:114].get_level_values("date")
    assert dates.is_monotonic_increasing


def test_fixed_top_30_colors_and_neutral_long_tail(prepared_data):
    _, _, _, ranking = prepared_data
    languages = ranking.head(100)["language"].tolist()
    colors_a = language_color_map(languages)
    colors_b = language_color_map(languages)

    assert colors_a == colors_b
    assert colors_a["JavaScript"] == FIXED_LANGUAGE_COLORS["JavaScript"]
    assert len(set(colors_a[language] for language in languages[:30])) == 30
    assert all(colors_a[language] == NEUTRAL for language in languages[30:])


def test_semantic_group_colors_for_embedding_tail(prepared_data):
    _, _, _, ranking = prepared_data
    languages = ranking.head(150)["language"].tolist()
    colors = language_color_map(languages, use_semantic_groups=True)

    assert colors["JavaScript"] == FIXED_LANGUAGE_COLORS["JavaScript"]
    assert colors["TeX"] == LANGUAGE_GROUP_COLORS["Document / typesetting"]
    assert colors["Roff"] == LANGUAGE_GROUP_COLORS["Document / typesetting"]
    assert colors["Smarty"] == LANGUAGE_GROUP_COLORS["Markup & templates"]
    assert colors["MATLAB"] == LANGUAGE_GROUP_COLORS["ML / scientific"]
    assert colors["CMake"] == LANGUAGE_GROUP_COLORS["Infra / config"]
    assert colors["Less"] == LANGUAGE_GROUP_COLORS["Web styling"]
    assert colors["Elixir"] == LANGUAGE_GROUP_COLORS["General-purpose"]


def test_projection_method_figure_adds_semantic_group_legend(prepared_data):
    _, _, shares, ranking = prepared_data
    languages = ranking.head(150)["language"].tolist()
    vectors = build_profile_vectors(shares, languages)
    embeddings = run_all_embeddings(
        vectors,
        ranking,
        seed=42,
        trimap_iterations=40,
        pacmap_iterations=(12, 12, 24),
    )
    colors = language_color_map(languages, use_semantic_groups=True)
    figure = projection_method_figure(
        embeddings,
        colors,
        "UMAP",
        use_semantic_groups=True,
    )
    legend_names = [trace.name for trace in figure.data if trace.showlegend]
    assert "Document / typesetting" in legend_names
    assert "General-purpose" in legend_names


def test_reducers_return_one_finite_point_per_language(prepared_data):
    _, _, shares, ranking = prepared_data
    languages = ranking.head(30)["language"].tolist()
    vectors = build_activity_vectors(shares, languages, ACTIVITY_PROFILES["All activity"])
    embeddings = embed_activity_vectors(
        vectors,
        seed=42,
        trimap_iterations=60,
        pacmap_iterations=(20, 20, 40),
    )

    assert set(embeddings) == {"UMAP", "TriMAP", "PaCMAP"}
    for embedding in embeddings.values():
        assert embedding.shape == (30, 2)
        assert np.isfinite(embedding).all()


def test_reducers_are_reproducible_with_a_fixed_seed(prepared_data):
    _, _, shares, ranking = prepared_data
    languages = ranking.head(18)["language"].tolist()
    vectors = build_activity_vectors(shares, languages, ACTIVITY_PROFILES["All activity"])
    parameters = {
        "seed": 42,
        "trimap_iterations": 40,
        "pacmap_iterations": (12, 12, 24),
    }

    first = embed_activity_vectors(vectors, **parameters)
    second = embed_activity_vectors(vectors, **parameters)

    for method in first:
        np.testing.assert_allclose(first[method], second[method], atol=1e-7)


def test_embedding_quality_metrics_are_finite(prepared_data):
    _, _, shares, ranking = prepared_data
    languages = ranking.head(24)["language"].tolist()
    vectors = {
        "All activity": build_activity_vectors(
            shares, languages, ACTIVITY_PROFILES["All activity"]
        )
    }
    embedding_frame = run_all_embeddings(
        vectors,
        ranking,
        seed=42,
        trimap_iterations=50,
        pacmap_iterations=(15, 15, 30),
    )
    scores = evaluate_embeddings(vectors, embedding_frame, n_neighbors=5)

    assert len(embedding_frame) == 24 * 3
    assert len(scores) == 3
    assert np.isfinite(
        scores[
            ["trustworthiness", "knn_preservation", "distance_spearman"]
        ].to_numpy()
    ).all()
    assert (
        scores[["trustworthiness", "knn_preservation"]].to_numpy() >= 0
    ).all()
    assert (
        scores[["trustworthiness", "knn_preservation"]].to_numpy() <= 1
    ).all()


def test_trailing_windows_use_only_current_and_prior_months(prepared_data):
    _, _, shares, _ = prepared_data
    python = shares.loc[shares["language"] == "Python"].sort_values("date")
    expected_3m = python["composite_share"].iloc[:3].mean()
    expected_12m = python["composite_share"].iloc[:12].mean()
    assert python["composite_share_3m"].iloc[2] == pytest.approx(expected_3m)
    assert python["composite_share_12m"].iloc[11] == pytest.approx(expected_12m)
    assert python["composite_share_smooth"].equals(python["composite_share_3m"])


def test_period_aggregation_scope_ties_and_partial_year(prepared_data):
    _, _, shares, _ = prepared_data
    annual = aggregate_period_shares(shares, "Pull Requests", "Year")
    assert annual["period"].nunique() == 10
    assert annual.loc[annual["period"].dt.year == 2025, "months"].max() == 6

    technologies = aggregate_period_shares(
        shares, "Composite", "Quarter", "Technology / artifacts"
    )
    assert set(technologies["category"]) == {"Technology / artifact"}
    assert technologies["period"].dt.month.isin([1, 4, 7, 10]).all()

    tied = shares.loc[
        (shares["date"] == shares["date"].min())
        & shares["language"].isin(["Python", "Java"])
    ].copy()
    tied["composite_share"] = 0.5
    ranked = aggregate_period_shares(tied, "Composite", "Month")
    assert set(ranked["rank"]) == {1.0}


def test_dynamic_rankings_and_turnover_respect_limits(prepared_data):
    _, _, shares, _ = prepared_data
    trajectories = ranking_trajectories(
        shares, "Issue Comments", "Quarter", endpoint_count=20
    )
    assert trajectories["language"].nunique() <= 40
    assert trajectories["language"].nunique() >= 20
    assert trajectories["granularity"].eq("Quarter").all()
    period_count = aggregate_period_shares(
        shares, "Issue Comments", "Quarter"
    )["period"].nunique()
    assert trajectories.groupby("language")["period"].nunique().eq(period_count).all()
    assert trajectories["rank"].max() > 20

    turnover = dominance_turnover(shares, "Issues", "Year", top_k=10)
    assert turnover["rank"].le(10).all()
    assert turnover["language"].nunique() >= 10


def test_structural_analysis_helpers(prepared_data):
    _, _, shares, _ = prepared_data

    composition = category_composition(shares, "Composite", window=3)
    raw_totals = composition.groupby("date")["raw_share"].sum()
    np.testing.assert_allclose(raw_totals, 1.0)
    languages = composition.loc[
        composition["category"] == "Programming language"
    ].sort_values("date")
    assert languages["share"].iloc[2] == pytest.approx(
        languages["raw_share"].iloc[:3].mean()
    )

    agreement = signal_rank_agreement(
        shares, "Composite", "Programming languages", window=3
    )
    assert agreement["signal"].nunique() == 11
    assert "Composite" not in set(agreement["signal"])
    assert agreement["agreement"].between(-1, 1).all()

    stability = rank_stability(
        shares, "Stars", "Quarter", "Technology / artifacts", count=40
    )
    assert len(stability) == 25
    assert set(stability["category"]) == {"Technology / artifact"}
    assert (stability["rank_volatility"] >= 0).all()

    diversity = ecosystem_diversity(
        shares, "Composite", "Programming languages", window=3
    )
    assert np.isfinite(
        diversity[
            [
                "effective_hhi",
                "effective_entropy",
                "raw_effective_hhi",
                "raw_effective_entropy",
            ]
        ].to_numpy()
    ).all()
    assert (
        diversity["raw_effective_hhi"] <= diversity["raw_effective_entropy"] + 1e-12
    ).all()
    assert (diversity[["effective_hhi", "effective_entropy"]] >= 1).all().all()


def test_scope_momentum_specialization_and_dominance(prepared_data):
    _, _, shares, ranking = prepared_data
    languages = filter_label_scope(shares, "Programming languages")
    assert set(languages["category"]) == {"Programming language"}

    momentum = ecosystem_momentum(shares, "Composite", count=40)
    assert len(momentum) == 40
    np.testing.assert_allclose(
        momentum["change"],
        momentum["current_share"] - momentum["previous_share"],
    )

    leaders = ranking.head(40)["language"].tolist()
    specialization = activity_specialization(shares, leaders)
    assert len(specialization) == 40 * len(METRICS)
    relative_means = specialization.groupby("language")["over_index"].mean()
    np.testing.assert_allclose(relative_means, 1.0)

    dominance = top_k_dominance(shares)
    assert dominance.columns.tolist() == [
        "date",
        "top_1_share",
        "top_5_share",
        "top_10_share",
    ]
    assert (dominance["top_1_share"] <= dominance["top_5_share"]).all()
    assert (dominance["top_5_share"] <= dominance["top_10_share"]).all()


def test_interactive_figures_have_control_bands_and_expected_limits(prepared_data):
    _, _, shares, ranking = prepared_data
    top_150 = ranking.head(CHART_LIMITS["embeddings"])["language"].tolist()
    colors = language_color_map(top_150)

    composite = composite_trend_figure(
        shares, top_150[: CHART_LIMITS["trajectories"]], colors
    )
    metrics = metric_trend_figure(
        shares, top_150[: CHART_LIMITS["trajectories"]], colors
    )
    ranking_figure = ranking_explorer_figure(shares, colors)
    turnover = dominance_turnover_figure(shares)
    momentum = ecosystem_momentum_figure(shares, colors)
    specialization = activity_specialization_figure(shares)

    assert len(composite.data) == CHART_LIMITS["trajectories"]
    assert len(metrics.layout.updatemenus) == 2
    assert len(ranking_figure.layout.updatemenus) == 2
    assert len(turnover.layout.updatemenus) == 2
    assert len(momentum.layout.updatemenus) == 2
    assert len(specialization.layout.updatemenus) == 2
    for figure in [
        composite,
        metrics,
        ranking_figure,
        turnover,
        momentum,
        specialization,
    ]:
        assert figure.layout.margin.t == 145
        assert figure.layout.title.yref == "container"
        assert figure.layout.title.y == pytest.approx(0.96)
        assert figure.layout.title.yanchor == "top"
        plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
        assert all(menu.yanchor == "bottom" for menu in figure.layout.updatemenus)
        assert all(menu.pad.t == 0 for menu in figure.layout.updatemenus)
        assert all(
            (menu.y - 1) * plot_height == pytest.approx(12)
            for menu in figure.layout.updatemenus
        )
        assert all(annotation.yanchor == "middle" for annotation in figure.layout.annotations)
        assert all(
            (annotation.y - 1) * plot_height == pytest.approx(60)
            for annotation in figure.layout.annotations
        )
        pio.to_json(figure)

    assert ranking_figure.layout.yaxis.autorange == "reversed"
    visible_names = {
        trace.name for trace in ranking_figure.data if trace.visible is True
    }
    assert len(visible_names) >= CHART_LIMITS["trajectory_endpoints"]
    assert "" not in visible_names
    assert all(trace.opacity is None for trace in ranking_figure.data)
    assert sum(trace.visible is True for trace in turnover.data) == 1
    assert all(trace.opacity is None for trace in turnover.data)
    assert sum(trace.visible is True for trace in momentum.data) == 1
    assert all(trace.opacity is None for trace in momentum.data)
    assert sum(trace.visible is True for trace in specialization.data) == 1
    assert all(trace.opacity is None for trace in specialization.data)
    assert "Makefile" not in set(specialization.data[1].y)
    assert set(specialization.data[1].y).issubset(
        set(
            filter_label_scope(shares, "Programming languages")[
                "language"
            ].unique()
        )
    )


def test_new_structural_figures_are_interactive_and_serializable(prepared_data):
    _, _, shares, ranking = prepared_data
    colors = language_color_map(ranking.head(150)["language"].tolist())
    figures = [
        category_composition_figure(shares),
        signal_agreement_figure(shares),
        rank_stability_figure(shares, colors),
        diversity_figure(shares),
    ]
    for figure in figures:
        assert len(figure.layout.updatemenus) == 2
        assert figure.layout.margin.t == 145
        assert all(menu.yanchor == "bottom" for menu in figure.layout.updatemenus)
        assert all(trace.opacity is None for trace in figure.data)
        pio.to_json(figure)

    category, agreement, stability, diversity = figures
    assert sum(trace.visible is True for trace in category.data) == 2
    assert sum(trace.visible is True for trace in agreement.data) == 1
    assert sum(trace.visible is True for trace in stability.data) == 1
    assert sum(trace.visible is True for trace in diversity.data) == 2


def test_projection_method_figure_is_full_width_and_profile_selectable(
    prepared_data,
):
    _, _, shares, ranking = prepared_data
    languages = ranking.head(18)["language"].tolist()
    vectors = build_profile_vectors(shares, languages)
    embeddings = run_all_embeddings(
        vectors,
        ranking,
        seed=42,
        trimap_iterations=40,
        pacmap_iterations=(12, 12, 24),
    )
    figure = projection_method_figure(
        embeddings, language_color_map(languages), "UMAP", label_count=5
    )
    assert len(figure.data) == len(ACTIVITY_PROFILES)
    assert len(figure.layout.updatemenus) == 1
    assert figure.layout.height == 760
    assert "activity-profile similarity" in figure.layout.title.text
    assert len(figure.layout.updatemenus[0].buttons[0].args) == 1
