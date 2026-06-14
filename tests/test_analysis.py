from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lang_ecosystem.analysis import (
    ACTIVITY_PROFILES,
    METRICS,
    add_monthly_shares,
    build_activity_vectors,
    build_profile_vectors,
    complete_month_grid,
    embed_activity_vectors,
    evaluate_embeddings,
    load_activity_data,
    rank_languages,
    run_all_embeddings,
)
from lang_ecosystem.visuals import FIXED_LANGUAGE_COLORS, NEUTRAL, language_color_map

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "github_language_activity_monthly.csv"


@pytest.fixture(scope="module")
def prepared_data():
    raw = load_activity_data(DATA_PATH)
    dense = complete_month_grid(raw)
    shares = add_monthly_shares(dense)
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
