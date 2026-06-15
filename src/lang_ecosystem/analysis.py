"""Data preparation and dimensionality-reduction utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pacmap
import trimap
import umap
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

METRICS = [
    "pushes",
    "pull_requests",
    "issues",
    "issue_comments",
    "stars",
    "forks",
    "creates",
    "contributors",
    "active_repos",
]

EVENT_METRICS = [
    "pushes",
    "pull_requests",
    "issues",
    "issue_comments",
    "stars",
    "forks",
    "creates",
]

CONTRIBUTION_METRICS = [
    "pushes",
    "pull_requests",
    "issues",
    "issue_comments",
    "creates",
]

COMMUNITY_METRICS = [
    "stars",
    "forks",
    "contributors",
    "active_repos",
]

SCORE_COLUMNS = {
    "Composite": "composite_share",
    "Contribution": "contribution_share",
    "Community": "community_share",
    **{metric.replace("_", " ").title(): f"{metric}_share" for metric in METRICS},
}

LABEL_SCOPES = {
    "All labels": None,
    "Programming languages": "Programming language",
    "Technology / artifacts": "Technology / artifact",
}

PERIOD_FREQUENCIES = {
    "Month": "MS",
    "Quarter": "QS",
    "Year": "YS",
}

ACTIVITY_PROFILES = {
    "All activity": METRICS,
    "Contribution": CONTRIBUTION_METRICS,
    "Reach / community": COMMUNITY_METRICS,
}

# These are spelling/case aliases, not semantic remappings.
LANGUAGE_ALIASES = {
    "Ecl": "ECL",
    "FORTRAN": "Fortran",
    "FreeBasic": "FreeBASIC",
    "HaXe": "Haxe",
    "Matlab": "MATLAB",
    "PAWN": "Pawn",
    "Perl6": "Perl 6",
    "Vim script": "Vim Script",
    "wdl": "WDL",
}

TECHNOLOGY_LABELS = {
    "ASP.NET",
    "Blade",
    "CMake",
    "CSS",
    "Dockerfile",
    "EJS",
    "Gherkin",
    "HCL",
    "HTML",
    "Handlebars",
    "Jinja",
    "Jupyter Notebook",
    "Less",
    "MDX",
    "Makefile",
    "Markdown",
    "Nunjucks",
    "Pug",
    "Rich Text Format",
    "Roff",
    "SCSS",
    "Smarty",
    "Svelte",
    "TeX",
    "Vue",
}


def _validate_columns(data: pd.DataFrame) -> None:
    required = {"language", "year", "month", *METRICS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_activity_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV, merge spelling aliases, and aggregate any collisions."""
    data = pd.read_csv(path)
    _validate_columns(data)
    data = data.copy()
    data["language"] = data["language"].replace(LANGUAGE_ALIASES)
    data["date"] = pd.to_datetime(
        {"year": data["year"], "month": data["month"], "day": 1}
    )
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="raise")

    group_columns = ["language", "year", "month", "date"]
    data = (
        data.groupby(group_columns, as_index=False, sort=True)[METRICS]
        .sum()
        .sort_values(["date", "language"], ignore_index=True)
    )
    if data.duplicated(["language", "date"]).any():
        raise ValueError("Alias cleanup left duplicate language-month rows")
    return data


def complete_month_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Create the full language x month grid and fill absent activity with zero."""
    _validate_columns(data)
    if "date" not in data:
        data = data.copy()
        data["date"] = pd.to_datetime(
            {"year": data["year"], "month": data["month"], "day": 1}
        )

    months = pd.date_range(data["date"].min(), data["date"].max(), freq="MS")
    languages = np.sort(data["language"].unique())
    complete_index = pd.MultiIndex.from_product(
        [languages, months], names=["language", "date"]
    )
    dense = (
        data.set_index(["language", "date"])[METRICS]
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    dense["year"] = dense["date"].dt.year
    dense["month"] = dense["date"].dt.month
    return dense[["language", "year", "month", "date", *METRICS]]


def add_monthly_shares(data: pd.DataFrame) -> pd.DataFrame:
    """Add within-month metric shares and equal-weight profile scores."""
    _validate_columns(data)
    if "date" not in data:
        raise ValueError("A date column is required; call complete_month_grid first")

    result = data.copy()
    monthly_totals = result.groupby("date")[METRICS].transform("sum")
    if (monthly_totals == 0).any().any():
        zero_metrics = monthly_totals.columns[(monthly_totals == 0).any()].tolist()
        raise ValueError(f"Metrics with zero monthly totals: {zero_metrics}")

    share_columns = []
    for metric in METRICS:
        share_column = f"{metric}_share"
        result[share_column] = result[metric] / monthly_totals[metric]
        share_columns.append(share_column)

    result["composite_share"] = result[share_columns].mean(axis=1)
    result["contribution_share"] = result[
        [f"{metric}_share" for metric in CONTRIBUTION_METRICS]
    ].mean(axis=1)
    result["community_share"] = result[
        [f"{metric}_share" for metric in COMMUNITY_METRICS]
    ].mean(axis=1)
    result["category"] = result["language"].map(classify_language)
    return result


def add_smoothed_shares(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    window: int = 3,
    center: bool = False,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Add rolling means while retaining the unsmoothed columns."""
    if window < 1:
        raise ValueError("window must be at least 1")
    columns = list(
        columns
        or [
            *(f"{metric}_share" for metric in METRICS),
            "composite_share",
            "contribution_share",
            "community_share",
        ]
    )
    result = data.sort_values(["language", "date"]).copy()
    smoothed = result.groupby("language", sort=False)[columns].transform(
        lambda values: values.rolling(
            window=window, center=center, min_periods=1
        ).mean()
    )
    suffix = suffix or ("smooth" if window == 3 else f"{window}m")
    smoothed.columns = [f"{column}_{suffix}" for column in columns]
    return pd.concat([result, smoothed], axis=1)


def add_trailing_shares(
    data: pd.DataFrame,
    windows: Sequence[int] = (3, 12),
) -> pd.DataFrame:
    """Add trailing rolling means for every share and composite score."""
    result = data.copy()
    for window in windows:
        result = add_smoothed_shares(
            result,
            window=window,
            center=False,
            suffix=f"{window}m",
        )
    if 3 in windows:
        share_columns = [
            *(f"{metric}_share" for metric in METRICS),
            "composite_share",
            "contribution_share",
            "community_share",
        ]
        for column in share_columns:
            result[f"{column}_smooth"] = result[f"{column}_3m"]
    return result


def filter_label_scope(data: pd.DataFrame, scope: str = "All labels") -> pd.DataFrame:
    """Filter a prepared frame to one of the supported label categories."""
    if scope not in LABEL_SCOPES:
        raise ValueError(f"Unknown label scope: {scope}")
    category = LABEL_SCOPES[scope]
    if category is None:
        return data.copy()
    if "category" not in data:
        raise ValueError("A category column is required for scoped analysis")
    return data.loc[data["category"] == category].copy()


def aggregate_period_shares(
    data: pd.DataFrame,
    score: str = "Composite",
    granularity: str = "Year",
    scope: str = "All labels",
) -> pd.DataFrame:
    """Aggregate monthly shares to periods and rank all labels in each period."""
    if score not in SCORE_COLUMNS:
        raise ValueError(f"Unknown score: {score}")
    if granularity not in PERIOD_FREQUENCIES:
        raise ValueError(f"Unknown granularity: {granularity}")
    subset = filter_label_scope(data, scope)
    score_column = SCORE_COLUMNS[score]
    if score_column not in subset:
        raise ValueError(f"Missing score column: {score_column}")

    subset["period"] = subset["date"].dt.to_period(
        {"Month": "M", "Quarter": "Q", "Year": "Y"}[granularity]
    ).dt.start_time
    period = (
        subset.groupby(["period", "language"], as_index=False)
        .agg(
            share=(score_column, "mean"),
            months=("date", "nunique"),
            category=("category", "first"),
        )
        .sort_values(["period", "share", "language"], ascending=[True, False, True])
    )
    period["rank"] = period.groupby("period")["share"].rank(
        method="min", ascending=False
    )
    period["score"] = score
    period["granularity"] = granularity
    period["scope"] = scope
    return period.reset_index(drop=True)


def ranking_trajectories(
    data: pd.DataFrame,
    score: str = "Composite",
    granularity: str = "Year",
    scope: str = "All labels",
    endpoint_count: int = 15,
) -> pd.DataFrame:
    """Return period ranks for languages in the start/end top-k union."""
    period = aggregate_period_shares(data, score, granularity, scope)
    first_period = period["period"].min()
    last_period = period["period"].max()

    def top_at(when: pd.Timestamp) -> list[str]:
        return (
            period.loc[period["period"] == when]
            .nsmallest(endpoint_count, "rank")["language"]
            .tolist()
        )

    leaders = list(dict.fromkeys(top_at(first_period) + top_at(last_period)))
    return period.loc[period["language"].isin(leaders)].copy()


def dominance_turnover(
    data: pd.DataFrame,
    score: str = "Composite",
    granularity: str = "Year",
    scope: str = "All labels",
    top_k: int = 10,
) -> pd.DataFrame:
    """Return the union of labels that enter the top-k in any period."""
    period = aggregate_period_shares(data, score, granularity, scope)
    turnover = period.loc[period["rank"] <= top_k].copy()
    order = (
        turnover.groupby("language")
        .agg(best_rank=("rank", "min"), first_period=("period", "min"))
        .sort_values(["best_rank", "first_period"])
        .index
    )
    turnover["language"] = pd.Categorical(
        turnover["language"], categories=order, ordered=True
    )
    return turnover.sort_values(["language", "period"]).reset_index(drop=True)


def category_composition(
    data: pd.DataFrame,
    score: str = "Composite",
    window: int = 3,
) -> pd.DataFrame:
    """Return category shares over time with optional trailing smoothing."""
    if score not in SCORE_COLUMNS:
        raise ValueError(f"Unknown score: {score}")
    if window < 1:
        raise ValueError("window must be at least 1")
    score_column = SCORE_COLUMNS[score]
    frame = (
        data.groupby(["date", "category"], as_index=False)[score_column]
        .sum()
        .rename(columns={score_column: "raw_share"})
        .sort_values(["category", "date"])
    )
    frame["share"] = frame.groupby("category", sort=False)["raw_share"].transform(
        lambda values: values.rolling(window=window, min_periods=1).mean()
    )
    frame["score"] = score
    frame["window"] = window
    return frame.reset_index(drop=True)


def signal_rank_agreement(
    data: pd.DataFrame,
    reference_score: str = "Composite",
    scope: str = "All labels",
    window: int = 3,
) -> pd.DataFrame:
    """Compare one signal's monthly ranking with every other signal."""
    if reference_score not in SCORE_COLUMNS:
        raise ValueError(f"Unknown score: {reference_score}")
    if window < 1:
        raise ValueError("window must be at least 1")
    subset = filter_label_scope(data, scope)
    score_names = list(SCORE_COLUMNS)
    rows = []
    for date, monthly in subset.groupby("date", sort=True):
        reference = monthly[SCORE_COLUMNS[reference_score]]
        for score in score_names:
            if score == reference_score:
                continue
            agreement = reference.corr(
                monthly[SCORE_COLUMNS[score]], method="spearman"
            )
            rows.append(
                {
                    "date": date,
                    "signal": score,
                    "raw_agreement": float(agreement),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["signal", "date"])
    frame["agreement"] = frame.groupby("signal", sort=False)[
        "raw_agreement"
    ].transform(lambda values: values.rolling(window=window, min_periods=1).mean())
    frame["reference_score"] = reference_score
    frame["scope"] = scope
    frame["window"] = window
    return frame.reset_index(drop=True)


def rank_stability(
    data: pd.DataFrame,
    score: str = "Composite",
    granularity: str = "Quarter",
    scope: str = "All labels",
    count: int = 40,
) -> pd.DataFrame:
    """Summarize prominence and rank volatility for leading ecosystems."""
    if count < 1:
        raise ValueError("count must be at least 1")
    period = aggregate_period_shares(data, score, granularity, scope)
    frame = (
        period.groupby("language", as_index=False)
        .agg(
            mean_share=("share", "mean"),
            mean_rank=("rank", "mean"),
            rank_volatility=("rank", "std"),
            best_rank=("rank", "min"),
            worst_rank=("rank", "max"),
            periods=("period", "nunique"),
            category=("category", "first"),
        )
        .sort_values(["mean_share", "language"], ascending=[False, True])
        .head(count)
    )
    frame["rank_volatility"] = frame["rank_volatility"].fillna(0)
    frame["score"] = score
    frame["granularity"] = granularity
    frame["scope"] = scope
    return frame.reset_index(drop=True)


def ecosystem_diversity(
    data: pd.DataFrame,
    score: str = "Composite",
    scope: str = "All labels",
    window: int = 3,
) -> pd.DataFrame:
    """Return effective ecosystem counts derived from HHI and entropy."""
    if score not in SCORE_COLUMNS:
        raise ValueError(f"Unknown score: {score}")
    if window < 1:
        raise ValueError("window must be at least 1")
    subset = filter_label_scope(data, scope)
    score_column = SCORE_COLUMNS[score]
    rows = []
    for date, monthly in subset.groupby("date", sort=True):
        shares = monthly[score_column].to_numpy(dtype=float)
        total = shares.sum()
        if total <= 0:
            raise ValueError(f"Non-positive scoped share total for {date:%Y-%m}")
        probabilities = shares / total
        positive = probabilities[probabilities > 0]
        hhi = float(np.square(probabilities).sum())
        entropy = float(-(positive * np.log(positive)).sum())
        rows.append(
            {
                "date": date,
                "raw_effective_hhi": 1.0 / hhi,
                "raw_effective_entropy": float(np.exp(entropy)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("date")
    for column in ["effective_hhi", "effective_entropy"]:
        frame[column] = frame[f"raw_{column}"].rolling(
            window=window, min_periods=1
        ).mean()
    frame["score"] = score
    frame["scope"] = scope
    frame["window"] = window
    return frame.reset_index(drop=True)


def ecosystem_momentum(
    data: pd.DataFrame,
    score: str = "Composite",
    scope: str = "All labels",
    window: int = 12,
    count: int = 40,
) -> pd.DataFrame:
    """Compare the latest window with the immediately preceding window."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if score not in SCORE_COLUMNS:
        raise ValueError(f"Unknown score: {score}")
    subset = filter_label_scope(data, scope)
    dates = np.sort(subset["date"].unique())
    if len(dates) < window * 2:
        raise ValueError("Not enough periods for two momentum windows")
    previous_dates = dates[-2 * window : -window]
    latest_dates = dates[-window:]
    score_column = SCORE_COLUMNS[score]
    previous = (
        subset.loc[subset["date"].isin(previous_dates)]
        .groupby("language")[score_column]
        .mean()
    )
    latest = (
        subset.loc[subset["date"].isin(latest_dates)]
        .groupby("language")[score_column]
        .mean()
    )
    frame = pd.concat(
        [previous.rename("previous_share"), latest.rename("current_share")],
        axis=1,
    ).fillna(0)
    frame["change"] = frame["current_share"] - frame["previous_share"]
    metadata = subset.groupby("language")["category"].first()
    frame["category"] = metadata
    frame = (
        frame.reset_index()
        .sort_values(["current_share", "language"], ascending=[False, True])
        .head(count)
    )
    frame["score"] = score
    frame["scope"] = scope
    return frame.reset_index(drop=True)


def activity_specialization(
    data: pd.DataFrame,
    languages: Sequence[str],
) -> pd.DataFrame:
    """Return absolute metric shares and each language's metric over-index."""
    columns = [f"{metric}_share" for metric in METRICS]
    absolute = (
        data.loc[data["language"].isin(languages)]
        .groupby("language")[columns]
        .mean()
        .reindex(languages)
    )
    absolute.columns = METRICS
    baseline = absolute.mean(axis=1).replace(0, np.nan)
    relative = absolute.div(baseline, axis=0)
    rows = []
    for language in absolute.index:
        for metric in METRICS:
            rows.append(
                {
                    "language": language,
                    "metric": metric,
                    "absolute_share": absolute.loc[language, metric],
                    "over_index": relative.loc[language, metric],
                }
            )
    return pd.DataFrame(rows)


def classify_language(language: str) -> str:
    """Classify GitHub labels for notebook filtering and hover context."""
    if language in TECHNOLOGY_LABELS:
        return "Technology / artifact"
    return "Programming language"


def rank_languages(data: pd.DataFrame) -> pd.DataFrame:
    """Rank labels by mean monthly equal-weight composite share."""
    required = {"language", "composite_share"}
    if missing := required.difference(data.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ranking = (
        data.groupby("language", as_index=False)
        .agg(
            mean_composite_share=("composite_share", "mean"),
            months_observed=("composite_share", lambda values: int((values > 0).sum())),
            category=("category", "first"),
        )
        .sort_values(
            ["mean_composite_share", "language"],
            ascending=[False, True],
            ignore_index=True,
        )
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def build_activity_vectors(
    data: pd.DataFrame,
    languages: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Build metric-major vectors with each metric's months in chronological order."""
    if not languages:
        raise ValueError("At least one language is required")
    if not metrics:
        raise ValueError("At least one metric is required")
    missing_metrics = set(metrics).difference(METRICS)
    if missing_metrics:
        raise ValueError(f"Unknown metrics: {sorted(missing_metrics)}")

    dates = pd.Index(np.sort(data["date"].unique()), name="date")
    blocks = []
    for metric in metrics:
        share_column = f"{metric}_share"
        if share_column not in data:
            raise ValueError(f"Missing share column: {share_column}")
        block = (
            data.pivot(index="language", columns="date", values=share_column)
            .reindex(index=list(languages), columns=dates, fill_value=0)
            .fillna(0)
        )
        block.columns = pd.MultiIndex.from_product(
            [[metric], dates], names=["metric", "date"]
        )
        blocks.append(block)

    vectors = pd.concat(blocks, axis=1)
    if vectors.isna().any().any():
        raise ValueError("Activity vectors contain missing values")
    return vectors.astype(float)


def build_profile_vectors(
    data: pd.DataFrame,
    languages: Sequence[str],
    profiles: Mapping[str, Sequence[str]] = ACTIVITY_PROFILES,
) -> dict[str, pd.DataFrame]:
    """Build one vector matrix for every configured activity profile."""
    return {
        profile: build_activity_vectors(data, languages, metrics)
        for profile, metrics in profiles.items()
    }


def _deterministic_trimap_inputs(
    matrix: np.ndarray,
    n_inliers: int,
    n_outliers: int,
    n_random: int,
    seed: int,
    weight_temperature: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce TriMAP triplet construction without unseeded parallel RNGs."""
    trimap_matrix = matrix.copy().astype(np.float32)
    used_svd = trimap_matrix.shape[1] > 100
    if used_svd:
        trimap_matrix -= np.mean(trimap_matrix, axis=0)
        trimap_matrix = TruncatedSVD(
            n_components=100, random_state=0
        ).fit_transform(trimap_matrix)
    else:
        trimap_matrix -= np.min(trimap_matrix)
        maximum = np.max(trimap_matrix)
        if maximum > 0:
            trimap_matrix /= maximum
        trimap_matrix -= np.mean(trimap_matrix, axis=0)
    trimap_matrix = trimap_matrix.astype(np.float32)

    n_samples = len(trimap_matrix)
    n_extra = min(n_inliers + 50, n_samples)
    neighbors_model = NearestNeighbors(
        n_neighbors=n_extra, metric="euclidean", algorithm="brute"
    ).fit(trimap_matrix)
    neighbor_distances, neighbors = neighbors_model.kneighbors(trimap_matrix)
    neighbors = neighbors.astype(np.int32)
    neighbor_distances = neighbor_distances.astype(np.float32)

    sigma = np.maximum(np.mean(neighbor_distances[:, 3:6], axis=1), 1e-10)
    probabilities = -np.square(neighbor_distances) / (
        sigma[:, np.newaxis] * sigma[neighbors]
    )
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_samples)
    triplet_rows: list[tuple[int, int, int]] = []
    weights: list[float] = []

    for anchor in range(n_samples):
        sorted_indices = np.argsort(-probabilities[anchor], kind="stable")
        for inlier_index in range(n_inliers):
            neighbor_position = sorted_indices[inlier_index + 1]
            similar = int(neighbors[anchor, neighbor_position])
            rejected = neighbors[anchor, sorted_indices[: inlier_index + 2]]
            candidates = np.setdiff1d(all_indices, rejected, assume_unique=False)
            sampled_outliers = rng.choice(
                candidates, size=n_outliers, replace=False
            )
            p_similar = probabilities[anchor, neighbor_position]
            for outlier in sampled_outliers:
                distance = np.linalg.norm(
                    trimap_matrix[anchor] - trimap_matrix[outlier]
                )
                p_outlier = -(distance**2) / (sigma[anchor] * sigma[outlier])
                triplet_rows.append((anchor, similar, int(outlier)))
                weights.append(float(p_similar - p_outlier))

    for anchor in range(n_samples):
        candidates = all_indices[all_indices != anchor]
        for _ in range(n_random):
            similar, outlier = rng.choice(candidates, size=2, replace=False)
            similar_distance = np.linalg.norm(
                trimap_matrix[anchor] - trimap_matrix[similar]
            )
            outlier_distance = np.linalg.norm(
                trimap_matrix[anchor] - trimap_matrix[outlier]
            )
            p_similar = -(similar_distance**2) / (
                sigma[anchor] * sigma[similar]
            )
            p_outlier = -(outlier_distance**2) / (
                sigma[anchor] * sigma[outlier]
            )
            if p_similar < p_outlier:
                similar, outlier = outlier, similar
                p_similar, p_outlier = p_outlier, p_similar
            triplet_rows.append((anchor, int(similar), int(outlier)))
            weights.append(0.1 * float(p_similar - p_outlier))

    triplets = np.asarray(triplet_rows, dtype=np.int32)
    triplet_weights = np.nan_to_num(np.asarray(weights, dtype=np.float32))
    triplet_weights -= np.min(triplet_weights)
    if np.isclose(weight_temperature, 1.0):
        triplet_weights = np.log1p(triplet_weights)
    else:
        triplet_weights = (
            np.power(1.0 + triplet_weights, 1.0 - weight_temperature) - 1.0
        ) / (1.0 - weight_temperature)

    return (
        trimap_matrix,
        triplets,
        triplet_weights.astype(np.float32),
    )


def embed_activity_vectors(
    vectors: pd.DataFrame,
    seed: int = 42,
    trimap_iterations: int = 400,
    pacmap_iterations: tuple[int, int, int] = (100, 100, 250),
) -> dict[str, np.ndarray]:
    """Fit UMAP, TriMAP, and PaCMAP to a language-vector matrix."""
    matrix = vectors.to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Vector matrix contains non-finite values")
    if len(matrix) < 4:
        raise ValueError("At least four language vectors are required")

    n_neighbors = min(15, len(matrix) - 1)
    embeddings: dict[str, np.ndarray] = {}
    embeddings["UMAP"] = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.12,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(matrix)

    n_inliers = min(12, len(matrix) - 2)
    n_outliers = min(4, max(1, len(matrix) - n_inliers - 2))
    n_random = min(3, len(matrix) - 2)
    (
        trimap_matrix,
        trimap_triplets,
        trimap_weights,
    ) = _deterministic_trimap_inputs(
        matrix,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        seed=seed,
    )
    embeddings["TriMAP"] = trimap.TRIMAP(
        n_dims=2,
        distance="euclidean",
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        n_iters=trimap_iterations,
        triplets=trimap_triplets,
        weights=trimap_weights,
        apply_pca=False,
        verbose=False,
    ).fit_transform(trimap_matrix, init="pca")

    pacmap_logger = logging.getLogger("pacmap.pacmap")
    previous_level = pacmap_logger.level
    pacmap_logger.setLevel(logging.ERROR)
    try:
        embeddings["PaCMAP"] = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=min(10, len(matrix) - 1),
            distance="euclidean",
            num_iters=pacmap_iterations,
            random_state=seed,
            apply_pca=True,
            verbose=False,
        ).fit_transform(matrix, init="pca")
    finally:
        pacmap_logger.setLevel(previous_level)

    for method, embedding in embeddings.items():
        if embedding.shape != (len(vectors), 2):
            raise ValueError(f"{method} returned shape {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise ValueError(f"{method} returned non-finite coordinates")
    return embeddings


def run_all_embeddings(
    vectors_by_profile: Mapping[str, pd.DataFrame],
    ranking: pd.DataFrame,
    seed: int = 42,
    trimap_iterations: int = 400,
    pacmap_iterations: tuple[int, int, int] = (100, 100, 250),
) -> pd.DataFrame:
    """Fit all reducers for every profile and return tidy coordinates."""
    metadata = ranking.set_index("language")
    rows = []
    for profile, vectors in vectors_by_profile.items():
        profile_embeddings = embed_activity_vectors(
            vectors,
            seed=seed,
            trimap_iterations=trimap_iterations,
            pacmap_iterations=pacmap_iterations,
        )
        for method, coordinates in profile_embeddings.items():
            frame = pd.DataFrame(
                coordinates, index=vectors.index, columns=["x", "y"]
            ).rename_axis("language")
            frame = frame.join(metadata, how="left").reset_index()
            frame["profile"] = profile
            frame["method"] = method
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _nearest_neighbor_preservation(
    high_dimensional: np.ndarray,
    embedding: np.ndarray,
    n_neighbors: int,
) -> float:
    high_neighbors = NearestNeighbors(
        n_neighbors=n_neighbors + 1, metric="euclidean"
    ).fit(high_dimensional)
    low_neighbors = NearestNeighbors(
        n_neighbors=n_neighbors + 1, metric="euclidean"
    ).fit(embedding)
    high_indices = high_neighbors.kneighbors(return_distance=False)[:, 1:]
    low_indices = low_neighbors.kneighbors(return_distance=False)[:, 1:]
    overlap = [
        len(set(high_row).intersection(low_row)) / n_neighbors
        for high_row, low_row in zip(high_indices, low_indices, strict=True)
    ]
    return float(np.mean(overlap))


def evaluate_embeddings(
    vectors_by_profile: Mapping[str, pd.DataFrame],
    embeddings: pd.DataFrame,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """Score embeddings using local and global structure diagnostics."""
    scores = []
    for profile, vectors in vectors_by_profile.items():
        high_dimensional = vectors.to_numpy(dtype=float)
        k = min(n_neighbors, len(vectors) - 2)
        for method in ["UMAP", "TriMAP", "PaCMAP"]:
            subset = (
                embeddings.loc[
                    (embeddings["profile"] == profile)
                    & (embeddings["method"] == method)
                ]
                .set_index("language")
                .reindex(vectors.index)
            )
            low_dimensional = subset[["x", "y"]].to_numpy(dtype=float)
            distance_correlation = spearmanr(
                pdist(high_dimensional, metric="euclidean"),
                pdist(low_dimensional, metric="euclidean"),
            ).statistic
            scores.append(
                {
                    "profile": profile,
                    "method": method,
                    "trustworthiness": trustworthiness(
                        high_dimensional,
                        low_dimensional,
                        n_neighbors=k,
                        metric="euclidean",
                    ),
                    "knn_preservation": _nearest_neighbor_preservation(
                        high_dimensional, low_dimensional, k
                    ),
                    "distance_spearman": float(distance_correlation),
                }
            )
    return pd.DataFrame(scores)


def top_k_dominance(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the monthly composite share held by leading labels."""
    records = []
    for date, group in data.groupby("date", sort=True):
        shares = np.sort(group["composite_share"].to_numpy(dtype=float))[::-1]
        records.append(
            {
                "date": date,
                "top_1_share": float(shares[:1].sum()),
                "top_5_share": float(shares[:5].sum()),
                "top_10_share": float(shares[:10].sum()),
            }
        )
    return pd.DataFrame(records)
