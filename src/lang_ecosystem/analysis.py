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
    center: bool = True,
) -> pd.DataFrame:
    """Add centered rolling means while retaining the unsmoothed columns."""
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
    smoothed.columns = [f"{column}_smooth" for column in columns]
    return pd.concat([result, smoothed], axis=1)


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


def monthly_concentration(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate concentration statistics from monthly composite shares."""
    records = []
    for date, group in data.groupby("date", sort=True):
        shares = np.sort(group["composite_share"].to_numpy(dtype=float))[::-1]
        hhi = float(np.square(shares).sum())
        records.append(
            {
                "date": date,
                "top_1_share": float(shares[:1].sum()),
                "top_5_share": float(shares[:5].sum()),
                "top_10_share": float(shares[:10].sum()),
                "hhi": hhi,
                "effective_languages": 1.0 / hhi,
            }
        )
    return pd.DataFrame(records)
