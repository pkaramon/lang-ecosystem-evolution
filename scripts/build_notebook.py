"""Build the polished analysis notebook from maintainable source cells."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "language_ecosystem_analysis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        """
        # Language Ecosystem Evolution on GitHub

        **Interactive analysis of monthly programming-language activity, January 2016 to June 2025**

        This notebook studies popularity, community behavior, concentration, and long-run
        similarity across GitHub language ecosystems. Every comparison uses **within-month
        proportions**, because the source dataset samples only the 15th day of each month.

        The notebook is deliberately split into a guided analytical story and interactive
        Plotly views. Reusable calculations and figure builders live in `src/lang_ecosystem`
        so the same work can later power a Streamlit application.
        """
    ),
    markdown(
        """
        ## 1. Setup and visual language

        The analysis uses one fixed color per leading language. Dense trajectories show
        20 labels, community explorers show 40, and dimensionality-reduction views use
        150. The 30 most prominent labels retain vivid colors; the longer tail is muted
        to keep the structure readable.
        """
    ),
    code(
        """
        from pathlib import Path
        import importlib
        import sys
        import warnings

        import numpy as np
        import pandas as pd
        import plotly.io as pio
        from IPython.display import Markdown, display

        warnings.filterwarnings("ignore", message="IProgress not found.*")

        ROOT = Path.cwd()
        if ROOT.name == "notebooks":
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT / "src"))

        # Jupyter keeps imported modules alive across reruns. Reload this local package so
        # regenerated notebooks always use the current source tree.
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == "lang_ecosystem" or module_name.startswith(
                "lang_ecosystem."
            ):
                del sys.modules[module_name]

        from lang_ecosystem.analysis import (
            ACTIVITY_PROFILES,
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
        from lang_ecosystem.visuals import (
            CHART_LIMITS,
            activity_specialization_figure,
            animated_activity_bubble,
            composite_trend_figure,
            concentration_figure,
            dominance_turnover_figure,
            ecosystem_momentum_figure,
            embedding_quality_figure,
            language_color_map,
            leaders_decliners_figure,
            metric_trend_figure,
            projection_method_figure,
            ranking_explorer_figure,
            sampling_bias_figure,
            stacked_area_figure,
        )

        pio.renderers.default = "notebook_connected"
        PLOT_CONFIG = {
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        }
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
        """
    ),
    markdown(
        """
        ## 2. Prepare comparable monthly activity

        The source contains nine activity measures: pushes, pull requests, issues, issue
        comments, stars, forks, creates, contributors, and active repositories.

        Preparation follows four rules:

        1. Merge only obvious case or spacing aliases.
        2. Build the complete language-by-month grid.
        3. Treat an absent row as zero activity observed on that sampled day.
        4. Divide each metric by its total in the same month.

        The composite popularity score is the equal-weight mean of those nine monthly
        proportions. It is not an estimate of total monthly GitHub activity.
        """
    ),
    code(
        """
        DATA_PATH = ROOT / "data" / "github_language_activity_monthly.csv"

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
        top_20 = top_150[: CHART_LIMITS["trajectories"]]
        colors = language_color_map(top_150)
        colors["Other"] = "#D7D1C8"

        summary = pd.DataFrame(
            {
                "Value": [
                    f"{activity['date'].min():%B %Y} - {activity['date'].max():%B %Y}",
                    activity["date"].nunique(),
                    activity["language"].nunique(),
                    len(activity),
                    len(METRICS),
                    len(top_150),
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
        display(summary)
        display(
            ranking.head(30).style.format(
                {"mean_composite_share": "{:.2%}", "months_observed": "{:,.0f}"}
            )
        )
        """
    ),
    code(
        '''
        share_columns = [f"{metric}_share" for metric in METRICS]
        share_check = activity.groupby("date")[share_columns].sum()
        max_error = (share_check - 1).abs().to_numpy().max()
        category_counts = ranking.head(CHART_LIMITS["embeddings"])["category"].value_counts()

        display(
            Markdown(
                f"""
                **Validation.** Every metric's monthly language shares sum to one
                (maximum floating-point error: `{max_error:.2e}`). The top-150 set contains
                **{category_counts.get('Programming language', 0)} programming-language
                labels** and **{category_counts.get('Technology / artifact', 0)}
                technology/artifact labels**.

                Key explorers provide explicit scope controls for all labels, programming
                languages only, and technology/artifact labels only. They default to all
                labels because the project brief covers both languages and technologies.
                """
            )
        )
        '''
    ),
    markdown(
        """
        ## 3. Why the raw counts cannot be compared directly

        The top panel below sums event rows observed on each sampled day. Weekend samples
        are visibly lower. The bottom panel uses within-month proportions instead, which
        removes the shared calendar-day volume shock and preserves the relative ecosystem
        composition.
        """
    ),
    code(
        """
        sampling_bias_figure(activity, top_20[:5], colors).show(config=PLOT_CONFIG)
        """
    ),
    markdown(
        """
        ## 4. Popularity and dominant technologies

        The next views move from the composite score to its individual ingredients.
        Hover for exact values, zoom into a period, and use the controls above each chart.
        """
    ),
    code(
        """
        composite_trend_figure(activity, top_20, colors).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        metric_trend_figure(activity, top_20, colors).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        stacked_area_figure(activity, top_20, colors).show(config=PLOT_CONFIG)
        """
    ),
    markdown(
        """
        The stacked view answers a market-share question; the ranking explorer answers a
        position question. Its controls independently change the activity signal, period,
        and label scope. Dynamic leaders are recalculated for each selected signal, so
        metric specialists are not excluded by the composite ranking.
        """
    ),
    code(
        """
        ranking_explorer_figure(activity, colors).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        dominance_turnover_figure(activity).show(config=PLOT_CONFIG)
        """
    ),
    markdown(
        """
        ## 5. Structural winners and decliners

        Comparing twelve-month windows is less sensitive to a single noisy sampled day.
        The chart reports changes in percentage points of composite activity share, not
        changes in raw event counts.
        """
    ),
    code(
        """
        leaders_decliners_figure(activity, count=10).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
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

        display(
            Markdown(
                "**Largest gains:** "
                + ", ".join(f"{name} ({value:+.2%})" for name, value in risers.items())
                + "\\n\\n**Largest declines:** "
                + ", ".join(
                    f"{name} ({value:+.2%})" for name, value in decliners.items()
                )
            )
        )
        """
    ),
    markdown(
        """
        ## 6. Community activity profiles

        Contribution activity and reach/community activity are related but not identical.
        Stars and forks can indicate attention and adoption; contributors and active
        repositories indicate participation breadth. The animation exposes movement over
        time, the specialization heatmap shows which signals define each ecosystem, and
        the momentum view separates current prominence from recent change.
        """
    ),
    code(
        """
        animated_activity_bubble(activity, top_40, colors).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        activity_specialization_figure(activity).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        ecosystem_momentum_figure(activity, colors).show(config=PLOT_CONFIG)
        """
    ),
    markdown(
        """
        ## 7. Is the ecosystem concentrating?

        Top-k shares show directly how much of the composite activity belongs to the
        leading one, five, and ten ecosystems. This avoids translating the distribution
        into a less interpretable synthetic count.
        """
    ),
    code(
        """
        concentration_figure(activity).show(config=PLOT_CONFIG)
        """
    ),
    markdown(
        """
        ## 8. Activity vectors

        Each point in the following projections represents **one language across the whole
        114-month history**. For each profile, the vector concatenates one chronological
        monthly-share time series per metric:

        - **All activity:** 9 metrics x 114 months = 1,026 features
        - **Contribution:** 5 metrics x 114 months = 570 features
        - **Reach / community:** 4 metrics x 114 months = 456 features

        These are the proportions defined earlier, used directly. No per-language
        standardization or alternative "shape" representation is applied.
        """
    ),
    code(
        """
        profile_vectors = build_profile_vectors(activity, top_150)
        vector_summary = pd.DataFrame(
            {
                "Languages": [matrix.shape[0] for matrix in profile_vectors.values()],
                "Features": [matrix.shape[1] for matrix in profile_vectors.values()],
                "First feature": [
                    f"{matrix.columns[0][0]} / {matrix.columns[0][1]:%Y-%m}"
                    for matrix in profile_vectors.values()
                ],
                "Last feature": [
                    f"{matrix.columns[-1][0]} / {matrix.columns[-1][1]:%Y-%m}"
                    for matrix in profile_vectors.values()
                ],
            },
            index=profile_vectors.keys(),
        )
        display(vector_summary)
        """
    ),
    markdown(
        """
        ## 9. UMAP, TriMAP, and PaCMAP

        All methods use two output dimensions and Euclidean distance. UMAP and PaCMAP use
        explicit random seeds. TriMAP 1.1.5 exposes no `random_state`; the helper replaces
        its unseeded approximate-neighbor and parallel-sampling step with deterministic
        exact neighbors and seeded weighted triplets before running the TriMAP optimizer.

        Coordinates and axis directions have no intrinsic meaning. Interpret neighborhoods
        and relative structure, not absolute x/y positions.
        """
    ),
    code(
        """
        embeddings = run_all_embeddings(
            profile_vectors,
            ranking,
            seed=42,
        )
        embedding_scores = evaluate_embeddings(
            profile_vectors,
            embeddings,
            n_neighbors=10,
        )

        display(
            embedding_scores.style.format(
                {
                    "trustworthiness": "{:.3f}",
                    "knn_preservation": "{:.3f}",
                    "distance_spearman": "{:.3f}",
                }
            )
        )
        """
    ),
    code(
        """
        projection_method_figure(
            embeddings, colors, "UMAP"
        ).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        projection_method_figure(
            embeddings, colors, "TriMAP"
        ).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        projection_method_figure(
            embeddings, colors, "PaCMAP"
        ).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        embedding_quality_figure(embedding_scores).show(config=PLOT_CONFIG)
        """
    ),
    code(
        """
        score_winners = (
            embedding_scores.set_index(["profile", "method"])[
                ["trustworthiness", "knn_preservation", "distance_spearman"]
            ]
            .groupby(level="profile")
            .idxmax()
        )
        winner_lines = []
        for profile in score_winners.index:
            local = score_winners.loc[profile, "trustworthiness"][1]
            neighbors = score_winners.loc[profile, "knn_preservation"][1]
            global_method = score_winners.loc[profile, "distance_spearman"][1]
            winner_lines.append(
                f"- **{profile}:** trustworthiness `{local}`, "
                f"k-NN preservation `{neighbors}`, global distance `{global_method}`"
            )
        display(
            Markdown(
                "**Best method by diagnostic (this dataset and parameterization):**\\n\\n"
                + "\\n".join(winner_lines)
            )
        )
        """
    ),
    markdown(
        """
        ## 10. Interpretation and limitations

        **What the analysis supports**

        - Data preparation creates a complete language-by-month grid and comparable
          within-month shares from the sampled repository events.
        - Popularity is multidimensional: code contribution, discussion, attention,
          participation breadth, and repository breadth do not always move together.
        - Monthly, quarterly, and annual shares and ranks reveal changes in dominant
          languages and technology/artifact labels more reliably than a sampled raw count.
        - Turnover, momentum, and specialization distinguish leadership changes from
          community behavior: a prominent ecosystem can decline while a smaller one
          over-indexes strongly on discussion, contribution, or adoption signals.
        - The three projections are complementary. UMAP emphasizes local neighborhoods,
          while TriMAP and PaCMAP optimize different combinations of local and global
          structure; the quality diagnostics make those trade-offs visible.
        - Technology/artifact labels remain in scope because GitHub's classification and
          the project brief include technologies as well as general-purpose languages.

        **What the analysis does not support**

        - The data is not a monthly census. It is one probe on the 15th of each month.
        - Only repositories present in the PR-derived language map are included.
        - Each repository has one dominant language, so multi-language repositories are
          simplified.
        - User migration between technologies cannot be inferred because the dataset has
          aggregate language-month counts, not actor-level transition histories.
        - Distances in a 2D embedding are approximations. Clusters should be checked against
          the original time series and embedding-quality scores before interpretation.

        The notebook therefore describes **relative activity among observed ecosystems**,
        not absolute language usage or developer population.
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)

nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
