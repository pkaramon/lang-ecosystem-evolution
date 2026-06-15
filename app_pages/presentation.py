"""Presentation-style introduction page for the Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_PATH = ROOT / "Screenshot 2026-06-15 at 23.53.53.png"
SLIDE_STATE_KEY = "presentation_slide_index"


@dataclass(frozen=True)
class Slide:
    title: str
    render: Callable[[], None]


DATA_SOURCE_LOGOS = [
    (
        "GH Archive",
        "https://img.shields.io/badge/GH_Archive-181717?style=for-the-badge&logo=github&logoColor=white",
        "Public GitHub event stream, partitioned as daily BigQuery tables.",
    ),
    (
        "BigQuery",
        "https://img.shields.io/badge/BigQuery-669DF6?style=for-the-badge&logo=googlebigquery&logoColor=white",
        "SQL engine used to scan GH Archive and aggregate language activity.",
    ),
]

TECHNOLOGY_GROUPS = [
    (
        "Data collection",
        [
            ("GH Archive", "https://img.shields.io/badge/GH_Archive-181717?style=for-the-badge&logo=github&logoColor=white"),
            ("BigQuery", "https://img.shields.io/badge/BigQuery-669DF6?style=for-the-badge&logo=googlebigquery&logoColor=white"),
        ],
    ),
    (
        "Analysis",
        [
            ("Python", "https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"),
            ("Pandas", "https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"),
            ("scikit-learn", "https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"),
        ],
    ),
    (
        "Visualization",
        [
            ("Streamlit", "https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"),
            ("Plotly", "https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"),
        ],
    ),
    (
        "Embeddings",
        [
            ("UMAP", "https://img.shields.io/badge/UMAP-6C5CE7?style=for-the-badge"),
            ("TriMAP", "https://img.shields.io/badge/TriMAP-00A884?style=for-the-badge"),
            ("PaCMAP", "https://img.shields.io/badge/PaCMAP-C44569?style=for-the-badge"),
        ],
    ),
]


def _render_presentation_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1400px;
                padding-top: 3.25rem !important;
                padding-bottom: 6rem !important;
            }
            .presentation-kicker {
                color: #6F6558;
                font-size: 0.85rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.45rem;
                font-weight: 700;
            }
            .presentation-hero {
                background:
                    radial-gradient(circle at top right, rgba(102, 157, 246, 0.18), transparent 32rem),
                    linear-gradient(135deg, #FBF7EF 0%, #E8E1D6 100%);
                border: 1px solid #D8D0C4;
                border-radius: 1.25rem;
                padding: 1.85rem 2.35rem;
                margin-bottom: 1.25rem;
                min-height: 10rem;
                box-shadow: 0 18px 45px rgba(61, 53, 44, 0.08);
            }
            .presentation-hero h1 {
                font-size: clamp(2.15rem, 4.2vw, 4.1rem);
                line-height: 0.95;
                margin: 0 0 1rem;
                letter-spacing: -0.06em;
            }
            .presentation-hero p {
                color: #4E473F;
                font-size: 1.08rem;
                max-width: 58rem;
                margin: 0;
            }
            .presentation-card {
                background-color: #FBF7EF;
                border: 1px solid #D8D0C4;
                border-radius: 1rem;
                padding: 1rem 1.1rem;
                min-height: 8.2rem;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                box-shadow: 0 12px 30px rgba(61, 53, 44, 0.06);
            }
            .presentation-card h3 {
                margin: 0 0 0.65rem;
                font-size: 1.15rem;
            }
            .presentation-card p {
                margin: 0;
                color: #5D554B;
                font-size: 1rem;
            }
            .presentation-stat {
                font-size: 2.15rem;
                font-weight: 700;
                line-height: 1.1;
                margin-bottom: 0.45rem;
                letter-spacing: -0.04em;
            }
            .presentation-stat-card {
                height: 8.6rem;
                min-height: 8.6rem;
            }
            .presentation-muted {
                color: #6F6558;
                font-size: 0.95rem;
            }
            .presentation-note {
                background-color: #EFE8DC;
                border: 1px solid #D8D0C4;
                border-radius: 1rem;
                color: #4E473F;
                font-size: 1rem;
                margin-top: 1.05rem;
                padding: 0.95rem 1.15rem;
            }
            .presentation-icon-card {
                align-items: flex-start;
                gap: 0.65rem;
                min-height: 8.5rem;
            }
            .presentation-icon-card img {
                height: 1.7rem;
                width: auto;
                max-width: 100%;
            }
            .technology-card {
                min-height: 12rem;
            }
            .technology-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                align-items: center;
            }
            .technology-badges img {
                height: 1.85rem;
                width: auto;
            }
            .presentation-list {
                background-color: #FBF7EF;
                border: 1px solid #D8D0C4;
                border-radius: 1rem;
                margin-top: 1rem;
                padding: 0.9rem 1.1rem;
            }
            .presentation-list ul {
                margin-bottom: 0;
                padding-left: 1.2rem;
            }
            .presentation-progress {
                color: #6F6558;
                text-align: center;
                font-size: 0.9rem;
                padding-bottom: 0.35rem;
            }
            div[class*="st-key-presentation_controls"] {
                position: fixed;
                right: 1.25rem;
                bottom: 1.25rem;
                z-index: 1000;
                width: 14rem;
                background: rgba(239, 232, 220, 0.94);
                border: 1px solid #D8D0C4;
                border-radius: 1rem;
                padding: 0.65rem;
                box-shadow: 0 16px 35px rgba(61, 53, 44, 0.18);
                backdrop-filter: blur(8px);
            }
            div[class*="st-key-presentation_controls"] [data-testid="stHorizontalBlock"] {
                gap: 0.5rem;
            }
            div[class*="st-key-presentation_controls"] button {
                min-height: 2.45rem;
                border-radius: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(kicker: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <section class="presentation-hero">
            <div class="presentation-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="presentation-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _icon_card(name: str, badge_url: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="presentation-card presentation-icon-card">
            <h3>{name}</h3>
            <img src="{badge_url}" alt="{name}">
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _technology_card(title: str, badges: list[tuple[str, str]]) -> None:
    badge_markup = "\n".join(
        f'<img src="{badge_url}" alt="{name}">' for name, badge_url in badges
    )
    st.markdown(
        f"""
        <div class="presentation-card technology-card">
            <h3>{title}</h3>
            <div class="technology-badges">{badge_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stat(value: str, label: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="presentation-card presentation-stat-card">
            <div class="presentation-stat">{value}</div>
            <strong>{label}</strong>
            <p class="presentation-muted">{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _note(body: str) -> None:
    st.markdown(f'<div class="presentation-note">{body}</div>', unsafe_allow_html=True)


def _render_title_slide() -> None:
    _hero(
        "Project introduction",
        "Language Ecosystem Evolution",
        "An interactive dashboard for studying how programming-language activity on GitHub "
        "changed from January 2016 to June 2025.",
    )
    cols = st.columns(3)
    with cols[0]:
        _stat("114", "monthly samples", "One sampled day: the 15th of each month.")
    with cols[1]:
        _stat("9", "activity metrics", "Pushes, PRs, issues, stars, forks, repos, and more.")
    with cols[2]:
        _stat("150", "languages embedded", "Top labels are projected with UMAP, TriMAP, and PaCMAP.")

    _note(
        "The dashboard is a story about relative ecosystem composition: we compare "
        "shares, trends, and rankings rather than raw GitHub event totals."
    )


def _render_technology_slide() -> None:
    _hero(
        "Technologies used",
        "From raw events to an interactive dashboard",
        "The project uses cloud SQL for extraction, Python for analysis, dimensionality "
        "reduction for similarity maps, and Streamlit for the final dashboard.",
    )
    cols = st.columns(4)
    for col, (title, badges) in zip(cols, TECHNOLOGY_GROUPS, strict=True):
        with col:
            _technology_card(title, badges)


def _render_data_source_slide() -> None:
    _hero(
        "Data source",
        "GH Archive, queried through BigQuery",
        "The dataset starts from public GitHub events and turns them into a monthly language "
        "activity table: one row per language, year, and month.",
    )

    logo_cols = st.columns(2)
    for col, (name, badge_url, body) in zip(logo_cols, DATA_SOURCE_LOGOS, strict=True):
        with col:
            _icon_card(name, badge_url, body)

    st.write("")
    cols = st.columns(2)
    with cols[0]:
        _card(
            "Sampling design",
            "We scan the 15th day of each month from January 2016 to June 2025. "
            "The same calendar position keeps samples comparable across time.",
        )
    with cols[1]:
        _card(
            "Language attribution",
            "Repository language is extracted from pull request payloads at "
            "`payload.pull_request.base.repo.language`, then persisted as a lookup table.",
        )

    _note(
        "Raw counts are sensitive to weekdays and weekends. The analysis therefore uses "
        "within-month shares, percentages, and trends."
    )


def _render_bigquery_limits_slide() -> None:
    _hero(
        "BigQuery limits",
        "The 1 TB/query limit shaped the dataset",
        "Reading GitHub's JSON payload column is expensive. The extraction had to be planned "
        "around BigQuery's free-tier ceiling, not written as one careless full scan.",
    )

    left, right = st.columns([0.9, 1.35], gap="large")
    with left:
        _stat("1 TB", "query ceiling", "Each expensive extraction query had to stay below this limit.")

        st.markdown(
            """
            <div class="presentation-list">
                <ul>
                    <li>Scan only the 15th day of each month, not every day.</li>
                    <li>Split language extraction into two date ranges.</li>
                    <li>Read the costly payload field once, then reuse the language lookup.</li>
                    <li>Aggregate all final activity metrics in one cheaper pass.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _note(
            "The dataset stops at June 2025 because GitHub later removed the embedded "
            "language field from event payloads."
        )

    with right:
        if SCREENSHOT_PATH.exists():
            st.image(
                str(SCREENSHOT_PATH),
                caption="The BigQuery editor shows the cost estimate before running the query.",
                use_container_width=True,
            )
        else:
            st.error(f"Screenshot not found: `{SCREENSHOT_PATH.name}`")


def _render_dashboard_slide() -> None:
    _hero(
        "Dashboard views",
        "What the analysis pages show",
        "The rest of the app turns the prepared dataset into trend charts, ecosystem "
        "comparisons, and low-dimensional maps of language behavior.",
    )
    first_row = st.columns(3)
    with first_row[0]:
        _card("Sampling", "Why the dashboard compares shares and trends instead of raw monthly counts.")
    with first_row[1]:
        _card("Popularity", "Composite language share, dominance, and changes in the leading languages.")
    with first_row[2]:
        _card("Winners", "Long-run risers and decliners across the 2016-2025 window.")

    st.write("")
    second_row = st.columns(3)
    with second_row[0]:
        _card("Community", "Contribution signals compared with reach signals like stars and forks.")
    with second_row[1]:
        _card("Concentration", "Whether activity is concentrated in a few languages or spread broadly.")
    with second_row[2]:
        _card("Embeddings", "UMAP, TriMAP, and PaCMAP views of language activity profiles.")


SLIDES = [
    Slide("Title", _render_title_slide),
    Slide("Technologies Used", _render_technology_slide),
    Slide("Data Source", _render_data_source_slide),
    Slide("BigQuery Limits", _render_bigquery_limits_slide),
    Slide("What The Dashboard Shows", _render_dashboard_slide),
]


def _set_slide(index: int) -> None:
    st.session_state[SLIDE_STATE_KEY] = max(0, min(index, len(SLIDES) - 1))


def _advance_slide(delta: int) -> None:
    _set_slide(st.session_state[SLIDE_STATE_KEY] + delta)


def _render_controls(current_index: int) -> None:
    with st.container(key="presentation_controls"):
        st.markdown(
            f'<div class="presentation-progress">{current_index + 1} / {len(SLIDES)}</div>',
            unsafe_allow_html=True,
        )
        previous_col, next_col = st.columns(2)
        with previous_col:
            st.button(
                "<-",
                disabled=current_index == 0,
                use_container_width=True,
                on_click=_advance_slide,
                args=(-1,),
            )
        with next_col:
            st.button(
                "->",
                disabled=current_index == len(SLIDES) - 1,
                use_container_width=True,
                on_click=_advance_slide,
                args=(1,),
            )


def render() -> None:
    _render_presentation_css()

    if SLIDE_STATE_KEY not in st.session_state:
        st.session_state[SLIDE_STATE_KEY] = 0

    current_index = st.session_state[SLIDE_STATE_KEY]
    SLIDES[current_index].render()
    _render_controls(current_index)
