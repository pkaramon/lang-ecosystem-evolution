# Language Ecosystem Evolution

Monthly programming-language activity on GitHub (Jan 2016 – Jun 2025), built from GH Archive via BigQuery.

## Roadmap

- **[Dataset docs](data/DATASET.md)** — what the data is, how it was sampled, pipeline, schema, and caveats
- **[CSV](data/github_language_activity_monthly.csv)** — monthly activity by language (pushes, PRs, stars, forks, etc.)
- **[Initial EDA notebook](notebooks/initial_data_quality_checks.ipynb)** — data quality checks and normalized Plotly charts (shares/ranks, not raw counts)

## Quick notes

- One sampled day per month (the 15th) — compare **shares and trends**, not raw counts across months
- Python 3.12+, managed with `uv` — see `pyproject.toml`
