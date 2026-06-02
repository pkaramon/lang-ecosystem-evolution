# Language Ecosystem Evolution

## Opis projektu

**5. Ewolucja ekosystemów języków programowania**

### Cel projektu

Celem projektu jest pokazanie zmian popularności języków programowania i technologii w czasie.

### Zakres projektu

Należy wykorzystać dane o aktywności w repozytoriach, aby porównać rozwój różnych języków, frameworków lub obszarów technologicznych.

### Wymagania

- przygotowanie danych o repozytoriach lub zdarzeniach,
- analiza popularności języków w czasie,
- porównanie aktywności społeczności,
- przygotowanie interaktywnych wykresów trendów,
- analiza zmian dominujących technologii,
- interpretacja wyników.

### Element związany z redukcją wymiaru

Należy przygotować wektory aktywności języków lub repozytoriów i porównać wyniki UMAP, TriMAP lub PaCMAP.

### Element rozszerzony

Można dodać analizę migracji użytkowników między technologiami.

### Przykładowe dane i narzędzia

GH Archive, BigQuery, Python, Vega-Lite.

---

Monthly programming-language activity on GitHub (Jan 2016 – Jun 2025), built from GH Archive via BigQuery.

## Roadmap

- **[Dataset docs](data/DATASET.md)** — what the data is, how it was sampled, pipeline, schema, and caveats
- **[CSV](data/github_language_activity_monthly.csv)** — monthly activity by language (pushes, PRs, stars, forks, etc.)
- **[Initial EDA notebook](notebooks/initial_data_quality_checks.ipynb)** — data quality checks and normalized Plotly charts (shares/ranks, not raw counts)

## Quick notes

- One sampled day per month (the 15th) — compare **shares and trends**, not raw counts across months
- Python 3.12+, managed with `uv` — see `pyproject.toml`
