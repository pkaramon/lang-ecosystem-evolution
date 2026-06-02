# GitHub Language Activity Dataset

## Overview

A monthly time series of programming language activity on GitHub, derived from public GH Archive event data via BigQuery. Each row represents aggregated activity for one language in one calendar month.

- Unit of observation: `(language, year, month)`
- Time span: January 2016 to June 2025 (114 monthly samples)
- Granularity: one sampled day per month (the 15th)
- Source: [GH Archive](https://www.gharchive.org/) via BigQuery, collected June 2026

---

## Sampling Design

Only the 15th of each month is scanned, not the full month. This reduces BigQuery costs by roughly 30x, since the `payload` column (which holds JSON event data) is extremely expensive to scan. Because the same calendar position is used every month, the sample remains consistent and comparable across time.

Important: always analyze this data in shares, percentages, or ratios. Raw counts are sensitive to whether the 15th fell on a weekday or weekend, so they should not be compared across months directly.

---

## The Language Field Problem

A repository's language is not a top-level column in the GitHub event schema. It had to be extracted from deep inside PR event payloads (`payload.pull_request.base.repo.language`). Two issues shaped the pipeline:

1. Reading this field requires scanning the entire `payload` column, which is costly.
2. GitHub removed the embedded language field from event payloads starting with a brownout on 8 September 2025 and officially on 7 October 2025. After this date, the field is null.

The dataset is therefore capped at June 2025 to ensure language attribution remains intact throughout.

---

## Pipeline

The pipeline is split into four queries to manage cost. The language field is read only once and stored, then all event counting is done cheaply without touching `payload`.

```
Q1  payload -> language map, 2016-2020        -+
Q2  payload -> language map, 2021-mid-2025     +-> Q3 merge -> repo_language table

Q4  all event types per language per month -> JOIN repo_language -> final CSV
```

Queries 1 and 2 are split at 2020/2021 to stay within BigQuery's 1 TB free-tier limit per query. Query 4 is the only one that produces the CSV and never touches `payload`.

The destination BigQuery dataset must be in the US multi-region to match `githubarchive`:

```sql
CREATE SCHEMA `gh` OPTIONS (location = 'US');
```

### Q1 -- language map, 2016-2020 (`gh.lang_a`)

```sql
CREATE OR REPLACE TABLE `gh.lang_a` AS
WITH lang_counts AS (
  SELECT
    repo.name AS repo_name,
    JSON_VALUE(payload, '$.pull_request.base.repo.language') AS language,
    COUNT(*) AS n
  FROM `githubarchive.day.20*`
  WHERE _TABLE_SUFFIX IN (
    '160115','160215','160315','160415','160515','160615','160715','160815','160915','161015','161115','161215',
    '170115','170215','170315','170415','170515','170615','170715','170815','170915','171015','171115','171215',
    '180115','180215','180315','180415','180515','180615','180715','180815','180915','181015','181115','181215',
    '190115','190215','190315','190415','190515','190615','190715','190815','190915','191015','191115','191215',
    '200115','200215','200315','200415','200515','200615','200715','200815','200915','201015','201115','201215'
  )
  AND type = 'PullRequestEvent'
  AND JSON_VALUE(payload, '$.pull_request.base.repo.language') IS NOT NULL
  GROUP BY repo_name, language
)
SELECT repo_name, language, n
FROM lang_counts
QUALIFY ROW_NUMBER() OVER (PARTITION BY repo_name ORDER BY n DESC) = 1;
```

### Q2 -- language map, 2021-mid-2025 (`gh.lang_b`)

```sql
CREATE OR REPLACE TABLE `gh.lang_b` AS
WITH lang_counts AS (
  SELECT
    repo.name AS repo_name,
    JSON_VALUE(payload, '$.pull_request.base.repo.language') AS language,
    COUNT(*) AS n
  FROM `githubarchive.day.20*`
  WHERE _TABLE_SUFFIX IN (
    '210115','210215','210315','210415','210515','210615','210715','210815','210915','211015','211115','211215',
    '220115','220215','220315','220415','220515','220615','220715','220815','220915','221015','221115','221215',
    '230115','230215','230315','230415','230515','230615','230715','230815','230915','231015','231115','231215',
    '240115','240215','240315','240415','240515','240615','240715','240815','240915','241015','241115','241215',
    '250115','250215','250315','250415','250515','250615'
  )
  AND type = 'PullRequestEvent'
  AND JSON_VALUE(payload, '$.pull_request.base.repo.language') IS NOT NULL
  GROUP BY repo_name, language
)
SELECT repo_name, language, n
FROM lang_counts
QUALIFY ROW_NUMBER() OVER (PARTITION BY repo_name ORDER BY n DESC) = 1;
```

### Q3 -- merge both halves (`gh.repo_language`)

```sql
CREATE OR REPLACE TABLE `gh.repo_language` AS
WITH u AS (
  SELECT repo_name, language, n FROM `gh.lang_a`
  UNION ALL
  SELECT repo_name, language, n FROM `gh.lang_b`
),
agg AS (
  SELECT repo_name, language, SUM(n) AS n
  FROM u
  GROUP BY repo_name, language
)
SELECT repo_name, language
FROM agg
QUALIFY ROW_NUMBER() OVER (PARTITION BY repo_name ORDER BY n DESC) = 1;
```

### Q4 -- activity table / final CSV (no payload scanned)

```sql
SELECT
  rl.language,
  EXTRACT(YEAR  FROM g.created_at) AS year,
  EXTRACT(MONTH FROM g.created_at) AS month,
  COUNTIF(g.type = 'PushEvent')          AS pushes,
  COUNTIF(g.type = 'PullRequestEvent')   AS pull_requests,
  COUNTIF(g.type = 'IssuesEvent')        AS issues,
  COUNTIF(g.type = 'IssueCommentEvent')  AS issue_comments,
  COUNTIF(g.type = 'WatchEvent')         AS stars,
  COUNTIF(g.type = 'ForkEvent')          AS forks,
  COUNTIF(g.type = 'CreateEvent')        AS creates,
  COUNT(DISTINCT g.actor.login)          AS contributors,
  COUNT(DISTINCT g.repo.name)            AS active_repos
FROM `githubarchive.day.20*` AS g
JOIN `gh.repo_language` AS rl
  ON g.repo.name = rl.repo_name
WHERE g._TABLE_SUFFIX IN (
  '160115','160215','160315','160415','160515','160615','160715','160815','160915','161015','161115','161215',
  '170115','170215','170315','170415','170515','170615','170715','170815','170915','171015','171115','171215',
  '180115','180215','180315','180415','180515','180615','180715','180815','180915','181015','181115','181215',
  '190115','190215','190315','190415','190515','190615','190715','190815','190915','191015','191115','191215',
  '200115','200215','200315','200415','200515','200615','200715','200815','200915','201015','201115','201215',
  '210115','210215','210315','210415','210515','210615','210715','210815','210915','211015','211115','211215',
  '220115','220215','220315','220415','220515','220615','220715','220815','220915','221015','221115','221215',
  '230115','230215','230315','230415','230515','230615','230715','230815','230915','231015','231115','231215',
  '240115','240215','240315','240415','240515','240615','240715','240815','240915','241015','241115','241215',
  '250115','250215','250315','250415','250515','250615'
)
GROUP BY language, year, month
ORDER BY year, month, pushes DESC;
```

---

## Schema

| Column | Type | Meaning |
|---|---|---|
| `language` | string | Primary language of the repository |
| `year` | int | Calendar year of the sampled day |
| `month` | int | Calendar month (1-12) |
| `pushes` | int | PushEvent count |
| `pull_requests` | int | PullRequestEvent count |
| `issues` | int | IssuesEvent count |
| `issue_comments` | int | IssueCommentEvent count |
| `stars` | int | WatchEvent count (GitHub stars) |
| `forks` | int | ForkEvent count |
| `creates` | int | CreateEvent count (repo/branch/tag creation) |
| `contributors` | int | Distinct active users that month for the language |
| `active_repos` | int | Distinct active repositories that month for the language |

All counts reflect the single sampled day, not the full month.

---

## Key Design Decisions

- The language for each repo is learned once from PR payloads and persisted to a lookup table. Q4 then joins against this table cheaply without re-reading payload.
- Where a repo has PRs attributed to multiple languages, the most frequent one wins (`ROW_NUMBER() OVER ... ORDER BY n DESC`). This is more reliable than `MAX(language)`, which would simply favor alphabetically later values.
- Counting multiple event types in Q4 (pushes, stars, forks, etc.) is essentially free since they are all `COUNTIF`s on the same `type` column already being scanned.

---

## Known Biases and Limitations

1. PR-participation bias: only repositories that had at least one language-tagged pull request on a sampled day are included. Solo or PR-less repos are excluded.
2. Figures are probes, not totals. Use shares and trends rather than raw counts.
3. Months where the 15th fell on a weekend show lower absolute activity. This does not affect relative measures.
4. Data ends at June 2025 due to the GitHub payload trimming that removed the language field.
5. Each repo is assigned exactly one language. Multi-language repos are collapsed to their dominant language.
6. Language strings come from GitHub's own classification, which includes labels like `Jupyter Notebook`, `HTML`, `Shell`, and `Dockerfile`. Filter these if you only want general-purpose programming languages.

---

## Reproducibility

- Run queries in order: Q1, Q2, Q3, Q4.
- Both the source (`githubarchive`) and destination dataset must be in the US multi-region.
- Export Q4 result as CSV.
- Sampling key: the 15th of each month, January 2016 through June 2025.
- Language extraction path: `payload.pull_request.base.repo.language` on `PullRequestEvent`.