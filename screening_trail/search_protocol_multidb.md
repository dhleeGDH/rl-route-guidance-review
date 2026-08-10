# Reproducible multi-database search protocol (supplementary)

This note records the reproducible cross-publisher queries referenced in Section II-C. The
manuscript charts N = 94 systems from a documented search whose IEEE Xplore component is
reproducible and whose cross-publisher coverage came from an earlier multi-database
collection. The queries below make the cross-publisher search itself reproducible and
executable; title-and-abstract screening and field-by-field full-text charting of the
returned pool to the verbatim-anchored standard of Section II-C remain future work.

## OpenAlex (REST API, no key required)

Endpoint: `https://api.openalex.org/works`

Filter (title and abstract), scope-matched, 2018-2026:

```
title_and_abstract.search:
  "reinforcement learning" AND
  ("route guidance" OR "vehicle navigation" OR "route planning") AND
  (vehicle OR traffic OR road)
from_publication_date:2018-01-01
to_publication_date:2026-12-31
```

Full request URL:

```
https://api.openalex.org/works?filter=title_and_abstract.search:%22reinforcement%20learning%22%20AND%20(%22route%20guidance%22%20OR%20%22vehicle%20navigation%22%20OR%20%22route%20planning%22)%20AND%20(vehicle%20OR%20traffic%20OR%20road),from_publication_date:2018-01-01,to_publication_date:2026-12-31&per_page=1
```

Candidate count at time of writing (`meta.count`): **506**.

Reference variants:
- `"reinforcement learning" AND "route guidance"` (no date filter): 59
- `"reinforcement learning" AND ("route guidance" OR "route planning") AND (vehicle OR traffic)` (no date filter): 364

## Semantic Scholar (Graph API, reproducible query)

Endpoint: `https://api.semanticscholar.org/graph/v1/paper/search`
Query: `reinforcement learning route guidance vehicle navigation route planning`
(bulk endpoint supports year and field filtering; an equivalent scope-matched pool is
returned). Included as a second reproducible, API-based source alongside OpenAlex.

## Note

These queries reproduce the *search*, not the charting. The prevalence figures in the
manuscript are stated at the corpus level precisely because screening and full-text charting
of the returned pool is the labor that a field-level estimate requires, not the availability
of a reproducible query.
