# Campaign Schema

## Canonical record shape

From `data/campaigns_v4.json` and scripts, each campaign includes:

- top-level
  - `campaign_id`, `title`, `description`, `vertical`, `landing_url`, `creative_format`, `embedding_text`
- taxonomy
  - `taxonomy.category`, `taxonomy.subcategory`, `taxonomy.leaf`
- brand
  - `brand.name`, `brand.product_line`
- targeting
  - `targeting.age_min`, `targeting.age_max`
  - `targeting.gender` (list)
  - `targeting.geo.include`, `targeting.geo.exclude`
  - `targeting.interests`
- pricing and budget
  - `pricing.price`, `pricing.price_bucket`
  - `budget.remaining`
- ranking helpers
  - `attributes`
  - `keywords` buckets (`brand_terms`, `model_terms`, `category_terms`, `feature_terms`, `benefit_terms`)
  - `negative_keywords`

## What is indexed for retrieval

`CampaignIndex` loads and aligns all retrieval artifacts in `data/`:

- Vector index: `faiss.index`
- Response surface: `campaigns_meta.json`
- Filter arrays: `ages_min.npy`, `ages_max.npy`, `genders.npy`, `bids.npy`, `budgets.npy`
- Targeting lists: `geo_include.json`, `geo_exclude.json`
- Ranking metadata: `campaign_taxonomy.json`, `brands.json`, `keyword_sets.json`, `negative_keywords.json`, `attributes.json`, `interests.json`

## What is used in ranking

Ranking components consume these schema fields:

- Taxonomy alignment: category/subcategory/leaf.
- Brand alignment: brand name and product line.
- Attribute alignment: structured attributes.
- Keyword overlap: five keyword buckets.
- Price alignment: `pricing.price` proxy in `bids.npy`.
- Demographic/location/interest boosts: targeting arrays.
- Contradiction penalties: negative keywords.

## How targeting works

Hard filters in `CampaignEligibilityFilter` run before ranking:

- geo include and geo exclude.
- age range with ±1 boundary tolerance.
- gender bitmask match (`all`, `male`, `female`).
- category-gender mismatch heuristic.
- negative keyword conflict.
- hard price max with 1.5x margin when explicit query max exists.

This separation keeps serveability constraints deterministic and independent from soft ranking boosts.

## Schema-to-response mapping

`campaigns_meta.json` controls exposed campaign fields. Ranker copies this object and appends scores.

Inference: this response-surface mapping is intentional to avoid leaking full internal campaign config to API clients.

## Related docs

- [04 Retrieval](../01-pipelines/04-retrieval.md)
- [05 Ranking and Scoring](../01-pipelines/05-ranking-and-scoring.md)
