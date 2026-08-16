# Transaction classification feedback lifecycle

Feature: `transaction-classification-feedback-v4-candidate-lifecycle`

Taxonomy: `transaction-categories-v1`

The application stores the backend prediction and the user's final category.
Feedback never updates the active model online:

```text
saved feedback -> immutable export -> human review -> isolated candidate
  -> feedback holdout + frozen challenge gates -> explicit promotion + archive
```

## Capture and eligibility

`POST /v1/transactions` repeats classification on the backend before saving, so
browser-supplied confidence, versions, or predictions are not trusted. The
database records predicted and final categories, input source, route, method,
confidence proxy, both model predictions and versions, agreement, review reason,
feedback status, taxonomy version, and timestamp.

| Status | Exported | Meaning |
|---|---|---|
| `accepted_explicit` | Yes | User actively confirmed the suggestion |
| `corrected` | Yes | User chose a different label |
| `manual` | Yes | User labeled a transaction without a suggestion |
| `accepted_implicit` | No | A prefilled suggestion was saved unchanged |
| `accepted` | No | Legacy record without explicit confirmation evidence |
| `unreviewed` | No | No final reviewed label |

Here, “reviewed feedback” means a category was actively selected, confirmed, or
corrected in the form. It does **not** mean the exported dataset has completed
privacy, ambiguity, and poisoning review.

## Export

```powershell
uv run financial-ai-export-category-feedback --version reviewed-v1
```

The command creates an immutable CSV and metadata JSON under
`data/runtime/ml/feedback`. Existing versions are never overwritten. Metadata
contains schema and taxonomy versions, SHA-256 checksum, counts, exclusions,
and the preparation report.

The CSV contains classifier text, target category, cash-flow direction, label
source, model scope, taxonomy version, and prediction model version. It omits
account/transaction identifiers, timestamps, amount magnitude, currency, and
notes. Free text may still contain sensitive information and must remain local
until reviewed or de-identified.

Preparation rejects missing or out-of-taxonomy labels, collapses exact
normalized duplicates with one label, and excludes every occurrence of text
with conflicting labels.

## Candidate evaluation

After reviewing the export:

```powershell
uv run financial-ai-train-feedback-candidate `
  --feedback-version reviewed-v1 `
  --candidate-version bilingual-feedback-v1
```

Defaults require at least 100 eligible expense rows, five examples per
represented category, and three represented expense categories. Product-rule
labels such as `income`, `fees`, and `investments` do not enter the expense
model.

Feedback is split deterministically by category: 80% augments controlled
training data and 20% remains an unseen feedback holdout. Exact overlap with the
fixed challenge is removed. Controlled test partitions and challenge rows never
enter training.

Each isolated candidate receives a model, compatible metadata, and evaluation
report under `data/runtime/ml/candidates`. Promotion requires all gates:

- frozen-challenge expense macro-F1 regression no greater than `0.01`;
- automatically accepted challenge accuracy regression no greater than `0.01`;
- feedback-holdout macro-F1 at least equal to the active model.

These thresholds are an experimental governance policy, not production evidence.
Repeated tuning against the frozen challenge would overfit the evaluation and
requires a new versioned protocol.

## Promotion and rollback

```powershell
uv run financial-ai-promote-category-model `
  --candidate-version bilingual-feedback-v1 `
  --yes
```

Promotion rejects failed gates, invalid candidate checksums, or an active model
that changed after evaluation. It archives the current artifact and writes a
receipt with old/new versions and hashes. Restart the API to clear the
in-process classifier cache.

## Failure behavior and limitations

Missing model artifacts do not block manual transaction creation; unresolved
classification is stored for review. Categories remain restricted to the
versioned taxonomy.

Feedback comes from explicit creation-time confirmation and later category
review. The current candidate pipeline promotes only the lexical expense model;
the semantic head requires its own future governed promotion protocol. A small
local sample should fail the minimum-data gate. The privacy-reduced export
cannot provide merchant-group isolation, there is no authenticated reviewer,
and confidence remains uncalibrated. Before production use, add authenticated
review, formal de-identification, poisoning controls, representative data,
calibration, and monitoring.
