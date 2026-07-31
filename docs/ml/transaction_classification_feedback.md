# Transaction classification feedback loop

Feature version: `transaction-classification-feedback-v4-candidate-lifecycle`

Taxonomy version: `transaction-categories-v1`

## Purpose

The transaction form requests an automatic category suggestion after the user
enters a description. The suggestion remains editable. Saving the transaction
persists both the backend model result and the final category selected by the
user, creating an auditable source for future model improvement.

This version collects feedback and supports a guarded offline candidate
lifecycle. It does not retrain or deploy models automatically.

## Versioned offline export

Reviewed feedback can be converted into an immutable local dataset snapshot:

```powershell
uv run financial-ai-export-category-feedback --version reviewed-v1
```

Without a global `uv` command, use the synchronized environment:

```powershell
.\.venv\Scripts\python.exe -m financial_ai.ml.feedback_export --version reviewed-v1
```

The command writes a CSV and matching metadata JSON below
`data/runtime/ml/feedback/`. Runtime snapshots are deliberately ignored by Git.
The caller supplies the version, and an existing version is never overwritten.
The metadata includes the schema version, canonical SHA-256 checksum, category
and feedback-status counts, exclusions, and the complete preparation report.

### Label eligibility

| Feedback status | Exported | Reason |
| --- | --- | --- |
| `accepted_explicit` | Yes | The user actively confirmed the suggestion |
| `corrected` | Yes | The user supplied a corrective label |
| `manual` | Yes | The user explicitly selected a label without a suggestion |
| `accepted_implicit` | No | Saving a prefilled value is weak evidence |
| `accepted` | No | Legacy records do not prove explicit confirmation |
| `unreviewed` | No | No final reviewed label exists |

Missing and out-of-taxonomy labels are excluded and counted. Exact normalized
text duplicates with the same label are collapsed. If the same normalized text
has conflicting labels, all affected records are excluded instead of choosing
an arbitrary target. These controls improve snapshot quality but do not replace
manual dataset review.

### Export schema and privacy boundary

The CSV contains only:

- `text` (transaction name plus counterparty, matching classifier input)
- `target_category`
- `cash_flow`
- `label_source`
- `model_scope`
- `taxonomy_version`
- `prediction_model_version`

Account, transaction, and classification identifiers, timestamps, amount
magnitude, currency, and notes are excluded. Free text can still contain
sensitive user-entered information, so snapshots must remain local and require
review or de-identification before wider use.

## Candidate training and evaluation

After reviewing a snapshot, train a new candidate without changing the active
model:

```powershell
uv run financial-ai-train-feedback-candidate `
  --feedback-version reviewed-v1 `
  --candidate-version bilingual-feedback-v1
```

The default quality gate requires at least 100 eligible expense rows, at least
five rows for every represented category, and at least three represented
expense categories. Product-rule categories such as `income`, `fees`, and
`investments` are excluded because the current learned model predicts only the
expense taxonomy.

Feedback is split deterministically and per category: 80% augments the existing
controlled English and German training partitions, while 20% remains an unseen
feedback holdout. Exact overlap with the fixed challenge set is excluded before
the split. The original controlled test partitions and fixed challenge rows are
never added to training.

Each candidate receives its own pickle, compatible model metadata, and JSON
evaluation report under `data/runtime/ml/candidates/`. The report contains input
checksums, row counts, configuration, holdout results, fixed-challenge results,
and individual promotion gates. Candidate and active models are compared on:

- hybrid expense macro-F1 on the frozen bilingual challenge;
- selective accuracy for automatically accepted challenge predictions;
- macro-F1 on the unseen feedback holdout.

Challenge macro-F1 and selective accuracy may regress by at most 0.01. Feedback
holdout macro-F1 must be at least as high as the active model. These thresholds
are an explicit learning-project policy, not evidence of production readiness.
Repeated tuning against the fixed challenge would itself create evaluation
overfitting, so threshold or feature changes require a separately versioned
evaluation design.

## Explicit promotion and rollback

An eligible candidate still requires an explicit command:

```powershell
uv run financial-ai-promote-category-model `
  --candidate-version bilingual-feedback-v1 `
  --yes
```

Promotion fails if a gate is false, the candidate checksum differs from its
report, or the active baseline changed after candidate evaluation. The previous
active artifact and metadata are copied to `data/runtime/ml/models/archive/`
before replacement, and a promotion receipt records both versions and hashes.
The candidate remains available for audit. Restart the API after promotion so
its in-process classifier cache loads the new artifact.

Without a global `uv` command, the equivalent module commands are:

```powershell
.\.venv\Scripts\python.exe -m financial_ai.ml.feedback_candidate `
  --feedback-version reviewed-v1 `
  --candidate-version bilingual-feedback-v1

.\.venv\Scripts\python.exe -m financial_ai.ml.feedback_promotion `
  --candidate-version bilingual-feedback-v1 `
  --yes
```

## User flow

1. The user enters a name/description and a signed non-zero amount.
2. The frontend waits 400 milliseconds to avoid a request for every keystroke.
3. `POST /v1/transactions/classify` returns a deterministic or ML suggestion.
4. The category field is populated only for an automatically accepted
   suggestion. A `needs_review` proposal is displayed but requires an explicit
   user selection.
5. The user can retain the suggestion, select another taxonomy category, or
   leave the transaction uncategorized.
6. `POST /v1/transactions` repeats classification on the backend and stores the
   trusted result with the final category.

Repeating classification at save time prevents browser-supplied confidence,
model version, or predicted categories from becoming trusted training metadata.

## Persisted record

Migration `0003_classification_feedback` creates
`transaction_classifications`. Each creation event records:

- predicted category
- final category
- deterministic, ML, or unresolved route
- classification method
- confidence proxy
- review flag and reason
- feedback status
- taxonomy and model versions
- creation timestamp

The transaction keeps the final category used by the product. Classification
records form the audit and future training-feedback history.

## Feedback statuses

| Status | Meaning |
| --- | --- |
| `accepted_implicit` | An automatically populated suggestion was saved unchanged |
| `accepted_explicit` | The user actively selected the same category as the suggestion |
| `corrected` | Submitted final category differs from the suggestion |
| `manual` | No automatic category was available |
| `unreviewed` | A suggestion existed but the transaction remained uncategorized |

The legacy value `accepted` remains readable for records created before this
distinction was introduced. `accepted_implicit` remains weaker training evidence
than an active confirmation or correction. Future dataset preparation must
retain this distinction and apply quality checks before using feedback labels.

## Failure behavior

If the local expense-model artifact is unavailable, manual transaction creation
still succeeds. The record contains no prediction, uses method `none`, sets
`needs_review`, and explains that the artifact was unavailable. Text-rule
matches do not depend on the model artifact.

The category field accepts only versioned product and expense taxonomy values.
New manual categories cannot silently create uncontrolled labels.

## Guarded retraining protocol

Feedback never updates the active model online. The implemented and planned
protocol is:

1. export eligible records into a versioned, checksummed local snapshot;
2. manually review sensitive, ambiguous, and potentially poisoned examples;
3. train a separately versioned candidate from eligible expense feedback;
4. evaluate it on a deterministic feedback holdout and frozen benchmark;
5. require every machine-readable promotion gate to pass;
6. explicitly promote the candidate after human review;
7. archive the previous artifact for rollback;
8. add authenticated review and formal de-identification before production use.

This prevents noisy labels, feedback poisoning, test leakage, and untraceable
model changes.

## Limitations

- Feedback is collected only when a transaction is created; editing existing
  transactions is not implemented yet.
- Exporting does not make a snapshot suitable for training automatically.
- The deterministic feedback split cannot provide true merchant-group isolation
  because the privacy-reduced export contains no merchant identifier.
- A small local feedback sample is expected to fail the minimum-data gate.
- No authenticated user or reviewer identity exists in the local phase.
- Descriptions and counterparties must be treated as potentially sensitive if
  real financial data is introduced later.
- Confidence remains uncalibrated and synthetic-data performance does not imply
  production readiness.
