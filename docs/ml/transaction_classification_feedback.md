# Transaction classification feedback loop

Feature version: `transaction-classification-feedback-v3-offline-export`

Taxonomy version: `transaction-categories-v1`

## Purpose

The transaction form requests an automatic category suggestion after the user
enters a description. The suggestion remains editable. Saving the transaction
persists both the backend model result and the final category selected by the
user, creating an auditable source for future model improvement.

This version collects feedback; it does not retrain or deploy models
automatically.

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

## Future retraining protocol

Feedback must not update the active model online. A later versioned pipeline
should:

1. export eligible records into a versioned, checksummed local snapshot;
2. manually review sensitive, ambiguous, and potentially poisoned examples;
3. create a separately approved, de-identified training-data release;
4. split it by merchant or provenance group before model development;
5. train a new candidate offline;
6. compare it against fixed acceptance criteria and the frozen benchmark;
7. deploy it as a new model version only after approval;
8. retain the previous artifact for rollback.

This prevents noisy labels, feedback poisoning, test leakage, and untraceable
model changes.

## Limitations

- Feedback is collected only when a transaction is created; editing existing
  transactions is not implemented yet.
- Exporting does not make a snapshot suitable for training automatically.
- No authenticated user or reviewer identity exists in the local phase.
- Descriptions and counterparties must be treated as potentially sensitive if
  real financial data is introduced later.
- Confidence remains uncalibrated and synthetic-data performance does not imply
  production readiness.
