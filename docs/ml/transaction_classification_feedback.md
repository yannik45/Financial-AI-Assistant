# Transaction classification feedback loop

Feature version: `transaction-classification-feedback-v1`  
Taxonomy version: `transaction-categories-v1`

## Purpose

The transaction form requests an automatic category suggestion after the user
enters a description. The suggestion remains editable. Saving the transaction
persists both the backend model result and the final category selected by the
user, creating an auditable source for future model improvement.

This version collects feedback; it does not retrain or deploy models
automatically.

## User flow

1. The user enters a name/description and a signed non-zero amount.
2. The frontend waits 400 milliseconds to avoid a request for every keystroke.
3. `POST /v1/transactions/classify` returns a deterministic or ML suggestion.
4. The category field is populated when a category is available.
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
| `accepted` | Submitted final category equals the backend suggestion |
| `corrected` | Submitted final category differs from the suggestion |
| `manual` | No automatic category was available |
| `unreviewed` | A suggestion existed but the transaction remained uncategorized |

`accepted` is currently implicit: the suggestion was submitted unchanged, but
the user did not press a separate confirmation control. It is therefore weaker
training evidence than an active correction. Future dataset preparation must
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

1. export eligible records without personal or sensitive data;
2. separate explicit corrections from implicit acceptances;
3. review ambiguous and low-confidence samples;
4. deduplicate and split by merchant or provenance group;
5. create a versioned training-data snapshot;
6. train a new candidate offline;
7. compare it against fixed acceptance criteria and an appropriate benchmark;
8. deploy it as a new model version only after approval;
9. retain the previous artifact for rollback.

This prevents noisy labels, feedback poisoning, test leakage, and untraceable
model changes.

## Limitations

- Feedback is collected only when a transaction is created; editing existing
  transactions is not implemented yet.
- No authenticated user or reviewer identity exists in the local phase.
- Descriptions and counterparties must be treated as potentially sensitive if
  real financial data is introduced later.
- Confidence remains uncalibrated and synthetic-data performance does not imply
  production readiness.
