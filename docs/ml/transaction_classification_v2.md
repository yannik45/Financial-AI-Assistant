# Transaction classification v2 model selection

Status: integrated experimental serving policy

Version 2 evaluates lexical and semantic evidence without routing by arbitrary
text length. The manual candidate is frozen multilingual E5 with a Logistic
Regression head. The selected bank policy runs both TF-IDF and E5 after
high-signal rules and automatically accepts only agreement. This document
describes the evaluated candidates and the policy served by the API.

## Evaluation protocol

Controlled English/German bank statements and `manual-short-v2` use declared
train, validation, and test partitions. Merchant, semantic-detail, and template
groups do not cross bank partitions. E5 is fitted only on in-scope bank training
rows, so manual-short measures transfer to concise user language rather than
memorization of those examples. Validation selects candidates and thresholds;
the manual test remains untouched until the decision is frozen. Dataset and
decision checksums bind the final report to that protocol.

## Validation decision

| Candidate | Bank accuracy | Bank coverage | Manual accuracy | Manual coverage |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 88.73% | 96.51% | 56.82% | 50.00% |
| multilingual E5 + Logistic Regression | 79.35% | 80.55% | 95.45% | 100.00% |

Coverage uses thresholds chosen for at least 90% accepted validation accuracy.
On the four manual `other` examples, E5 rejected 100% and TF-IDF rejected 75%.
`other` is excluded from multiclass training and measured as an out-of-scope
slice.

TF-IDF's mean merchant-group accuracy is 89.07%. Of 44 held-out merchant groups,
81.82% are entirely correct and 6.82% entirely wrong. Several complete failures
come from synthetic groups whose merchant, semantic detail, and statement
format are simultaneously unseen. They are retained as stress tests rather
than treated as independent real-world errors.

`manual-short-v2` followed a documented taxonomy audit after inspecting v1
validation errors. Metrics across those dataset versions therefore do not show
model improvement. No candidate was changed to target the two remaining v2
manual errors. The exact pre-test selection is stored in
`data/evaluation/transaction_categories/classification_v2_model_selection.json`.
Its bank threshold records the controlled-validation decision made at that
time, but was later rejected for serving after the seeded demo-bank acceptance
check. The record remains immutable because its checksum anchors the manual
final-test result.

## Frozen manual-short test

The fixed E5 candidate and validation-selected threshold were evaluated once on
the previously unused `manual-short-v2` test partition:

| Metric | Result |
|---|---:|
| In-scope rows | 44 |
| Accuracy | 95.45% |
| Macro-F1 | 95.38% |
| Automatic coverage | 97.73% |
| Automatically accepted accuracy | 95.35% |
| `other` rows | 4 |
| `other` rejection rate | 100.00% |

Known-concept/new-phrase accuracy is 90.91%; novel-concept accuracy is 100%.
The result is checksum-bound to the dataset and pre-test selection record in
`classification_v2_manual_test.json`. It is retained without post-test model,
threshold, rule, or dataset changes.

## Feature-fusion experiment

A preselected candidate concatenated normalized character TF-IDF features with
384-dimensional frozen E5 embeddings and trained one Logistic Regression head.
It improved controlled bank validation accuracy to 91.49%, but manual accuracy
fell to 81.82%. On the seeded demo-bank benchmark it reached 63.00% raw hybrid
accuracy; a conservative 0.65 threshold retained 100% accepted accuracy at only
44.07% coverage. The candidate is retained as a rejected experiment and is not
evaluated on the frozen manual test.

An agreement policy is the stronger bank-serving candidate: high-signal rules
run first, then TF-IDF and E5 must predict the same category for automatic
acceptance. It reaches 97.90% accepted accuracy at 76.07% coverage on controlled
bank validation and 93.74% at 62.80% coverage across 1,500 seeded demo-bank
transactions. This policy uses semantic and lexical evidence for every bank
text rather than routing by source or arbitrary text length. Manual entry keeps
the separately frozen E5 suggestion because it is editable and has a different
interaction contract. The policy decision and rejected alternatives are stored
in `classification_v2_bank_policy_selection.json`.

## Serving contract

High-signal rules run first. Remaining bank-feed outflows are encoded in one
batch and automatically categorized only when TF-IDF and E5 agree. A
disagreement stores the E5 suggestion and both model versions for review while
leaving the ledger category unset. Manual entry uses the E5 candidate as an
editable suggestion with its frozen validation threshold.

The API reports `ready`, `degraded`, or `unavailable` at
`GET /v1/transactions/classification/status`. If the semantic artifact or
encoder is unavailable, classification falls back to the conservative v1
TF-IDF threshold and reports degraded status.

## Limitations

All training and evaluation text is synthetic or manually curated. In
particular, 95.45% on 44 in-scope manual test rows must not be interpreted as
broad free-form accuracy: the small curated slice cannot represent the wording,
misspellings, mixed languages, and ambiguity seen during open-ended use. Scores
are uncalibrated, and the bank split intentionally imposes a strong distribution
shift. Independent privacy-reviewed bank data and a larger frozen free-form
evaluation are required before making product-performance claims.

Ad hoc prompts are useful for discovering missing evaluation slices, but are
not added directly to training after a failure. A later iteration should first
version and freeze a broader dataset, then compare data augmentation, a
fine-tuned encoder, and the existing baseline without tuning against its test
partition.
