# German synthetic transaction training data v2

Generator version: `german-training-generator-v2`  
Base generator version: `german-training-generator-v1`  
Taxonomy version: `transaction-categories-v1`  
Default seed: `20260730`  
Default size: 12,000 rows  
Status: validation completed; test and external challenge remain frozen

## Versioning policy

Published generator versions are immutable reproducibility contracts. Material
changes to merchants, semantic detail groups, formats, split assignments, or
generation rules require a new generator version and new output filename. A
versioned module contains the rules; generated CSV and metadata artifacts remain
under `data/runtime/` and are not committed.

Version 2 imports stable shared primitives from version 1 and records this base
version in its metadata. Version 1 remains available to reproduce its diagnostic
result and is not overwritten by version 2.

## Motivation

Version 1 produced 100% merchant-holdout validation Accuracy and Macro-F1. The
score was caused by category-specific detail phrases and statement templates
remaining shared across train and validation. Version 2 makes the development
evaluation harder without inspecting German challenge errors.

## Data design

The default version 2 dataset contains:

- 1,000 examples per category and 12,000 rows in total;
- 16 fictional merchant families per category;
- 8 semantic detail groups per category;
- 8 global bank-statement format groups with two surface templates each;
- five deterministic casing, spacing, and abbreviation styles;
- synthetic cities, payment types, and references;
- no real account, customer, IBAN, or personal data.

The model input remains only `description`. Provenance columns are used for
quality checks and splitting, not as model features.

| Provenance column | Purpose |
| --- | --- |
| `merchant_group` | Identifies a fictional merchant family |
| `detail_group` | Identifies a category-specific semantic phrase family |
| `format_group` | Identifies a bank-statement wrapper family |
| `template_id` | Identifies the surface template inside a format family |
| `split` | Immutable generator-assigned train, validation, or test partition |

## Split protocol

Version 2 assigns provenance groups before generating rows. The default split
contains 750 train, 125 validation, and 125 test examples per category:

| Dimension | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Merchant groups per category | 12 | 2 | 2 |
| Detail groups per category | 6 | 1 | 1 |
| Global format groups | 6 | 1 | 1 |
| Rows per category | 750 | 125 | 125 |

No merchant, detail, or format group overlaps between the three partitions.
Unlike a random row split, validation therefore combines unseen merchants,
unseen semantic phrases, and an unseen statement format.

The internal test partition and external German challenge set remain untouched
during version 2 development. Challenge descriptions and merchant groups are
also checked for exact disjointness from generated training data.

## Reproducibility

Generate the version 2 runtime artifact:

```powershell
uv run python -m financial_ai.ml.german_training_generator_v2
```

Outputs:

```text
data/runtime/ml/transaction_categories/german_training_v2.csv
data/runtime/ml/transaction_categories/german_training_v2.metadata.json
```

The default generated CSV SHA-256 is
`7b13897e48c58fd443a33627f32ad264c446af84488eaddf3cedbc2eae6c0045`.

## Validation result

The unchanged character TF-IDF and balanced logistic-regression configuration
was fitted on the 9,000 version 2 train rows and evaluated on the 1,500 version
2 validation rows.

| Metric | Result |
| --- | ---: |
| Accuracy | 87.20% |
| Macro-F1 | 86.09% |
| Incorrect predictions | 192 |

Per-category validation results:

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| education | 100.00% | 47.20% | 64.13% | 125 |
| transport | 100.00% | 47.20% | 64.13% | 125 |
| groceries | 100.00% | 52.00% | 68.42% | 125 |
| travel | 65.45% | 100.00% | 79.11% | 125 |
| other | 67.57% | 100.00% | 80.65% | 125 |
| dining | 79.11% | 100.00% | 88.34% | 125 |
| healthcare | 79.11% | 100.00% | 88.34% | 125 |
| entertainment | 100.00% | 100.00% | 100.00% | 125 |
| housing | 100.00% | 100.00% | 100.00% | 125 |
| insurance | 100.00% | 100.00% | 100.00% | 125 |
| shopping | 100.00% | 100.00% | 100.00% | 125 |
| utilities | 100.00% | 100.00% | 100.00% | 125 |

Observed validation confusions:

| Actual | Predicted | Count |
| --- | --- | ---: |
| transport | travel | 66 |
| education | healthcare | 33 |
| education | other | 33 |
| groceries | dining | 33 |
| groceries | other | 27 |

The lower score is expected and desirable relative to version 1: it exposes
generalization weaknesses that the earlier merchant-only holdout hid. It is
still a synthetic validation result and cannot establish real German banking
performance.

## Next decision gate

Before touching the internal test or external challenge, review whether the v2
generator and unchanged baseline are accepted as the fixed development setup.
If accepted, refit once on v2 train plus validation, evaluate the v2 test once,
then evaluate the German challenge once. Later changes require a new model or
generator version rather than repeated tuning on these frozen evaluations.
