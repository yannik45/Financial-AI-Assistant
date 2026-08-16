# German synthetic transaction training data

Generator version: `german-training-generator-v1`  
Taxonomy version: `transaction-categories-v1`  
Default seed: `20260730`  
Default size: 12,000 rows

Status: retained as the initial diagnostic generator. Development continues in
[`german-training-generator-v2`](german_transaction_training_v2.md).

## Purpose

The generator creates a reproducible German-language training baseline for the
12 expense categories. Generated rows are synthetic and must not be represented
as real bank or customer transactions.

The generator source is versioned, while generated CSV and metadata files are
written below `data/runtime/` and are not committed. This avoids storing a large
derived artifact whose exact contents can be reproduced from code and a seed.

## Why 12,000 rows

Version 1 creates 1,000 examples per category. This is large enough to fit and
compare the character TF-IDF and logistic-regression baseline without making
class frequency the primary limitation. The size is an engineering starting
point, not a statistical guarantee. Template and merchant diversity matter more
than producing additional near-duplicate rows.

The current generator contains:

- 12 balanced categories;
- 8 fictional merchant families per category (96 total);
- 8 bank-description templates;
- German payment wrappers, cities, purposes, and synthetic references;
- 12,000 unique descriptions with the default settings.

Future versions should increase independently reviewed merchant and template
diversity before increasing the row count substantially.

## Generated schema

| Column | Purpose |
| --- | --- |
| `example_id` | Stable row identifier for the selected seed and size |
| `description` | Synthetic bank-transaction description used as model input |
| `target_category` | Label from `transaction-categories-v1` |
| `language` | Always `de` |
| `template_id` | Template provenance for leakage analysis |
| `merchant_group` | Fictional merchant-family group for grouped splitting |

`template_id` and `merchant_group` are provenance fields, not model features.
The first learned baseline continues to use only `description` as input.

## Reproducibility

Generate the default dataset from the repository root:

```powershell
uv run python -m financial_ai.ml.transaction_classification.data.german_training_generator
```

Outputs:

```text
data/runtime/ml/transaction_categories/german_training_v1.csv
data/runtime/ml/transaction_categories/german_training_v1.metadata.json
```

The metadata records generator version, taxonomy version, seed, row count, rows
per category, and the generated CSV SHA-256 checksum.

## Leakage controls

- German challenge descriptions are never used as generator inputs.
- Generated merchant groups use a dedicated `generated_` namespace and do not
  overlap challenge merchant groups.
- Automated tests check exact description and merchant-group disjointness.
- The 120-row German challenge set remains evaluation-only.
- Model development must not tune generator rules against individual challenge
  errors. Material changes require a new documented generator/model version.

Training and internal validation splits must keep each `merchant_group` within
one split. Random row splitting would place near-identical rows from one merchant
family on both sides and overstate generalization.

## Limitations

- The data is template-generated rather than sampled from real German banks.
- Eight merchant families per category provide limited entity diversity.
- Synthetic references can create uniqueness without adding semantic diversity.
- Templates may be easier for a character model than real truncated statements.
- Results must be reported separately from English and real-data evaluations.
- Production claims require a lawful, licensed, anonymized real-data benchmark.

## Diagnostic validation result

The unchanged character TF-IDF and logistic-regression pipeline reached 100%
Accuracy and 100% Macro-F1 on the merchant-group validation split. This is not
treated as model quality evidence. Merchant groups were separate, but category
details and statement templates still overlapped between fitting and validation
data. The result demonstrated that version 1 measured generator-pattern recall
rather than sufficiently difficult generalization. No version 1 test or German
challenge evaluation was performed for the German-trained model.
