# Controlled English synthetic transaction training data v1

Generator version: `controlled-english-training-generator-v1`  
Taxonomy version: `transaction-categories-v1`  
Default seed: `20260730`  
Default size: 12,000 rows  
Status: final controlled test completed and frozen

## Purpose

This generator provides an English benchmark with the same row counts,
provenance fields, and holdout rules as German generator v2. It supplements the
legacy English synthetic dataset; it does not replace or relabel that benchmark.

## Legacy-train pattern analysis

Only the 25,644-row legacy English grouped train partition was analyzed. Legacy
validation and test rows were not used to design this generator. Aggregate train
properties were:

| Property | Value |
| --- | ---: |
| Median description length | 34 characters |
| 10th–90th length interval | 21–59 characters |
| Median word count | 5 |
| Rows containing digits | 66.79% |
| Rows with bracketed debit prefix | 94.48% |
| Rows with `WEB` identifier pattern | 11.39% |
| Rows with `PPD` identifier pattern | 8.64% |
| Rows with hash or star reference | 25.09% |
| Rows with long numeric reference | 50.33% |

These statistics informed general format families only. No legacy description or
merchant was copied into controlled train, validation, or test data.

## Controlled design

The generator contains 1,000 examples per category and uses the same controlled
structure as German v2:

| Dimension | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Merchant groups per category | 12 | 2 | 2 |
| Detail groups per category | 6 | 1 | 1 |
| Global format groups | 6 | 1 | 1 |
| Rows per category | 750 | 125 | 125 |

The complete runtime artifact contains 12,000 unique descriptions, 192 merchant
groups, 96 category-specific detail groups, and 8 global format groups. No group
overlaps between train, validation, and test.

Format families include bracketed debit descriptions, `WEB` and `PPD`
identifiers, processor wrappers, card purchases, recurring charges, transfers,
and compact descriptions. Merchants and references are fictional.

## Reproducibility

```powershell
uv run python -m financial_ai.ml.transaction_classification.english_training_generator_v1
```

Outputs:

```text
data/runtime/ml/transaction_categories/english_training_v1.csv
data/runtime/ml/transaction_categories/english_training_v1.metadata.json
```

Default CSV SHA-256:
`6e15882563e85956efa9782d36e411bc39b6fffe06f43e806199bea498bca4a5`.

## Character baseline validation

The unchanged character TF-IDF and balanced logistic-regression model was fitted
on 9,000 controlled English train rows and evaluated on 1,500 validation rows:

| Metric | Result |
| --- | ---: |
| Accuracy | 92.73% |
| Macro-F1 | 92.41% |

The result is materially below the 97.57% / 97.66% legacy grouped validation
result. This supports the conclusion that explicit merchant, semantic-detail,
and format holdouts are harder than the legacy text-derived grouping heuristic.

## Final controlled test

After model selection was frozen, the unchanged character model was refitted on
10,500 controlled English train-plus-validation rows and evaluated once on the
1,500-row controlled test:

| Metric | Validation | Final test |
| --- | ---: | ---: |
| Accuracy | 92.73% | 75.67% |
| Macro-F1 | 92.41% | 72.91% |

`groceries` has 0% test recall and F1; `utilities` is the next weakest category
at 47.83% F1. The large validation-test gap shows that the test detail and format
holdouts are substantially harder. These errors are reported without modifying
the generator, split, features, or classifier.

## Limitations

- Data remains synthetic and template-generated.
- Aggregate legacy format statistics do not establish real-bank realism.
- Generated semantic groups reflect manually authored taxonomy assumptions.
- The controlled English test has been evaluated once and is now frozen for
  reporting rather than further model selection.
- Production claims require a lawful, licensed, anonymized external benchmark.
