# Multilingual transaction classification

Experiment status: final controlled tests completed and frozen  
Model configuration: unchanged character TF-IDF and balanced logistic regression  
Languages: English and German  
Taxonomy: `transaction-categories-v1`

## Purpose

This experiment measures whether one classifier can retain the established
English baseline while learning the German v2 synthetic training distribution.
English and German metrics are always reported separately; a pooled metric could
hide a material regression in one language.

The internal English test, German v2 test, and external German challenge set
remain frozen during multilingual development.

## Natural-size concatenation baseline

The first multilingual baseline concatenates the two training partitions without
resampling or language weights:

| Training source | Rows |
| --- | ---: |
| English grouped train | 25,644 |
| German v2 train | 9,000 |
| Combined | 34,644 |

Only `description` and `target_category` are passed to the model. English source
metadata and German provenance fields are excluded from model features.

Separate validation results:

| Model | Validation language | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| English-only | English | 97.57% | 97.66% |
| German-only v2 | German | 87.20% | 86.09% |
| Multilingual natural-size | English | 97.52% | 97.62% |
| Multilingual natural-size | German | 84.60% | 83.30% |

The multilingual model retains essentially all English validation performance,
but German Accuracy falls by 2.60 percentage points and German Macro-F1 by 2.79
percentage points relative to the German-only model. The training mixture
contains approximately 2.85 English rows for every German row, so the result is
consistent with language-volume dominance. It does not prove that row count is
the only cause.

Weakest German validation categories for the natural-size baseline:

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| transport | 100.00% | 40.00% | 57.14% | 125 |
| education | 100.00% | 47.20% | 64.13% | 125 |
| groceries | 100.00% | 49.60% | 66.31% | 125 |
| other | 62.19% | 100.00% | 76.69% | 125 |
| healthcare | 75.00% | 79.20% | 77.04% | 125 |
| travel | 65.26% | 99.20% | 78.73% | 125 |

## Next validation experiment

Before opening any test set, compare a deterministic language-balanced baseline.
Downsample the larger English train partition to 9,000 rows using a fixed seed,
retain all 9,000 German train rows, fit the unchanged pipeline, and evaluate the
same English and German validation partitions separately.

This is a development comparison rather than a final model. The choice between
natural-size and language-balanced training must be fixed using validation only.

## Category- and language-balanced validation

The balanced variant deterministically samples the English train partition to
match the German row count within every target category. It therefore uses 9,000
English and 9,000 German rows. Sampling is stable when source-row order changes,
and only `description` and `target_category` enter the combined model data.

| Model | Validation language | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Multilingual natural-size | English | 97.52% | 97.62% |
| Multilingual natural-size | German | 84.60% | 83.30% |
| Multilingual balanced 1:1 | English | 91.67% | 91.80% |
| Multilingual balanced 1:1 | German | 86.40% | 85.16% |

Relative to natural-size concatenation, 1:1 balancing improves German Accuracy
by 1.80 percentage points and German Macro-F1 by 1.86 points. English Accuracy
falls by 5.85 points and English Macro-F1 by 5.82 points. Relative to the
German-only model, the balanced multilingual model still trails by 0.80 points
of Accuracy and 0.93 points of Macro-F1.

The 1:1 legacy/German variant is not accepted as a final multilingual candidate:
its modest German gain does not justify the substantially larger English
regression. More importantly, the comparison mixes different evaluation designs.
The proposed 2:1 experiment was therefore cancelled in favor of a controlled
English benchmark with the same provenance holdouts as German v2.

## Controlled bilingual comparison

Controlled English v1 and German v2 each contain 9,000 train, 1,500 validation,
and 1,500 frozen test rows. Both split merchant, semantic-detail, and global
format groups before generating rows. This makes the language comparison more
meaningful, although both datasets remain synthetic.

Character-TF-IDF validation results:

| Model | English Accuracy / Macro-F1 | German Accuracy / Macro-F1 |
| --- | ---: | ---: |
| English-only | 92.73% / 92.41% | not measured |
| German-only | not measured | 87.20% / 86.09% |
| Bilingual controlled 1:1 | 93.53% / 93.21% | 85.00% / 84.51% |

The bilingual character model improves controlled English by 0.80 percentage
points of Accuracy and 0.79 points of Macro-F1. German falls by 2.20 points of
Accuracy and 1.58 points of Macro-F1 relative to German-only. This is a smaller
and more interpretable trade-off than the legacy/German size comparison.

## Word plus character feature comparison

A second linear model combines the existing character TF-IDF features with word
unigrams and bigrams. Logistic Regression, class balancing, random seed, and all
data splits remain unchanged.

| Model | English Accuracy / Macro-F1 | German Accuracy / Macro-F1 |
| --- | ---: | ---: |
| Word+Character English-only | 93.40% / 93.08% | not measured |
| Word+Character German-only | not measured | 77.13% / 74.75% |
| Word+Character bilingual 1:1 | 95.20% / 94.92% | 79.53% / 76.09% |

Word features improve English but materially reduce German holdout performance.
The likely explanation is that exact words and word pairs overfit training detail
groups, while character n-grams transfer better to unseen German compounds and
surface variants. This is an inference from the controlled results, not proof of
causality.

## Validation decision

The Word+Character candidate is rejected because of its German regression. The
Character-TF-IDF bilingual 1:1 model is selected as the final multilingual
candidate. English-only and German-only character models remain specialist
reference baselines.

No controlled test or external challenge result was inspected during this model
selection. The next gate is to refit the three predeclared character candidates
on their respective train-plus-validation data, then evaluate each frozen test
once and evaluate external challenges according to their documented protocols.

## Final frozen evaluation

The three predeclared character models were refitted after validation decisions
were frozen:

| Candidate | Fit rows |
| --- | ---: |
| Controlled English-only | 10,500 |
| German-only v2 | 10,500 |
| Controlled bilingual 1:1 | 21,000 |

Final controlled-test results:

| Model | English Accuracy / Macro-F1 | German Accuracy / Macro-F1 |
| --- | ---: | ---: |
| English-only | 75.67% / 72.91% | not evaluated |
| German-only | not evaluated | 80.27% / 77.79% |
| Controlled bilingual | 82.13% / 81.71% | 75.73% / 71.27% |

The bilingual model improves controlled English test performance relative to the
English specialist but trails the German specialist by 4.53 percentage points of
Accuracy and 6.52 points of Macro-F1 on the German controlled test. The
multilingual candidate remains useful as the single-model reference; specialist
models remain stronger references for German controlled-test performance.

Validation-to-test gaps are substantial:

| Candidate and language | Validation Accuracy | Test Accuracy | Gap |
| --- | ---: | ---: | ---: |
| English-only / English | 92.73% | 75.67% | -17.06 pp |
| German-only / German | 87.20% | 80.27% | -6.93 pp |
| Bilingual / English | 93.53% | 82.13% | -11.40 pp |
| Bilingual / German | 85.00% | 75.73% | -9.27 pp |

The gaps indicate that the frozen test provenance groups are harder than the
validation groups. No generator or model change was made after observing them.

German Challenge v1 results:

| Model | Overall Accuracy / Macro-F1 | Local | International |
| --- | ---: | ---: | ---: |
| German-only | 75.00% / 74.96% | 77.65% / 77.64% | 68.57% / 68.33% |
| Controlled bilingual | 75.00% / 74.64% | 75.29% / 74.80% | 74.29% / 73.81% |

The bilingual model is more even across merchant scopes, while overall challenge
Accuracy is identical. The challenge contains only 120 synthetic examples, so
these differences are descriptive rather than statistically conclusive.

The machine-readable runtime report is generated by:

```powershell
uv run python -m financial_ai.ml.multilingual_final_evaluation
```

Output:
`data/runtime/ml/transaction_categories/final_model_evaluation_v1.json`  
SHA-256:
`914044ec5c9934f66ed84c64584788f633f0668745e9f91e526a9dca68348bd5`

All controlled test and challenge results in this section are now frozen. Future
embedding or model experiments require a new versioned evaluation protocol and
must not tune against these reported errors.
