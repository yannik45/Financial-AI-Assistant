# ML documentation

The current product baseline is a text-first bilingual hybrid: transparent
rules handle high-signal cases and a character TF-IDF Logistic Regression model
classifies unmatched expenses. Suggestions are editable, uncertain cases can
abstain, and model changes follow an offline gated lifecycle. It is an
experimental synthetic-data baseline, not a production classifier.

## Start here

| Document | What it answers |
|---|---|
| [Category taxonomy](transaction_categories.md) | Which labels exist and where are their boundaries? |
| [Classifier service](transaction_classifier_service.md) | What does the current API classifier do? |
| [Frozen product evaluation](text_classification_evaluation.md) | How do rules, ML, and the hybrid compare? |
| [Feedback lifecycle](transaction_classification_feedback.md) | How are corrections exported, evaluated, and promoted safely? |
| [Market forecast data foundation](market_forecast_data_foundation.md) | How will daily market observations become reproducible, leakage-aware datasets? |

## Reproducibility records

These documents preserve experiment decisions and results. They are evidence,
not alternative product instructions.

| Record | Status |
|---|---|
| [Multilingual classification](multilingual_transaction_classification.md) | Frozen model-selection experiment |
| [Controlled English training](controlled_english_training.md) | Frozen English generator and evaluation |
| [German training v2](german_transaction_training_v2.md) | Frozen German generator used by the model |
| [German challenge](german_transaction_challenge.md) | Frozen German-only evaluation |
| [Original English methodology](transaction_classification.md) | Historical external-dataset baseline |
| [German training v1](german_transaction_training.md) | Historical, superseded generator |

Corresponding modules and tests remain versioned so the results can be
reproduced. New product code should use the active artifact builder and service,
not historical evaluation runners.
