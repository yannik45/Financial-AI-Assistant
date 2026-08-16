# ML documentation

The repository contains two applied-ML tracks: a bilingual hybrid transaction
classifier with governed feedback, and offline market-volatility forecasting
with leakage-aware temporal evaluation. Both retain transparent references,
versioned inputs, and explicit production-readiness limits.

Each track separates `data`, `modeling`, and `evaluation` modules. Transaction
classification additionally isolates its online `core` and offline `feedback`
lifecycle; market-forecast serving remains in `financial_ai.market_forecast_service`.

## Start here

| Document | What it answers |
|---|---|
| [Category taxonomy](transaction_categories.md) | Which labels exist and where are their boundaries? |
| [Classifier service](transaction_classifier_service.md) | What does the current API classifier do? |
| [Classification v2 selection](transaction_classification_v2.md) | Why are bank and manual descriptions routed to different models? |
| [Frozen product evaluation](text_classification_evaluation.md) | How do rules, ML, and the hybrid compare? |
| [Feedback lifecycle](transaction_classification_feedback.md) | How are corrections exported, evaluated, and promoted safely? |
| [Market forecast data foundation](market_forecast_data_foundation.md) | How do daily market observations become reproducible, leakage-aware datasets? |
| [Market volatility baseline](market_volatility_baseline.md) | Do naive, statistical, and regularized linear forecasts beat simple references? |
| [Market volatility boosting](market_volatility_boosting.md) | Does a preselected nonlinear model improve the frozen references, and where does it fail? |

Aggregate market-forecast evidence is stored under
[`data/evaluation/market_forecast`](../../data/evaluation/market_forecast/);
full datasets and reports remain ignored runtime artifacts.

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
