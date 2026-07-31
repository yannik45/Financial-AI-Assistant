# Transaction classifier service

Service version: `transaction-classifier-service-v2-text-first`
Expense model: `transaction-category-char-tfidf-bilingual-v1`
Taxonomy: `transaction-categories-v1`
Status: experimental portfolio baseline, not production-ready

## Purpose

The service suggests a category from transaction name/description,
counterparty, and signed amount. A user does not need to select a detailed
transaction type before receiving a useful suggestion. Stored transaction types
remain optional source metadata and are never classifier features.

```text
name + counterparty + signed amount
  -> small bilingual text-rule baseline
  -> unmatched outflow: expense ML model
  -> unmatched inflow: needs review
```

The signed amount provides only cash-flow direction. Negative amounts are
outgoing and positive amounts are incoming. The amount magnitude is not used as
an ML feature.

## Why the contract changed

Version 1 routed `salary`, `fee`, and similar cases from a manually selected
transaction type. That produced good deterministic results but poor product
behavior: the user effectively supplied the answer before classification. The
text-first version removes this target leakage from the interaction and makes
the experiment a more meaningful demonstration of classification trade-offs.

## Hybrid baseline

The current system is deliberately hybrid:

1. A short, version-controlled bilingual rule list handles high-signal phrases
   such as `salary`, `income`, `house payment`, `rent`, `bank fee`, `tax`,
   `insurance`, `ATM`, `Gehalt`, and `Miete`.
2. An unmatched outgoing transaction is passed to the existing bilingual
   character TF-IDF and balanced Logistic Regression expense model.
3. An unmatched incoming transaction is marked `needs_review`; the expense-only
   model is not allowed to invent an income category.

Rules use word/phrase boundaries, so a phrase such as `coffee shop` does not
match the standalone `fee` rule. The list is intentionally small and auditable;
it is a reference baseline, not an attempt to encode every merchant.

Text-rule responses use route `text_rule`, method `keyword_rule`, and no numeric
confidence. ML responses expose the uncalibrated `predict_proba` score and use
the configured review threshold.

## Model artifact

The expense model is fitted on controlled English v1 and German v2
`train + validation` partitions. Frozen test rows are excluded. Build the local
artifact with either:

```powershell
uv run financial-ai-build-category-model
```

or, when an existing `.venv` is available but `uv` is not on `PATH`:

```powershell
.\.venv\Scripts\python.exe -m financial_ai.ml.category_artifact
```

The generated pickle and checksum-verified JSON metadata remain under
`data/runtime/ml/models` and are excluded from Git. Pickle must never be loaded
from an untrusted location.

## API

`POST /v1/transactions/classify`

```json
{
  "description": "House Payment",
  "amount": "-950.00",
  "counterparty": "Demo Property Management"
}
```

Example response:

```json
{
  "category": "housing",
  "route": "text_rule",
  "classification_method": "keyword_rule",
  "confidence": null,
  "needs_review": false,
  "reason": "Category matched a reviewable text rule in the experimental baseline.",
  "taxonomy_version": "transaction-categories-v1",
  "model_version": null
}
```

The endpoint rejects blank descriptions and zero amounts. If an unmatched
outflow needs the model but its artifact is missing, the endpoint returns
`503 category_model_unavailable`. Text rules remain available without it.

## Interpretation and recruiter-facing scope

This implementation demonstrates:

- separation between product input and labels;
- a transparent rule baseline combined with a learned model;
- signed cash-flow context without using amount magnitude as a shortcut;
- versioned model artifacts and taxonomy;
- abstention/review behavior;
- feedback provenance;
- explicit limitations and reproducible tests.

It must not be presented as production performance. Training data is synthetic,
free-form coverage is limited, rule coverage is intentionally incomplete, and
the ML confidence is uncalibrated. A production candidate requires lawful real
or independently realistic data, representative multilingual evaluation,
calibration, monitoring, privacy controls, and deployment governance.
