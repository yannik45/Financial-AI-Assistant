# Transaction classifier service

Service version: `transaction-classifier-service-v1`  
Model version: `transaction-category-char-tfidf-bilingual-v1`  
Taxonomy version: `transaction-categories-v1`

## Purpose

The service turns structured transaction types and merchant descriptions into a
versioned product category. It deliberately keeps deterministic business rules
separate from the expense machine-learning model.

```text
transaction type
  -> deterministic product category
  -> expense model
  -> needs review
```

The service does not calculate financial metrics and does not use an LLM.

## Routing contract

| Transaction type | Route | Product category |
| --- | --- | --- |
| `salary`, `interest` | deterministic | `income` |
| `dividend`, `security_buy`, `security_sell` | deterministic | `investments` |
| `fee` | deterministic | `fees` |
| `tax` | deterministic | `taxes` |
| `deposit` | deterministic | `savings` |
| `cash_withdrawal` | deterministic | `cash` |
| `card_payment`, `direct_debit` | expense model | one `ExpenseCategory` |
| `transfer`, `withdrawal` | needs review | none |

Transfers remain unresolved because an own-account transfer and a payment to a
third party cannot be distinguished safely from the transaction type alone.

## Model artifact

The selected bilingual character TF-IDF and balanced Logistic Regression model
is fitted on the controlled English v1 and German v2 `train + validation`
partitions. Frozen test rows are excluded.

Generate the local artifact from the repository root after generating both
controlled datasets:

```powershell
uv run financial-ai-build-category-model
```

The command writes generated local files under:

```text
data/runtime/ml/models/transaction_category_bilingual_v1.pkl
data/runtime/ml/models/transaction_category_bilingual_v1.json
```

The JSON sidecar records the model and taxonomy versions, training row count,
languages, source checksums, creation time, and artifact checksum. The loader
rejects a checksum or taxonomy mismatch. Pickle is unsafe for untrusted input;
the application only loads its locally generated, checksum-verified artifact.
Model files remain outside Git because they are reproducible runtime artifacts.

## API

`POST /v1/transactions/classify`

Example request:

```json
{
  "transaction_type": "card_payment",
  "description": "REWE MARKT 1842 BERLIN",
  "counterparty": "REWE"
}
```

Example response shape:

```json
{
  "category": "groceries",
  "route": "expense_model",
  "classification_method": "ml",
  "confidence": 0.82,
  "needs_review": false,
  "reason": "Expense category predicted by the versioned model artifact.",
  "taxonomy_version": "transaction-categories-v1",
  "model_version": "transaction-category-char-tfidf-bilingual-v1"
}
```

Deterministic results use confidence `1.0` and no model version. Ambiguous
routes return no category or confidence and set `needs_review` to `true`.

Expense predictions below the configured review threshold, `0.65` by default,
retain the predicted category but set `needs_review` to `true`. Logistic
Regression `predict_proba` is used as a confidence proxy; it has not yet been
calibrated and must not be interpreted as a guaranteed real-world probability.

If an expense request arrives before the artifact is built, the API returns
`503 category_model_unavailable`. Deterministic routes do not load the artifact
and remain available.

## Configuration

Environment variables:

- `FINANCIAL_AI_CATEGORY_MODEL_ARTIFACT_PATH`
- `FINANCIAL_AI_CATEGORY_MODEL_METADATA_PATH`
- `FINANCIAL_AI_CATEGORY_REVIEW_THRESHOLD`

## Current limitations

- Training data is synthetic and does not establish production performance.
- Confidence is not calibrated on real banking data.
- The endpoint predicts a category but does not yet persist user corrections.
- Manual transaction creation does not automatically overwrite a supplied
  category.
- Model loading is local; a deployment will need a controlled artifact build,
  registry, or immutable container packaging workflow.
