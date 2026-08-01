# Transaction classifier service

Service: `transaction-classifier-service-v3-direction-aware`

Model: `transaction-category-char-tfidf-bilingual-v1`

Taxonomy: `transaction-categories-v1`

Status: experimental synthetic-data baseline

## Contract

`POST /v1/transactions/classify` suggests a category from description,
optional counterparty, and signed amount. Negative values are outflows and
positive values are inflows; amount magnitude and stored transaction type are
not model features.

```text
text + counterparty + direction
  -> bilingual high-signal rules
  -> unmatched outflow: expense ML model
  -> unmatched inflow: abstain for review
```

This avoids target leakage from asking a user to select a detailed transaction
type before classification. Direction-inconsistent rule matches, such as a
negative salary, also abstain.

## Hybrid behavior

The small version-controlled rule list covers phrases such as salary, rent,
fees, tax, insurance, ATM, `Gehalt`, and `Miete`. Boundary-aware matching avoids
substrings such as `fee` inside `coffee`. Rules are an auditable reference, not
a merchant dictionary.

Unmatched outflows use character TF-IDF and class-balanced Logistic Regression.
Rule results have no numeric confidence. ML results expose an uncalibrated
`predict_proba` score and abstain below the configured review threshold.

The API rejects blank descriptions and zero amounts. If a required model
artifact is absent, model-routed classification returns
`503 category_model_unavailable`; rule matches remain available.

## Artifact

The expense model is trained on the controlled English v1 and German v2 train
plus validation partitions. Frozen test data is excluded. Build it with:

```powershell
uv run financial-ai-bootstrap-category-model
```

Generated pickle and checksum-verified metadata live under
`data/runtime/ml/models` and are ignored by Git. Metadata records input hashes,
versions, model configuration, random state, and library versions. Never load a
pickle from an untrusted location.

The transaction form keeps suggestions editable. Saving repeats classification
on the backend and stores trusted prediction provenance with the user's final
category. See the [feedback lifecycle](transaction_classification_feedback.md)
and the live OpenAPI page at `/docs` for the exact request and response schema.

## Limitations

Training data is synthetic, free-form coverage is limited, and probabilities
are uncalibrated. Production use would require lawful representative data,
multilingual external evaluation, probability calibration, privacy controls,
monitoring, and deployment governance. Current benchmark results are documented
in [text classification evaluation](text_classification_evaluation.md).
