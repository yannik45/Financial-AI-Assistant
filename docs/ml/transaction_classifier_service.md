# Transaction classifier service

Service: `transaction-classifier-service-v4-semantic-agreement`

Models: `transaction-category-char-tfidf-bilingual-v1` and
`transaction-category-e5-head-v1`

Taxonomy: `transaction-categories-v1`

Status: experimental synthetic-data baseline

The evaluated semantic and agreement policy is documented in
[transaction classification v2](transaction_classification_v2.md).

Category boundaries and precedence rules are defined in the
[transaction category taxonomy](transaction_categories.md).

## Contract

`POST /v1/transactions/classify` suggests a category from description,
optional counterparty, and signed amount. Negative values are outflows and
positive values are inflows; amount magnitude and stored transaction type are
not model features.

```text
text + counterparty + direction
  -> bilingual high-signal rules
  -> unmatched bank-feed outflow: TF-IDF + E5 agreement
  -> unmatched manual outflow: editable E5 suggestion
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

Bank-feed auto-acceptance requires agreement between character TF-IDF and a
Logistic Regression head over frozen multilingual E5 embeddings. Manual input
uses the semantic candidate because its wording differs from bank statements.
Scores remain uncalibrated and are not presented as probabilities.

The API rejects blank descriptions and zero amounts. A missing semantic artifact
activates the explicit TF-IDF fallback; a missing lexical artifact returns
`503 category_model_unavailable` for model-routed requests. Rules remain available.

## Artifact

The lexical artifact uses controlled English v1 and German v2 train plus
validation partitions. The semantic head uses only in-scope bank training rows;
manual-short and all validation/test rows remain excluded. Build both with:

```powershell
uv run financial-ai-bootstrap-category-model
```

Generated pickle, safe NumPy arrays, and checksum-verified metadata live under
`data/runtime/ml/models` and are ignored by Git. Metadata records input hashes,
versions, model configuration, random state, and library versions. Never load a
pickle from an untrusted location.

The bootstrap is idempotent: valid artifacts are reused. Semantic dependencies
are installed with `uv sync --all-groups --extra semantic`. If semantic startup
cannot complete, the API remains available in an explicit TF-IDF fallback mode.

The transaction form keeps suggestions editable. Saving repeats classification
on the backend and stores trusted prediction provenance with the user's final
category. See the [feedback lifecycle](transaction_classification_feedback.md)
and the live OpenAPI page at `/docs` for the exact request and response schema.

## Interactive bank-feed simulation

`POST /v1/transactions/demo-bank-feed` generates a synthetic monthly bank feed
for a cash account. A random seed varies merchant text, counterparties, dates,
and plausible amounts; supplying the same seed and month reproduces the same
scenario and is idempotent for that account.

New rows are classified in one batch. Ledger creation remains atomic, and
classification provenance records the input source, both candidate predictions,
model versions, and agreement decision. Review corrections are persisted as
strong feedback labels.

The generator's expected categories are used only to report interactive demo
accuracy. They are never classifier inputs or training labels. This simulation
is useful for product exploration and error discovery, but it does not replace
the frozen challenge-set evaluation.

## Limitations

Training data is synthetic, open-ended manual wording is not represented by a
large independent test, and scores are uncalibrated. The strong small
manual-short result is therefore not evidence of broad free-form reliability.
Production use would require lawful representative data,
multilingual external evaluation, probability calibration, privacy controls,
monitoring, and deployment governance. Current benchmark results are documented
in [text classification evaluation](text_classification_evaluation.md).
