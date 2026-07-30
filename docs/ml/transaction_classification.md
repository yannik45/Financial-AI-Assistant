# Transaction classification methodology

Status: baseline development  
Last updated: 2026-07-30

## Objective

The transaction classifier predicts one versioned expense category from a bank
transaction description. It is a decision-support component for transaction
organization, not a financial calculation or financial-advice system.

Structured transaction types that can be assigned reliably from source data
remain deterministic and outside this model. The category definitions and
boundary rules are specified in `docs/ml/transaction_categories.md`.

## Current scope and assumptions

- Input: one English transaction description.
- Output: exactly one `transaction-categories-v1` expense label.
- Training data is synthetic and must not be presented as real customer data.
- No real personal or account data is used.
- The first model is a transparent local baseline, not a production-ready
  classifier.
- German descriptions are outside the current training scope. A separate German
  challenge set and deterministic synthetic generator are planned.

## Dataset

The current source is the MIT-licensed synthetic
`DoDataThings/us-bank-transaction-categories-v2` dataset. Provenance, revision,
retrieval date, and SHA-256 checksum are recorded in
`data/external/transaction_categories/metadata.json`.

The raw CSV contains 68,000 rows, 17 source categories, and two columns:
`description` and `category`. The raw file is downloaded to the ignored
`data/runtime` directory and is never committed.

The downloader pins an immutable source revision and verifies the file checksum
before loading. Schema validation rejects missing columns, additional columns,
unknown source categories, and missing or blank descriptions.

## Label mapping and exclusions

Source categories are mapped into the internal taxonomy before training. Notable
decisions include:

- `Rent` and `Mortgage` map to `housing`.
- `Subscription` maps to `other` in version 1 because the source label mixes
  entertainment, software, and other recurring services.
- `Income`, `Transfer`, and `Fees` are excluded because they are outside the
  expense-category model and should preferentially use structured transaction
  information.
- `Personal Care` is excluded because its examples span several internal
  categories and cannot be mapped consistently without more context.

These exclusions remove 16,000 rows and leave 52,000 mapped rows.

## Ambiguity and duplicate handling

An exact description is considered conflicting when it has more than one target
label. All occurrences of a conflicting description are excluded rather than
forcing an arbitrary label.

The source data contained four conflicting descriptions across 12 rows. They
were variants of `RITE AID` and `WALGREENS`, labeled as both `healthcare` and
`shopping`. This ambiguity is plausible because the transaction text does not
identify the purchased items.

After conflict removal, exact duplicate descriptions with the same label are
reduced to their first occurrence. For this synthetic dataset, retaining every
duplicate would overweight repeated generator outputs and could inflate random
split results.

The prepared dataset contains:

- 36,617 unique descriptions;
- 12 target categories;
- zero exact duplicate descriptions;
- zero conflicting descriptions.

The original raw data remains unchanged. All filtering is performed on the
derived training DataFrame.

## Split protocol

The prepared data is split reproducibly with random seed `42` and stratification
by `target_category`:

| Split | Share | Rows | Purpose |
| --- | ---: | ---: | --- |
| Train | 70% | 25,631 | Fit model parameters |
| Validation | 15% | 5,493 | Compare baselines and model choices |
| Test | 15% | 5,493 | Final evaluation only |

Descriptions do not overlap across these splits. The test split has not been
used for baseline selection or reporting.

The current random stratified split does not fully prevent generator-pattern or
merchant-family leakage. Different descriptions derived from the same merchant
or template may still occur in separate splits. A grouped challenge split by
merchant or generator template is required before making generalization claims.

## Evaluation metrics

The primary comparison metrics are:

- Accuracy: overall fraction of correct predictions.
- Macro-F1: unweighted mean of per-category F1 scores.

Macro-F1 is essential because deduplication produces unequal category sizes and
accuracy alone can hide poor performance on smaller categories. Per-category
precision, recall, F1, and a confusion matrix will be added for trained models.

## Baselines

All values below are measured on the validation split. The test split remains
untouched.

| Baseline | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Majority class (`shopping`) | 10.83% | 1.63% |
| Ordered keyword rules | 24.74% | 26.03% |
| Character TF-IDF + logistic regression | 99.25% | 99.28% |

### Majority-class baseline

The scikit-learn `DummyClassifier` with `strategy="most_frequent"` always
predicts the most common training label. It establishes a minimum reference and
does not use transaction text.

### Keyword baseline

The keyword classifier uses a deliberately small ordered rule set. It
case-normalizes descriptions, selects the first matching rule, and falls back to
`other`. The rules represent broad domain terms such as `rent`, `mortgage`,
`hotel`, `pharmacy`, and `supermarket`; they are not intended to enumerate all
merchants.

This baseline is deterministic and explainable but has known limitations:

- coverage requires manual maintenance;
- ambiguous keywords depend on rule precedence;
- unseen merchants and synonyms usually fall back to `other`;
- language transfer is weak;
- rules must not be tuned by inspecting test examples.

Its purpose is to quantify whether a learned model provides value beyond a
small hand-written ruleset.

## Learned baseline

The first learned model uses a scikit-learn pipeline:

1. transaction description text;
2. `char_wb` TF-IDF character n-grams from 3 to 5 characters;
3. class-balanced logistic regression;
4. category prediction and confidence scores.

The vectorizer uses `min_df=2` and sublinear term frequency. The logistic
regression uses `class_weight="balanced"`, `max_iter=1000`, and random seed `42`.
The fitted validation model contains 52,365 TF-IDF features.

Character n-grams are used because bank descriptions contain abbreviations,
prefixes, identifiers, spelling variants, and compressed merchant names. Model
selection will use only train and validation data. The test split will be
evaluated once after the approach and hyperparameters are fixed.

The 99% validation result is not treated as production-level evidence. It is
unusually high for transaction categorization and is consistent with the known
risk that related merchant names and synthetic generator templates occur across
the random train and validation splits. A grouped merchant/template challenge
split must be evaluated before claiming generalization to unseen descriptions.

## Reproducibility and future work

- Dataset source revision and checksum are pinned.
- Preprocessing and split behavior are covered by automated tests.
- The split seed is fixed and versioned.
- Python dependencies are locked with `uv.lock`.
- Model artifacts will record dataset checksum, taxonomy version, split seed,
  feature configuration, model parameters, library versions, and metrics.
- A German synthetic dataset must model German bank-statement formats rather
  than merely translate US descriptions.
- German and English evaluation results must be reported separately.
- Grouped merchant/template evaluation is required in addition to the current
  random stratified baseline.
