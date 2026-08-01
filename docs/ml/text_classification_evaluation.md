# Text classification evaluation

Evaluation version: `text-classification-evaluation-v2`

Challenge version: `text-classification-challenge-v2`

Model: `transaction-category-char-tfidf-bilingual-v1`

Status: frozen development benchmark; not production evidence

## Objective

This evaluation closes the current transaction-classification experiment with
a product-level comparison of three approaches:

1. `text_rules_only`: the small, reviewable bilingual phrase baseline;
2. `tfidf_only`: the bilingual character TF-IDF and Logistic Regression expense
   model without product rules, while retaining the cash-flow gate that prevents
   positive transactions from being forced into expense labels;
3. `hybrid`: the current product routing of text rules, expense model, and
   review for unsupported inputs.

The comparison deliberately reports abstention behavior. A classifier that
returns fewer but safer automatic suggestions can be more useful than one that
always guesses.

## Frozen challenge set

The committed CSV contains 252 manually authored examples:

- 18 product categories;
- English and German;
- seven examples per category and language;
- easy, medium, and hard descriptions;
- explicit ambiguity flags;
- positive and negative signed cash-flow context.
- non-empty counterparties on a declared subset of cases.

The CSV and metadata are stored in
`data/evaluation/transaction_categories/`. Metadata includes a SHA-256 checksum
calculated after canonical LF line-ending normalization so integrity checks are
stable on Windows and Linux.
Exact development examples such as `Salary`, `House Payment`, `Coffee Shop`,
`Amazon`, and `Überweisung Mama` remain regression-test inputs and are excluded
from the challenge.

Version 1 remains committed as an immutable historical snapshot. A repository
audit found that its dividend-credit descriptions incorrectly had negative
amounts and that every counterparty was empty. Version 2 supersedes it rather
than silently rewriting a frozen benchmark. It corrects distribution cash flows,
adds counterparty coverage, verifies committed checksums on load, and evaluates
the direction-aware rule behavior.

The set was designed alongside the application and is therefore a development
benchmark, not an independent real-world test set. It cannot establish
production accuracy. The dataset was frozen before the first result run, and no
rules, model parameters, or challenge labels were tuned after observing the
results below.

Generate and validate the snapshot with:

```powershell
uv run financial-ai-build-text-challenge
```

or with an existing local environment:

```powershell
.\.venv\Scripts\python.exe -m financial_ai.ml.text_classification_challenge
```

## Evaluation design

The full-system view includes all 18 categories. The expense-only view contains
the 12 labels the learned model was trained to predict. This distinction is
essential: the TF-IDF model cannot emit income, investment, fee, tax, savings,
or cash labels, and unmatched positive transactions are intentionally sent to
review rather than forced into an expense class.

Reported metrics are:

- **Accuracy:** correct predictions divided by all examples. Abstentions count
  as incorrect in this metric.
- **Macro-F1:** the unweighted mean of per-category F1, so large categories do
  not hide weak smaller categories.
- **Precision and recall per category:** prediction purity and recovered share
  of the category, respectively.
- **Prediction coverage:** share with any proposed category, including reviewed
  low-confidence proposals.
- **Review rate:** share marked `needs_review`.
- **Auto-acceptance rate:** share with a prediction above the current acceptance
  policy or covered by a rule.
- **Selective accuracy:** accuracy only among automatically accepted examples.
- **Rule coverage:** share handled by a text rule.

The report also contains confusion matrices and slices by language, difficulty,
and ambiguity. The review threshold is `0.65`.

`predict_proba` output is used only as an experimental ranking score. It is not
claimed to be calibrated probability. Calibration curves are deferred until an
independent, representative calibration dataset exists; calibrating and judging
the model on this same development challenge would leak evaluation information.

## Frozen results

### Overall comparison

| Strategy | Accuracy | Macro-F1 | Prediction coverage | Review rate | Auto-acceptance | Selective accuracy | Rule coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text rules only | 32.1% | 46.1% | 33.7% | 66.3% | 33.7% | 95.3% | 33.7% |
| TF-IDF only | 57.1% | 47.6% | 92.9% | 56.8% | 43.3% | 96.3% | 0.0% |
| Hybrid | 69.8% | 68.4% | 96.0% | 39.3% | 60.7% | 96.1% | 33.3% |

### Expense-only comparison

| Strategy | Accuracy | Macro-F1 |
|---|---:|---:|
| Text rules only | 28.6% | 42.7% |
| TF-IDF only | 85.7% | 84.6% |
| Hybrid | 85.1% | 84.8% |

The hybrid improves broad product coverage and expense accuracy over either
component alone across the complete taxonomy. Its 69.8% overall accuracy is not
production-ready. The more useful experimental finding is that it automatically
accepts 60.7% of examples at 96.1% selective accuracy, while routing the
remainder to review.

### Hybrid slices

| Slice | Accuracy | Macro-F1 | Review rate | Selective accuracy |
|---|---:|---:|---:|---:|
| English | 72.2% | 71.9% | 30.2% | 95.5% |
| German | 67.5% | 64.8% | 48.4% | 96.9% |
| Easy | 83.3% | 84.0% | 23.6% | 98.2% |
| Medium | 63.9% | 60.7% | 40.7% | 93.8% |
| Hard | 65.3% | 60.3% | 52.8% | 97.1% |

The decreasing accuracy and increasing review rate from easy to hard are
directionally sensible. German coverage is lower than English coverage. Weak
full-system categories include `other`, `fees`, `taxes`, `savings`, and
`investments`; these are candidates for future data work, not post-benchmark
tuning in this version.

### Historical v1 record

Version 1 reported 62.3% hybrid accuracy, 60.8% Macro-F1, 54.4%
auto-acceptance, and 93.4% selective accuracy. Those numbers remain historical
and must not be compared as a model improvement: v2 corrects evaluation inputs
and cash-flow behavior rather than changing the learned model.

## Reproduction and artifacts

Bootstrap the local model first, then run:

```powershell
uv run financial-ai-bootstrap-category-model
uv run financial-ai-evaluate-text-classification
```

Windows fallback:

```powershell
.\.venv\Scripts\python.exe -m financial_ai.ml.category_bootstrap
.\.venv\Scripts\python.exe -m financial_ai.ml.text_classification_evaluation
```

The generated JSON report under `data/runtime/ml/transaction_categories/`
records the evaluation and challenge versions, threshold, model version, model
artifact checksum, all aggregate metrics, per-category metrics, and confusion
matrices. It is reproducible and intentionally ignored instead of adding a
large generated file to Git. The model pickle is also an ignored local artifact
because pickle files must not be accepted from untrusted sources. The compact
frozen results relevant to review are recorded in this document.

## Limitations and next decision

- The challenge is manually authored and small.
- Training data is controlled/synthetic rather than representative bank data.
- Rules and challenge text share human vocabulary and are not independent.
- No probability calibration, temporal validation, subgroup fairness study, or
  production drift analysis has been performed.
- Selective accuracy is measured on the same development benchmark and will
  need confirmation on independently collected data.

This version freezes classification feature work. Reviewed feedback can enter
the guarded offline candidate lifecycle, but must not trigger automatic online
retraining. Containerization is complete; the next planned ML feature is
transaction fraud/risk scoring.
