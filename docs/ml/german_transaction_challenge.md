# German transaction challenge set

Dataset version: `german-transaction-challenge-v1`  
Status: initial zero-shot evaluation completed  
Language: German (`de`)  
Data type: manually reviewed synthetic evaluation data

Versioned data file:
`data/evaluation/transaction_categories/german_challenge_v1.csv`

## Purpose

This challenge set measures how the English transaction classifier behaves on
German bank-transaction descriptions before any German examples are used for
training. It is an out-of-language evaluation set, not training data.

The initial English model must be evaluated unchanged. German challenge results
must not be used to claim performance on real German bank data.

## Target size and balance

Version 1 targets 120 descriptions:

- 12 `transaction-categories-v1` target categories;
- 10 examples per category;
- a documented mix of German-local and international merchants;
- no real customer, account, IBAN, reference, or personal data.

The versioned file contains 120 unique descriptions with exactly 10 examples
per category. Its merchant-scope slices contain 85 `german_local` and 35
`international` examples. Slice sizes are reported explicitly because they are
not balanced and must not be compared as if they had equal statistical weight.

The set remains synthetic even when it uses recognizable transaction-format
patterns or merchant names.

## Schema

| Column | Description |
| --- | --- |
| `scenario_id` | Stable unique identifier such as `de_groceries_001` |
| `description` | Synthetic German transaction description shown to the model |
| `target_category` | Label from `transaction-categories-v1` |
| `language` | Must be `de` |
| `merchant_group` | Stable synthetic grouping key for related descriptions |
| `merchant_scope` | Either `german_local` or `international` |

## Labeling and review rules

- Apply the existing versioned taxonomy and boundary rules.
- Include only examples whose category is defensible from the available text.
- Do not include the target label as an artificial annotation in the text.
- Avoid making every example trivial by naming the category directly.
- Include realistic German formats such as card payments, SEPA direct debits,
  standing orders, payment-processor wrappers, and shortened merchant text.
- Keep identifiers synthetic and non-functional.
- Review duplicate descriptions, conflicting labels, category balance, and
  merchant-scope balance before evaluation.

## Evaluation protocol

1. Freeze the English TF-IDF and logistic-regression baseline.
2. Evaluate it on this German set without fitting or parameter changes.
3. Report Accuracy, Macro-F1, per-category metrics, and confusion matrix.
4. Report `german_local` and `international` slices separately.
5. Treat the results as zero-shot language-transfer evidence only.
6. Build a separate deterministic German training generator afterward.
7. Never add challenge examples to training data.

## English baseline zero-shot result

The frozen English character TF-IDF and balanced logistic-regression pipeline
was refitted on the existing grouped English train and validation partitions
(31,119 rows). No German challenge description was used for fitting, feature
selection, parameter selection, or taxonomy changes.

| Evaluation slice | Rows | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Overall | 120 | 27.50% | 24.87% |
| German-local merchants | 85 | 23.53% | 19.70% |
| International merchants | 35 | 37.14% | 32.47% |

The international slice performs better, consistent with the model recognizing
some merchant names or character patterns already represented in English data.
This difference must not be interpreted as general German-language transfer.
The slices are small and unequal, so their scores are descriptive rather than
statistically precise comparisons.

Overall per-category results:

| Category | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| dining | 15.79% | 90.00% | 26.87% | 10 |
| education | 100.00% | 20.00% | 33.33% | 10 |
| entertainment | 33.33% | 60.00% | 42.86% | 10 |
| groceries | 75.00% | 30.00% | 42.86% | 10 |
| healthcare | 100.00% | 10.00% | 18.18% | 10 |
| housing | 0.00% | 0.00% | 0.00% | 10 |
| insurance | 0.00% | 0.00% | 0.00% | 10 |
| other | 22.22% | 20.00% | 21.05% | 10 |
| shopping | 25.00% | 40.00% | 30.77% | 10 |
| transport | 25.00% | 10.00% | 14.29% | 10 |
| travel | 50.00% | 30.00% | 37.50% | 10 |
| utilities | 66.67% | 20.00% | 30.77% | 10 |

The model predicts `dining` for 57 of 120 descriptions while only 10 are
actually dining transactions. It never predicts `housing` or `insurance`.
These errors show that the English model is not suitable for German transaction
classification and provide the reference that a German or multilingual model
must improve upon.

The evaluation is reproducible from the repository root after the pinned source
dataset is available locally:

```powershell
uv run python -m financial_ai.ml.german_evaluation
```

## Known limitations

- The challenge set is small and synthetic.
- Manual examples cannot represent every German bank or payment format.
- International merchants may be recognized without German-language transfer.
- Results depend on taxonomy clarity and scenario-review quality.
- A future real-data evaluation requires a lawful, licensed, anonymized source
  and a separate documented protocol.

## Final German-trained model results

After validation-based model selection was frozen, German-only and controlled
bilingual character models were evaluated once on this challenge:

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| English-only zero-shot | 27.50% | 24.87% |
| German-only | 75.00% | 74.96% |
| Controlled bilingual | 75.00% | 74.64% |

The German-only model scores 77.65% / 77.64% on the German-local slice and
68.57% / 68.33% on the international slice. The bilingual model scores 75.29% /
74.80% locally and 74.29% / 73.81% internationally. The bilingual model is more
balanced across merchant scopes, while neither result supports production use.

These results are frozen. Challenge errors must not be used to revise the current
generator or model while continuing to describe this set as untouched.
