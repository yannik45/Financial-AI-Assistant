# Transaction categorization dataset

The transaction categorization baseline uses the synthetic
[`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)
dataset.

The raw CSV is not committed. Download the pinned and checksum-verified version
to the ignored `data/runtime` directory from the repository root:

```powershell
.\.venv\Scripts\python.exe -m financial_ai.ml.category_dataset
```

Expected local path:

```text
data/runtime/ml/transaction_categories/transactions-synthetic.csv
```

The source data is synthetic and modeled after US bank-statement descriptions.
It must not be presented as real customer transaction data. See `metadata.json`
for provenance and integrity information.
