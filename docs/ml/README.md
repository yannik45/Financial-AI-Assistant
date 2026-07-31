# ML documentation status

This index separates the active product contract from frozen evaluation records
and historical experiments. Historical code and documents are retained because
they show the decisions, failed assumptions, and validation steps that led to
the current baseline; they are not alternative production paths.

| Document | Status | Purpose |
|---|---|---|
| `transaction_categories.md` | Active contract | Versioned product taxonomy and boundaries |
| `transaction_classifier_service.md` | Active contract | Current text-first hybrid service |
| `transaction_classification_feedback.md` | Active contract | Current feedback capture and offline-use policy |
| `text_classification_evaluation.md` | Active frozen benchmark | Current product-level evaluation and limitations |
| `multilingual_transaction_classification.md` | Frozen experiment | Selection of the bilingual model artifact |
| `controlled_english_training.md` | Frozen experiment | Controlled English generator and test |
| `german_transaction_training_v2.md` | Frozen experiment | Current German generator used by the artifact |
| `german_transaction_challenge.md` | Frozen experiment | Earlier German-only challenge |
| `transaction_classification.md` | Historical experiment | Original external English baseline and evolution |
| `german_transaction_training.md` | Historical experiment | Superseded German generator v1 |

The corresponding Python modules and tests remain intentionally versioned for
reproducibility. New product code should depend on the active artifact builder
and service modules, not on historical evaluation runners.
