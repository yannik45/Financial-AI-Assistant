import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from financial_ai.ml.transaction_classification.core.contracts import (
    ClassificationRoute,
    route_transaction_text,
)
from financial_ai.ml.transaction_classification.data.classification_v2_dataset import (
    load_classification_v2_dataset,
)
from financial_ai.ml.transaction_classification.evaluation.semantic_baseline import (
    SemanticValidationRun,
    load_sentence_encoder,
    run_semantic_validation_with_predictions,
    run_tfidf_validation_with_predictions,
)

DEFAULT_DIAGNOSTICS_DIRECTORY = Path(
    "data/runtime/ml/transaction_categories/validation_diagnostics"
)


@dataclass(frozen=True)
class HybridEvaluation:
    rows: int
    rules_accepted: int
    model_accepted: int
    needs_review: int
    automatic_coverage: float
    automatically_accepted_accuracy: float | None
    raw_hybrid_accuracy: float


@dataclass(frozen=True)
class GroupGeneralizationEvaluation:
    groups: int
    mean_group_accuracy: float
    fully_correct_group_rate: float
    fully_failed_group_rate: float


def build_bank_prediction_rows(validation_run: SemanticValidationRun) -> pd.DataFrame:
    validation = validation_run.validation
    mask = validation["input_slice"].eq("bank_feed") & validation["target_category"].ne("other")
    rows = validation.loc[mask].copy()
    rows["predicted_category"] = validation_run.predictions[mask]
    rows["confidence"] = validation_run.confidence[mask]
    rows["correct"] = rows["target_category"].eq(rows["predicted_category"])
    threshold = validation_run.report.validation["bank_feed_in_scope"].threshold
    rows["model_accepted"] = rows["confidence"].ge(threshold)
    return rows


def evaluate_hybrid_bank_predictions(rows: pd.DataFrame) -> tuple[HybridEvaluation, pd.DataFrame]:
    evaluated = rows.copy()
    routes: list[str] = []
    final_predictions: list[str] = []
    automatically_accepted: list[bool] = []
    for row in evaluated.itertuples(index=False):
        decision = route_transaction_text(row.description, Decimal("-1"))
        if decision.route is ClassificationRoute.TEXT_RULE:
            routes.append("text_rule")
            final_predictions.append(decision.category.value)
            automatically_accepted.append(True)
        elif row.model_accepted:
            routes.append("expense_model")
            final_predictions.append(row.predicted_category)
            automatically_accepted.append(True)
        else:
            routes.append("needs_review")
            final_predictions.append(row.predicted_category)
            automatically_accepted.append(False)

    evaluated["hybrid_route"] = routes
    evaluated["hybrid_prediction"] = final_predictions
    evaluated["automatically_accepted"] = automatically_accepted
    evaluated["hybrid_correct"] = evaluated["target_category"].eq(evaluated["hybrid_prediction"])
    accepted = evaluated["automatically_accepted"]
    accepted_accuracy = (
        float(evaluated.loc[accepted, "hybrid_correct"].mean()) if accepted.any() else None
    )
    report = HybridEvaluation(
        rows=len(evaluated),
        rules_accepted=int(evaluated["hybrid_route"].eq("text_rule").sum()),
        model_accepted=int(evaluated["hybrid_route"].eq("expense_model").sum()),
        needs_review=int(evaluated["hybrid_route"].eq("needs_review").sum()),
        automatic_coverage=float(accepted.mean()),
        automatically_accepted_accuracy=accepted_accuracy,
        raw_hybrid_accuracy=float(evaluated["hybrid_correct"].mean()),
    )
    return report, evaluated


def build_group_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dimension in (
        "target_category",
        "language",
        "template_id",
        "merchant_group",
        "detail_group",
        "format_group",
    ):
        for value, group in rows.groupby(dimension, dropna=False):
            records.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "rows": len(group),
                    "errors": int((~group["correct"]).sum()),
                    "error_rate": float((~group["correct"]).mean()),
                    "mean_confidence": float(group["confidence"].mean()),
                    "false_automatic_acceptances": int(
                        ((~group["correct"]) & group["model_accepted"]).sum()
                    ),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["error_rate", "errors", "dimension", "value"],
        ascending=[False, False, True, True],
    )


def evaluate_group_generalization(
    rows: pd.DataFrame,
) -> dict[str, GroupGeneralizationEvaluation]:
    evaluations = {}
    for dimension in ("merchant_group", "detail_group", "format_group"):
        group_accuracy = rows.groupby(dimension, dropna=False)["correct"].mean()
        evaluations[dimension] = GroupGeneralizationEvaluation(
            groups=len(group_accuracy),
            mean_group_accuracy=float(group_accuracy.mean()),
            fully_correct_group_rate=float(group_accuracy.eq(1.0).mean()),
            fully_failed_group_rate=float(group_accuracy.eq(0.0).mean()),
        )
    return evaluations


def write_candidate_diagnostics(
    name: str,
    validation_run: SemanticValidationRun,
    destination: Path = DEFAULT_DIAGNOSTICS_DIRECTORY,
) -> HybridEvaluation:
    destination.mkdir(parents=True, exist_ok=True)
    rows = build_bank_prediction_rows(validation_run)
    hybrid_report, hybrid_rows = evaluate_hybrid_bank_predictions(rows)
    rows.to_csv(destination / f"{name}_bank_predictions.csv", index=False)
    build_group_diagnostics(rows).to_csv(destination / f"{name}_bank_groups.csv", index=False)
    confusion = pd.crosstab(
        rows["target_category"],
        rows["predicted_category"],
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    )
    confusion.to_csv(destination / f"{name}_bank_confusion.csv")
    hybrid_rows.to_csv(destination / f"{name}_hybrid_predictions.csv", index=False)
    (destination / f"{name}_hybrid_summary.json").write_text(
        json.dumps(asdict(hybrid_report), indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / f"{name}_group_summary.json").write_text(
        json.dumps(
            {
                dimension: asdict(evaluation)
                for dimension, evaluation in evaluate_group_generalization(rows).items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return hybrid_report


def run() -> None:
    dataset = load_classification_v2_dataset()
    tfidf = run_tfidf_validation_with_predictions(dataset)
    encoder = load_sentence_encoder()
    semantic = run_semantic_validation_with_predictions(dataset, encoder)
    reports = {
        "tfidf": write_candidate_diagnostics("tfidf", tfidf),
        "semantic": write_candidate_diagnostics("semantic", semantic),
    }
    print(json.dumps({name: asdict(report) for name, report in reports.items()}, indent=2))


if __name__ == "__main__":
    run()
