import json
from pathlib import Path

EVIDENCE_DIRECTORY = Path("data/evaluation/market_forecast")
DATASET_SHA256 = "ed1bf70f3e673c081cf5cfe1865b1145f058f2b480404b0a29404493d2460cec"


def load_evidence(filename: str) -> dict[str, object]:
    return json.loads((EVIDENCE_DIRECTORY / filename).read_text(encoding="utf-8"))


def test_published_market_forecast_evidence_preserves_evaluation_chain():
    selection = load_evidence("inner_cv_v1.summary.json")
    validation = load_evidence("outer_validation_v1.summary.json")
    diagnostics = load_evidence("outer_validation_diagnostics_v1.summary.json")
    final_test = load_evidence("final_test_v1.summary.json")

    selection_sha256 = selection["source_runtime_report"]["canonical_sha256"]
    validation_sha256 = validation["source_runtime_report"]["canonical_sha256"]
    assert validation["selection_report_sha256"] == selection_sha256
    assert diagnostics["selection_report_sha256"] == selection_sha256
    assert final_test["validation_report_sha256"] == validation_sha256

    assert selection["dataset"]["sha256"] == DATASET_SHA256
    assert validation["dataset"]["sha256"] == DATASET_SHA256
    assert diagnostics["dataset_sha256"] == DATASET_SHA256
    assert final_test["dataset"]["sha256"] == DATASET_SHA256

    for evidence in (validation, final_test):
        assert evidence["selected_model"] == {
            "candidate": "flexible",
            "boosting_rounds": 144,
        }
    assert final_test["model_selection_complete"] is True
    assert final_test["further_tuning_allowed"] is False
