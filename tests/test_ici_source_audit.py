import gc
import weakref

import pandas as pd
from scripts import audit_ici_source_locators as audit


def test_source_match_uses_only_reported_ci_as_evidence():
    row = pd.Series(
        {
            "metric": "ORR",
            "value": "",
            "value_status": "numeric",
            "unit": "percent",
            "ci_low": 10,
            "ci_high": 20,
            "metric_n": 42,
            "responders": "",
            "source_n": 42,
            "timepoint": "",
            "setting": "",
            "ci_basis": "computed_wilson",
        }
    )
    block = audit.SourceBlock(
        "Table 3",
        "Objective response rate; n=42; confidence interval 10 to 20.",
        "table",
    )

    assert audit._match_block(row, block) == (0, ())
    row["ci_basis"] = "reported"
    score, signals = audit._match_block(row, block)
    assert score >= 11
    assert "ci" in signals


def test_ci_completion_is_idempotent():
    estimates = pd.DataFrame(
        [
            {
                "estimate_id": "ICI-fixture",
                "metric": "ORR",
                "value": 20,
                "value_status": "numeric",
                "ci_low": "",
                "ci_low_status": "not_extracted",
                "ci_high": "",
                "ci_high_status": "not_extracted",
                "ci_basis": "not_extracted",
                "metric_n": 10,
                "responders": 2,
                "source_verified": True,
                "source_locator_status": "verified",
                "value_basis": "reported",
                "note": "",
            }
        ]
    )
    for column in ("ci_low", "ci_high"):
        estimates[column] = estimates[column].astype("string[pyarrow]")

    first = audit.complete_value_and_ci_provenance(estimates)
    second = audit.complete_value_and_ci_provenance(first)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "ci_basis"] == "computed_wilson"
    assert first.loc[0, "ci_low"] == 5.7
    assert first.loc[0, "ci_high"] == 51.0


def test_ci_completion_preserves_curated_clopper_pearson_interval():
    estimates = pd.DataFrame(
        [
            {
                "estimate_id": "ICI-exact",
                "metric": "ORR",
                "value": 0,
                "value_status": "numeric",
                "ci_low": 0.0,
                "ci_low_status": "numeric",
                "ci_high": 28.4914,
                "ci_high_status": "numeric",
                "ci_basis": "computed_clopper_pearson",
                "metric_n": 11,
                "responders": 0,
                "source_verified": True,
                "source_locator_status": "verified",
                "value_basis": "computed_from_counts",
                "note": "two-sided exact interval calculated from 0/11",
            }
        ]
    )

    result = audit.complete_value_and_ci_provenance(estimates)

    assert result.loc[0, "ci_low"] == 0.0
    assert result.loc[0, "ci_high"] == 28.4914
    assert result.loc[0, "ci_basis"] == "computed_clopper_pearson"


def test_build_audit_releases_each_document_and_preserves_row_order(monkeypatch, tmp_path):
    prior_document = None
    loaded_refs = []

    def fake_source_document(ref, **_kwargs):
        nonlocal prior_document
        gc.collect()
        if prior_document is not None:
            assert prior_document() is None
        document = audit.SourceDocument(f"https://example.org/{ref}", "fixture", ())
        prior_document = weakref.ref(document)
        loaded_refs.append(ref)
        return document

    monkeypatch.setattr(audit, "_id_conversion", lambda _refs, _cache: {})
    monkeypatch.setattr(audit, "_pubmed_documents", lambda _pmids, _cache: {})
    monkeypatch.setattr(audit, "_source_document", fake_source_document)

    estimates = pd.DataFrame(
        [
            {
                "estimate_id": "ICI-1",
                "ref": "PMID:1",
                "value_basis": "reported",
                "source_verified": True,
                "metric": "ORR",
                "value": 10,
                "value_status": "numeric",
                "unit": "percent",
                "metric_n": 10,
                "responders": 1,
                "source_n": 10,
                "ci_low": "",
                "ci_high": "",
                "timepoint": "",
                "setting": "cohort one",
                "note": "",
            },
            {
                "estimate_id": "ICI-2",
                "ref": "PMID:2",
                "value_basis": "reported",
                "source_verified": True,
                "metric": "ORR",
                "value": 20,
                "value_status": "numeric",
                "unit": "percent",
                "metric_n": 10,
                "responders": 2,
                "source_n": 10,
                "ci_low": "",
                "ci_high": "",
                "timepoint": "",
                "setting": "cohort two",
                "note": "",
            },
            {
                "estimate_id": "ICI-3",
                "ref": "PMID:1",
                "value_basis": "reported",
                "source_verified": True,
                "metric": "DCR",
                "value": 30,
                "value_status": "numeric",
                "unit": "percent",
                "metric_n": 10,
                "responders": 3,
                "source_n": 10,
                "ci_low": "",
                "ci_high": "",
                "timepoint": "",
                "setting": "cohort one",
                "note": "",
            },
        ]
    )

    result = audit.build_audit(estimates, tmp_path)

    assert loaded_refs == ["PMID:1", "PMID:2"]
    assert result["estimate_id"].tolist() == ["ICI-1", "ICI-2", "ICI-3"]
