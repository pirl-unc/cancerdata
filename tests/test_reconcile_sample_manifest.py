import pandas as pd
import pytest
from scripts.reconcile_sample_manifest import reconcile_sample_manifest, replace_manifest_sources


def test_reconcile_sample_manifest_uses_unique_pipeline_and_canonical_lineage():
    samples = pd.DataFrame(
        {
            "source_cohort": ["BEATAML", "BEATAML", "AMBIGUOUS"],
            "processing_pipeline": ["legacy_v1", "legacy_v1", "keep_this"],
            "lineage_label": ["LAML_ELN_Fav", "", "BL"],
        }
    )
    availability = pd.DataFrame(
        {
            "source_cohort": ["BEATAML", "BEATAML", "AMBIGUOUS", "AMBIGUOUS"],
            "processing_pipeline": ["current", "current", "pipeline_a", "pipeline_b"],
        }
    )

    reconciled = reconcile_sample_manifest(samples, availability)

    assert reconciled["processing_pipeline"].tolist() == ["current", "current", "keep_this"]
    assert reconciled["lineage_label"].tolist() == ["LAML_ELNfav", "", "BL"]


def test_reconcile_sample_manifest_rejects_unknown_lineage():
    samples = pd.DataFrame(
        {
            "source_cohort": ["SOURCE"],
            "processing_pipeline": ["pipeline"],
            "lineage_label": ["NOT_A_CANCER_CODE"],
        }
    )
    availability = pd.DataFrame({"source_cohort": ["SOURCE"], "processing_pipeline": ["pipeline"]})

    with pytest.raises(ValueError, match="unresolved lineage labels"):
        reconcile_sample_manifest(samples, availability)


def test_replace_manifest_sources_is_idempotent_and_preserves_public_schema():
    columns = ["cancer_code", "source_cohort", "sample_id", "processing_pipeline", "lineage_label"]
    samples = pd.DataFrame(
        [
            ["OLD", "SOURCE_A", "a1", "old", "OLD"],
            ["KEEP", "SOURCE_B", "b1", "keep", "KEEP"],
        ],
        columns=columns,
    )
    update = pd.DataFrame(
        [["NEW", "SOURCE_A", "a2", "new", "NEW", "ignored"]],
        columns=[*columns, "source_record"],
    )

    merged = replace_manifest_sources(samples, [update])
    repeated = replace_manifest_sources(merged, [update])

    assert list(merged.columns) == columns
    assert merged.to_dict("records") == repeated.to_dict("records")
    assert set(merged["sample_id"]) == {"a2", "b1"}
