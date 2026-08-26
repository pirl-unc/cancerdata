import pandas as pd
import pytest

import oncoref
from oncoref import therapy_evidence


def test_therapy_evidence_schema_ids_and_sources():
    frame = therapy_evidence.therapy_benefit_toxicity_evidence()

    assert len(frame) == 7
    assert tuple(frame.columns) == therapy_evidence.THERAPY_EVIDENCE_COLUMNS
    assert frame["evidence_id"].is_unique
    assert frame["evidence_id"].astype(str).str.startswith("therapy_").all()
    assert set(frame["benefit_tier"]) <= set(therapy_evidence.THERAPY_BENEFIT_TIERS)
    assert set(frame["toxicity_tier"]) <= set(therapy_evidence.THERAPY_TOXICITY_TIERS)
    assert set(frame["evidence_transfer"]) <= set(therapy_evidence.THERAPY_EVIDENCE_TRANSFER_VALUES)
    assert frame["source_token"].astype(str).str.len().gt(0).all()
    assert frame["source_anchor"].astype(str).str.len().gt(0).all()
    assert frame["source_url"].astype(str).str.startswith(("https://", "http://")).all()


def test_therapy_evidence_filters_preserve_parent_rows():
    gist = therapy_evidence.therapy_benefit_toxicity_evidence(
        agent="IMATINIB",
        cancer_code="sarc",
        subtype="GIST",
        line_of_therapy="STANDARD_OR_FRONTLINE",
    )
    assert gist["evidence_id"].tolist() == ["therapy_imatinib_sarc_gist_label"]

    subtype_only = therapy_evidence.therapy_benefit_toxicity_evidence(subtype="gist")
    assert subtype_only["evidence_id"].tolist() == ["therapy_imatinib_sarc_gist_label"]

    ovarian = therapy_evidence.therapy_benefit_toxicity_evidence(
        cancer_code="OV", subtype="not_curated"
    )
    assert set(ovarian["evidence_id"]) == {
        "therapy_olaparib_ov_solo1",
        "therapy_pembrolizumab_ov_msih_label",
    }


def test_therapy_evidence_can_exclude_transferred_rows():
    default = therapy_evidence.therapy_benefit_toxicity_evidence(
        agent="pembrolizumab", cancer_code="OV"
    )
    strict = therapy_evidence.therapy_benefit_toxicity_evidence(
        agent="pembrolizumab", cancer_code="OV", include_transferred=False
    )

    assert default["evidence_transfer"].tolist() == ["cross_indication"]
    assert strict.empty


def test_postmarket_signal_is_not_an_incidence_estimate():
    row = therapy_evidence.therapy_benefit_toxicity_evidence(source_type="postmarket_signal").iloc[
        0
    ]
    assert row["evidence_transfer"] == "safety_signal_only"
    assert pd.isna(row["grade3_plus_ae_rate"])
    assert pd.isna(row["discontinuation_rate"])


def test_therapy_evidence_validation_rejects_incidence_on_postmarket_signal():
    frame = therapy_evidence.therapy_benefit_toxicity_evidence()
    frame.loc[frame["source_type"].eq("postmarket_signal"), "source_type"] = "POSTMARKET_SIGNAL"
    signal = frame["source_type"].eq("POSTMARKET_SIGNAL")
    frame.loc[signal, "grade3_plus_ae_rate"] = "4%"

    with pytest.raises(ValueError, match="must not provide incidence-like"):
        therapy_evidence._validate_therapy_evidence(frame)


@pytest.mark.parametrize(
    "column,value",
    [
        ("benefit_tier", "miraculous"),
        ("toxicity_tier", "catastrophic"),
        ("evidence_transfer", "assumed_equivalent"),
    ],
)
def test_therapy_evidence_validation_rejects_unknown_enumerations(column, value):
    frame = therapy_evidence.therapy_benefit_toxicity_evidence()
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=f"invalid {column}"):
        therapy_evidence._validate_therapy_evidence(frame)


def test_therapy_evidence_public_exports():
    assert oncoref.therapy_benefit_toxicity_evidence is (
        therapy_evidence.therapy_benefit_toxicity_evidence
    )
    assert oncoref.therapy_evidence is therapy_evidence
