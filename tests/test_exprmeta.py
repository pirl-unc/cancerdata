# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expression-source candidates + frameshift burden (#35, R-exprmeta)."""

import pytest

from oncoref import cancer_type_registry, tmb
from oncoref.expression_registry import expression_source_candidates


def test_frameshift_burden():
    m = tmb.cancer_frameshift_burden()
    assert m and all(isinstance(v, int) for v in m.values())
    assert tmb.cancer_frameshift_burden("KIRC") == 2
    # alias resolves (kidney_clear -> KIRC)
    assert tmb.cancer_frameshift_burden("kidney_clear") == 2
    # unknown code raises (mirrors cancer_tmb); a valid-but-unmapped code -> None
    with pytest.raises(ValueError):
        tmb.cancer_frameshift_burden("NOT_A_CODE")
    unmapped = next(c for c in cancer_type_registry()["code"] if c not in m)
    assert tmb.cancer_frameshift_burden(unmapped, inherit=False) is None


def test_expression_source_candidates():
    df = expression_source_candidates()
    assert {"cancer_code", "source_status", "reference_code"} <= set(df.columns)
    one = expression_source_candidates("BRCA_Basal")
    assert len(one) >= 1 and (one["cancer_code"] == "BRCA_Basal").all()

    mmnst = expression_source_candidates("SARC_MMNST").iloc[0]
    assert mmnst["source_status"] == "direct_reference_available"
    assert mmnst["reference_code"] == "SARC_MMNST"
    assert mmnst["source_project"] == "NCBI SRA Gene Feature counts"
    assert mmnst["source_cohort"] == "SRP493407_MMNST_2024"
    assert mmnst["accession"] == "PRJNA1083972"
    assert int(mmnst["estimated_samples"]) == 3

    mplps = expression_source_candidates("SARC_MPLPS").iloc[0]
    assert mplps["source_status"] == "source_needed"
    assert mplps["reference_code"] == "SARC_LPS"
    assert mplps["accession"] == "DOI:10.1038/s41379-021-00862-2"
    assert int(mplps["estimated_samples"]) == 12
    assert "no RNA expression matrix" in mplps["source_scope"]
