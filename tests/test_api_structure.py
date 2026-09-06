import pandas as pd
import pytest

import oncoref
from oncoref import antigen_coverage, apd1, cta_coverage, cta_peptides, ici, ici_response


def test_semantic_modules_are_top_level_facades():
    for name in (
        "antigen_coverage",
        "cancer_ontology",
        "cohorts",
        "cta_coverage",
        "cta_peptides",
        "expression_builders",
        "expression_engine",
        "ici_response",
        "source_matrices",
    ):
        assert hasattr(oncoref, name)
        assert name in oncoref.__all__


def test_expression_engine_builder_audit_surface_is_public():
    assert callable(oncoref.expression_engine.detect_source_row_id_type)
    assert callable(oncoref.expression_engine.map_source_gene_rows)
    assert callable(oncoref.expression_engine.canonicalize_source_gene_matrix)
    assert callable(oncoref.expression_engine.coerce_source_expression_values)
    assert callable(oncoref.expression_engine.source_gene_mapping_stats)
    assert (
        "source_expression_sample_with_max"
        in oncoref.expression_engine.SOURCE_GENE_MAPPING_AUDIT_COLUMNS
    )
    assert "source_gene_mapping_schema_version" not in (
        oncoref.expression_engine.SOURCE_GENE_MAPPING_AUDIT_COLUMNS
    )
    assert "source_value_parse_schema_version" not in (
        oncoref.expression_engine.SOURCE_VALUE_PARSE_DIAGNOSTIC_COLUMNS
    )


def test_clearer_names_stay_in_semantic_modules_not_flat_namespace():
    for name in (
        "best_available_ici_response",
        "ici_response_by_regimen",
        "cta_specific_9mer_count_map",
        "cta_addressable_fraction",
        "addressable_antigen_fraction",
    ):
        assert not hasattr(oncoref, name)


def test_ici_response_facade_delegates_to_compatibility_api():
    assert ici_response.DEFAULT_ICI_REGIMEN_PRIORITY == ici.REGIMEN_FALLBACK
    assert ici_response.ICI_REGIMEN_LABELS == ici.REGIMEN_LABELS
    assert ici_response.RESPONSE_PROPORTION_METRICS == ici.PROPORTION_METRICS

    assert ici_response.apd1_response("SKCM") == apd1.cancer_apd1_response("SKCM")
    assert ici_response.best_available_ici_response("SKCM") == ici.cancer_ici_response("SKCM")
    assert ici_response.best_available_ici_response_record("SKCM") == (
        ici.cancer_ici_response_record("SKCM")
    )
    assert ici_response.ici_response_source("COAD_MSI") == ici.resolve_ici_response_source(
        "COAD_MSI"
    )
    assert ici_response.ici_response_by_regimen("SKCM") == ici.cancer_ici_response(
        "SKCM", fallback=False
    )
    assert ici_response.ici_response_records_by_regimen("SKCM") == (
        ici.cancer_ici_response_record("SKCM", fallback=False)
    )
    assert ici_response.selected_ici_regimen("SKCM") == ici.cancer_ici_regimen("SKCM")
    pd.testing.assert_frame_equal(
        ici_response.ici_response_anchor_df(), ici.cancer_ici_response_df()
    )
    pd.testing.assert_frame_equal(
        ici_response.ici_response_estimates_df(),
        ici.cancer_ici_response_estimates_df(),
    )
    pd.testing.assert_frame_equal(
        ici_response.ici_source_locator_audit_df(),
        ici.cancer_ici_source_locator_audit_df(),
    )


@pytest.mark.parametrize("code, orr", [("LUAD_EGFR", 19.0), ("STAD_CIN", 12.0)])
def test_selected_ici_regimen_defaults_to_inherited_response(code, orr):
    assert ici_response.best_available_ici_response(code) == orr
    assert ici_response.selected_ici_regimen(code) == "PD-1"
    # The legacy helper deliberately retains its source-scope-only contract.
    assert ici.cancer_ici_regimen(code) is None


@pytest.mark.parametrize("inherit", [False, True])
def test_selected_ici_regimen_matches_values_records_and_sources_across_registry(inherit):
    for code in oncoref.cancer_type_registry()["code"]:
        value = ici_response.best_available_ici_response(code, inherit=inherit)
        record = ici_response.best_available_ici_response_record(code, inherit=inherit)
        source = ici_response.ici_response_source(code, inherit=inherit)
        regimen = ici_response.selected_ici_regimen(code, inherit=inherit)
        assert regimen == source["selected_regimen"], code
        assert regimen == (record["regimen"] if record else None), code
        assert (regimen is None) == (value is None), code


def test_cta_peptides_clear_count_map_name(monkeypatch):
    monkeypatch.setattr(
        cta_peptides,
        "cta_specific_9mer_weights",
        lambda *, k=9, by="proteoform_key": {f"{by}:{k}": 7},
    )
    assert cta_peptides.cta_specific_9mer_count_map(k=3, by="symbol") == {"symbol:3": 7}


def test_cta_coverage_names_make_cta_default_explicit(monkeypatch):
    calls = []

    def fake_addressable(cancer_type, *, threshold_tpm, proteoform, **kwargs):
        calls.append((cancer_type, threshold_tpm, proteoform, kwargs))
        return 0.5

    monkeypatch.setattr(cta_coverage, "addressable_fraction", fake_addressable)
    assert cta_coverage.cta_addressable_fraction("LUAD", threshold_tpm=5) == 0.5
    assert calls == [("LUAD", 5, True, {})]


def test_antigen_coverage_requires_explicit_gene_panel(monkeypatch):
    calls = []

    def fake_addressable(cancer_type, *, threshold_tpm, gene_ids, proteoform):
        calls.append((cancer_type, threshold_tpm, gene_ids, proteoform))
        return 0.25

    monkeypatch.setattr(antigen_coverage, "addressable_fraction", fake_addressable)
    assert (
        antigen_coverage.addressable_antigen_fraction(
            "LUAD", gene_ids={"ENSG00000141510"}, threshold_tpm=2
        )
        == 0.25
    )
    assert calls == [("LUAD", 2, {"ENSG00000141510"}, True)]
