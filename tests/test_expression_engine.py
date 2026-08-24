# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd
import pytest

from oncoref import expression_engine as ee


def test_aggregate_sums_transcripts_per_gene():
    df = pd.DataFrame(
        {
            "transcript_id": ["ENST1.2", "ENST2", "ENST3", "ENSTX"],
            "tpm": [10.0, 5.0, 20.0, 7.0],
        }
    )
    tx_map = {"ENST1": "GENEA", "ENST2": "GENEA", "ENST3": "GENEB"}  # ENSTX unknown
    out = ee.aggregate_transcripts_to_genes(df, tx_map)
    by = dict(zip(out["gene"], out["TPM"]))
    assert by["GENEA"] == 15.0  # ENST1(versioned) + ENST2
    assert by["GENEB"] == 20.0
    assert by["unresolved"] == 7.0  # ENSTX kept, not dropped
    stats = out.attrs["aggregation_stats"]
    assert stats["unresolved_tpm"] == 7.0
    assert stats["n_genes"] == 2
    assert stats["unresolved_fraction"] == pytest.approx(7.0 / 42.0)


def test_find_column_absorbs_naming():
    df = pd.DataFrame({"Target_ID": ["t"], "TPM": [1.0]})
    assert ee.find_column(df, ["transcript", "target_id"], "tx") == "Target_ID"
    assert ee.find_column(df, ["tpm"], "TPM") == "TPM"
    with pytest.raises(ValueError, match="no column for"):
        ee.find_column(df, ["nope"], "missing")


def test_find_column_respects_candidate_priority():
    # Frame carries both a transcript id and a 'name' column; the higher-priority
    # candidate (transcript_id) must win regardless of column order.
    df = pd.DataFrame({"name": ["n"], "transcript_id": ["t"]})
    assert ee.find_column(df, ["transcript_id", "name"], "tx") == "transcript_id"
    # column order reversed -> still resolves by candidate priority, not column order
    df2 = pd.DataFrame({"transcript_id": ["t"], "name": ["n"]})
    assert ee.find_column(df2, ["transcript_id", "name"], "tx") == "transcript_id"


def test_expanded_tx_map_versionless():
    m = ee.expanded_tx_map({"ENST9.3": "G"})
    assert m["ENST9.3"] == "G"
    assert m["ENST9"] == "G"  # versionless key added


def test_default_map_is_oncoref_extra_tx():
    # The default map comes from oncoref's curated extra-tx-mappings.
    df = pd.DataFrame({"transcript_id": ["ENST00000264036"], "tpm": [99.0]})
    out = ee.aggregate_transcripts_to_genes(df)
    assert "MCAM" in set(out["gene"])  # ENST00000264036 -> MCAM in extra-tx-mappings


def test_detect_source_row_id_type():
    assert ee.detect_source_row_id_type(["ENSG00000141510.17", "ENSG00000278311"]) == (
        "ensembl_gene_id"
    )
    assert ee.detect_source_row_id_type(["ENSG00000182162.11_PAR_Y"]) == "ensembl_gene_id"
    assert ee.detect_source_row_id_type(["ENST00000264036", "ENST00000367770.8"]) == (
        "ensembl_transcript_id"
    )
    assert ee.detect_source_row_id_type(["TP53", "GGNBP2"]) == "symbol"
    assert ee.detect_source_row_id_type(["7SK", "7SL", "5S_rRNA", "5_8S_rRNA"]) == "symbol"
    assert ee.detect_source_row_id_type(["7157", "1234"]) == "entrez_id"
    assert ee.detect_source_row_id_type(["ENSG00000141510", "TP53"]) == "mixed"


def test_map_source_gene_rows_resolves_ids_symbols_and_transcripts():
    df = pd.DataFrame(
        {
            "gene_id": [
                "ENSG00000005955.7",  # old GGNBP2 id -> canonical alias
                "TP53",
                "7157",
                "ENST00000264036",
                "NOT_A_REAL_GENE",
            ],
            "s1": [2.0, 3.0, 5.0, 4.0, 10.0],
        }
    )
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["s1"])
    by_id = audit.set_index("source_row_id")

    assert by_id.loc["ENSG00000005955.7", "canonical_ensembl_gene_id"] == "ENSG00000278311"
    assert by_id.loc["ENSG00000005955.7", "mapping_method"] == "ensembl_gene_id_alias"
    assert by_id.loc["TP53", "canonical_ensembl_gene_id"] == "ENSG00000141510"
    assert by_id.loc["TP53", "mapping_method"] == "symbol"
    assert by_id.loc["7157", "canonical_ensembl_gene_id"] == "ENSG00000141510"
    assert by_id.loc["7157", "mapping_method"] == "entrez_dbxrefs"
    assert by_id.loc["ENST00000264036", "mapping_method"] == "extra_transcript_mapping"
    assert by_id.loc["NOT_A_REAL_GENE", "mapping_status"] == "unresolved"
    assert bool(by_id.loc["NOT_A_REAL_GENE", "high_expression_unresolved"]) is True

    stats = ee.source_gene_mapping_stats(audit)
    assert stats["n_source_rows"] == 5
    assert stats["n_resolved_rows"] == 4
    assert stats["n_high_expression_unresolved_rows"] == 1


def test_source_audit_outputs_clean_unversioned_contract_and_expression_triage():
    df = pd.DataFrame(
        {"gene_id": ["TP53", "NOT_A_REAL_GENE", "ZERO"], "sample_a": ["1.5", "0", "0"]}
    )

    audit = ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["sample_a"])
    coerced, diagnostics = ee.coerce_source_expression_values(df, value_cols=["sample_a"])
    matrix, matrix_audit = ee.canonicalize_source_gene_matrix(
        df, row_id_col="gene_id", value_cols=["sample_a"]
    )

    assert "schema_version" not in audit.attrs
    assert "source_gene_mapping_schema_version" not in audit.columns
    assert "schema_version" not in diagnostics.attrs
    assert "source_value_parse_schema_version" not in diagnostics.columns
    assert "source_gene_mapping_schema_version" not in matrix_audit.columns
    assert audit.loc[0, "source_expression_max"] == 1.5
    assert audit.loc[0, "source_expression_sample_with_max"] == "sample_a"
    assert audit.loc[0, "source_expression_nonzero_samples"] == 1
    assert audit.loc[1, "source_expression_max"] == 0.0
    assert pd.isna(audit.loc[1, "source_expression_sample_with_max"])
    assert audit.loc[1, "source_expression_nonzero_samples"] == 0
    parse = matrix.attrs["source_value_parse_diagnostics"]
    assert "source_value_parse_schema_version" not in parse.columns
    assert coerced["sample_a"].tolist() == [1.5, 0.0, 0.0]


def test_map_source_gene_rows_falls_back_from_noncanonical_transcript_gene_id():
    df = pd.DataFrame({"gene_id": ["ENST00000644628"], "s1": [11.0]})
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["s1"])
    row = audit.iloc[0]

    assert row["mapping_status"] == "resolved"
    assert row["mapping_method"] == "extra_transcript_mapping"
    assert row["canonical_ensembl_gene_id"] == "ENSG00000171862"
    assert row["canonical_symbol"] == "PTEN"


def test_map_source_gene_rows_normalizes_case_varied_ensembl_identifiers():
    df = pd.DataFrame(
        {
            "gene_id": ["ensg00000141510.17", "enst00000644628"],
            "s1": [3.0, 4.0],
        }
    )
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["s1"])
    by_id = audit.set_index("source_row_id")

    assert by_id.loc["ensg00000141510.17", "mapping_status"] == "resolved"
    assert by_id.loc["ensg00000141510.17", "canonical_ensembl_gene_id"] == "ENSG00000141510"
    assert by_id.loc["enst00000644628", "mapping_status"] == "resolved"
    assert by_id.loc["enst00000644628", "canonical_ensembl_gene_id"] == "ENSG00000171862"

    matrix, _ = ee.canonicalize_source_gene_matrix(df, row_id_col="gene_id", value_cols=["s1"])
    assert set(matrix["Ensembl_Gene_ID"]) == {"ENSG00000141510", "ENSG00000171862"}


def test_map_source_gene_rows_folds_pseudoautosomal_ensembl_suffixes():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG00000182162", "ENSG00000182162.11_PAR_Y"],
            "s1": [2.0, 3.0],
        }
    )
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["s1"])
    by_id = audit.set_index("source_row_id")

    assert by_id.loc["ENSG00000182162.11_PAR_Y", "source_row_id_type"] == "ensembl_gene_id"
    assert by_id.loc["ENSG00000182162.11_PAR_Y", "canonical_ensembl_gene_id"] == ("ENSG00000182162")
    assert by_id.loc["ENSG00000182162.11_PAR_Y", "canonical_symbol"] == "P2RY8"

    matrix, _ = ee.canonicalize_source_gene_matrix(df, row_id_col="gene_id", value_cols=["s1"])
    row = matrix.set_index("Ensembl_Gene_ID").loc["ENSG00000182162"]
    assert row["Symbol"] == "P2RY8"
    assert row["s1"] == 5.0


def test_canonicalize_source_gene_matrix_counts_identical_par_y_symbol_alias_once():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG00000182162.11", "ENSG00000182162.11"],
            "gene_symbol": ["P2RY8", "PAR_Y_P2RY8"],
            "s1": [2.0, 2.0],
            "s2": [3.0, 3.0],
        }
    )

    matrix, audit = ee.canonicalize_source_gene_matrix(
        df,
        row_id_col="gene_id",
        symbol_col="gene_symbol",
        value_cols=["s1", "s2"],
    )

    row = matrix.set_index("Ensembl_Gene_ID").loc["ENSG00000182162"]
    assert row["Symbol"] == "P2RY8"
    assert row["s1"] == 2.0
    assert row["s2"] == 3.0
    assert audit["mapping_status"].tolist() == ["resolved", "resolved"]
    assert audit["mapping_method"].tolist() == [
        "ensembl_gene_id",
        "ensembl_gene_id_par_y_symbol_alias",
    ]


def test_canonicalize_source_gene_matrix_rejects_unequal_par_y_symbol_alias():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG00000182162.11", "ENSG00000182162.11"],
            "gene_symbol": ["P2RY8", "PAR_Y_P2RY8"],
            "s1": [2.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="unequal values for pseudoautosomal symbol aliases"):
        ee.canonicalize_source_gene_matrix(
            df,
            row_id_col="gene_id",
            symbol_col="gene_symbol",
            value_cols=["s1"],
        )


def test_map_source_gene_rows_resolves_leading_digit_ncrna_symbols():
    df = pd.DataFrame({"gene_id": ["7SL", "5_8S_rRNA", "45S"], "s1": [3.0, 4.0, 5.0]})
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["s1"])
    by_id = audit.set_index("source_row_id")

    assert by_id.loc["7SL", "source_row_id_type"] == "symbol"
    assert by_id.loc["7SL", "canonical_ensembl_gene_id"] == "ENSG00000276168"
    assert by_id.loc["7SL", "canonical_symbol"] == "RN7SL1"
    assert by_id.loc["5_8S_rRNA", "source_row_id_type"] == "symbol"
    assert by_id.loc["5_8S_rRNA", "mapping_status"] == "ambiguous"
    assert by_id.loc["5_8S_rRNA", "unresolved_reason"] == "ambiguous_symbol"
    assert by_id.loc["45S", "source_row_id_type"] == "symbol"
    assert by_id.loc["45S", "unresolved_reason"] == "unknown_symbol"


def test_map_source_gene_rows_preserves_identifier_unresolved_reasons():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG99999999999", "999999999999"],
            "s1": [10.0, 20.0],
        }
    )
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["s1"])
    by_id = audit.set_index("source_row_id")

    assert by_id.loc["ENSG99999999999", "unresolved_reason"] == "unknown_ensembl_gene_id"
    assert by_id.loc["ENSG99999999999", "mapping_method"] == "unresolved"
    assert by_id.loc["999999999999", "unresolved_reason"] == "unknown_entrez_id"
    assert by_id.loc["999999999999", "mapping_method"] == "unresolved"


def test_map_source_gene_rows_accepts_symbol_only_matrix_by_default():
    df = pd.DataFrame({"Symbol": ["TP53", "PTEN"], "sample_a": [5.0, 7.0]})
    audit = ee.map_source_gene_rows(df)
    by_id = audit.set_index("source_row_id")

    assert by_id.loc["TP53", "canonical_ensembl_gene_id"] == "ENSG00000141510"
    assert by_id.loc["PTEN", "canonical_ensembl_gene_id"] == "ENSG00000171862"

    matrix, _ = ee.canonicalize_source_gene_matrix(df)
    assert set(matrix["Symbol"]) == {"TP53", "PTEN"}
    assert matrix["sample_a"].sum() == 12.0


def test_map_source_gene_rows_treats_duplicate_synonym_alias_as_ambiguous():
    df = pd.DataFrame({"gene_id": ["A3"], "sample_a": [9.0]})
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id")
    row = audit.iloc[0]

    assert row["mapping_status"] == "ambiguous"
    assert row["mapping_method"] == "ambiguous"
    assert row["unresolved_reason"] == "ambiguous_symbol"
    assert pd.isna(row["canonical_ensembl_gene_id"])


def test_map_source_gene_rows_handles_duplicate_dataframe_index_labels():
    df = pd.DataFrame(
        {"gene_id": ["TP53", "PTEN"], "sample_a": [5.0, 7.0]},
        index=["dup", "dup"],
    )
    audit = ee.map_source_gene_rows(df, row_id_col="gene_id")

    assert audit["source_row_index"].tolist() == ["dup", "dup"]
    assert audit["source_value_sum"].tolist() == [5.0, 7.0]
    matrix, _ = ee.canonicalize_source_gene_matrix(df, row_id_col="gene_id")
    assert matrix["sample_a"].sum() == 12.0


def test_canonicalize_source_gene_matrix_sums_duplicate_canonical_ids():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG00000005955.7", "ENSG00000278311", "TP53", "NOT_A_REAL_GENE"],
            "sample_a": [1.0, 2.0, 5.0, 100.0],
            "sample_b": [10.0, 20.0, 50.0, 200.0],
        }
    )
    matrix, audit = ee.canonicalize_source_gene_matrix(
        df, row_id_col="gene_id", value_cols=["sample_a", "sample_b"]
    )
    by_gene = matrix.set_index("Ensembl_Gene_ID")

    assert by_gene.loc["ENSG00000278311", "Symbol"] == "GGNBP2"
    assert by_gene.loc["ENSG00000278311", "sample_a"] == 3.0
    assert by_gene.loc["ENSG00000278311", "sample_b"] == 30.0
    assert by_gene.loc["ENSG00000141510", "sample_a"] == 5.0
    assert "NOT_A_REAL_GENE" in set(
        audit.loc[audit["mapping_status"] == "unresolved", "source_row_id"]
    )
    assert matrix.attrs["source_gene_mapping_stats"]["n_unresolved_rows"] == 1


def test_coerce_source_expression_values_distinguishes_missing_nonparse_and_zero():
    df = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG00000141510", "ENSG00000278311", "ENSG00000000003"],
            "s1": ["0", "", None],
            "s2": ["1.5", "bad", "0"],
        }
    )
    out, diag = ee.coerce_source_expression_values(df, value_cols=["s1", "s2"])
    by_col = diag.set_index("value_col")

    assert out["s1"].isna().sum() == 2
    assert out["s2"].isna().sum() == 1
    assert by_col.loc["s1", "n_literal_zero"] == 1
    assert by_col.loc["s1", "n_parse_missing"] == 1
    assert by_col.loc["s1", "n_input_missing"] == 1
    assert by_col.loc["s2", "n_literal_zero"] == 1
    assert by_col.loc["s2", "n_parse_missing"] == 1


def test_coerce_source_expression_values_default_preserves_source_identifiers():
    df = pd.DataFrame(
        {
            "gene_id": ["ENSG00000141510", "ENSG00000278311"],
            "symbol": ["TP53", "GGNBP2"],
            "sample_a": ["1.5", "0"],
        }
    )
    out, diag = ee.coerce_source_expression_values(df)

    assert out["gene_id"].tolist() == ["ENSG00000141510", "ENSG00000278311"]
    assert out["symbol"].tolist() == ["TP53", "GGNBP2"]
    assert out["sample_a"].tolist() == [1.5, 0.0]
    assert diag["value_col"].tolist() == ["sample_a"]


def test_source_expression_helpers_reject_missing_requested_value_columns():
    df = pd.DataFrame({"gene_id": ["TP53"], "sample_a": [1.0]})

    with pytest.raises(ValueError, match="missing"):
        ee.map_source_gene_rows(df, row_id_col="gene_id", value_cols=["sample_b"])
    with pytest.raises(ValueError, match="missing"):
        ee.canonicalize_source_gene_matrix(df, row_id_col="gene_id", value_cols=["sample_b"])
    with pytest.raises(ValueError, match="missing"):
        ee.coerce_source_expression_values(df, value_cols=["sample_b"])
