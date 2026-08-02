# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from oncoref import gene_qc as qc


def test_symbol_regex_classification():
    cases = {
        "MT-CO1": "mt_dna",
        "MT-RNR1": "mt_dna",
        "MTCO1P12": "mt_like_pseudogene",
        "RNA5SP1": "rrna_like",
        "RPL13": "ribosomal_protein",
        "RPL13AP5": "ribosomal_protein_pseudogene",
        "MALAT1": "polyadenylation_bias_lncrna",
        "NEAT1": "polyadenylation_bias_lncrna",
        "SNORD3A": "protocol_sensitive_structural_rna",
        "H2BC1": "histone",
        "HBA1": "hemoglobin",
        "IGHV1-2": "immune_receptor",
        "IGKC": "immune_receptor",
        "TRBV2": "immune_receptor",
        "PRAME": "other",
        "": "other",
    }
    for symbol, group in cases.items():
        assert qc.classify_gene_qc(symbol).group == group, symbol


def test_is_rescue_feature_is_technical_only():
    # technical RNA -> rescued; ribosomal proteins / real genes -> not.
    assert qc.is_rescue_feature("MT-CO1")
    assert qc.is_rescue_feature("MALAT1")
    assert qc.is_rescue_feature("RNA5SP1")
    assert not qc.is_rescue_feature("RPL13")  # ribosomal protein is kept, not technical
    assert not qc.is_rescue_feature("PRAME")


def test_ensembl_id_first_lookup():
    # A real mitochondrial gene id (from oncoref's panel) classifies via ENSG
    # even with a mismatched/absent symbol — ENSG-first is rename-stable.
    from oncoref import gene_families

    mt_ids = list(gene_families.gene_family_ids("mitochondrial"))
    assert mt_ids
    cls = qc.classify_gene_qc(symbol="WRONGSYMBOL", ensembl_id=mt_ids[0])
    assert cls.group == "mt_dna"
    assert qc.is_rescue_feature(ensembl_id=mt_ids[0])


def test_symbol_path_returns_coarse_labels():
    # Global technical symbols use the same refined label with or without an ENSG.
    assert qc.classify_gene_qc("MT-TA").label == "mitochondrial transcript"
    assert qc.classify_gene_qc("SNORD3A").label == "small nucleolar RNA (C/D box)"
    assert qc.classify_gene_qc("RNA18S5").label == "18S rRNA-like"


def test_family_path_refines_label():
    # Through the family classifier (ENSG-first path), the label is refined.
    assert qc._family_to_qc_class("mitochondrial", "MT-TA").label == "mitochondrial tRNA"
    assert (
        qc._family_to_qc_class("small_noncoding_rna", "SNORD3A").label
        == "small nucleolar RNA (C/D box)"
    )
