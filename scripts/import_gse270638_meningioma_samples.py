#!/usr/bin/env python3
"""Build the GSE270638 meningioma sample manifest from GEO metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_builders import (
    GeoMatrixSource,
    geo_matrix_source_from_registry,
    sample_columns,
    verify_source_file_integrity,
)
from oncoref.expression_source_adapters import parse_geo_soft_samples

SOURCE_ID = "gse270638-meningioma"
COHORT = "GSE270638_MENINGIOMA_2024"
PIPELINE = "gse270638_htseq_raw_counts_gene_length_tpm_ensembl112_clean_tpm_16_9_75"
MANIFEST_COLUMNS = [
    "cancer_code",
    "source_cohort",
    "source_project",
    "case_id",
    "sample_id",
    "source_file_id",
    "source_file_name",
    "source_project_id",
    "sample_type",
    "primary_diagnosis",
    "md5sum",
    "file_size",
    "workflow_type",
    "raw_unit",
    "processing_pipeline",
    "source_url",
    "lineage_evidence_source",
    "included",
    "exclusion_reason",
    "lineage_label",
]


def build_manifest(
    soft_path: Path,
    matrix_path: Path,
    qc_path: Path,
    *,
    source: GeoMatrixSource | None = None,
) -> pd.DataFrame:
    """Return one auditable manifest row per public matrix column."""
    source = source or geo_matrix_source_from_registry(SOURCE_ID)
    verify_source_file_integrity(
        soft_path,
        label=f"{source.source_cohort} GEO SOFT",
        expected_bytes=source.soft_file_bytes,
        expected_md5=source.soft_file_md5,
    )
    metadata = parse_geo_soft_samples(soft_path)
    by_title = {values["title"]: values for values in metadata.values()}
    matrix = pd.read_parquet(matrix_path)
    samples = sample_columns(matrix)
    if len(samples) != 384 or set(samples) != set(by_title):
        raise ValueError(
            "GSE270638 matrix/SOFT identity mismatch: "
            f"matrix={len(samples)}, metadata={len(by_title)}"
        )

    qc = pd.read_csv(qc_path, keep_default_na=False).set_index("sample_id")
    if set(samples) != set(qc.index):
        raise ValueError("GSE270638 matrix/QC sample identities do not match")

    records = []
    for sample_id in samples:
        row = metadata[by_title[sample_id]["geo_accession"]]
        status = str(qc.loc[sample_id, "sample_qc_status"])
        reasons = str(qc.loc[sample_id, "sample_qc_reasons"])
        records.append(
            {
                "cancer_code": "MENINGIOMA",
                "source_cohort": COHORT,
                "source_project": "GEO",
                "case_id": sample_id,
                "sample_id": sample_id,
                "source_file_id": row["geo_accession"],
                "source_file_name": "GSE270638_MNG_RNAseq_Htseq_RawCounts.txt.gz",
                "source_project_id": "GSE270638",
                "sample_type": "Tumor",
                "primary_diagnosis": "Meningioma",
                "md5sum": "",
                "file_size": 20298059,
                "workflow_type": "STAR alignment; HTSeq counts",
                "raw_unit": "raw counts",
                "processing_pipeline": PIPELINE,
                "source_url": (
                    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
                    + row["geo_accession"]
                ),
                "lineage_evidence_source": (
                    f"{row['geo_accession']} GEO source/tissue metadata: meningioma"
                ),
                "included": status == "pass",
                "exclusion_reason": "" if status == "pass" else f"sample_qc_fail:{reasons}",
                "lineage_label": "MENINGIOMA",
            }
        )
    return pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--qc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Optional expression_sources.yaml path; defaults to the packaged registry.",
    )
    args = parser.parse_args()

    source = geo_matrix_source_from_registry(SOURCE_ID, registry_path=args.registry)
    manifest = build_manifest(args.soft, args.matrix, args.qc, source=source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(
        f"wrote {len(manifest)} GSE270638 rows "
        f"({int(manifest['included'].sum())} included) to {args.output}"
    )


if __name__ == "__main__":
    main()
