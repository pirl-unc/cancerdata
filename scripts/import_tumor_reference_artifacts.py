# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Import the tumor-reference artifacts that were historically owned by Trufflepig.

This is an offline migration boundary, not a second decomposition pipeline. The
TCGA artifact is copied byte-for-byte after validation. The legacy subtype table
is also preserved except for its BeatAML rows: those rows contain negative Q1
values and are rebuilt as high-purity passthrough summaries from Oncoref's clean,
per-sample BeatAML matrices.

No network tools, Salmon binaries, or Trufflepig imports are used.

Run::

    python scripts/import_tumor_reference_artifacts.py \
      --tcga ~/code/trufflepig/trufflepig/data/tcga-deconvolved-expression.csv.gz \
      --subtype ~/code/trufflepig/trufflepig/data/subtype-deconvolved-expression.csv.gz \
      --beataml-matrix LAML_APL=~/.cache/oncoref/source-matrices/v5.22.9/LAML_APL.parquet \
      --beataml-matrix LAML_ELNadv=~/.cache/oncoref/source-matrices/v5.22.9/LAML_ELNadv.parquet \
      --beataml-matrix LAML_ELNfav=~/.cache/oncoref/source-matrices/v5.22.9/LAML_ELNfav.parquet \
      --beataml-matrix LAML_ELNint=~/.cache/oncoref/source-matrices/v5.22.9/LAML_ELNint.parquet \
      --sample-qc ~/.cache/oncoref/bundled_data/v5.23.14/source-matrix-sample-qc.csv \
      --source-commit f4a87c39b1c8b8939e89778113614a9f2c303d59 \
      --output-dir /tmp/oncoref-data-v5.23.15-source
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.tumor_references import (
    _SUBTYPE_COLUMNS,
    _TCGA_COLUMNS,
    TUMOR_REFERENCE_PROVENANCE_COLUMNS,
    _canonicalize_codes,
    _validate_reference,
    _with_stable_schema,
)
from oncoref.version import SOURCE_MATRIX_VERSION

TCGA_ARTIFACT = "tcga-deconvolved-expression.csv.gz"
SUBTYPE_ARTIFACT = "subtype-deconvolved-expression.csv.gz"
PROVENANCE_ARTIFACT = "tumor-reference-expression-provenance.csv"
BEATAML_SOURCE_COHORT = "BEATAML_OHSU_2022"
BEATAML_SUBTYPES = ("LAML_APL", "LAML_ELNadv", "LAML_ELNfav", "LAML_ELNint")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
    ):
        frame.to_csv(text, index=False, lineterminator="\n")


def _read_reference(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "Ensembl_Gene_ID": "string",
            "symbol": "string",
            "cancer_code": "string",
            "subtype": "string",
            "source_cohort": "string",
        },
    )


def _validated_reference(
    frame: pd.DataFrame,
    *,
    dataset: str,
    columns: tuple[str, ...],
    identity_columns: tuple[str, ...],
) -> pd.DataFrame:
    frame = _with_stable_schema(frame, columns)
    frame = _canonicalize_codes(frame)
    return _validate_reference(
        frame,
        dataset=dataset,
        identity_columns=identity_columns,
    )


def summarize_passthrough_matrix(
    matrix: pd.DataFrame,
    *,
    subtype_code: str,
    min_tpm: float = 0.01,
) -> pd.DataFrame:
    """Aggregate one clean BeatAML gene-by-sample matrix without deconvolution."""
    required = {"Ensembl_Gene_ID", "Symbol"}
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise ValueError(f"{subtype_code} matrix lacks required columns: {missing}")
    sample_columns = [column for column in matrix.columns if column not in required]
    if not sample_columns:
        raise ValueError(f"{subtype_code} matrix has no sample columns")

    symbols = matrix["Symbol"].astype("string").str.strip()
    keep = symbols.notna() & symbols.ne("")
    values = matrix.loc[keep, sample_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{subtype_code} matrix contains missing or non-finite TPM")
    if (values < 0).any().any():
        raise ValueError(f"{subtype_code} matrix contains negative TPM")

    gene_rows = pd.DataFrame(
        {
            "Ensembl_Gene_ID": matrix.loc[keep, "Ensembl_Gene_ID"].astype("string"),
            "symbol": symbols.loc[keep],
        }
    )
    collapsed = values.groupby(gene_rows["symbol"], sort=False).sum()
    identifiers = gene_rows.groupby("symbol", sort=False)["Ensembl_Gene_ID"].agg(
        lambda series: _unambiguous_identifier(series)
    )
    collapsed = collapsed.loc[collapsed.max(axis=1) >= min_tpm]

    out = pd.DataFrame(
        {
            "Ensembl_Gene_ID": identifiers.reindex(collapsed.index).array,
            "symbol": collapsed.index.astype("string"),
            "cancer_code": "LAML",
            "subtype": subtype_code,
            "source_cohort": BEATAML_SOURCE_COHORT,
            "tumor_tpm_median": collapsed.median(axis=1).to_numpy(),
            "tumor_tpm_q1": collapsed.quantile(0.25, axis=1).to_numpy(),
            "tumor_tpm_q3": collapsed.quantile(0.75, axis=1).to_numpy(),
            "n_samples": len(sample_columns),
        }
    )
    return _validated_reference(
        out,
        dataset=f"{subtype_code} passthrough summary",
        columns=_SUBTYPE_COLUMNS,
        identity_columns=("cancer_code", "subtype", "source_cohort"),
    )


def _unambiguous_identifier(series: pd.Series) -> object:
    identifiers = series.dropna().astype(str).str.strip().str.split(".").str[0]
    identifiers = identifiers[identifiers.ne("")].drop_duplicates()
    return identifiers.iloc[0] if len(identifiers) == 1 else pd.NA


def _provenance_rows(
    frame: pd.DataFrame,
    *,
    artifact: str,
    derivation_method: str,
    derivation_status: str,
    source_scale: str,
    processing_pipeline: str,
    source_artifact: str,
    source_artifact_sha256: str,
    source_commit: str | None,
    sample_qc_policy: str,
    sample_qc_artifact: str | None,
    sample_qc_artifact_sha256: str | None,
    output_artifact_sha256: str,
    notes: str,
) -> list[dict]:
    group_columns = ["cancer_code"]
    if "subtype" in frame.columns:
        group_columns.append("subtype")
    if "source_cohort" in frame.columns:
        group_columns.append("source_cohort")
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key))
        rows.append(
            {
                "artifact": artifact.removesuffix(".csv.gz"),
                "cancer_code": identity["cancer_code"],
                "subtype": identity.get("subtype", pd.NA),
                "source_cohort": identity.get("source_cohort", "TCGA_XENA_TOIL_RSEM"),
                "derivation_method": derivation_method,
                "derivation_status": derivation_status,
                "source_scale": source_scale,
                "processing_pipeline": processing_pipeline,
                "source_artifact": source_artifact,
                "source_artifact_sha256": source_artifact_sha256,
                "source_artifact_commit": source_commit,
                "source_matrix_version": SOURCE_MATRIX_VERSION
                if derivation_status == "rebuilt_from_oncoref_source_matrix"
                else pd.NA,
                "sample_qc_policy": sample_qc_policy,
                "sample_qc_artifact": sample_qc_artifact,
                "sample_qc_artifact_sha256": sample_qc_artifact_sha256,
                "output_artifact_sha256": output_artifact_sha256,
                "n_genes": len(group),
                "n_reference_samples": int(pd.to_numeric(group["n_samples"]).max()),
                "notes": notes,
            }
        )
    return rows


def _pass_sample_columns(
    sample_qc: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    subtype_code: str,
) -> list[str]:
    required = {"cancer_code", "source_cohort", "sample_id", "sample_qc_status"}
    missing = sorted(required - set(sample_qc.columns))
    if missing:
        raise ValueError(f"sample QC manifest lacks required columns: {missing}")

    records = sample_qc[
        sample_qc["cancer_code"].astype(str).eq(subtype_code)
        & sample_qc["source_cohort"].astype(str).eq(BEATAML_SOURCE_COHORT)
    ].copy()
    if records.empty:
        raise ValueError(f"sample QC manifest has no records for {subtype_code}")
    if records["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"sample QC manifest has duplicate sample IDs for {subtype_code}")

    matrix_samples = [
        column for column in matrix.columns if column not in {"Ensembl_Gene_ID", "Symbol"}
    ]
    manifest_samples = set(records["sample_id"].astype(str))
    if set(matrix_samples) != manifest_samples:
        missing_qc = sorted(set(matrix_samples) - manifest_samples)
        missing_matrix = sorted(manifest_samples - set(matrix_samples))
        raise ValueError(
            f"{subtype_code} matrix/QC sample mismatch: "
            f"missing QC={missing_qc[:8]}, missing matrix={missing_matrix[:8]}"
        )

    pass_samples = set(
        records.loc[records["sample_qc_status"].astype(str).eq("pass"), "sample_id"].astype(str)
    )
    if not pass_samples:
        raise ValueError(f"sample QC manifest has no passing samples for {subtype_code}")
    return [sample for sample in matrix_samples if sample in pass_samples]


def import_artifacts(
    *,
    tcga_path: Path,
    subtype_path: Path,
    beataml_matrices: dict[str, Path],
    sample_qc_path: Path,
    output_dir: Path,
    source_commit: str,
) -> tuple[Path, Path, Path]:
    """Validate, repair, and stage the three downloadable bundle files."""
    missing_subtypes = sorted(set(BEATAML_SUBTYPES) - set(beataml_matrices))
    if missing_subtypes:
        raise ValueError(f"missing BeatAML matrices: {missing_subtypes}")
    unexpected_subtypes = sorted(set(beataml_matrices) - set(BEATAML_SUBTYPES))
    if unexpected_subtypes:
        raise ValueError(f"unexpected BeatAML matrices: {unexpected_subtypes}")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source_commit must be a full 40-character lowercase Git commit")

    tcga = _validated_reference(
        _read_reference(tcga_path),
        dataset=TCGA_ARTIFACT,
        columns=_TCGA_COLUMNS,
        identity_columns=("cancer_code",),
    )
    legacy_subtype = _read_reference(subtype_path)
    legacy_subtype = legacy_subtype[
        legacy_subtype["source_cohort"].astype(str).ne(BEATAML_SOURCE_COHORT)
    ].copy()
    legacy_subtype = _validated_reference(
        legacy_subtype,
        dataset=SUBTYPE_ARTIFACT,
        columns=_SUBTYPE_COLUMNS,
        identity_columns=("cancer_code", "subtype", "source_cohort"),
    )

    sample_qc = pd.read_csv(sample_qc_path, low_memory=False)
    sample_qc_hash = sha256_file(sample_qc_path)
    repaired_parts = []
    for subtype_code in BEATAML_SUBTYPES:
        matrix = pd.read_parquet(beataml_matrices[subtype_code])
        pass_samples = _pass_sample_columns(sample_qc, matrix, subtype_code=subtype_code)
        repaired_parts.append(
            summarize_passthrough_matrix(
                matrix[["Ensembl_Gene_ID", "Symbol", *pass_samples]],
                subtype_code=subtype_code,
            )
        )
    repaired = pd.concat(repaired_parts, ignore_index=True)
    for part in (legacy_subtype, repaired):
        for column in ("Ensembl_Gene_ID", "symbol", "cancer_code", "subtype", "source_cohort"):
            part[column] = part[column].astype("string")
    subtype = pd.concat([legacy_subtype, repaired], ignore_index=True)
    subtype = _validated_reference(
        subtype,
        dataset=SUBTYPE_ARTIFACT,
        columns=_SUBTYPE_COLUMNS,
        identity_columns=("cancer_code", "subtype", "source_cohort"),
    )
    subtype = subtype.sort_values(
        ["source_cohort", "cancer_code", "subtype", "symbol", "Ensembl_Gene_ID"],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    tcga_output = output_dir / TCGA_ARTIFACT
    subtype_output = output_dir / SUBTYPE_ARTIFACT
    provenance_output = output_dir / PROVENANCE_ARTIFACT
    shutil.copyfile(tcga_path, tcga_output)
    _write_deterministic_gzip(subtype, subtype_output)

    tcga_hash = sha256_file(tcga_path)
    subtype_hash = sha256_file(subtype_path)
    tcga_output_hash = sha256_file(tcga_output)
    subtype_output_hash = sha256_file(subtype_output)
    provenance = _provenance_rows(
        tcga,
        artifact=TCGA_ARTIFACT,
        derivation_method="tme_deconvolution",
        derivation_status="migrated_pinned_artifact",
        source_scale="tumor_attributed_tpm",
        processing_pipeline="trufflepig.tcga_decompose",
        source_artifact=tcga_path.name,
        source_artifact_sha256=tcga_hash,
        source_commit=source_commit,
        sample_qc_policy="legacy_artifact",
        sample_qc_artifact=None,
        sample_qc_artifact_sha256=None,
        output_artifact_sha256=tcga_output_hash,
        notes="Pinned Trufflepig output; Oncoref does not reimplement the decomposition model.",
    )
    provenance.extend(
        _provenance_rows(
            legacy_subtype,
            artifact=SUBTYPE_ARTIFACT,
            derivation_method="observed_tpm_passthrough",
            derivation_status="legacy_artifact_source_inferred",
            source_scale="observed_tpm",
            processing_pipeline="trufflepig legacy subtype artifact",
            source_artifact=subtype_path.name,
            source_artifact_sha256=subtype_hash,
            source_commit=source_commit,
            sample_qc_policy="legacy_artifact",
            sample_qc_artifact=None,
            sample_qc_artifact_sha256=None,
            output_artifact_sha256=subtype_output_hash,
            notes="Source-level passthrough summary; not labeled as TME-deconvolved.",
        )
    )
    for subtype_code, matrix_path in beataml_matrices.items():
        part = repaired[repaired["subtype"].astype(str).eq(subtype_code)]
        provenance.extend(
            _provenance_rows(
                part,
                artifact=SUBTYPE_ARTIFACT,
                derivation_method="high_purity_passthrough",
                derivation_status="rebuilt_from_oncoref_source_matrix",
                source_scale="clean_tpm_16_9_75",
                processing_pipeline="oncoref.import_tumor_reference_artifacts",
                source_artifact=matrix_path.name,
                source_artifact_sha256=sha256_file(matrix_path),
                source_commit=None,
                sample_qc_policy="pass",
                sample_qc_artifact=sample_qc_path.name,
                sample_qc_artifact_sha256=sample_qc_hash,
                output_artifact_sha256=subtype_output_hash,
                notes="Observed clean TPM is used as tumor TPM for high-purity BeatAML samples.",
            )
        )
    provenance_frame = pd.DataFrame(provenance).sort_values(
        ["artifact", "source_cohort", "cancer_code", "subtype"],
        na_position="last",
        kind="stable",
    )
    provenance_frame[list(TUMOR_REFERENCE_PROVENANCE_COLUMNS)].to_csv(
        provenance_output, index=False, lineterminator="\n"
    )
    return tcga_output, subtype_output, provenance_output


def _parse_matrix(value: str) -> tuple[str, Path]:
    code, separator, path = value.partition("=")
    if not separator or code not in BEATAML_SUBTYPES or not path:
        raise argparse.ArgumentTypeError(
            "BeatAML matrices must use CANONICAL_SUBTYPE=/path, where subtype is "
            + ", ".join(BEATAML_SUBTYPES)
        )
    return code, Path(path).expanduser()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcga", type=Path, required=True)
    parser.add_argument("--subtype", type=Path, required=True)
    parser.add_argument("--beataml-matrix", action="append", type=_parse_matrix, default=[])
    parser.add_argument("--sample-qc", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = import_artifacts(
        tcga_path=args.tcga.expanduser(),
        subtype_path=args.subtype.expanduser(),
        beataml_matrices=dict(args.beataml_matrix),
        sample_qc_path=args.sample_qc.expanduser(),
        output_dir=args.output_dir.expanduser(),
        source_commit=args.source_commit,
    )
    for path in outputs:
        print(f"{path}: {sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
