#!/usr/bin/env python3
"""Merge a targeted expression rebuild into an existing complete data bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.representative_partitions import (
    REPRESENTATIVE_PARTITION_COHORT_COLUMNS,
    assign_representative_partitions,
    representative_partition_build_metadata,
    representative_partition_cohort_metadata,
)

PROVENANCE_PATH = Path("cancer-reference-expression-representatives/_provenance.csv")
BUILD_METADATA_CSV = Path("expression-artifact-build-metadata.csv")
BUILD_METADATA_JSON = Path("expression-artifact-build-metadata.json")
SAMPLE_QC_PATH = Path("source-matrix-sample-qc.csv")
GLOBAL_PATHS = {PROVENANCE_PATH, BUILD_METADATA_CSV, BUILD_METADATA_JSON, SAMPLE_QC_PATH}


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    shutil.copy2(source, target)


def _replace_codes(base: pd.DataFrame, update: pd.DataFrame, codes: set[str]) -> pd.DataFrame:
    if "cancer_code" not in base or "cancer_code" not in update:
        raise ValueError("incremental expression metadata lacks cancer_code")
    retained = base.loc[~base["cancer_code"].astype(str).isin(codes)]
    return pd.concat([retained, update], ignore_index=True)


def merge_expression_artifact_update(base: Path, update: Path, out: Path) -> set[str]:
    """Hardlink-clone ``base`` and atomically replace the cohorts in ``update``."""
    if out.exists():
        raise FileExistsError(f"incremental output already exists: {out}")
    for root, label in ((base, "base"), (update, "update")):
        missing = sorted(str(path) for path in GLOBAL_PATHS if not (root / path).is_file())
        if missing:
            raise FileNotFoundError(f"{label} artifact tree lacks files: {missing}")

    update_build = pd.read_csv(update / BUILD_METADATA_CSV, keep_default_na=False)
    codes = set(update_build["cancer_code"].astype(str))
    if not codes:
        raise ValueError("incremental expression update contains no cohorts")
    shutil.copytree(base, out, copy_function=os.link)

    for source in sorted(path for path in update.rglob("*") if path.is_file()):
        relative = source.relative_to(update)
        if relative in GLOBAL_PATHS:
            continue
        _replace_file(source, out / relative)

    base_provenance = pd.read_csv(base / PROVENANCE_PATH, keep_default_na=False)
    update_provenance = pd.read_csv(update / PROVENANCE_PATH, keep_default_na=False)
    provenance = _replace_codes(base_provenance, update_provenance, codes)
    partition_columns = set(REPRESENTATIVE_PARTITION_COHORT_COLUMNS[1:]) | {
        "partition_role",
        "partition_reason",
    }
    provenance = provenance.drop(
        columns=[column for column in partition_columns if column in provenance],
    )
    provenance = assign_representative_partitions(provenance)
    provenance_path = out / PROVENANCE_PATH
    provenance_path.unlink(missing_ok=True)
    provenance.to_csv(provenance_path, index=False)

    base_qc = pd.read_csv(base / SAMPLE_QC_PATH, keep_default_na=False)
    update_qc = pd.read_csv(update / SAMPLE_QC_PATH, keep_default_na=False)
    qc = _replace_codes(base_qc, update_qc, codes)
    qc_path = out / SAMPLE_QC_PATH
    qc_path.unlink(missing_ok=True)
    qc.to_csv(qc_path, index=False)

    base_build = pd.read_csv(base / BUILD_METADATA_CSV, keep_default_na=False)
    build = _replace_codes(base_build, update_build, codes)
    cohort_partition_columns = set(REPRESENTATIVE_PARTITION_COHORT_COLUMNS[1:])
    build = build.drop(
        columns=[column for column in cohort_partition_columns if column in build],
    ).merge(
        representative_partition_cohort_metadata(provenance),
        on="cancer_code",
        how="left",
        validate="one_to_one",
    )
    build = build.sort_values("cancer_code", kind="stable").reset_index(drop=True)
    build_path = out / BUILD_METADATA_CSV
    build_path.unlink(missing_ok=True)
    build.to_csv(build_path, index=False)

    metadata = json.loads((base / BUILD_METADATA_JSON).read_text())
    metadata.update(
        {
            "n_cohorts": len(build),
            "n_source_samples": int(pd.to_numeric(build["n_source_samples"]).sum()),
            "n_cohort_samples": int(pd.to_numeric(build["n_cohort_samples"]).sum()),
            "n_negative_values_clipped": int(
                pd.to_numeric(build["n_negative_values_clipped"]).sum()
            ),
            "sample_qc_fallbacks": int(
                build["sample_qc_fallback_reason"].astype(str).str.strip().ne("").sum()
            ),
            "representative_partition": representative_partition_build_metadata(provenance),
        }
    )
    metadata_path = out / BUILD_METADATA_JSON
    metadata_path.unlink(missing_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--update", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    codes = merge_expression_artifact_update(args.base, args.update, args.out)
    print(f"merged {len(codes)} cohorts ({', '.join(sorted(codes))}) -> {args.out}")


if __name__ == "__main__":
    main()
