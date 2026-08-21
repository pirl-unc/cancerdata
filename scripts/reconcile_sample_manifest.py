#!/usr/bin/env python3
"""Reconcile the sample manifest with canonical registry-owned metadata."""

from __future__ import annotations

import argparse
import gzip
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.cancer_types import resolve_cancer_type


def _unique_pipeline_by_source(availability: pd.DataFrame) -> dict[str, str]:
    pipelines = {}
    for source_cohort, rows in availability.groupby("source_cohort", sort=False):
        values = {
            str(value).strip()
            for value in rows["processing_pipeline"]
            if pd.notna(value) and str(value).strip()
        }
        if len(values) == 1:
            pipelines[str(source_cohort)] = values.pop()
    return pipelines


def _canonical_lineage_labels(labels: pd.Series) -> pd.Series:
    canonical = labels.copy()
    replacements = {}
    unresolved = []
    for label in sorted(
        {str(value).strip() for value in labels if pd.notna(value) and str(value).strip()}
    ):
        resolved = resolve_cancer_type(label, strict=False)
        if resolved is None:
            unresolved.append(label)
        else:
            replacements[label] = resolved
    if unresolved:
        raise ValueError("unresolved lineage labels: " + ", ".join(unresolved))
    nonempty = canonical.notna() & canonical.astype(str).str.strip().ne("")
    canonical.loc[nonempty] = canonical.loc[nonempty].astype(str).str.strip().map(replacements)
    return canonical


def reconcile_sample_manifest(
    samples: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    """Return sample rows aligned to availability pipelines and canonical codes."""
    required_sample_columns = {"source_cohort", "processing_pipeline", "lineage_label"}
    required_availability_columns = {"source_cohort", "processing_pipeline"}
    missing_samples = required_sample_columns - set(samples.columns)
    missing_availability = required_availability_columns - set(availability.columns)
    if missing_samples:
        raise ValueError(f"sample manifest is missing columns: {sorted(missing_samples)}")
    if missing_availability:
        raise ValueError(
            f"availability manifest is missing columns: {sorted(missing_availability)}"
        )

    out = samples.copy()
    pipeline_by_source = _unique_pipeline_by_source(availability)
    current_pipeline = out["processing_pipeline"].copy()
    reconciled_pipeline = out["source_cohort"].astype(str).map(pipeline_by_source)
    out["processing_pipeline"] = reconciled_pipeline.fillna(current_pipeline)
    out["lineage_label"] = _canonical_lineage_labels(out["lineage_label"])
    return out


def replace_manifest_sources(samples: pd.DataFrame, updates: list[pd.DataFrame]) -> pd.DataFrame:
    """Replace complete physical-source manifests while preserving the public schema."""
    if not updates:
        return samples.copy()
    required = set(samples.columns)
    normalized = []
    replaced_sources: set[str] = set()
    for update in updates:
        missing = sorted(required - set(update.columns))
        if missing:
            raise ValueError(f"sample-manifest update lacks columns: {missing}")
        current = update[list(samples.columns)].copy()
        sources = {
            str(value).strip()
            for value in current["source_cohort"]
            if pd.notna(value) and str(value).strip()
        }
        if not sources:
            raise ValueError("sample-manifest update has no source_cohort")
        overlap = replaced_sources & sources
        if overlap:
            raise ValueError(f"sample-manifest update repeats sources: {sorted(overlap)}")
        replaced_sources.update(sources)
        normalized.append(current)
    retained = samples.loc[~samples["source_cohort"].astype(str).isin(replaced_sources)]
    return pd.concat([retained, *normalized], ignore_index=True)


def _write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=0,
        ) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text,
    ):
        frame.to_csv(text, index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("oncoref/data/cancer-reference-expression-samples.csv.gz"),
    )
    parser.add_argument(
        "--availability",
        type=Path,
        default=Path("oncoref/data/cancer-reference-expression-availability.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("oncoref/data/cancer-reference-expression-samples.csv.gz"),
    )
    parser.add_argument(
        "--append",
        action="append",
        default=[],
        type=Path,
        help="Complete source-manifest CSV to add or replace (repeatable)",
    )
    args = parser.parse_args()

    samples = pd.read_csv(args.samples, dtype=str, keep_default_na=False)
    availability = pd.read_csv(args.availability, dtype=str, keep_default_na=False)
    updates = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in args.append]
    samples = replace_manifest_sources(samples, updates)
    reconciled = reconcile_sample_manifest(samples, availability)
    _write_deterministic_gzip_csv(reconciled, args.output)
    print(f"wrote {len(reconciled)} reconciled sample rows to {args.output}")


if __name__ == "__main__":
    main()
