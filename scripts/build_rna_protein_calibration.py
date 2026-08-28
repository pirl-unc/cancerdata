#!/usr/bin/env python
"""Acquire and standardize the matched CPTAC inputs for RNA/protein calibration."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.gene_ids import canonical_gene_ids
from oncoref.rna_protein import (
    CPTAC_CALIBRATION_COHORTS,
    rna_protein_calibration_sources,
)

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_ATTEMPTS = 4


@dataclass(frozen=True)
class MatchedCptacCohort:
    """Canonical matched matrices plus the decisions needed to audit their join."""

    rna: pd.DataFrame
    protein: pd.DataFrame
    samples: pd.DataFrame
    genes: pd.DataFrame
    excluded_genes: pd.DataFrame


def file_md5(path: Path) -> str:
    """Return the lowercase MD5 digest used by the immutable Zenodo record."""

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_file(path: Path, source: pd.Series) -> None:
    """Reject a source file whose byte size or digest differs from the manifest."""

    expected_size = int(source["byte_size"])
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"{source['source_id']} expected {expected_size} bytes at {path}")
    expected_md5 = str(source["checksum"]).removeprefix("md5:")
    observed_md5 = file_md5(path)
    if observed_md5 != expected_md5:
        raise ValueError(
            f"{source['source_id']} MD5 mismatch: expected {expected_md5}, got {observed_md5}"
        )


def _fetch_range(url: str, start: int, end: int, path: Path) -> None:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                content_range = response.headers.get("Content-Range")
                if (
                    response.status != 206
                    or not content_range
                    or not content_range.startswith(f"bytes {start}-{end}/")
                ):
                    raise ValueError(
                        f"source did not honor byte range {start}-{end}: {content_range!r}"
                    )
                with path.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            if path.stat().st_size != expected:
                raise ValueError(
                    f"range {start}-{end} returned {path.stat().st_size} bytes, expected {expected}"
                )
            return
        except Exception as error:
            last_error = error
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            if attempt + 1 < DOWNLOAD_ATTEMPTS:
                time.sleep(0.25 * (2**attempt))
    raise RuntimeError(f"failed to fetch byte range {start}-{end}") from last_error


def fetch_source_file(
    source: pd.Series,
    cache_dir: Path,
    *,
    workers: int = 8,
    force: bool = False,
) -> Path:
    """Download one manifest row atomically using validated HTTP byte ranges."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / str(source["source_file"])
    if target.exists() and not force:
        verify_source_file(target, source)
        return target

    total = int(source["byte_size"])
    ranges = [
        (start, min(total - 1, start + DOWNLOAD_CHUNK_BYTES - 1))
        for start in range(0, total, DOWNLOAD_CHUNK_BYTES)
    ]
    part_dir = cache_dir / ".parts" / str(source["source_id"])
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [part_dir / f"{index:04d}.part" for index in range(len(ranges))]

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(ranges)))) as executor:
        futures = [
            executor.submit(_fetch_range, str(source["source_url"]), start, end, part_path)
            for (start, end), part_path in zip(ranges, part_paths)
        ]
        for future in futures:
            future.result()

    partial = target.with_suffix(target.suffix + ".partial")
    with partial.open("wb") as output:
        for part_path in part_paths:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, output)
    verify_source_file(partial, source)
    partial.replace(target)

    for part_path in part_paths:
        part_path.unlink()
    with contextlib.suppress(OSError):
        part_dir.rmdir()
        part_dir.parent.rmdir()
    return target


def load_matched_cptac_cohort(rna_path: Path, protein_path: Path) -> MatchedCptacCohort:
    """Join one BCM RNA/protein tumor pair in canonical gene and patient space.

    RNA ``_T`` columns are tumors; ``_A`` adjacent-normal columns are excluded.
    Protein columns already contain tumor patient IDs. Only their deterministic
    intersection is retained. Multi-gene groups and unmapped identifiers are
    excluded. If multiple source identifiers collapse to one canonical gene, all
    colliding rows are excluded instead of inventing an aggregation on a log scale.
    Missing protein values remain missing and define observed/not-observed outcomes
    for the later calibration model.
    """

    rna_raw = pd.read_csv(rna_path, sep="\t", index_col=0, low_memory=False)
    protein_raw = pd.read_csv(protein_path, sep="\t", index_col=0, low_memory=False)
    rna_raw.columns = rna_raw.columns.astype(str).str.strip()
    protein_raw.columns = protein_raw.columns.astype(str).str.strip()

    rna = rna_raw.loc[:, rna_raw.columns.str.endswith("_T")].copy()
    rna.columns = rna.columns.str.removesuffix("_T")
    if rna.columns.duplicated().any() or protein_raw.columns.duplicated().any():
        raise ValueError("CPTAC sample identifiers must be unique after tumor normalization")
    matched_samples = sorted(set(rna.columns) & set(protein_raw.columns))
    if not matched_samples:
        raise ValueError("CPTAC RNA and protein matrices have no matched tumor samples")
    rna = rna.loc[:, matched_samples].apply(pd.to_numeric, errors="coerce")
    protein = protein_raw.loc[:, matched_samples].apply(pd.to_numeric, errors="coerce")

    source_gene_ids = sorted(set(rna.index.astype(str)) & set(protein.index.astype(str)))
    mapping = pd.DataFrame({"source_gene_id": source_gene_ids})
    mapping["exclusion_reason"] = ""
    multi_gene = mapping["source_gene_id"].str.contains(r"[;,|]", regex=True)
    mapping.loc[multi_gene, "exclusion_reason"] = "multi_gene_group"
    candidates = mapping.loc[~multi_gene, "source_gene_id"].tolist()
    canonical = canonical_gene_ids(candidates)
    mapping.loc[~multi_gene, "canonical_gene_id"] = canonical
    unmapped = mapping["canonical_gene_id"].isna() & mapping["exclusion_reason"].eq("")
    mapping.loc[unmapped, "exclusion_reason"] = "unmapped_gene_id"
    duplicated = mapping["canonical_gene_id"].notna() & mapping["canonical_gene_id"].duplicated(
        keep=False
    )
    mapping.loc[duplicated, "exclusion_reason"] = "duplicate_canonical_gene"

    kept = mapping.loc[mapping["exclusion_reason"].eq("")].copy()
    kept = kept.sort_values("canonical_gene_id").reset_index(drop=True)
    excluded = mapping.loc[~mapping["exclusion_reason"].eq("")].copy()
    excluded = excluded.sort_values(["exclusion_reason", "source_gene_id"]).reset_index(drop=True)
    if kept.empty:
        raise ValueError("no canonical genes remain after CPTAC identifier validation")

    rna = rna.loc[kept["source_gene_id"]].copy()
    protein = protein.loc[kept["source_gene_id"]].copy()
    rna.index = kept["canonical_gene_id"]
    protein.index = kept["canonical_gene_id"]
    rna.index.name = protein.index.name = "canonical_gene_id"

    samples = pd.DataFrame(
        {
            "sample_id": matched_samples,
            "rna_source_sample_id": [f"{sample}_T" for sample in matched_samples],
            "protein_source_sample_id": matched_samples,
        }
    )
    genes = kept[["canonical_gene_id", "source_gene_id"]].copy()
    return MatchedCptacCohort(rna, protein, samples, genes, excluded)


def build_cohort_inputs(
    cohort: str,
    *,
    cache_dir: Path,
    output_dir: Path,
    download: bool,
    force: bool,
    workers: int,
) -> dict:
    """Verify, standardize, and write one deterministic CPTAC cohort input set."""

    sources = rna_protein_calibration_sources(cptac_cohort=cohort)
    paths: dict[str, Path] = {}
    for _, source in sources.iterrows():
        path = cache_dir / str(source["source_file"])
        if download:
            path = fetch_source_file(source, cache_dir, workers=workers, force=force)
        verify_source_file(path, source)
        paths[str(source["modality"])] = path

    matched = load_matched_cptac_cohort(paths["rna"], paths["protein"])
    cohort_dir = output_dir / cohort.lower()
    cohort_dir.mkdir(parents=True, exist_ok=True)
    matched.rna.to_parquet(cohort_dir / "rna.parquet")
    matched.protein.to_parquet(cohort_dir / "protein.parquet")
    matched.samples.to_csv(cohort_dir / "samples.csv", index=False)
    matched.genes.to_csv(cohort_dir / "genes.csv", index=False)
    matched.excluded_genes.to_csv(cohort_dir / "excluded-genes.csv", index=False)

    metadata = {
        "cptac_cohort": cohort,
        "cancer_code": sources["cancer_code"].iloc[0],
        "n_matched_samples": len(matched.samples),
        "n_canonical_genes": len(matched.genes),
        "protein_missing_fraction": float(matched.protein.isna().to_numpy().mean()),
        "excluded_gene_counts": matched.excluded_genes["exclusion_reason"].value_counts().to_dict(),
        "sample_join_policy": "RNA tumor suffix _T stripped; exact patient-ID intersection",
        "multi_gene_policy": "exclude source identifiers containing ; , or |",
        "duplicate_gene_policy": "exclude every row in a duplicate canonical-gene collision",
        "missing_protein_policy": "preserve as not-observed outcome; never impute",
        "sources": sources[
            ["source_id", "source_record", "source_file", "byte_size", "checksum"]
        ].to_dict(orient="records"),
    }
    metadata_path = cohort_dir / "build-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=CPTAC_CALIBRATION_COHORTS)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    metadata = build_cohort_inputs(
        args.cohort,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        download=args.download,
        force=args.force,
        workers=args.workers,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
