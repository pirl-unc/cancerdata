# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end rebuild of the per-cohort expression artifacts from per-sample matrices.

Ties the pieces together so oncoref can **regenerate** (not just hold) its
expression bundle:

    raw per-sample TPM matrices  (candidate source cohorts per cancer code)
        -> select ONE source per code  (the source-matrices.csv choice; never pool)
        -> clean_tpm                   (two-compartment biological view)
        -> sample QC filter            (pass by default; opt out with --sample-qc all)
        -> drop technical genes        (biology-only, matching the shipped artifact)
        -> percentile vectors / n=5 representatives / within-sample top-fractions

Input matrices are discovered under ``--cache`` as
``<cohort>/derived/<NAME>_per_sample_tpm.parquet``. The derived ``<NAME>`` is mapped
to a registered source-matrix cancer code case-insensitively (``tcga_acc`` -> ``ACC``,
``LAML_ELNadv`` kept). A code with several candidate source cohorts is
resolved to the single one recorded in ``source-matrices.csv`` (pirlygenes selects
one source per code; it never pools) — so the artifacts match the shipped reference.

Outputs land under ``--out`` (a staging dir, NOT ``oncoref/data`` — the artifacts
are large and ship via the release tarball, so they're never committed):

    <out>/clean/<CODE>.parquet                                 (QC-filtered clean TPM)
    <out>/cancer-reference-expression/<SOURCE>.csv.gz          (raw + clean summary rows)
    <out>/cancer-reference-expression-percentiles/<CODE>.parquet      (biology-only)
    <out>/cancer-reference-expression-representatives/<CODE>.parquet + _provenance.csv
    <out>/cancer-reference-expression-within-sample-top5/<CODE>.parquet (biology-only)
    <out>/source-matrix-sample-qc.csv                          (per-sample QC manifest)
    <out>/expression-artifact-build-metadata.csv               (per-cohort provenance)
    <out>/expression-artifact-build-metadata.json              (QC/build policy metadata)

``--validate`` additionally correlates each rebuilt percentile vector against the
reference artifact in ``--ref`` and prints the per-code agreement.

Run:
    python scripts/rebuild_expression_artifacts.py \
        --cache ~/.cache/pirlygenes/expression \
        --ref   ~/code/pirlygenes/pirlygenes/data/cancer-reference-expression-percentiles \
        --out   ~/.cache/oncoref/rebuild-staging [--limit N] [--validate]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.cancer_types import cohort_source_version
from oncoref.expression import (
    EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION,
    SAMPLE_EXPRESSION_QC_POLICY_VERSION,
    SHARD_DATASETS,
    _canonicalize_gene_rows,
    _validate_sample_qc,
    sample_columns,
    sample_expression_qc_from_matrix,
)
from oncoref.expression_builders import (
    cohort_medoids,
    cohort_percentile_vectors,
    summarize_source_matrix,
    within_sample_top_fractions,
)
from oncoref.expression_registry import expression_source_registry_entries
from oncoref.gene_families import clean_tpm_censored_gene_ids
from oncoref.load_dataset import get_data
from oncoref.normalization import clean_tpm
from oncoref.representative_partitions import (
    assign_representative_partitions,
    representative_partition_build_metadata,
    representative_partition_cohort_metadata,
)
from oncoref.source_matrices import registry as source_registry
from oncoref.source_matrices import source_sample_namespace

# Rebuilt artifacts must land in the exact directories the reader resolves; derive every
# name from the shared registry so producer and reader can't drift. Proteoform shards are
# built at "cta" scope (the only scope generated today).
_PCT_DS = SHARD_DATASETS["percentiles"]
_WS_DS = SHARD_DATASETS["within_sample"]
_PROTEOFORM_SCOPE = "cta"
_SUMMARY_DIR = "cancer-reference-expression"

_BASE = ["Ensembl_Gene_ID", "Symbol"]


@dataclass(frozen=True)
class _SummarySourceMetadata:
    """Static provenance copied into a rebuilt reference-summary shard."""

    source_cohort: str
    source_project: str | None
    source_version: str | None
    processing_pipeline: str | None
    notes: str
    tumor_origin: str
    metastasis_site: str | None
    unit: str = "TPM"
    citation: str | None = None
    pipeline_stem: str = ""


@dataclass(frozen=True)
class _RepresentativeAdjudication:
    """Reviewed provenance override for one physical source sample."""

    source_project: str
    source_diagnosis: str
    source_morphology: str
    representative_role: str
    benchmark_eligible: bool
    review_source: str
    review_note: str


_REPRESENTATIVE_ADJUDICATION_COLUMNS = (
    "source_group_id",
    "source_project",
    "source_diagnosis",
    "source_morphology",
    "representative_role",
    "benchmark_eligible",
    "review_source",
    "review_note",
)


def _optional_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _required_adjudication_text(row: dict, column: str) -> str:
    value = _optional_text(row[column])
    if value is None:
        raise ValueError(f"representative source adjudication has blank {column}")
    return value


def _representative_adjudications() -> dict[str, _RepresentativeAdjudication]:
    """Load reviewed source-sample decisions keyed by stable source-group ID."""

    df = get_data("representative-source-adjudications")
    missing = sorted(set(_REPRESENTATIVE_ADJUDICATION_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"representative source adjudications lack columns: {missing}")
    if df["source_group_id"].astype(str).duplicated().any():
        raise ValueError("representative source adjudications contain duplicate source_group_id")

    out: dict[str, _RepresentativeAdjudication] = {}
    for row in df.to_dict("records"):
        source_group_id = _required_adjudication_text(row, "source_group_id")
        benchmark_text = str(row["benchmark_eligible"]).strip().lower()
        if benchmark_text not in {"true", "false"}:
            raise ValueError("representative benchmark_eligible must be true or false")
        out[source_group_id] = _RepresentativeAdjudication(
            source_project=_required_adjudication_text(row, "source_project"),
            source_diagnosis=_required_adjudication_text(row, "source_diagnosis"),
            source_morphology=_required_adjudication_text(row, "source_morphology"),
            representative_role=_required_adjudication_text(row, "representative_role"),
            benchmark_eligible=benchmark_text == "true",
            review_source=_required_adjudication_text(row, "review_source"),
            review_note=_required_adjudication_text(row, "review_note"),
        )
    return out


def _representative_provenance_fields(
    *,
    source_group_id: str,
    default_source_project: str | None,
    default_benchmark_eligible: bool,
    adjudications: dict[str, _RepresentativeAdjudication],
) -> dict:
    """Resolve normal QC provenance or one explicit reviewed override."""

    default_role = "standard" if default_benchmark_eligible else "source_qc_fallback_audit_only"
    adjudication = adjudications.get(source_group_id)
    if adjudication is None:
        return {
            "source_project": default_source_project,
            "source_diagnosis": None,
            "source_morphology": None,
            "representative_role": default_role,
            "benchmark_eligible": default_benchmark_eligible,
            "review_source": None,
            "review_note": None,
        }
    if adjudication.benchmark_eligible and not default_benchmark_eligible:
        raise ValueError(
            f"{source_group_id}: representative source adjudication cannot override "
            "source QC to make a sample benchmark-eligible"
        )
    return {
        "source_project": adjudication.source_project,
        "source_diagnosis": adjudication.source_diagnosis,
        "source_morphology": adjudication.source_morphology,
        "representative_role": adjudication.representative_role,
        "benchmark_eligible": adjudication.benchmark_eligible,
        "review_source": adjudication.review_source,
        "review_note": adjudication.review_note,
    }


def _summary_source_metadata(code: str, source_cohort: str) -> _SummarySourceMetadata:
    """Load static summary provenance without reusing the stale sample count."""
    registry_matches = [
        entry
        for entry in expression_source_registry_entries()
        if code in {str(value) for value in entry.get("cancer_codes", [])}
        and str(entry.get("source_cohort") or "") == source_cohort
    ]
    if len(registry_matches) > 1:
        raise ValueError(f"{code}: multiple expression sources for {source_cohort}")
    if registry_matches:
        entry = registry_matches[0]
        return _SummarySourceMetadata(
            source_cohort=source_cohort,
            source_project=_optional_text(entry.get("source_project")),
            source_version=_optional_text(entry.get("source_version")),
            processing_pipeline=_optional_text(entry.get("processing_pipeline")),
            notes=_optional_text(entry.get("notes") or entry.get("special_handling")) or "",
            tumor_origin=_optional_text(entry.get("tumor_origin")) or "primary",
            metastasis_site=_optional_text(entry.get("metastasis_site")),
            unit=_optional_text(entry.get("unit")) or "TPM",
            citation=_optional_text(entry.get("citation")),
            pipeline_stem=_optional_text(entry.get("pipeline_stem")) or "",
        )

    # Some legacy source-matrix labels predate the source registry's canonical
    # cohort ids. Its compact summary manifest remains the static-provenance fallback.
    availability = get_data("cancer-reference-expression-availability")
    matches = availability[
        availability["cancer_code"].astype(str).eq(code)
        & availability["source_cohort"].astype(str).eq(source_cohort)
    ]
    if len(matches) > 1:
        raise ValueError(f"{code}: multiple summary metadata rows for {source_cohort}")
    if matches.empty:
        return _SummarySourceMetadata(
            source_cohort=source_cohort,
            source_project=None,
            source_version=str(cohort_source_version(code)),
            processing_pipeline=None,
            notes="",
            tumor_origin="primary",
            metastasis_site=None,
        )

    row = matches.iloc[0]
    return _SummarySourceMetadata(
        source_cohort=source_cohort,
        source_project=_optional_text(row.get("source_project")),
        source_version=_optional_text(row.get("source_version")),
        processing_pipeline=_optional_text(row.get("processing_pipeline")),
        notes=_optional_text(row.get("notes")) or "",
        tumor_origin=_optional_text(row.get("tumor_origin")) or "primary",
        metastasis_site=_optional_text(row.get("metastasis_site")),
    )


def _summary_shard_name(source_cohort: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", source_cohort):
        raise ValueError(f"source cohort is not safe as a summary shard name: {source_cohort!r}")
    return f"{source_cohort}.csv.gz"


def _code_key(name: str) -> str:
    """Normalize a matrix/reference name to a case-insensitive join key."""
    n = name[5:] if name.lower().startswith("tcga_") else name
    return n.replace("_", "").lower()


def _source_key(name: str) -> str:
    """Normalize a cohort directory or registry source-cohort id to a join key —
    by GSE accession when present, else alphanumeric-only (so ``treehouse-polya-25-01``
    and ``TREEHOUSE_POLYA_25_01`` match, and ``gse75885-sarc`` matches its registry
    id ``GSE75885_DELESPAUL_2017`` on the shared GSE accession)."""
    m = re.search(r"GSE\d+", name.upper())
    return m.group() if m else re.sub(r"[^A-Z0-9]", "", name.upper())


def _parquet_sample_count(path: Path) -> int:
    """Fast sample-column count from parquet metadata."""
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    return len([c for c in names if c not in _BASE])


def discover(cache: Path, cancer_codes: list[str]) -> dict[str, list[tuple[str, Path]]]:
    """Map each reference cancer code to its candidate ``(cohort_dir, matrix path)``."""
    code_by_key = {_code_key(code): code for code in cancer_codes}
    by_code: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    unmatched = []
    for m in cache.glob("*/derived/*_per_sample_tpm.parquet"):
        cohort_dir = m.parent.parent.name
        stem = m.name.replace("_per_sample_tpm.parquet", "")
        code = code_by_key.get(_code_key(stem))
        if code is None:
            unmatched.append(stem)
            continue
        by_code[code].append((cohort_dir, m))
    if unmatched:
        print(f"  note: {len(unmatched)} matrices matched no reference code: {unmatched}")
    return dict(by_code)


def _select_source(
    code: str,
    candidates: list[tuple[str, Path]],
    code_to_source: dict,
    code_to_n_samples: dict | None = None,
) -> Path:
    """Pick the single source matrix for a code — never pool.

    pirlygenes selects exactly one source cohort per code (RNA-seq over microarray
    proxy, then a primary-tumor source, then most samples); oncoref's shipped
    ``source-matrices.csv`` already records that choice as ``code -> source_cohort``.
    So with a single candidate we use it; with several we keep the one whose cohort
    directory matches the registry's source_cohort. This replaces the old concat-pool
    (which over-counted multi-source codes) — single-source codes were always a no-op."""
    if len(candidates) == 1:
        return candidates[0][1]
    src = code_to_source.get(code)
    if src is not None:
        want = _source_key(src)
        hits = [p for d, p in candidates if _source_key(d) == want]
        if len(hits) == 1:
            return hits[0]
        # Some source registry rows name an aggregate release with a year suffix
        # (e.g. GEO_HEME_2022), while the staged cache directory uses the family
        # label (geo-heme). Prefer that explicit aggregate directory over
        # duplicate per-study cache aliases when it is unique.
        want_without_trailing_digits = want.rstrip("0123456789")
        if want_without_trailing_digits and want_without_trailing_digits != want:
            hits = [p for d, p in candidates if _source_key(d) == want_without_trailing_digits]
            if len(hits) == 1:
                return hits[0]
    if code_to_n_samples:
        want_n = code_to_n_samples.get(code)
        if want_n is not None:
            count_hits = []
            for _d, p in candidates:
                if _parquet_sample_count(p) == want_n:
                    count_hits.append(p)
            if len(count_hits) == 1:
                return count_hits[0]
    # Fall back to the most-sampled source so a registry miss still picks one source,
    # never a pool. (Not expected for the shipped registry.)
    print(
        f"  warn: {code} has {len(candidates)} sources, no unique registry match; "
        f"using the largest",
        flush=True,
    )
    return max(candidates, key=lambda c: pd.read_parquet(c[1]).shape[1])[1]


def read_raw(path: Path) -> pd.DataFrame:
    """Canonical raw per-sample matrix for one source (genes x samples + ids).

    Canonicalize the raw matrix first (sum alt-haplotype/patch copies in LINEAR TPM,
    relabel retired ids), exactly as the runtime `_load_per_sample_matrix` does, so the
    shipped percentile/representative shards are natively dense in the canonical gene-ID
    space and match the on-the-fly recompute path (oncoref#135 item 6)."""
    raw = pd.read_parquet(path)
    return _canonicalize_gene_rows(raw, sample_cols=sample_columns(raw)).reset_index(drop=True)


def _clip_negative_expression(df: pd.DataFrame, sample_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """TPM-like source matrices should be nonnegative; clip invalid negatives to zero."""
    if not sample_cols:
        return df, 0
    values = df[sample_cols]
    negative = values < 0
    n_negative = int(negative.to_numpy().sum())
    if not n_negative:
        return df, 0
    out = df.copy()
    out.loc[:, sample_cols] = values.clip(lower=0)
    return out, n_negative


def build_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean-TPM matrix for one source's per-sample matrix (genes x samples + ids)."""
    samples = [c for c in raw.columns if c not in _BASE]
    gene_table = raw[_BASE]
    clean = clean_tpm(raw[samples], gene_table=gene_table)
    return pd.concat([gene_table.reset_index(drop=True), clean.reset_index(drop=True)], axis=1)


def _drop_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the clean-TPM censored (technical + ribosomal) genes, so the percentile /
    within-sample artifacts describe the biological view pirlygenes ships. clean_tpm
    has already deflated these into the technical compartment; dropping the rows
    doesn't change any biological gene's percentile (they're per-row independent)."""
    censored = clean_tpm_censored_gene_ids()
    unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    return df[~unversioned.isin(censored)].reset_index(drop=True)


def _sample_selection_for_qc(
    samples: list[str], qc: pd.DataFrame, sample_qc: str
) -> tuple[list[str], str, str]:
    mode = _validate_sample_qc(sample_qc)
    if mode == "all":
        return samples, "all", ""
    if qc.empty:
        return [], mode, ""
    status = dict(zip(qc["sample_id"].astype(str), qc["sample_qc_status"].astype(str)))
    if mode == "pass":
        allowed = {"pass"}
    else:
        allowed = {"pass", "warn"}
    selected = [s for s in samples if status.get(str(s)) in allowed]
    if selected or mode != "pass":
        return selected, mode, ""

    # Source-aware escape hatch: microarray/TPM-proxy sources are flagged warn
    # because their absolute scale is not linear RNA-seq TPM, but dropping the
    # entire cohort would erase a curated reference. Keep warn samples only when
    # every source sample is warn for the explicit proxy-scale reason.
    reasons = set()
    if "sample_qc_reasons" in qc.columns:
        for value in qc["sample_qc_reasons"].fillna("").astype(str):
            reasons.update(part for part in value.split(";") if part)
    scale_classes = set(qc.get("source_scale_class", pd.Series(dtype=str)).fillna("").astype(str))
    statuses = set(qc["sample_qc_status"].astype(str))
    if statuses <= {"warn"} and (
        "nonlinear_or_proxy_expression_scale" in reasons or "microarray_tpm_proxy" in scale_classes
    ):
        warn_samples = [s for s in samples if status.get(str(s)) == "warn"]
        return warn_samples, "pass_or_warn", "no_pass_samples_tpm_proxy_source"
    concentration_only = {"high_top_gene_fraction", "high_top10_gene_fraction"}
    if statuses <= {"fail"} and reasons and reasons <= concentration_only:
        fail_samples = [s for s in samples if status.get(str(s)) == "fail"]
        return fail_samples, "all", "no_pass_samples_high_concentration_source"
    return [], mode, ""


def _qc_counts(qc: pd.DataFrame) -> dict[str, int]:
    counts = qc["sample_qc_status"].value_counts().to_dict() if not qc.empty else {}
    return {f"n_qc_{name}": int(counts.get(name, 0)) for name in ("pass", "warn", "fail")}


def rebuild(
    cache: Path,
    ref: Path,
    out: Path,
    *,
    limit: int | None,
    validate: bool,
    sample_qc: str = "pass",
) -> None:
    sample_qc = _validate_sample_qc(sample_qc)
    reg = source_registry()
    by_code = discover(cache, reg["cancer_code"].astype(str).tolist())
    code_to_source = dict(zip(reg["cancer_code"].astype(str), reg["source_cohort"].astype(str)))
    code_to_n_samples = (
        dict(zip(reg["cancer_code"].astype(str), reg["n_samples"].astype(int)))
        if "n_samples" in reg.columns
        else {}
    )
    codes = sorted(by_code)
    if limit:
        codes = codes[:limit]
    print(f"rebuilding {len(codes)} cohorts -> {out}", flush=True)

    clean_dir = out / "clean"
    pct_dir = out / _PCT_DS.gene_dir
    pct_pf_dir = out / _PCT_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE)
    rep_dir = out / SHARD_DATASETS["representatives"].gene_dir
    summary_dir = out / _SUMMARY_DIR
    ws_dir = out / _WS_DS.gene_dir
    ws_pf_dir = out / _WS_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE)
    for d in (clean_dir, pct_dir, pct_pf_dir, rep_dir, summary_dir, ws_dir, ws_pf_dir):
        d.mkdir(parents=True, exist_ok=True)

    provenance: list[dict] = []
    adjudications = _representative_adjudications()
    qc_manifest: list[pd.DataFrame] = []
    build_rows: list[dict] = []
    summary_frames: list[pd.DataFrame] = []
    corrs: list[float] = []
    for code in codes:
        source_path = _select_source(code, by_code[code], code_to_source, code_to_n_samples)
        raw_df = read_raw(source_path)
        source_samples = [c for c in raw_df.columns if c not in _BASE]
        raw_df, n_negative_values_clipped = _clip_negative_expression(raw_df, source_samples)
        qc = sample_expression_qc_from_matrix(raw_df, cancer_type=code)
        if not qc.empty:
            qc = qc.assign(source_matrix_path=str(source_path))
            qc_manifest.append(qc)
        samples, effective_sample_qc, sample_qc_fallback_reason = _sample_selection_for_qc(
            source_samples, qc, sample_qc
        )
        if not samples:
            raise ValueError(f"{code}: sample_qc={sample_qc!r} leaves no source samples")

        all_sample_clean_df = build_clean(raw_df)
        clean_df = all_sample_clean_df[[*_BASE, *samples]].copy()
        clean_df.to_parquet(clean_dir / f"{code}.parquet", index=False, compression="zstd")

        source_cohort = code_to_source.get(code, code)
        source_metadata = _summary_source_metadata(code, source_cohort)
        summary_frames.append(
            summarize_source_matrix(
                raw_df,
                cancer_code=code,
                source=source_metadata,
                clean_matrix=all_sample_clean_df,
            )
        )

        # Biological view (technical genes dropped) for the percentile, within-sample,
        # and representative-selection geometry, matching the shared contract:
        # choose representatives on biological signal but store full clean TPM vectors.
        bio_df = _drop_technical(clean_df)
        pct = cohort_percentile_vectors(bio_df, samples)
        pct.to_parquet(pct_dir / f"{code}.parquet", index=False, compression="zstd")

        # Representatives keep the full gene set (real per-sample vectors), while
        # the medoid/farthest-first distances are computed on biological genes only.
        reps = cohort_medoids(clean_df, sample_cols=samples, k=5, selection_df=bio_df)
        rep_cols = [c for c in reps.columns if c not in _BASE]
        rep_ids = [f"{code}__rep{i}" for i in range(1, len(rep_cols) + 1)]
        reps = reps.rename(columns=dict(zip(rep_cols, rep_ids)))
        reps.to_parquet(rep_dir / f"{code}.parquet", index=False, compression="zstd")
        source_version = cohort_source_version(code)
        qc_counts = _qc_counts(qc)
        sample_qc_by_id = dict(
            zip(
                qc.get("sample_id", pd.Series(dtype=str)).astype(str),
                qc.get("sample_qc_status", pd.Series(dtype=str)).astype(str),
            )
        )
        qc_by_id = (
            qc.set_index(qc["sample_id"].astype(str), drop=False).to_dict("index")
            if not qc.empty and "sample_id" in qc
            else {}
        )
        build_row = {
            "cancer_code": code,
            "source_cohort": source_cohort,
            "build_source_cohort": source_cohort,
            "source_version": source_version,
            "source_matrix_path": str(source_path),
            "sample_qc": sample_qc,
            "sample_qc_effective": effective_sample_qc,
            "sample_qc_fallback_reason": sample_qc_fallback_reason,
            "sample_qc_policy_version": SAMPLE_EXPRESSION_QC_POLICY_VERSION,
            "n_source_samples": len(source_samples),
            "n_cohort_samples": len(samples),
            "n_negative_values_clipped": n_negative_values_clipped,
            **qc_counts,
        }
        build_rows.append(build_row)
        for rep_id, source_sample in zip(rep_ids, rep_cols):
            sample_namespace = source_sample_namespace(source_cohort)
            source_group_id = f"{sample_namespace}:{source_sample}"
            source_sample_qc = sample_qc_by_id.get(str(source_sample), effective_sample_qc)
            source_qc = qc_by_id.get(str(source_sample), {})
            source_scale_value = source_qc.get("source_scale_class")
            source_scale_class = (
                str(source_scale_value) if pd.notna(source_scale_value) else "unknown"
            )
            linear_value = source_qc.get("linear_tpm_comparable")
            linear_tpm_comparable = bool(linear_value) if pd.notna(linear_value) else False
            floor_value = source_qc.get("recommended_for_absolute_tpm_floor")
            recommended_for_absolute_tpm_floor = (
                bool(floor_value) if pd.notna(floor_value) else False
            )
            reasons_value = source_qc.get("sample_qc_reasons")
            source_sample_qc_reasons = str(reasons_value) if pd.notna(reasons_value) else ""
            benchmark_eligible = bool(
                effective_sample_qc in {"pass", "pass_or_warn"}
                and source_sample_qc in {"pass", "warn"}
            )
            reviewed_fields = _representative_provenance_fields(
                source_group_id=source_group_id,
                default_source_project=source_metadata.source_project,
                default_benchmark_eligible=benchmark_eligible,
                adjudications=adjudications,
            )
            provenance.append(
                {
                    "cancer_code": code,
                    "representative_id": rep_id,
                    "source_cohort": source_cohort,
                    "source_version": source_version,  # harmonized Ensembl release
                    "source_matrix_path": str(source_path),
                    "source_sample": source_sample,
                    "source_group_id": source_group_id,
                    # Row-level sample_qc is the selected source sample's actual
                    # status. Preserve the requested and effective artifact policies
                    # separately so an all-fail fallback can never look like a pass.
                    "sample_qc": source_sample_qc,
                    "sample_qc_requested": sample_qc,
                    "source_sample_qc": source_sample_qc,
                    "sample_qc_effective": effective_sample_qc,
                    "sample_qc_fallback_reason": sample_qc_fallback_reason,
                    "sample_qc_policy_version": SAMPLE_EXPRESSION_QC_POLICY_VERSION,
                    "source_sample_qc_reasons": source_sample_qc_reasons,
                    "n_source_samples": len(source_samples),
                    "n_cohort_samples": len(samples),
                    "n_negative_values_clipped": n_negative_values_clipped,
                    "source_scale_class": source_scale_class,
                    "linear_tpm_comparable": linear_tpm_comparable,
                    "recommended_for_absolute_tpm_floor": (recommended_for_absolute_tpm_floor),
                    "selection_scale_class": source_scale_class,
                    **reviewed_fields,
                    **qc_counts,
                }
            )

        ws = within_sample_top_fractions(bio_df, samples)
        ws.to_parquet(ws_dir / f"{code}.parquet", index=False, compression="zstd")

        # Proteoform key space: collapse identical-protein members per sample, then
        # build the same percentile + within-sample summaries on the reduced space so
        # every downstream read can compare/quantify/plot on one collapsed key space.
        from oncoref.proteoforms import collapse_to_proteoforms

        bio_pf = collapse_to_proteoforms(bio_df, scope=_PROTEOFORM_SCOPE, sample_cols=samples)
        pf_samples = sample_columns(bio_pf)
        cohort_percentile_vectors(bio_pf, pf_samples).to_parquet(
            pct_pf_dir / f"{code}.parquet", index=False, compression="zstd"
        )
        within_sample_top_fractions(bio_pf, pf_samples).to_parquet(
            ws_pf_dir / f"{code}.parquet", index=False, compression="zstd"
        )

        msg = f"  {code}: {len(samples)} samples, {len(pct)} genes"
        if validate:
            corr = _validate_one(pct, ref / f"{code}.parquet")
            if corr is not None:
                corrs.append(corr)
                msg += f"  p95-corr={corr:.4f}"
        if len(samples) != len(source_samples) or sample_qc_fallback_reason:
            msg += f"  sample_qc={effective_sample_qc}:{len(samples)}/{len(source_samples)}"
            if sample_qc_fallback_reason:
                msg += f" ({sample_qc_fallback_reason})"
        if n_negative_values_clipped:
            msg += f"  clipped_negative_values={n_negative_values_clipped}"
        print(msg, flush=True)

    if summary_frames:
        summary_rows = pd.concat(summary_frames, ignore_index=True)
        for source_cohort, rows in summary_rows.groupby("source_cohort", sort=True, dropna=False):
            source_name = str(source_cohort)
            rows.to_csv(
                summary_dir / _summary_shard_name(source_name),
                index=False,
                compression="gzip",
            )

    provenance_df = assign_representative_partitions(pd.DataFrame(provenance))
    provenance_df.to_csv(rep_dir / "_provenance.csv", index=False)
    if qc_manifest:
        pd.concat(qc_manifest, ignore_index=True).to_csv(
            out / "source-matrix-sample-qc.csv", index=False
        )
    partition_by_cohort = representative_partition_cohort_metadata(provenance_df)
    build_df = pd.DataFrame(build_rows).merge(
        partition_by_cohort,
        on="cancer_code",
        how="left",
        validate="one_to_one",
    )
    build_df.to_csv(out / "expression-artifact-build-metadata.csv", index=False)
    metadata = {
        "artifact": "expression-derived-shards",
        "schema_version": EXPRESSION_ARTIFACT_BUILD_METADATA_SCHEMA_VERSION,
        "sample_qc": sample_qc,
        "sample_qc_policy_version": SAMPLE_EXPRESSION_QC_POLICY_VERSION,
        "sample_qc_fallbacks": int(
            sum(1 for row in build_rows if row.get("sample_qc_fallback_reason"))
        ),
        "sample_qc_manifest": "source-matrix-sample-qc.csv",
        "cohort_metadata": "expression-artifact-build-metadata.csv",
        "n_cohorts": len(build_rows),
        "n_source_samples": int(sum(row["n_source_samples"] for row in build_rows)),
        "n_cohort_samples": int(sum(row["n_cohort_samples"] for row in build_rows)),
        "n_negative_values_clipped": int(
            sum(row["n_negative_values_clipped"] for row in build_rows)
        ),
        "representative_partition": representative_partition_build_metadata(provenance_df),
        "derived_artifacts": [
            "clean",
            _SUMMARY_DIR,
            _PCT_DS.gene_dir,
            _PCT_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE),
            SHARD_DATASETS["representatives"].gene_dir,
            _WS_DS.gene_dir,
            _WS_DS.subdir(proteoform=True, scope=_PROTEOFORM_SCOPE),
        ],
    }
    (out / "expression-artifact-build-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    if validate and corrs:
        # nan-robust: a cohort whose reference vector is constant/empty yields a
        # nan correlation that must not poison the summary.
        arr = np.array(corrs)
        finite = arr[~np.isnan(arr)]
        nan_n = len(arr) - len(finite)
        ge99 = int((finite >= 0.99).sum())
        worst = sorted(zip(codes, corrs), key=lambda t: (np.isnan(t[1]), t[1]))[:8]
        print(
            f"\nvalidation: {len(finite)} cohorts vs reference ({nan_n} undefined)  "
            f"p95-corr median={np.median(finite):.4f} mean={finite.mean():.4f} "
            f"min={finite.min():.4f}  (>={0.99}: {ge99}/{len(finite)})",
            flush=True,
        )
        print("  lowest agreement: " + ", ".join(f"{c}={v:.3f}" for c, v in worst), flush=True)
    print(f"\ndone -> {out}", flush=True)


def _validate_one(pct: pd.DataFrame, ref_path: Path) -> float | None:
    """Pearson correlation of rebuilt vs reference p95 (in TPM space)."""
    if not ref_path.exists():
        return None
    ref = pd.read_parquet(ref_path).set_index("Ensembl_Gene_ID")
    mine = pct.set_index("Ensembl_Gene_ID")
    common = mine.index.intersection(ref.index)
    if len(common) < 100 or "p95" not in ref.columns:
        return None
    a = np.expm1(mine.loc[common, "p95"].astype("float32").to_numpy())
    b = np.expm1(ref.loc[common, "p95"].astype("float32").to_numpy())
    mask = (a > 0) | (b > 0)
    if mask.sum() < 100:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path, help="Per-sample matrix cache root")
    p.add_argument(
        "--ref", required=True, type=Path, help="Reference percentile dir (validation only)"
    )
    p.add_argument("--out", required=True, type=Path, help="Staging output dir (not oncoref/data)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N codes (a test run)")
    p.add_argument("--validate", action="store_true", help="Correlate vs the reference artifacts")
    p.add_argument(
        "--sample-qc",
        choices=("pass", "pass_or_warn", "all"),
        default="pass",
        help="Which source-matrix samples feed rebuilt artifacts (default: pass)",
    )
    args = p.parse_args(argv)
    rebuild(
        args.cache.expanduser(),
        args.ref.expanduser(),
        args.out.expanduser(),
        limit=args.limit,
        validate=args.validate,
        sample_qc=args.sample_qc,
    )


if __name__ == "__main__":
    main()
