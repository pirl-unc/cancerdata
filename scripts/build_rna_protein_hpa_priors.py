#!/usr/bin/env python
"""Build checksum-pinned HPA v23 normal-tissue RNA/IHC weak priors."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.gene_ids import canonical_gene_ids, canonical_gene_symbols
from oncoref.proteoforms import (
    canonical_proteoform_ids,
    proteoform_groups,
    proteoform_symbol,
)
from oncoref.rna_protein import (
    HPA_PRIOR_COLUMNS,
    HPA_PRIOR_SOURCE_CLASS,
    HPA_PRIOR_VERSION,
    rna_protein_hpa_prior_sources,
)

MIN_MATCHED_TISSUES = 10
MIN_DETECTION_EVENTS = 5
LOGISTIC_RIDGE = 1.0
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_TOLERANCE = 1e-10
IHC_ORDINAL_LEVELS = {
    "Not detected": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}
EXPECTED_EXACT_TISSUE_LABELS = 43


def load_pinned_hpa_archives(archive_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify and parse the exact HPA archives declared by the public manifest."""

    tables = {}
    for source in rna_protein_hpa_prior_sources().to_dict(orient="records"):
        archive = archive_dir / str(source["archive_filename"])
        if not archive.is_file():
            raise FileNotFoundError(f"missing pinned HPA archive: {archive}")
        payload = archive.read_bytes()
        if len(payload) != int(source["archive_byte_size"]):
            raise ValueError(f"{archive.name}: archive byte size does not match the manifest")
        if hashlib.sha256(payload).hexdigest() != source["archive_sha256"]:
            raise ValueError(f"{archive.name}: archive SHA-256 does not match the manifest")
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            expected_member = str(source["extracted_filename"])
            if expected_member not in zipped.namelist():
                raise ValueError(f"{archive.name}: missing expected member {expected_member}")
            extracted = zipped.read(expected_member)
        if len(extracted) != int(source["extracted_byte_size"]):
            raise ValueError(f"{expected_member}: extracted byte size does not match the manifest")
        if hashlib.sha256(extracted).hexdigest() != source["extracted_sha256"]:
            raise ValueError(f"{expected_member}: extracted SHA-256 does not match the manifest")
        tables[str(source["modality"])] = pd.read_csv(io.BytesIO(extracted), sep="\t")
    return tables["rna"], tables["protein"]


def _detection_auc(rna: np.ndarray, detected: np.ndarray) -> float:
    n_detected = int(detected.sum())
    n_not_detected = int((~detected).sum())
    if not n_detected or not n_not_detected:
        return math.nan
    ranks = pd.Series(rna).rank(method="average").to_numpy(dtype=float)
    rank_sum = float(ranks[detected].sum())
    return (rank_sum - n_detected * (n_detected + 1) / 2) / (n_detected * n_not_detected)


def fit_ihc_detection_prior(rna: np.ndarray, ihc_ordinal: np.ndarray) -> dict[str, float | str]:
    """Fit a weak logistic prior for any ordinal IHC detection across tissues."""

    valid = np.isfinite(rna) & np.isfinite(ihc_ordinal)
    x = rna[valid]
    ordinal = ihc_ordinal[valid]
    detected = ordinal > 0
    n_detected = int(detected.sum())
    n_not_detected = int((~detected).sum())
    out: dict[str, float | str] = {
        "detection_prior_status": "",
        "detection_logit_intercept": math.nan,
        "detection_logit_slope": math.nan,
        "rna_at_50pct_ihc_detection": math.nan,
        "detection_auc": _detection_auc(x, detected),
        "detection_brier_score_in_sample": math.nan,
        "rna_ihc_spearman_rho": math.nan,
    }
    if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(ordinal) > 0:
        out["rna_ihc_spearman_rho"] = float(
            pd.Series(x).rank(method="average").corr(pd.Series(ordinal).rank(method="average"))
        )
    if len(x) < MIN_MATCHED_TISSUES:
        out["detection_prior_status"] = "insufficient_tissues"
        return out
    if not n_detected:
        out["detection_prior_status"] = "all_not_detected"
        return out
    if not n_not_detected:
        out["detection_prior_status"] = "all_detected"
        return out
    if min(n_detected, n_not_detected) < MIN_DETECTION_EVENTS:
        out["detection_prior_status"] = "insufficient_events"
        return out

    mean = float(x.mean())
    scale = float(x.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        out["detection_prior_status"] = "constant_rna"
        return out
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    response = detected.astype(float)
    beta = np.array([math.log(n_detected / n_not_detected), 0.0], dtype=float)
    converged = False
    for _ in range(LOGISTIC_MAX_ITERATIONS):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30.0, 30.0)))
        weights = probability * (1.0 - probability)
        hessian = design.T @ (design * weights[:, None])
        hessian[1, 1] += LOGISTIC_RIDGE
        gradient = design.T @ (response - probability)
        gradient[1] -= LOGISTIC_RIDGE * beta[1]
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta += step
        if float(np.max(np.abs(step))) < LOGISTIC_TOLERANCE:
            converged = True
            break
    if not converged or not np.isfinite(beta).all():
        out["detection_prior_status"] = "fit_failed"
        return out

    slope = float(beta[1] / scale)
    intercept = float(beta[0] - beta[1] * mean / scale)
    probability = 1.0 / (1.0 + np.exp(-np.clip(intercept + slope * x, -30.0, 30.0)))
    threshold = -intercept / slope if slope > 0 else math.nan
    if not np.isfinite(threshold) or threshold < x.min() or threshold > x.max():
        threshold = math.nan
    out.update(
        {
            "detection_prior_status": "fit",
            "detection_logit_intercept": intercept,
            "detection_logit_slope": slope,
            "rna_at_50pct_ihc_detection": float(threshold),
            "detection_brier_score_in_sample": float(np.mean((probability - response) ** 2)),
        }
    )
    return out


def canonical_hpa_tissue_pairs(
    rna: pd.DataFrame, ihc: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return exact-label RNA/IHC pairs plus per-gene IHC audit counts."""

    if tuple(rna.columns) != ("Gene", "Gene name", "Tissue", "nTPM"):
        raise ValueError("unexpected HPA RNA consensus columns")
    if tuple(ihc.columns) != (
        "Gene",
        "Gene name",
        "Tissue",
        "Cell type",
        "Level",
        "Reliability",
    ):
        raise ValueError("unexpected HPA normal-tissue IHC columns")
    if rna.duplicated(["Gene", "Tissue"]).any():
        raise ValueError("HPA RNA consensus must be unique by source gene and tissue")
    if ihc.duplicated(["Gene", "Tissue", "Cell type"]).any():
        raise ValueError("HPA IHC must be unique by source gene, tissue, and cell type")
    if pd.to_numeric(rna["nTPM"], errors="coerce").isna().any() or rna["nTPM"].lt(0).any():
        raise ValueError("HPA RNA nTPM values must be finite and nonnegative")

    exact_tissues = sorted(set(rna["Tissue"].dropna()) & set(ihc["Tissue"].dropna()))
    if len(exact_tissues) != EXPECTED_EXACT_TISSUE_LABELS:
        raise ValueError(
            f"expected {EXPECTED_EXACT_TISSUE_LABELS} exact HPA tissue labels, "
            f"found {len(exact_tissues)}"
        )
    rna = rna.loc[rna["Tissue"].isin(exact_tissues)].copy()
    ihc = ihc.loc[ihc["Tissue"].isin(exact_tissues)].copy()

    source_ids = sorted(set(rna["Gene"].astype(str)) | set(ihc["Gene"].astype(str)))
    mapped = pd.DataFrame(
        {
            "source_gene_id": source_ids,
            "canonical_gene_id": canonical_gene_ids(source_ids),
        }
    ).dropna()
    collision_counts = mapped.groupby("canonical_gene_id")["source_gene_id"].nunique()
    collided = set(collision_counts[collision_counts.gt(1)].index)
    mapped = mapped.loc[~mapped["canonical_gene_id"].isin(collided)]
    source_to_canonical = dict(zip(mapped["source_gene_id"], mapped["canonical_gene_id"]))
    rna["canonical_gene_id"] = rna["Gene"].map(source_to_canonical)
    ihc["canonical_gene_id"] = ihc["Gene"].map(source_to_canonical)
    rna = rna.dropna(subset=["canonical_gene_id"])
    ihc = ihc.dropna(subset=["canonical_gene_id"])

    reliability_counts = ihc.groupby("canonical_gene_id")["Reliability"].nunique(dropna=False)
    if reliability_counts.gt(1).any() or ihc["Reliability"].isna().any():
        raise ValueError("HPA IHC reliability must be one nonmissing category per gene")

    ihc["ihc_ordinal"] = ihc["Level"].map(IHC_ORDINAL_LEVELS)
    audit = (
        ihc.assign(nonordinal=ihc["ihc_ordinal"].isna().astype(int))
        .groupby("canonical_gene_id", sort=True)
        .agg(
            n_ihc_cell_type_observations=("Cell type", "size"),
            n_nonordinal_ihc_observations_excluded=("nonordinal", "sum"),
            ihc_reliability=("Reliability", "first"),
        )
        .reset_index()
    )
    tissue_ihc = (
        ihc.dropna(subset=["ihc_ordinal"])
        .groupby(["canonical_gene_id", "Tissue"], sort=True, as_index=False)
        .agg(ihc_ordinal=("ihc_ordinal", "max"))
    )
    rna = rna.rename(
        columns={"Gene": "source_gene_id", "Gene name": "source_gene_symbol", "nTPM": "rna_ntpm"}
    )
    pairs = rna.merge(
        tissue_ihc,
        on=["canonical_gene_id", "Tissue"],
        how="inner",
        validate="one_to_one",
    )
    pairs["rna_log2_ntpm_plus_1"] = np.log2(pairs["rna_ntpm"].astype(float) + 1.0)
    return pairs, audit, len(collided)


def build_hpa_priors(rna: pd.DataFrame, ihc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build one canonical gene-level weak prior from paired HPA tissues."""

    pairs, audit, n_canonical_collisions_excluded = canonical_hpa_tissue_pairs(rna, ihc)
    audit = audit.set_index("canonical_gene_id")
    gene_ids = sorted(pairs["canonical_gene_id"].unique())
    proteoform_map = canonical_proteoform_ids(gene_ids)
    proteoform_registry = proteoform_groups(scope="genome")
    proteoform_registry["canonical_proteoform_id"] = proteoform_registry["proteoform_id"].map(
        proteoform_symbol
    )
    proteoform_counts = proteoform_registry.groupby("canonical_proteoform_id")[
        "member_gene_id"
    ].nunique()
    canonical_symbols = dict(zip(gene_ids, canonical_gene_symbols(gene_ids)))
    sources = rna_protein_hpa_prior_sources().set_index("modality")

    rows = []
    for gene_id, gene_pairs in pairs.groupby("canonical_gene_id", sort=True):
        gene_pairs = gene_pairs.sort_values("Tissue")
        x = gene_pairs["rna_log2_ntpm_plus_1"].to_numpy(dtype=float)
        ordinal = gene_pairs["ihc_ordinal"].to_numpy(dtype=float)
        level_counts = pd.Series(ordinal.astype(int)).value_counts().to_dict()
        n_detected = int((ordinal > 0).sum())
        source_gene_ids = gene_pairs["source_gene_id"].unique()
        if len(source_gene_ids) != 1:
            raise ValueError(f"{gene_id}: multiple HPA source genes survived canonicalization")
        proteoform_id = proteoform_map[gene_id]
        row = {
            "prior_version": HPA_PRIOR_VERSION,
            "source_class": HPA_PRIOR_SOURCE_CLASS,
            "canonical_gene_id": gene_id,
            "canonical_proteoform_id": proteoform_id,
            "proteoform_member_count": int(proteoform_counts.get(proteoform_id, 1)),
            "source_gene_id": str(source_gene_ids[0]),
            "gene_symbol": canonical_symbols[gene_id],
            "ihc_reliability": str(audit.loc[gene_id, "ihc_reliability"]),
            "rna_scale": "log2_hpa_consensus_ntpm_plus_1",
            "protein_scale": "max_ordinal_ihc_across_cell_types",
            "tissue_join_policy": "exact_hpa_v23_tissue_label_only",
            "ihc_aggregation_policy": "max_high3_medium2_low1_not_detected0_exclude_nonordinal",
            "n_matched_tissues": len(gene_pairs),
            "n_ihc_cell_type_observations": int(audit.loc[gene_id, "n_ihc_cell_type_observations"]),
            "n_nonordinal_ihc_observations_excluded": int(
                audit.loc[gene_id, "n_nonordinal_ihc_observations_excluded"]
            ),
            "n_not_detected_tissues": int(level_counts.get(0, 0)),
            "n_low_tissues": int(level_counts.get(1, 0)),
            "n_medium_tissues": int(level_counts.get(2, 0)),
            "n_high_tissues": int(level_counts.get(3, 0)),
            "n_detected_tissues": n_detected,
            "ihc_detected_tissue_rate": float(n_detected / len(gene_pairs)),
            "rna_min": float(x.min()),
            "rna_max": float(x.max()),
            "rna_source_id": str(sources.loc["rna", "source_id"]),
            "protein_source_id": str(sources.loc["protein", "source_id"]),
            "source_version": "v23",
            "model_scope": "normal_tissue_weak_prior_no_patient_matching",
        }
        row.update(fit_ihc_detection_prior(x, ordinal))
        rows.append(row)
    result = (
        pd.DataFrame(rows, columns=HPA_PRIOR_COLUMNS)
        .sort_values("canonical_gene_id")
        .reset_index(drop=True)
    )
    metadata = {
        "prior_version": HPA_PRIOR_VERSION,
        "n_rows": len(result),
        "n_genes": result["canonical_gene_id"].nunique(),
        "n_exact_tissue_labels": EXPECTED_EXACT_TISSUE_LABELS,
        "n_canonical_collisions_excluded": n_canonical_collisions_excluded,
        "n_nonordinal_ihc_observations_excluded": int(
            result["n_nonordinal_ihc_observations_excluded"].sum()
        ),
        "detection_prior_status_counts": result["detection_prior_status"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "ihc_reliability_counts": result["ihc_reliability"].value_counts().sort_index().to_dict(),
    }
    return result, metadata


def write_deterministic_gzip_csv(frame: pd.DataFrame, output: Path) -> str:
    """Write a stable gzip CSV and return its SHA-256 digest."""

    csv_bytes = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.10g",
        na_rep="",
    ).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as stream:
        stream.write(csv_bytes)
    payload = buffer.getvalue()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(output)
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rna, ihc = load_pinned_hpa_archives(args.archive_dir)
    priors, metadata = build_hpa_priors(rna, ihc)
    metadata["output"] = str(args.output)
    metadata["sha256"] = write_deterministic_gzip_csv(priors, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
