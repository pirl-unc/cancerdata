#!/usr/bin/env python
"""Fit deterministic cohort-specific CPTAC RNA/protein calibration models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.proteoforms import (
    canonical_proteoform_ids,
    proteoform_groups,
    proteoform_symbol,
)
from oncoref.rna_protein import (
    CPTAC_CALIBRATION_COHORTS,
    CPTAC_CALIBRATION_COLUMNS,
    CPTAC_CALIBRATION_SAMPLE_COLUMNS,
    CPTAC_CALIBRATION_VERSION,
    RNA_PROTEIN_SOURCE_SCALES,
    rna_protein_calibration_sources,
)

MIN_QUANTITATIVE_PAIRS = 10
MIN_DETECTION_EVENTS = 5
LOGISTIC_RIDGE = 1.0
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_TOLERANCE = 1e-10


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return math.nan, math.nan
    return float(finite.min()), float(finite.max())


def _detection_auc(rna: np.ndarray, detected: np.ndarray) -> float:
    n_detected = int(detected.sum())
    n_missing = int((~detected).sum())
    if not n_detected or not n_missing:
        return math.nan
    ranks = pd.Series(rna).rank(method="average").to_numpy(dtype=float)
    rank_sum = float(ranks[detected].sum())
    return (rank_sum - n_detected * (n_detected + 1) / 2) / (n_detected * n_missing)


def fit_detection_model(rna: np.ndarray, protein: np.ndarray) -> dict[str, float | str]:
    """Fit an L2-regularized logistic model for TMT observation vs source RNA."""

    valid_rna = np.isfinite(rna)
    x = rna[valid_rna]
    detected = np.isfinite(protein[valid_rna])
    n_detected = int(detected.sum())
    n_missing = int((~detected).sum())
    out: dict[str, float | str] = {
        "detection_model_status": "",
        "detection_logit_intercept": math.nan,
        "detection_logit_slope": math.nan,
        "rna_at_50pct_detection": math.nan,
        "detection_auc": _detection_auc(x, detected),
        "detection_brier_score_in_sample": math.nan,
    }
    if not len(x) or not n_detected:
        out["detection_model_status"] = "all_missing"
        return out
    if not n_missing:
        out["detection_model_status"] = "all_detected"
        return out
    if min(n_detected, n_missing) < MIN_DETECTION_EVENTS:
        out["detection_model_status"] = "insufficient_events"
        return out

    mean = float(x.mean())
    scale = float(x.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        out["detection_model_status"] = "constant_rna"
        return out
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    response = detected.astype(float)
    beta = np.array([math.log(n_detected / n_missing), 0.0], dtype=float)
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
        out["detection_model_status"] = "fit_failed"
        return out

    slope = float(beta[1] / scale)
    intercept = float(beta[0] - beta[1] * mean / scale)
    probability = 1.0 / (1.0 + np.exp(-np.clip(intercept + slope * x, -30.0, 30.0)))
    threshold = -intercept / slope if slope > 0 else math.nan
    if not np.isfinite(threshold) or threshold < x.min() or threshold > x.max():
        threshold = math.nan
    out.update(
        {
            "detection_model_status": "fit",
            "detection_logit_intercept": intercept,
            "detection_logit_slope": slope,
            "rna_at_50pct_detection": float(threshold),
            "detection_brier_score_in_sample": float(np.mean((probability - response) ** 2)),
        }
    )
    return out


def fit_quantitative_model(rna: np.ndarray, protein: np.ndarray) -> dict[str, float | str]:
    """Fit observed TMT abundance on source RNA without imputing missing protein."""

    valid = np.isfinite(rna) & np.isfinite(protein)
    x = rna[valid]
    y = protein[valid]
    out: dict[str, float | str] = {
        "quantitative_model_status": "",
        "quantitative_intercept": math.nan,
        "quantitative_slope": math.nan,
        "quantitative_slope_standard_error": math.nan,
        "pearson_r": math.nan,
        "r_squared_in_sample": math.nan,
        "rmse_in_sample": math.nan,
        "rmse_leave_one_out": math.nan,
    }
    if len(x) < MIN_QUANTITATIVE_PAIRS:
        out["quantitative_model_status"] = "insufficient_pairs"
        return out
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    sum_xx = float(x_centered @ x_centered)
    sum_yy = float(y_centered @ y_centered)
    if sum_xx <= 0:
        out["quantitative_model_status"] = "constant_rna"
        return out
    if sum_yy <= 0:
        out["quantitative_model_status"] = "constant_protein"
        return out
    slope = float((x_centered @ y_centered) / sum_xx)
    intercept = float(y.mean() - slope * x.mean())
    residuals = y - (intercept + slope * x)
    sum_squared_error = float(residuals @ residuals)
    pearson = float((x_centered @ y_centered) / math.sqrt(sum_xx * sum_yy))
    slope_se = math.sqrt((sum_squared_error / (len(x) - 2)) / sum_xx)
    leverage = 1.0 / len(x) + (x_centered**2) / sum_xx
    leave_one_out_denominator = 1.0 - leverage
    if (leave_one_out_denominator <= np.finfo(float).eps).any():
        rmse_leave_one_out = math.nan
    else:
        leave_one_out_residuals = residuals / leave_one_out_denominator
        rmse_leave_one_out = math.sqrt(
            float(leave_one_out_residuals @ leave_one_out_residuals) / len(x)
        )
    out.update(
        {
            "quantitative_model_status": "fit",
            "quantitative_intercept": intercept,
            "quantitative_slope": slope,
            "quantitative_slope_standard_error": float(slope_se),
            "pearson_r": pearson,
            "r_squared_in_sample": float(max(0.0, 1.0 - sum_squared_error / sum_yy)),
            "rmse_in_sample": float(math.sqrt(sum_squared_error / len(x))),
            "rmse_leave_one_out": float(rmse_leave_one_out),
        }
    )
    return out


def fit_cohort_calibrations(cohort_dir: Path) -> pd.DataFrame:
    """Fit every canonical gene in one standardized cohort directory."""

    metadata = json.loads((cohort_dir / "build-metadata.json").read_text())
    cohort = str(metadata["cptac_cohort"])
    cancer_code = str(metadata["cancer_code"])
    rna = pd.read_parquet(cohort_dir / "rna.parquet")
    protein = pd.read_parquet(cohort_dir / "protein.parquet")
    genes = pd.read_csv(cohort_dir / "genes.csv", dtype=str)
    if not rna.index.equals(protein.index) or not rna.columns.equals(protein.columns):
        raise ValueError(f"{cohort}: standardized RNA/protein axes differ")
    if genes["canonical_gene_id"].tolist() != rna.index.astype(str).tolist():
        raise ValueError(f"{cohort}: gene manifest does not match matrix order")

    gene_ids = genes["canonical_gene_id"].tolist()
    proteoform_map = canonical_proteoform_ids(gene_ids)
    genes["canonical_proteoform_id"] = genes["canonical_gene_id"].map(proteoform_map)
    proteoform_registry = proteoform_groups(scope="genome")
    proteoform_registry["canonical_proteoform_id"] = proteoform_registry["proteoform_id"].map(
        proteoform_symbol
    )
    proteoform_counts = proteoform_registry.groupby("canonical_proteoform_id")[
        "member_gene_id"
    ].nunique()
    genes["proteoform_member_count"] = (
        genes["canonical_proteoform_id"].map(proteoform_counts).fillna(1).astype(int)
    )
    sources = rna_protein_calibration_sources(cptac_cohort=cohort).set_index("modality")

    rows: list[dict] = []
    rna_values = rna.to_numpy(dtype=float)
    protein_values = protein.to_numpy(dtype=float)
    for index, gene in genes.iterrows():
        x = rna_values[index]
        y = protein_values[index]
        valid_rna = np.isfinite(x)
        valid_protein = valid_rna & np.isfinite(y)
        rna_min, rna_max = _finite_range(x)
        protein_min, protein_max = _finite_range(y[valid_rna])
        row = {
            "calibration_version": CPTAC_CALIBRATION_VERSION,
            "source_class": "cptac_matched_quantitative",
            "cptac_cohort": cohort,
            "cancer_code": cancer_code,
            "canonical_gene_id": gene["canonical_gene_id"],
            "canonical_proteoform_id": gene["canonical_proteoform_id"],
            "proteoform_member_count": int(gene["proteoform_member_count"]),
            "source_gene_id": gene["source_gene_id"],
            "rna_scale": RNA_PROTEIN_SOURCE_SCALES["rna"],
            "protein_scale": RNA_PROTEIN_SOURCE_SCALES["protein"],
            "n_matched_samples": len(rna.columns),
            "n_rna_observed": int(valid_rna.sum()),
            "n_protein_observed": int(valid_protein.sum()),
            "protein_detection_rate": float(valid_protein.sum() / valid_rna.sum()),
            "rna_min": rna_min,
            "rna_max": rna_max,
            "protein_min": protein_min,
            "protein_max": protein_max,
            "rna_source_id": str(sources.loc["rna", "source_id"]),
            "protein_source_id": str(sources.loc["protein", "source_id"]),
            "source_record": str(sources.loc["rna", "source_record"]),
            "model_scope": "within_cptac_cohort_no_cross_cohort_pooling",
        }
        row.update(fit_detection_model(x, y))
        row.update(fit_quantitative_model(x, y))
        rows.append(row)
    return pd.DataFrame(rows, columns=CPTAC_CALIBRATION_COLUMNS)


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


def build_sample_manifest(input_dir: Path) -> pd.DataFrame:
    """Combine the exact source-sample joins used by every cohort model."""

    frames = []
    for cohort in CPTAC_CALIBRATION_COHORTS:
        cohort_dir = input_dir / cohort.lower()
        metadata = json.loads((cohort_dir / "build-metadata.json").read_text())
        samples = pd.read_csv(cohort_dir / "samples.csv", dtype=str)
        sources = rna_protein_calibration_sources(cptac_cohort=cohort).set_index("modality")
        samples.insert(0, "cancer_code", str(metadata["cancer_code"]))
        samples.insert(0, "cptac_cohort", cohort)
        samples.insert(0, "source_class", "cptac_matched_quantitative")
        samples.insert(0, "calibration_version", CPTAC_CALIBRATION_VERSION)
        samples["rna_source_id"] = str(sources.loc["rna", "source_id"])
        samples["protein_source_id"] = str(sources.loc["protein", "source_id"])
        samples["source_record"] = str(sources.loc["rna", "source_record"])
        samples["model_role"] = "cohort_calibration_observation"
        frames.append(samples.loc[:, CPTAC_CALIBRATION_SAMPLE_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["cptac_cohort", "sample_id"]).reset_index(drop=True)


def fit_all_calibrations(input_dir: Path, output: Path, samples_output: Path) -> dict:
    """Fit, validate, sort, and write all ten cohort-specific model tables."""

    frames = []
    for cohort in CPTAC_CALIBRATION_COHORTS:
        cohort_dir = input_dir / cohort.lower()
        if not cohort_dir.is_dir():
            raise FileNotFoundError(f"missing standardized cohort directory: {cohort_dir}")
        frames.append(fit_cohort_calibrations(cohort_dir))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["cptac_cohort", "canonical_gene_id"]).reset_index(drop=True)
    if combined.duplicated(["cptac_cohort", "canonical_gene_id"]).any():
        raise ValueError("duplicate cohort/gene calibration rows")
    digest = write_deterministic_gzip_csv(combined, output)
    samples = build_sample_manifest(input_dir)
    samples_digest = write_deterministic_gzip_csv(samples, samples_output)
    return {
        "calibration_version": CPTAC_CALIBRATION_VERSION,
        "output": str(output),
        "sha256": digest,
        "samples_output": str(samples_output),
        "samples_sha256": samples_digest,
        "n_rows": len(combined),
        "n_cohorts": combined["cptac_cohort"].nunique(),
        "n_genes": combined["canonical_gene_id"].nunique(),
        "n_matched_samples": int(
            combined.groupby("cptac_cohort")["n_matched_samples"].first().sum()
        ),
        "detection_model_status_counts": combined["detection_model_status"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "quantitative_model_status_counts": combined["quantitative_model_status"]
        .value_counts()
        .sort_index()
        .to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            fit_all_calibrations(args.input_dir, args.output, args.samples_output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
