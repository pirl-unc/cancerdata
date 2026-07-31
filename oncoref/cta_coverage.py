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

"""CTA antigen-coverage helpers over per-sample expression matrices."""

from __future__ import annotations

from .coverage import (
    DEFAULT_EXPRESSED_TPM,
    DEFAULT_WITHIN_SAMPLE_PERCENTILES,
    addressable_fraction,
    addressable_fraction_by_cohort,
    cta_patient_fractions,
    greedy_coverage,
    mean_antigens_per_patient,
    mean_antigens_per_patient_by_cohort,
    within_sample_percentile_addressable_fraction_by_cohort,
    within_sample_percentile_coverage_sweep,
)


def cta_addressable_fraction(
    cancer_type, *, threshold_tpm: float = DEFAULT_EXPRESSED_TPM, proteoform: bool = True
) -> float:
    """Fraction of patients expressing at least one CTA above ``threshold_tpm``."""
    return addressable_fraction(cancer_type, threshold_tpm=threshold_tpm, proteoform=proteoform)


def cta_greedy_coverage(
    cancer_type,
    *,
    threshold_tpm: float = DEFAULT_EXPRESSED_TPM,
    max_genes: int | None = None,
    proteoform: bool = True,
):
    """Greedy CTA set-cover order for a cohort."""
    return greedy_coverage(
        cancer_type, threshold_tpm=threshold_tpm, max_genes=max_genes, proteoform=proteoform
    )


def cta_mean_antigens_per_patient(
    cancer_type, *, threshold_tpm: float = DEFAULT_EXPRESSED_TPM, proteoform: bool = True
) -> float:
    """Mean number of CTAs expressed per patient in a cohort."""
    return mean_antigens_per_patient(
        cancer_type, threshold_tpm=threshold_tpm, proteoform=proteoform
    )


def cta_addressable_fraction_by_cohort(
    cohorts=None, *, threshold_tpm: float = DEFAULT_EXPRESSED_TPM, proteoform: bool = True
):
    """CTA addressable fraction for each cached or requested cohort."""
    return addressable_fraction_by_cohort(
        cohorts, threshold_tpm=threshold_tpm, proteoform=proteoform
    )


def cta_mean_antigens_per_patient_by_cohort(
    cohorts=None, *, threshold_tpm: float = DEFAULT_EXPRESSED_TPM, proteoform: bool = True
):
    """Mean number of expressed CTAs per patient for each cached or requested cohort."""
    return mean_antigens_per_patient_by_cohort(
        cohorts, threshold_tpm=threshold_tpm, proteoform=proteoform
    )


def cta_within_sample_percentile_coverage(
    cohorts=None,
    *,
    percentiles=DEFAULT_WITHIN_SAMPLE_PERCENTILES,
    proteoform: bool = True,
    sample_qc: str = "pass",
    on_missing: str = "raise",
):
    """Greedy CTA patient coverage at tumor-specific p90/p95 rank cutoffs."""
    return within_sample_percentile_coverage_sweep(
        cohorts,
        percentiles=percentiles,
        proteoform=proteoform,
        sample_qc=sample_qc,
        on_missing=on_missing,
    )


def cta_within_sample_percentile_addressable_fraction_by_cohort(
    cohorts=None,
    *,
    percentile: float = 0.95,
    proteoform: bool = True,
    sample_qc: str = "pass",
    coverage=None,
    on_missing: str = "raise",
):
    """CTA-positive patient union per cohort at a within-tumor rank cutoff."""
    return within_sample_percentile_addressable_fraction_by_cohort(
        cohorts,
        percentile=percentile,
        proteoform=proteoform,
        sample_qc=sample_qc,
        coverage=coverage,
        on_missing=on_missing,
    )


__all__ = [
    "DEFAULT_EXPRESSED_TPM",
    "DEFAULT_WITHIN_SAMPLE_PERCENTILES",
    "cta_addressable_fraction",
    "cta_addressable_fraction_by_cohort",
    "cta_greedy_coverage",
    "cta_mean_antigens_per_patient",
    "cta_mean_antigens_per_patient_by_cohort",
    "cta_patient_fractions",
    "cta_within_sample_percentile_addressable_fraction_by_cohort",
    "cta_within_sample_percentile_coverage",
]
