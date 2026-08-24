#!/usr/bin/env python3
"""Build the PRJNA994918 VSCC sample and molecular-provenance manifests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oncoref.expression_builders import (
    SraNcbiCountSource,
    sra_ncbi_count_source_from_registry,
    sra_ncbi_count_url,
)

SOURCE_ID = "prjna994918-vscc"
SOURCE_COHORT = "SRP449588_VSCC_2024"
PIPELINE = (
    "ncbi_sra_gene_feature_counts_refseq2025_gene_length_tpm_oncoref_canonical_clean_tpm_16_9_75"
)
EVIDENCE_SOURCE = "Van Arsdale et al. 2024 PMID:38898085 Tables 1-2; PRJNA994918"


@dataclass(frozen=True)
class VsccTumor:
    """One pathology-confirmed VSCC in the publication audit."""

    number: int
    age_years: int
    disease_status: str
    prior_therapy: bool
    hpv_types: tuple[str, ...] = ()
    integration_site: str = ""
    run_accession: str = ""
    counts_bytes: int = 0
    metastatic_site_biopsy: bool = False

    @property
    def tumor_id(self) -> str:
        return f"Tumor {self.number}"

    @property
    def expression_available(self) -> bool:
        return bool(self.run_accession)


VSCC_TUMORS: tuple[VsccTumor, ...] = (
    VsccTumor(1, 66, "Primary", False, run_accession="SRR25281082", counts_bytes=704619),
    VsccTumor(
        2,
        65,
        "Primary",
        False,
        ("HPV16",),
        "NCKAP1 intron 1",
        "SRR25281081",
        723521,
        True,
    ),
    VsccTumor(3, 74, "Primary", True, run_accession="SRR25281080", counts_bytes=708640),
    VsccTumor(
        4,
        45,
        "Primary",
        False,
        ("HPV16",),
        "C5orf67 intron 2",
        "SRR25281079",
        714856,
    ),
    VsccTumor(
        5,
        43,
        "Recurrence",
        True,
        ("HPV16",),
        "LRP1B introns 8-11",
        "SRR25281078",
        704033,
    ),
    VsccTumor(6, 64, "Primary", False, run_accession="SRR25281077", counts_bytes=706354),
    VsccTumor(7, 83, "Recurrence", True, run_accession="SRR25281076", counts_bytes=687999),
    VsccTumor(8, 81, "Recurrence", False, run_accession="SRR25281074", counts_bytes=713097),
    VsccTumor(9, 75, "Primary", False, run_accession="SRR25281073", counts_bytes=719549),
    VsccTumor(10, 85, "Recurrence", False, ("HPV6", "HPV16")),
    VsccTumor(11, 61, "Primary", False),
    VsccTumor(12, 77, "Recurrence", False),
    VsccTumor(13, 53, "Recurrence", False, ("HPV53", "HPV62")),
)


def vscc_clinical_audit() -> pd.DataFrame:
    """The 13-tumor clinical/HPV audit transcribed from publication Tables 1-2."""
    return pd.DataFrame(
        [
            {
                "tumor_id": tumor.tumor_id,
                "age_years": tumor.age_years,
                "disease_status": tumor.disease_status,
                "prior_therapy": tumor.prior_therapy,
                "metastatic_site_biopsy": tumor.metastatic_site_biopsy,
                "hpv_types": ";".join(tumor.hpv_types),
                "integration_site": tumor.integration_site,
                "run_accession": tumor.run_accession,
                "counts_bytes": tumor.counts_bytes,
                "expression_available": tumor.expression_available,
            }
            for tumor in VSCC_TUMORS
        ]
    )


def reference_sample_manifest(source: SraNcbiCountSource | None = None) -> pd.DataFrame:
    """Return the nine included RNA libraries in the public sample-manifest schema."""
    source = source or sra_ncbi_count_source_from_registry(SOURCE_ID)
    runs = {run.accession: run for run in source.runs}
    rows = []
    for tumor in VSCC_TUMORS:
        if not tumor.expression_available:
            continue
        run = runs[tumor.run_accession]
        sample_type = (
            "Metastatic-site biopsy"
            if tumor.metastatic_site_biopsy
            else f"{tumor.disease_status} Tumor"
        )
        rows.append(
            {
                "cancer_code": "VSCC",
                "source_cohort": source.source_cohort,
                "source_project": source.source_project,
                "case_id": run.biosample,
                "sample_id": run.accession,
                "source_file_id": run.analysis_accession,
                "source_file_name": f"{run.accession}.ncbi-gene-counts.tsv",
                "source_project_id": source.bioproject,
                "sample_type": sample_type,
                "primary_diagnosis": "Invasive vulvar squamous cell carcinoma",
                "md5sum": run.counts_md5,
                "file_size": tumor.counts_bytes,
                "workflow_type": "NCBI SRA Gene Feature RNA-seq counts",
                "raw_unit": "raw counts",
                "processing_pipeline": source.processing_pipeline or PIPELINE,
                "source_url": sra_ncbi_count_url(source, run),
                "lineage_evidence_source": (
                    "PMID:38898085 Table 1 squamous histology plus PRJNA994918 run-to-tumor mapping"
                ),
                "included": True,
                "exclusion_reason": "",
                "lineage_label": "VSCC",
            }
        )
    return pd.DataFrame(rows)


def molecular_provenance() -> pd.DataFrame:
    """Return direct HPV evidence for all 13 pathology-confirmed study tumors."""
    run_to_source = {
        run.accession: run for run in sra_ncbi_count_source_from_registry(SOURCE_ID).runs
    }
    rows = []
    for tumor in VSCC_TUMORS:
        run = run_to_source.get(tumor.run_accession)
        integrated = bool(tumor.integration_site)
        hpv_positive = bool(tumor.hpv_types)
        if integrated:
            driver_event = f"HPV16 integration in {tumor.integration_site}"
            driver_class = "viral_integration"
            molecular_status = "hpv16_integrated"
            assay = "HPV hybridization capture sequencing; junction PCR"
            orthogonal_confirmed = True
        elif hpv_positive:
            driver_event = ""
            driver_class = ""
            molecular_status = "hpv_coinfection_no_integration"
            assay = "HPV hybridization capture sequencing; type-specific PCR"
            orthogonal_confirmed = True
        else:
            driver_event = ""
            driver_class = ""
            molecular_status = "hpv_dna_negative"
            assay = "HPV hybridization capture sequencing"
            orthogonal_confirmed = False

        site = "metastatic site (site not reported)" if tumor.metastatic_site_biopsy else "vulva"
        prior = "yes" if tumor.prior_therapy else "no"
        expression_note = (
            f"RNA library {run.accession} is included in the direct reference."
            if run is not None
            else "No RIN-qualified RNA library was published for this tumor."
        )
        hpv_note = (
            f"Directly detected {'/'.join(tumor.hpv_types)}; "
            + (
                f"integration at {tumor.integration_site} was PCR-confirmed."
                if integrated
                else "no human-virus integration junction was detected."
            )
            if hpv_positive
            else "No HPV DNA was detected by the 143-type hybridization-capture panel."
        )
        rows.append(
            {
                "sample_id": tumor.tumor_id,
                "donor_id": tumor.tumor_id,
                "library_id": run.accession if run is not None else tumor.tumor_id,
                "cancer_code": "VSCC",
                "source_id": SOURCE_ID,
                "source_cohort": SOURCE_COHORT,
                "diagnosis_label": (
                    f"invasive vulvar squamous cell carcinoma; {tumor.disease_status.lower()}"
                ),
                "anatomic_site": site,
                "age_at_diagnosis": f"{tumor.age_years} years",
                "driver_event": driver_event,
                "driver_class": driver_class,
                "molecular_status": molecular_status,
                "assay": assay,
                "evidence_source": EVIDENCE_SOURCE,
                "access_level": "public",
                "expression_available": run is not None,
                "orthogonal_confirmed": orthogonal_confirmed,
                "notes": (
                    f"Disease status at acquisition={tumor.disease_status}; prior "
                    f"chemotherapy/radiation={prior}. {hpv_note} {expression_note}"
                ),
            }
        )
    return pd.DataFrame(rows)


def update_molecular_provenance(path: Path) -> int:
    """Replace this source's rows in the packaged molecular-provenance CSV."""
    existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    update = molecular_provenance()
    missing = sorted(set(existing.columns) - set(update.columns))
    if missing:
        raise ValueError(f"generated provenance lacks columns: {missing}")
    retained = existing.loc[~existing["source_id"].eq(SOURCE_ID)]
    combined = pd.concat([retained, update[existing.columns]], ignore_index=True)
    combined.to_csv(path, index=False, lineterminator="\n")
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-output",
        type=Path,
        required=True,
        help="Write a source-scoped sample-manifest CSV for reconcile_sample_manifest.py.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("oncoref/data/cancer-reference-sample-molecular-provenance.csv"),
    )
    args = parser.parse_args()

    samples = reference_sample_manifest()
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.sample_output, index=False, lineterminator="\n")
    total = update_molecular_provenance(args.provenance)
    print(
        f"wrote {len(samples)} VSCC sample rows to {args.sample_output} and "
        f"13 molecular-provenance rows ({total} total) to {args.provenance}"
    )


if __name__ == "__main__":
    main()
