#!/usr/bin/env python3
"""Import the OpenPBTA DIPG audit manifest into molecular provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCE_ID = "openpbta-v23-dipg-h3k27"
H3_SOURCE_DIAGNOSIS = "Diffuse midline glioma, H3 K28-mutant"


def molecular_provenance(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return public molecular-provenance rows for the 51-row DIPG audit."""
    required = {
        "sample_id",
        "case_id",
        "source_record",
        "source_cohort",
        "primary_diagnosis",
        "anatomic_site",
        "age_at_diagnosis",
        "integrated_diagnosis",
        "RNA_library",
        "included",
        "exclusion_reason",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"DIPG sample manifest lacks columns: {missing}")

    rows = []
    for row in manifest.itertuples(index=False):
        integrated = (
            "" if pd.isna(row.integrated_diagnosis) else str(row.integrated_diagnosis).strip()
        )
        harmonized = "" if pd.isna(row.primary_diagnosis) else str(row.primary_diagnosis).strip()
        h3_altered = integrated == H3_SOURCE_DIAGNOSIS and harmonized == H3_SOURCE_DIAGNOSIS
        if h3_altered:
            molecular_status = "h3_k27_altered"
            driver_event = "H3 K27 alteration"
            driver_class = "histone H3 K27-altered"
        elif "H3 wildtype" in integrated:
            molecular_status = "h3_wild_type"
            driver_event = driver_class = ""
        elif "IDH-mutant" in integrated:
            molecular_status = "idh_mutant_not_h3_k27_altered"
            driver_event = driver_class = ""
        else:
            molecular_status = "not_molecularly_classified"
            driver_event = driver_class = ""

        included = str(row.included).strip().lower() in {"1", "true"}
        evidence = (
            "OpenPBTA integrated and harmonized diagnoses are both H3 K28-mutant "
            "(the source label for the canonical H3 K27-altered entity)."
            if h3_altered
            else f"OpenPBTA integrated diagnosis={integrated or 'not available'}."
        )
        disposition = (
            "Initial solid-tumor direct reference-only profile."
            if included
            else f"Excluded from the reference ({row.exclusion_reason})."
        )
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "donor_id": str(row.case_id),
                "library_id": str(row.source_record),
                "cancer_code": "DIPG",
                "source_id": SOURCE_ID,
                "source_cohort": str(row.source_cohort),
                "diagnosis_label": harmonized,
                "anatomic_site": str(row.anatomic_site),
                "age_at_diagnosis": str(row.age_at_diagnosis),
                "driver_event": driver_event,
                "driver_class": driver_class,
                "molecular_status": molecular_status,
                "assay": f"{row.RNA_library} bulk RNA-seq; RSEM gene-level TPM",
                "evidence_source": (
                    "OpenPBTA release-v23-20230115 pbta-histologies.tsv; PMID:37492101"
                ),
                "access_level": "public",
                "expression_available": included,
                "orthogonal_confirmed": h3_altered,
                "notes": f"{evidence} {disposition}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("oncoref/data/cancer-reference-sample-molecular-provenance.csv"),
    )
    args = parser.parse_args()

    existing = pd.read_csv(args.provenance, dtype=str, keep_default_na=False)
    update = molecular_provenance(pd.read_csv(args.manifest, keep_default_na=False))
    missing = sorted(set(existing.columns) - set(update.columns))
    if missing:
        raise ValueError(f"generated provenance lacks columns: {missing}")
    retained = existing.loc[~existing["source_id"].eq(SOURCE_ID)]
    combined = pd.concat([retained, update[existing.columns]], ignore_index=True)
    combined.to_csv(args.provenance, index=False, lineterminator="\n")
    print(f"wrote {len(update)} DIPG rows ({len(combined)} total) to {args.provenance}")


if __name__ == "__main__":
    main()
