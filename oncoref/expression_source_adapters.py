# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Small adapters for public expression sources with nonstandard metadata.

Each adapter is responsible only for extracting the source-specific sample
labels or file layout. Normalization, canonical gene mapping, sample QC, summary
statistics, and artifact naming remain in :mod:`oncoref.expression_builders`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .expression_builders import (
    GeoMatrixSource,
    SourceMatrixBuildResult,
    build_canonical_source_matrices,
    build_gdc_source_matrices,
    ensembl_gene_lengths_kb,
    gdc_source_from_registry,
    prepare_source_matrix,
)
from .expression_engine import (
    canonicalize_source_gene_matrix,
    coerce_source_expression_values,
    sample_columns,
)
from .expression_registry import expression_source_registry_entries

GSE75885_EXPRESSION_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75885/suppl/"
    "GSE75885_Expression_117_sarcomas.tsv.gz"
)
GSE75885_SERIES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75885/matrix/GSE75885_series_matrix.txt.gz"
)
GSE75885_LABEL_TO_CODE = {
    "Liposarcoma - dedifferentiated": "SARC_DDLPS",
    "Liposarcoma - pleomorphic": "SARC_PLEOLPS",
    "Low grade fibromyxoid sarcoma": "SARC_LGFMS",
}

DRMETRICS_COUNTS_URL = (
    "https://raw.githubusercontent.com/IARCbioinfo/DRMetrics/NextJournalH/"
    "data/read_counts_all.txt.zip"
)
DRMETRICS_ATTRIBUTES_URL = (
    "https://raw.githubusercontent.com/IARCbioinfo/DRMetrics/NextJournalH/data/Attributes.txt.zip"
)
DRMETRICS_HISTOLOGY_TO_CODE = {
    "Typical": "NET_LUNG",
    "Atypical": "NET_LUNG",
    "Carcinoid": "NET_LUNG",
    "Supra_carcinoid": "NET_LUNG",
    "LCNEC": "NEC_LUNG_LARGECELL",
}

TARGET_ALL_SAMPLE_MATRICES = {
    "TARGET_ALL_P1": {
        "file_id": "c5b8499d-8777-4ad5-bb6b-48cfeca68aed",
        "members": ("TARGET_ALL_SampleMatrix_Phase1_20190606.xlsx",),
    },
    "TARGET_ALL_P2": {
        "file_id": "313fd46b-8111-45dd-88df-4e637a4d2249",
        "members": (
            "TARGET_ALL_SampleMatrix_Phase2_Discovery_20190606.xlsx",
            "TARGET_ALL_SampleMatrix_Phase2_Validation_20190606.xlsx",
        ),
    },
}
CBIOPORTAL_NBL_MYCN_URL = (
    "https://www.cbioportal.org/api/studies/nbl_target_2018_pub/"
    "clinical-data?clinicalDataType=SAMPLE&attributeId=MYCN"
)
CTCL_RAW_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171811/suppl/GSE171811_RAW.tar"
CTCL_SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171811/soft/GSE171811_family.soft.gz"
)
CTCL_SAMPLE_NAME = re.compile(r"^(GSM\d+)_(.+)_(Blood|Skin)_GEX\.tsv\.gz$")
HCL_ZENODO_URL = "https://zenodo.org/records/14917813/files/50_de_analysis.zip?download=1"
HCL_ZENODO_MD5 = "4f7676aa9dc44e9aa90e74495b5d9229"
HCL_ZENODO_BYTES = 68_163_890
HCL_MATRIX_MEMBER = "50_de_analysis/pseudobulk/bulk_response_t0_bulk_df.tsv"
HCL_SAMPLESHEET_MEMBER = "50_de_analysis/pseudobulk/bulk_response_t0_samplesheet.csv"
HCL_ALL_DONORS_MEMBER = "50_de_analysis/pseudobulk/bulk_response_all_timepoints_samplesheet.csv"
HCL_MARKERS = ("ANXA1", "MS4A1", "CD22", "IL2RA", "ITGAE", "ITGAX")


@dataclass(frozen=True)
class GeoMicroarrayGroup:
    """One GEO series/platform combination and its cancer-code filters."""

    source_id: str
    accession: str
    platform_id: str
    platform_name: str
    sample_patterns: Mapping[str, str | None]
    expected_samples: Mapping[str, int]
    log2_transformed: bool = True
    symbol_platform_id: str | None = None


MICROARRAY_GROUPS = {
    "cmn": GeoMicroarrayGroup(
        source_id="gse11482-cmn",
        accession="GSE11482",
        platform_id="GPL96",
        platform_name="Affymetrix HG-U133A",
        sample_patterns={"CMN": r"(?i)cellular mesoblastic nephroma"},
        expected_samples={"CMN": 12},
    ),
    "mtc": GeoMicroarrayGroup(
        source_id="gse32662-mtc",
        accession="GSE32662",
        platform_id="GPL6480",
        platform_name="Agilent Human Genome 4x44K v1",
        sample_patterns={"MTC": None},
        expected_samples={"MTC": 52},
    ),
    "ess": GeoMicroarrayGroup(
        source_id="gse85383-ess",
        accession="GSE85383",
        platform_id="GPL22303",
        platform_name="Agilent SurePrint G3 Human Gene Expression v3",
        sample_patterns={
            "SARC_ESS_LG": r"(?i)low.?grade",
            "SARC_ESS_HG": r"(?i)high.?grade",
        },
        expected_samples={"SARC_ESS_LG": 9, "SARC_ESS_HG": 4},
        symbol_platform_id="GPL13497",
    ),
    "lps": GeoMicroarrayGroup(
        source_id="gse30929-lps",
        accession="GSE30929",
        platform_id="GPL96",
        platform_name="Affymetrix HG-U133A",
        sample_patterns={
            "SARC_WDLPS": r"(?i)well-differentiated",
            "SARC_DDLPS": r"(?i)dedifferentiated",
            "SARC_PLEOLPS": r"(?i)pleomorphic",
            "SARC_MYXLPS": r"(?i)myxoid",
        },
        expected_samples={
            "SARC_WDLPS": 52,
            "SARC_DDLPS": 40,
            "SARC_PLEOLPS": 20,
            "SARC_MYXLPS": 28,
        },
    ),
}


def _registry_entry(source_id: str) -> dict:
    matches = [
        entry for entry in expression_source_registry_entries() if str(entry.get("id")) == source_id
    ]
    if len(matches) != 1:
        raise KeyError(f"expected one expression source {source_id!r}; found {len(matches)}")
    return matches[0]


def _download(url: str, path: Path, *, force: bool = False) -> Path:
    """Download one cacheable public source file atomically."""
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(path)
    return path


def _extract_single_file(archive: Path, output_dir: Path) -> Path:
    """Extract the sole regular file in a ZIP without accepting unsafe paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = [info for info in zipped.infolist() if not info.is_dir()]
        if len(members) != 1:
            raise ValueError(f"{archive} contains {len(members)} files; expected exactly one")
        member = members[0]
        name = Path(member.filename)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"{archive} contains unsafe path {member.filename!r}")
        path = output_dir / name.name
        if not path.exists() or path.stat().st_size != member.file_size:
            temporary = path.with_suffix(path.suffix + ".tmp")
            with zipped.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            temporary.replace(path)
    return path


def geo_sample_titles(series_matrix_path: str | Path) -> dict[str, str]:
    """Return ``sample_id -> title suffix`` from one GEO series matrix.

    The supported title format is ``<sample-id> - <label>``. Malformed or
    duplicate sample IDs fail explicitly instead of silently changing routing.
    """
    title_line = ""
    with gzip.open(Path(series_matrix_path), "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                title_line = line
                break
    if not title_line:
        raise ValueError(f"{series_matrix_path} has no !Sample_title row")

    titles: dict[str, str] = {}
    for title in re.findall(r'"([^"]+)"', title_line):
        if " - " not in title:
            raise ValueError(f"unsupported GEO sample title {title!r}")
        sample_id, label = (part.strip() for part in title.split(" - ", 1))
        if not sample_id or not label:
            raise ValueError(f"unsupported GEO sample title {title!r}")
        if sample_id in titles:
            raise ValueError(f"duplicate GEO sample title for {sample_id!r}")
        titles[sample_id] = label
    if not titles:
        raise ValueError(f"{series_matrix_path} has no quoted sample titles")
    return titles


def gse75885_routed_samples(
    samples: Iterable[str],
    sample_titles: Mapping[str, str],
) -> dict[str, list[str]]:
    """Route only the three GSE75885 histologies owned by this source."""
    routed = {code: [] for code in GSE75885_LABEL_TO_CODE.values()}
    for sample in samples:
        sample_id = str(sample)
        if sample_id not in sample_titles:
            raise ValueError(f"GSE75885 expression sample {sample_id!r} has no GEO title")
        code = GSE75885_LABEL_TO_CODE.get(str(sample_titles[sample_id]))
        if code is not None:
            routed[code].append(sample_id)
    return routed


def build_gse75885_source_matrices(
    *,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    expression_path: str | Path | None = None,
    series_matrix_path: str | Path | None = None,
    force_download: bool = False,
    high_expression_threshold: float = 1.0,
) -> SourceMatrixBuildResult:
    """Build the three registry-owned GSE75885 sarcoma matrices."""
    entry = _registry_entry("gse75885-sarc")
    cache = Path(cache_dir)
    source = GeoMatrixSource(
        cancer_code=[str(code) for code in entry["cancer_codes"]],
        source_cohort=str(entry["source_cohort"]),
        source_project=str(entry.get("source_project") or "GEO"),
        citation=str(entry.get("citation") or ""),
        file_url=GSE75885_EXPRESSION_URL,
        file_name="GSE75885_Expression_117_sarcomas.tsv.gz",
        unit="RPKM",
        expected_source_samples=117,
        expected_samples_by_code={"SARC_DDLPS": 19, "SARC_LGFMS": 2, "SARC_PLEOLPS": 4},
        notes=str(entry.get("special_handling") or ""),
        tumor_origin="mixed",
        source_type=str(entry.get("source_type") or "geo-rnaseq"),
    )
    matrix, audit, diagnostics = prepare_source_matrix(
        source,
        cache_dir=cache,
        source_path=expression_path,
        force_download=force_download,
        high_expression_threshold=high_expression_threshold,
    )
    series_path = (
        Path(series_matrix_path)
        if series_matrix_path is not None
        else _download(
            GSE75885_SERIES_URL,
            cache / "GSE75885_series_matrix.txt.gz",
            force=force_download,
        )
    )
    routed = gse75885_routed_samples(sample_columns(matrix), geo_sample_titles(series_path))
    return build_canonical_source_matrices(
        source,
        matrix,
        routed_samples=routed,
        output_dir=Path(output_dir) if output_dir is not None else cache / "derived",
        mapping_audit=audit,
        parse_diagnostics=diagnostics,
    )


def drmetrics_routed_samples(
    samples: Iterable[str],
    attributes: pd.DataFrame,
) -> dict[str, list[str]]:
    """Route DRMetrics samples by its published simplified histology."""
    required = {"Sample_ID", "Histopathology_simplified"}
    missing = sorted(required - set(attributes.columns))
    if missing:
        raise ValueError(f"DRMetrics attributes lack columns: {missing}")
    if attributes["Sample_ID"].astype(str).duplicated().any():
        raise ValueError("DRMetrics attributes contain duplicate Sample_ID values")
    sample_to_histology = dict(
        zip(
            attributes["Sample_ID"].astype(str),
            attributes["Histopathology_simplified"].astype(str),
        )
    )
    routed = {code: [] for code in set(DRMETRICS_HISTOLOGY_TO_CODE.values())}
    for sample in samples:
        sample_id = str(sample)
        if sample_id not in sample_to_histology:
            raise ValueError(f"DRMetrics expression sample {sample_id!r} has no attributes row")
        code = DRMETRICS_HISTOLOGY_TO_CODE.get(sample_to_histology[sample_id])
        if code is not None:
            routed[code].append(sample_id)
    return routed


def build_drmetrics_source_matrices(
    *,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    counts_path: str | Path | None = None,
    attributes_path: str | Path | None = None,
    force_download: bool = False,
    ensembl_release: int = 112,
    high_expression_threshold: float = 1.0,
) -> SourceMatrixBuildResult:
    """Build lung carcinoid and LCNEC matrices from the DRMetrics release."""
    entry = _registry_entry("drmetrics-lnen-2020")
    cache = Path(cache_dir)
    if counts_path is None:
        archive = _download(
            DRMETRICS_COUNTS_URL,
            cache / "read_counts_all.txt.zip",
            force=force_download,
        )
        counts = _extract_single_file(archive, cache)
    else:
        counts = Path(counts_path)
    if attributes_path is None:
        archive = _download(
            DRMETRICS_ATTRIBUTES_URL,
            cache / "Attributes.txt.zip",
            force=force_download,
        )
        attributes = _extract_single_file(archive, cache)
    else:
        attributes = Path(attributes_path)

    gene_ids = pd.read_csv(counts, sep=r"\s+", usecols=[0], dtype=str).iloc[:, 0]
    lengths = ensembl_gene_lengths_kb(gene_ids, release=ensembl_release)
    source = GeoMatrixSource(
        cancer_code=[str(code) for code in entry["cancer_codes"]],
        source_cohort=str(entry["source_cohort"]),
        source_project="IARC pan-LNEN",
        citation=str(entry.get("citation") or ""),
        file_name=counts.name,
        unit="raw_counts",
        gene_id_col="gene_id",
        sep=r"\s+",
        expected_source_samples=238,
        expected_samples_by_code={"NET_LUNG": 118, "NEC_LUNG_LARGECELL": 69},
        notes=str(entry.get("special_handling") or ""),
        tumor_origin="primary",
        source_type=str(entry.get("source_type") or "github-release"),
    )
    matrix, audit, diagnostics = prepare_source_matrix(
        source,
        cache_dir=cache,
        source_path=counts,
        gene_lengths_kb=lengths,
        high_expression_threshold=high_expression_threshold,
    )
    attrs = pd.read_csv(
        attributes,
        sep="\t",
        usecols=["Sample_ID", "Histopathology_simplified"],
        dtype=str,
    )
    routed = drmetrics_routed_samples(sample_columns(matrix), attrs)
    return build_canonical_source_matrices(
        source,
        matrix,
        routed_samples=routed,
        output_dir=Path(output_dir) if output_dir is not None else cache / "derived",
        mapping_audit=audit,
        parse_diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class TargetAllLineageAssignment:
    """One TARGET ALL case's explicit B/T lineage evidence."""

    cancer_code: str
    raw_label: str
    evidence_source: str


def parse_target_all_lineage_label(comment: object) -> tuple[str, str]:
    """Return the raw ``Cell of origin`` value and its canonical ALL code."""
    text = "" if comment is None else str(comment)
    match = re.search(r"Cell of origin:\s*([^;\n\r]+)", text)
    if match is None:
        return "", ""
    raw_label = match.group(1).strip()
    if raw_label.lower().startswith("indeterminate"):
        return raw_label, ""
    if re.match(r"^B(\b|-|\s+precursor|\s+cell)", raw_label, flags=re.IGNORECASE):
        return raw_label, "B_ALL"
    if re.match(r"^T(\b|-|\s+cell)", raw_label, flags=re.IGNORECASE):
        return raw_label, "T_ALL"
    return raw_label, ""


def _extract_named_tar_members(
    archive: Path,
    member_names: Iterable[str],
    output_dir: Path,
) -> list[Path]:
    """Extract an explicit member allowlist without trusting archive paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with tarfile.open(archive, "r:gz") as tar:
        available = {member.name: member for member in tar.getmembers() if member.isfile()}
        for name in member_names:
            if name not in available:
                raise ValueError(f"{archive} lacks required member {name!r}")
            path = output_dir / Path(name).name
            member = available[name]
            if not path.exists() or path.stat().st_size != member.size:
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"{archive} member {name!r} cannot be read")
                temporary = path.with_suffix(path.suffix + ".tmp")
                with extracted, temporary.open("wb") as destination:
                    shutil.copyfileobj(extracted, destination)
                temporary.replace(path)
            paths.append(path)
    return paths


def target_all_lineage_assignments(
    sample_matrices: Iterable[tuple[str, str | Path]],
) -> dict[str, TargetAllLineageAssignment]:
    """Resolve case-level B/T assignments from the public TARGET matrices."""
    rows = []
    for evidence_name, path in sample_matrices:
        matrix = pd.read_excel(Path(path), sheet_name="Sample Names")
        required = {"Case USI", "Comments"}
        missing = sorted(required - set(matrix.columns))
        if missing:
            raise ValueError(f"{path} lacks TARGET lineage columns: {missing}")
        for _index, row in matrix.loc[matrix["Case USI"].notna()].iterrows():
            case_id = str(row["Case USI"])
            raw_label, cancer_code = parse_target_all_lineage_label(row["Comments"])
            if cancer_code:
                rows.append(
                    {
                        "case_id": case_id,
                        "raw_label": raw_label,
                        "cancer_code": cancer_code,
                        "evidence_source": str(evidence_name),
                    }
                )

    if not rows:
        raise ValueError("TARGET ALL sample matrices contain no explicit B/T lineage assignments")
    assignments: dict[str, TargetAllLineageAssignment] = {}
    for case_id, group in pd.DataFrame(rows).groupby("case_id", sort=False):
        codes = sorted(set(group["cancer_code"].astype(str)))
        if len(codes) != 1:
            raise ValueError(f"TARGET ALL case {case_id!r} has conflicting lineages: {codes}")
        labels = sorted(set(group["raw_label"].astype(str)))
        evidence = sorted(set(group["evidence_source"].astype(str)))
        assignments[str(case_id)] = TargetAllLineageAssignment(
            cancer_code=codes[0],
            raw_label="; ".join(labels),
            evidence_source=(f"{'; '.join(evidence)}; Cell of origin: {'; '.join(labels)}"),
        )
    return assignments


def fetch_target_all_lineages(
    cache_dir: str | Path,
    *,
    force_download: bool = False,
) -> dict[str, TargetAllLineageAssignment]:
    """Download the public TARGET phase sample matrices and resolve lineage."""
    cache = Path(cache_dir) / "sample_matrices"
    matrices = []
    for archive_name, record in TARGET_ALL_SAMPLE_MATRICES.items():
        archive = _download(
            f"https://api.gdc.cancer.gov/data/{record['file_id']}",
            cache / f"{archive_name}.tar.gz",
            force=force_download,
        )
        paths = _extract_named_tar_members(
            archive,
            record["members"],
            cache / archive_name,
        )
        matrices.extend((f"{archive_name} {path.stem} sample matrix", path) for path in paths)
    return target_all_lineage_assignments(matrices)


def build_target_all_source_matrices(
    *,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    lineage_assignments: Mapping[str, TargetAllLineageAssignment] | None = None,
    manifest: pd.DataFrame | None = None,
    file_paths: Mapping[str, str | Path] | None = None,
    force_download: bool = False,
    high_expression_threshold: float = 1.0,
) -> SourceMatrixBuildResult:
    """Build TARGET ALL matrices using its authoritative phase lineage tables."""
    cache = Path(cache_dir)
    assignments = (
        dict(lineage_assignments)
        if lineage_assignments is not None
        else fetch_target_all_lineages(cache, force_download=force_download)
    )
    base = gdc_source_from_registry("target-all")

    def route(row: Mapping) -> str | None:
        assignment = assignments.get(str(row.get("case_id") or ""))
        return assignment.cancer_code if assignment is not None else None

    def evidence(row: Mapping) -> str | None:
        assignment = assignments.get(str(row.get("case_id") or ""))
        return assignment.evidence_source if assignment is not None else None

    source = replace(
        base,
        primary_sample_types=(
            "Primary Blood Derived Cancer - Bone Marrow",
            "Primary Blood Derived Cancer - Peripheral Blood",
        ),
        primary_diagnosis_contains=("acute lymphocytic leukemia",),
        sample_to_cancer_code=route,
        sample_lineage_evidence=evidence,
        expected_n={"B_ALL": 154, "T_ALL": 264},
    )
    return build_gdc_source_matrices(
        source,
        cache_dir=cache,
        output_dir=output_dir,
        manifest=manifest,
        file_paths=file_paths,
        force_download=force_download,
        high_expression_threshold=high_expression_threshold,
    )


def fetch_nbl_mycn_status(
    cache_path: str | Path,
    *,
    force_download: bool = False,
) -> dict[str, str]:
    """Return case-level ``amp``, ``nonamp``, or ``unknown`` MYCN calls."""
    cache = Path(cache_path)
    if cache.exists() and cache.stat().st_size > 0 and not force_download:
        rows = pd.read_csv(cache, dtype=str)
    else:
        with urllib.request.urlopen(CBIOPORTAL_NBL_MYCN_URL, timeout=120) as response:
            payload = json.load(response)
        rows = pd.DataFrame(
            {
                "case_id": str(item.get("patientId") or ""),
                "raw_mycn_status": str(item.get("value") or ""),
            }
            for item in payload
        )

    required = {"case_id", "raw_mycn_status"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"NBL MYCN table lacks columns: {missing}")

    normalized = {"Amplified": "amp", "Not Amplified": "nonamp", "Unknown": "unknown"}
    rows = rows.copy()
    rows["mycn_status"] = rows["raw_mycn_status"].map(normalized)
    if rows["mycn_status"].isna().any():
        unexpected = sorted(set(rows.loc[rows["mycn_status"].isna(), "raw_mycn_status"]))
        raise ValueError(f"unsupported cBioPortal MYCN values: {unexpected}")
    conflicts = {
        case_id: sorted(set(group["mycn_status"]))
        for case_id, group in rows.groupby("case_id")
        if group["mycn_status"].nunique() != 1
    }
    if conflicts:
        raise ValueError(f"conflicting cBioPortal MYCN values: {conflicts}")
    unique = rows.drop_duplicates("case_id")[["case_id", "raw_mycn_status", "mycn_status"]]
    if not cache.exists() or force_download:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(cache.suffix + ".tmp")
        unique.to_csv(temporary, index=False)
        temporary.replace(cache)
    return dict(zip(unique["case_id"].astype(str), unique["mycn_status"].astype(str)))


def nbl_mycn_routed_samples(
    samples: Iterable[str],
    case_status: Mapping[str, str],
) -> dict[str, list[str]]:
    """Route every NBL sample to the parent and exactly one MYCN child."""
    routed = {"NBL": [], "NBL_MYCNamp": [], "NBL_MYCNnonamp": []}
    for sample in samples:
        sample_id = str(sample)
        case_id = "-".join(sample_id.split("-")[:3])
        status = case_status.get(case_id)
        if status not in {"amp", "nonamp", "unknown"}:
            raise ValueError(f"NBL sample {sample_id!r} lacks a supported MYCN call")
        routed["NBL"].append(sample_id)
        child = "NBL_MYCNamp" if status == "amp" else "NBL_MYCNnonamp"
        routed[child].append(sample_id)
    return routed


def build_target_nbl_source_matrices(
    *,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    manifest: pd.DataFrame | None = None,
    file_paths: Mapping[str, str | Path] | None = None,
    mycn_status: Mapping[str, str] | None = None,
    force_download: bool = False,
    high_expression_threshold: float = 1.0,
) -> SourceMatrixBuildResult:
    """Build the TARGET NBL parent and its mutually exclusive MYCN children."""
    cache = Path(cache_dir)
    out_dir = Path(output_dir) if output_dir is not None else cache / "derived"
    base = gdc_source_from_registry("target-nbl")
    parent_source = replace(
        base,
        cancer_code="NBL",
        primary_sample_types=(),
        primary_diagnosis_contains=("neuroblastoma", "ganglioneuroblastoma"),
        sample_to_cancer_code=None,
        expected_n={"NBL": 155},
        tumor_origin="mixed",
    )
    parent = build_gdc_source_matrices(
        parent_source,
        cache_dir=cache,
        output_dir=out_dir,
        manifest=manifest,
        file_paths=file_paths,
        force_download=force_download,
        high_expression_threshold=high_expression_threshold,
    )
    statuses = (
        dict(mycn_status)
        if mycn_status is not None
        else fetch_nbl_mycn_status(
            cache / "cbioportal_nbl_mycn.csv",
            force_download=force_download,
        )
    )
    matrix = parent.matrices["NBL"]
    routed = nbl_mycn_routed_samples(sample_columns(matrix), statuses)
    source = GeoMatrixSource(
        cancer_code=["NBL", "NBL_MYCNamp", "NBL_MYCNnonamp"],
        source_cohort=base.source_cohort,
        source_project=base.source_project,
        citation=base.citation,
        file_name="GDC TARGET-NBL STAR-counts files",
        unit="TPM",
        expected_source_samples=155,
        expected_samples_by_code={"NBL": 155, "NBL_MYCNamp": 33, "NBL_MYCNnonamp": 122},
        source_scale_class="linear_rnaseq_tpm",
        linear_tpm_comparable=True,
        tpm_proxy=False,
        notes=(
            "One deterministic GDC sample per TARGET-NBL case. The released cohort "
            "contains 153 primary and two recurrent samples. MYCN children use "
            "cBioPortal nbl_target_2018_pub; its one explicit Unknown call is retained "
            "in the non-amplified child to preserve the published 33/122 split."
        ),
        tumor_origin="mixed",
        source_type="gdc",
    )
    result = build_canonical_source_matrices(
        source,
        matrix,
        routed_samples=routed,
        output_dir=out_dir,
        mapping_audit=parent.mapping_audit,
        parse_diagnostics=parent.parse_diagnostics,
    )

    evidence_rows = []
    for sample in sample_columns(matrix):
        case_id = "-".join(str(sample).split("-")[:3])
        status = statuses[case_id]
        evidence_rows.append(
            {
                "sample_id": str(sample),
                "case_id": case_id,
                "mycn_status": status,
                "child_cancer_code": ("NBL_MYCNamp" if status == "amp" else "NBL_MYCNnonamp"),
                "evidence_source": (
                    "cBioPortal nbl_target_2018_pub sample clinical attribute MYCN"
                ),
            }
        )
    evidence_path = out_dir / "target_nbl_2018_mycn_routing.csv"
    temporary = evidence_path.with_suffix(evidence_path.suffix + ".tmp")
    pd.DataFrame(evidence_rows).to_csv(temporary, index=False)
    temporary.replace(evidence_path)
    return SourceMatrixBuildResult(
        source=result.source,
        matrices=result.matrices,
        matrix_paths=result.matrix_paths,
        summary_rows=result.summary_rows,
        mapping_audit=result.mapping_audit,
        parse_diagnostics=result.parse_diagnostics,
        sample_qc=result.sample_qc,
        sidecar_paths={
            **result.sidecar_paths,
            "gdc_sample_manifest": parent.sidecar_paths["gdc_sample_manifest"],
            "mycn_routing": evidence_path,
        },
    )


@dataclass(frozen=True)
class CtclSampleRecord:
    """Paired GEX/TCR-beta matrices for one CTCL study specimen."""

    gsm_id: str
    case_id: str
    compartment: str
    gex_name: str
    tcrb_name: str

    @property
    def is_healthy_control(self) -> bool:
        return self.case_id.upper().startswith("HC")


def parse_geo_soft_samples(path: str | Path) -> dict[str, dict[str, str]]:
    """Parse the small sample-metadata subset needed from a GEO family SOFT."""
    samples: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    with gzip.open(Path(path), "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                current = {"geo_accession": line.split(" = ", 1)[1]}
                samples[current["geo_accession"]] = current
                continue
            if current is None:
                continue
            if line.startswith("!Sample_title = "):
                current["title"] = line.split(" = ", 1)[1]
            elif line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split(" = ", 1)[1]
                key, separator, field_value = value.partition(":")
                if separator:
                    current[key.strip().lower()] = field_value.strip()
    if not samples:
        raise ValueError(f"{path} contains no GEO samples")
    return samples


def ctcl_sample_records(tar_names: Iterable[str]) -> list[CtclSampleRecord]:
    """Pair each CTCL GEX member with its matching TCR-beta member."""
    names = {str(name) for name in tar_names}
    records = []
    for gex_name in sorted(name for name in names if name.endswith("_GEX.tsv.gz")):
        match = CTCL_SAMPLE_NAME.fullmatch(gex_name)
        if match is None:
            raise ValueError(f"unsupported CTCL GEX member name {gex_name!r}")
        tcrb_name = gex_name.replace("_GEX.tsv.gz", "_TCRb.tsv.gz")
        if tcrb_name not in names:
            raise ValueError(f"CTCL GEX member {gex_name!r} has no matching TCR-beta matrix")
        records.append(
            CtclSampleRecord(
                gsm_id=match.group(1),
                case_id=match.group(2),
                compartment=match.group(3),
                gex_name=gex_name,
                tcrb_name=tcrb_name,
            )
        )
    if not records:
        raise ValueError("CTCL archive contains no paired GEX/TCR-beta samples")
    return records


@contextmanager
def _tar_gzip_rows(tar: tarfile.TarFile, member_name: str):
    """Yield tab-separated rows from one gzip-compressed TAR member."""
    member = tar.extractfile(member_name)
    if member is None:
        raise ValueError(f"CTCL archive member {member_name!r} cannot be read")
    with (
        member,
        gzip.GzipFile(fileobj=member) as compressed,
        io.TextIOWrapper(compressed) as text,
    ):
        yield csv.reader(text, delimiter="\t")


def _ctcl_top_clone(tar: tarfile.TarFile, member_name: str) -> tuple[str, int, tuple[str, ...]]:
    with _tar_gzip_rows(tar, member_name) as reader:
        header = next(reader)
        ranked = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{member_name} contains a malformed clone row")
            try:
                count = sum(int(value) for value in row[1:] if value)
            except ValueError as error:
                raise ValueError(f"{member_name} has a non-integer TCR count") from error
            ranked.append((count, row[0]))
    if not ranked:
        raise ValueError(f"{member_name} has no TCR clones")
    ranked.sort(reverse=True)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        raise ValueError(f"{member_name} has a tied dominant TCR-beta clone")
    return ranked[0][1], ranked[0][0], tuple(header[1:])


def _ctcl_selected_cells(
    tar: tarfile.TarFile,
    member_name: str,
    clone: str,
) -> tuple[list[int], tuple[str, ...]]:
    with _tar_gzip_rows(tar, member_name) as reader:
        header = next(reader)
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{member_name} contains a malformed clone row")
            if row[0] != clone:
                continue
            indices = [
                index for index, value in enumerate(row[1:], start=1) if value and value != "0"
            ]
            return indices, tuple(header[1:])
    return [], tuple(header[1:])


def _add_ctcl_gex_counts(
    tar: tarfile.TarFile,
    record: CtclSampleRecord,
    selected_indices: list[int],
    tcr_cell_ids: tuple[str, ...],
    counts_by_symbol: dict[str, defaultdict[str, float]],
) -> None:
    with _tar_gzip_rows(tar, record.gex_name) as reader:
        header = next(reader)
        if tuple(header[1:]) != tcr_cell_ids:
            raise ValueError(
                f"{record.gex_name} and {record.tcrb_name} have different cell columns"
            )
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{record.gex_name} contains a malformed gene row")
            count = sum(float(row[index]) for index in selected_indices)
            if count > 0:
                counts_by_symbol[row[0]][record.case_id] += count


def ctcl_case_pseudobulk(
    raw_tar_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select dominant-clone cells and return case counts plus provenance."""
    with tarfile.open(Path(raw_tar_path)) as tar:
        records = ctcl_sample_records(tar.getnames())
        blood_clones: dict[str, set[str]] = defaultdict(set)
        for record in records:
            if record.is_healthy_control or record.compartment != "Blood":
                continue
            clone, _count, _cell_ids = _ctcl_top_clone(tar, record.tcrb_name)
            blood_clones[record.case_id].add(clone)
        conflicts = {
            case_id: sorted(clones) for case_id, clones in blood_clones.items() if len(clones) != 1
        }
        if conflicts:
            raise ValueError(f"CTCL cases lack one unambiguous blood TCR-beta clone: {conflicts}")
        case_clone = {case_id: next(iter(clones)) for case_id, clones in blood_clones.items()}

        counts_by_symbol: dict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        manifest_rows = []
        for record in records:
            clone = case_clone.get(record.case_id, "")
            selected_indices: list[int] = []
            cell_ids: tuple[str, ...] = ()
            if not record.is_healthy_control:
                if not clone:
                    raise ValueError(f"CTCL case {record.case_id!r} has no blood clone")
                selected_indices, cell_ids = _ctcl_selected_cells(
                    tar,
                    record.tcrb_name,
                    clone,
                )
                _add_ctcl_gex_counts(
                    tar,
                    record,
                    selected_indices,
                    cell_ids,
                    counts_by_symbol,
                )
            else:
                _clone, _count, cell_ids = _ctcl_top_clone(tar, record.tcrb_name)
            manifest_rows.append(
                {
                    "cancer_code": "CTCL",
                    "source_cohort": "GSE171811_ECCITE_CTCL",
                    "case_id": record.case_id,
                    "sample_id": f"{record.gsm_id}_{record.case_id}_{record.compartment}",
                    "source_file_id": record.gsm_id,
                    "source_file_name": f"GSE171811_RAW.tar:{record.gex_name}",
                    "sample_type": record.compartment,
                    "raw_unit": "scRNA UMI counts",
                    "lineage_evidence_source": (
                        f"dominant blood TCR-beta clone {clone}; "
                        f"{len(selected_indices)} of {len(cell_ids)} cells selected"
                        if not record.is_healthy_control
                        else "healthy-control specimen excluded"
                    ),
                    "included": not record.is_healthy_control and bool(selected_indices),
                    "exclusion_reason": ("healthy_control" if record.is_healthy_control else ""),
                    "lineage_label": "CTCL" if not record.is_healthy_control else "",
                }
            )

    cases = sorted(case_clone)
    counts = pd.DataFrame(
        (
            {
                "source_symbol": symbol,
                **{case: values.get(case, 0.0) for case in cases},
            }
            for symbol, values in counts_by_symbol.items()
        )
    )
    if counts.empty or len(cases) != 7:
        raise ValueError(f"CTCL pseudobulk produced {len(cases)} cases and {len(counts)} genes")
    return counts, pd.DataFrame(manifest_rows)


def build_ctcl_source_matrices(
    *,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    raw_tar_path: str | Path | None = None,
    soft_path: str | Path | None = None,
    force_download: bool = False,
    high_expression_threshold: float = 1.0,
) -> SourceMatrixBuildResult:
    """Build seven CTCL case pseudobulks from GSE171811 GEX/TCR-beta data."""
    entry = _registry_entry("gse171811-ctcl")
    cache = Path(cache_dir)
    raw_tar = (
        Path(raw_tar_path)
        if raw_tar_path is not None
        else _download(
            CTCL_RAW_URL,
            cache / "GSE171811_RAW.tar",
            force=force_download,
        )
    )
    soft = (
        Path(soft_path)
        if soft_path is not None
        else _download(
            CTCL_SOFT_URL,
            cache / "GSE171811_family.soft.gz",
            force=force_download,
        )
    )
    counts, manifest = ctcl_case_pseudobulk(raw_tar)
    sample_metadata = parse_geo_soft_samples(soft)
    manifest["primary_diagnosis"] = manifest["source_file_id"].map(
        lambda gsm: sample_metadata.get(str(gsm), {}).get("disease state", "")
    )
    manifest["sample_type"] = manifest.apply(
        lambda row: sample_metadata.get(str(row["source_file_id"]), {}).get(
            "tissue",
            row["sample_type"],
        ),
        axis=1,
    )

    value_cols = [column for column in counts.columns if column != "source_symbol"]
    values = counts[value_cols].apply(pd.to_numeric, errors="raise")
    totals = values.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("CTCL pseudobulk contains a case with no selected UMI counts")
    ntpm = values.div(totals, axis=1) * 1_000_000.0
    raw = pd.concat([counts[["source_symbol"]], ntpm], axis=1)
    _, diagnostics = coerce_source_expression_values(
        raw,
        value_cols=value_cols,
        row_id_col="source_symbol",
    )
    matrix, audit = canonicalize_source_gene_matrix(
        raw,
        row_id_col="source_symbol",
        value_cols=value_cols,
        high_expression_threshold=high_expression_threshold,
    )
    source = GeoMatrixSource(
        cancer_code="CTCL",
        source_cohort=str(entry["source_cohort"]),
        source_project="GEO",
        citation=str(entry.get("citation") or ""),
        file_name=raw_tar.name,
        unit="TPM",
        expected_source_samples=7,
        expected_samples_by_code={"CTCL": 7},
        source_scale_class="scrna_tcr_pseudobulk_ntpm",
        linear_tpm_comparable=False,
        tpm_proxy=True,
        native_unit=str(entry.get("unit") or "nTPM (pseudobulk)"),
        notes=(
            "GSE171811 CTCL ECCITE-seq. Each disease patient's dominant blood "
            "TCR-beta clonotype selects cells from available blood/skin specimens; "
            "UMIs are pseudobulked per case and scaled to nTPM. This rank-preserving "
            "single-cell pseudobulk is not directly comparable to bulk RNA-seq TPM."
        ),
        processing_pipeline=(
            "gse171811_ctcl_scrna_tcrb_pseudobulk_ntpm_ensembl112_clean_tpm_16_9_75"
        ),
        tumor_origin="primary",
        source_type="geo-scrna-pseudobulk",
    )
    out_dir = Path(output_dir) if output_dir is not None else cache / "derived"
    result = build_canonical_source_matrices(
        source,
        matrix,
        routed_samples={"CTCL": sample_columns(matrix)},
        output_dir=out_dir,
        mapping_audit=audit,
        parse_diagnostics=diagnostics,
    )
    manifest_path = out_dir / "gse171811_eccite_ctcl_sample_manifest.csv"
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest.to_csv(temporary, index=False)
    temporary.replace(manifest_path)
    return SourceMatrixBuildResult(
        source=result.source,
        matrices=result.matrices,
        matrix_paths=result.matrix_paths,
        summary_rows=result.summary_rows,
        mapping_audit=result.mapping_audit,
        parse_diagnostics=result.parse_diagnostics,
        sample_qc=result.sample_qc,
        sidecar_paths={**result.sidecar_paths, "sample_manifest": manifest_path},
    )


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hcl_t0_pseudobulk(archive_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the five published pretreatment HCL donor pseudobulks.

    The Zenodo artifact assigns local ``sample_N`` labels independently in each
    analysis table. Columns are therefore renamed to donor IDs so the selected
    matrix is stable and auditable. P1 is retained in the manifest as excluded:
    it appears in the all-timepoints table but has no T0 pseudobulk.
    """
    archive = Path(archive_path)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        required_members = {
            HCL_MATRIX_MEMBER,
            HCL_SAMPLESHEET_MEMBER,
            HCL_ALL_DONORS_MEMBER,
        }
        missing_members = sorted(required_members - names)
        if missing_members:
            raise ValueError(f"HCL archive lacks required members: {missing_members}")
        with zipped.open(HCL_MATRIX_MEMBER) as handle:
            raw = pd.read_csv(handle, sep="\t")
        with zipped.open(HCL_SAMPLESHEET_MEMBER) as handle:
            selected = pd.read_csv(handle)
        with zipped.open(HCL_ALL_DONORS_MEMBER) as handle:
            all_donors = pd.read_csv(handle)

    required_sample_columns = {"sample_id", "patient", "response", "sex", "age", "n_obs"}
    for label, table in (("T0", selected), ("all-donor", all_donors)):
        missing = sorted(required_sample_columns - set(table.columns))
        if missing:
            raise ValueError(f"HCL {label} samplesheet lacks columns: {missing}")
        if table["patient"].astype(str).duplicated().any():
            raise ValueError(f"HCL {label} samplesheet contains duplicate donors")

    selected_ids = selected["sample_id"].astype(str).tolist()
    selected_donors = selected["patient"].astype(str).tolist()
    if set(selected_donors) != {"P2", "P3", "P4", "P5", "P6"}:
        raise ValueError(f"HCL T0 donor set changed: {sorted(selected_donors)}")
    all_donor_ids = set(all_donors["patient"].astype(str))
    if all_donor_ids != {"P1", "P2", "P3", "P4", "P5", "P6"}:
        raise ValueError(f"HCL study donor set changed: {sorted(all_donor_ids)}")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("HCL T0 samplesheet contains duplicate source sample IDs")

    required_matrix_columns = ["gene_id", "gene_name", *selected_ids]
    if list(raw.columns) != required_matrix_columns:
        raise ValueError(
            f"HCL T0 matrix columns do not match its samplesheet: {list(raw.columns)!r}"
        )
    source_symbols = raw["gene_name"].astype(str).str.strip()
    if source_symbols.eq("").any() or source_symbols.duplicated().any():
        raise ValueError("HCL T0 matrix contains blank or duplicate gene symbols")
    if not raw["gene_id"].astype(str).str.strip().eq(source_symbols).all():
        raise ValueError("HCL T0 matrix gene_id and gene_name columns disagree")

    values = raw[selected_ids].apply(pd.to_numeric, errors="raise")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("HCL T0 matrix contains non-finite or negative counts")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("HCL T0 matrix contains non-integral UMI counts")
    marker_counts = raw.assign(_symbol=source_symbols).set_index("_symbol").reindex(HCL_MARKERS)
    if marker_counts["gene_name"].isna().any():
        missing_markers = marker_counts.index[marker_counts["gene_name"].isna()].tolist()
        raise ValueError(f"HCL T0 matrix lacks marker genes: {missing_markers}")
    if (marker_counts[selected_ids].apply(pd.to_numeric) <= 0).any().any():
        raise ValueError("HCL T0 matrix has a non-positive HCL marker count")

    donor_by_sample = dict(zip(selected_ids, selected_donors))
    counts = pd.DataFrame({"source_symbol": source_symbols})
    for sample_id in selected_ids:
        counts[donor_by_sample[sample_id]] = values[sample_id].to_numpy(dtype=float)

    selected_by_donor = selected.assign(
        patient=selected["patient"].astype(str),
        sample_id=selected["sample_id"].astype(str),
    ).set_index("patient")
    manifest_rows = []
    for donor in sorted(all_donor_ids):
        included = donor in selected_by_donor.index
        row = (
            selected_by_donor.loc[donor] if included else all_donors.set_index("patient").loc[donor]
        )
        source_sample_id = str(row["sample_id"])
        n_cells = int(row["n_obs"])
        manifest_rows.append(
            {
                "cancer_code": "HCL",
                "source_cohort": "ZENODO_14917813_BOHN_2026_HCL",
                "source_project": "Zenodo",
                "case_id": donor,
                "sample_id": donor,
                "source_file_id": source_sample_id,
                "source_file_name": (
                    f"50_de_analysis.zip:{HCL_MATRIX_MEMBER}"
                    if included
                    else f"50_de_analysis.zip:{HCL_ALL_DONORS_MEMBER}"
                ),
                "source_project_id": "DOI:10.5281/zenodo.14917813",
                "sample_type": "author-annotated malignant HCL cells",
                "md5sum": HCL_ZENODO_MD5,
                "file_size": HCL_ZENODO_BYTES,
                "workflow_type": "author-generated scRNA malignant-cell donor pseudobulk",
                "raw_unit": "scRNA UMI counts",
                "processing_pipeline": (
                    "zenodo_14917813_hcl_t0_malignant_cell_pseudobulk_ntpm_"
                    "ensembl112_clean_tpm_16_9_75"
                ),
                "source_url": "https://zenodo.org/records/14917813",
                "lineage_evidence_source": (
                    f"author cell_type=HCL cell; pretreatment T0; {n_cells} cells pseudobulked"
                    if included
                    else "study donor present, but no pretreatment T0 pseudobulk is published"
                ),
                "included": included,
                "exclusion_reason": "" if included else "no_pretreatment_t0_pseudobulk",
                "lineage_label": "HCL" if included else "",
                "primary_diagnosis": "classic hairy cell leukemia (author diagnosis)",
                "treatment_status": "pretreatment T0" if included else "unresolved",
                "molecular_evidence_source": (
                    "No per-donor BRAF call used; BRAF V600E is not inferred from diagnosis "
                    "or expression"
                ),
                "response_group": str(row["response"]),
                "sex": str(row["sex"]),
                "age": int(row["age"]),
                "n_cells": n_cells,
                "source_record": "DOI:10.5281/zenodo.14917813",
            }
        )
    return counts, pd.DataFrame(manifest_rows)


def build_hcl_source_matrices(
    *,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    archive_path: str | Path | None = None,
    force_download: bool = False,
    high_expression_threshold: float = 1.0,
) -> SourceMatrixBuildResult:
    """Build five donor-level HCL T0 malignant-cell pseudobulks from Zenodo."""
    entry = _registry_entry("zenodo-14917813-hcl")
    cache = Path(cache_dir)
    archive = (
        Path(archive_path)
        if archive_path is not None
        else _download(
            str(entry.get("file_url") or HCL_ZENODO_URL),
            cache / str(entry.get("file_name") or "50_de_analysis.zip"),
            force=force_download,
        )
    )
    expected_bytes = int(entry.get("file_bytes") or HCL_ZENODO_BYTES)
    observed_bytes = archive.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(f"HCL Zenodo archive size mismatch: {observed_bytes} != {expected_bytes}")
    expected_md5 = str(entry.get("file_md5") or HCL_ZENODO_MD5).lower()
    observed_md5 = _file_md5(archive)
    if observed_md5 != expected_md5:
        raise ValueError(f"HCL Zenodo archive MD5 mismatch: {observed_md5} != {expected_md5}")

    counts, manifest = hcl_t0_pseudobulk(archive)
    value_cols = [column for column in counts.columns if column != "source_symbol"]
    values = counts[value_cols].apply(pd.to_numeric, errors="raise")
    totals = values.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("HCL pseudobulk contains a donor with no selected UMI counts")
    ntpm = values.div(totals, axis=1) * 1_000_000.0
    raw = pd.concat([counts[["source_symbol"]], ntpm], axis=1)
    _, diagnostics = coerce_source_expression_values(
        raw,
        value_cols=value_cols,
        row_id_col="source_symbol",
    )
    matrix, audit = canonicalize_source_gene_matrix(
        raw,
        row_id_col="source_symbol",
        value_cols=value_cols,
        high_expression_threshold=high_expression_threshold,
    )
    marker_rows = matrix[matrix["Symbol"].isin(HCL_MARKERS)].copy()
    if set(marker_rows["Symbol"].astype(str)) != set(HCL_MARKERS):
        missing = sorted(set(HCL_MARKERS) - set(marker_rows["Symbol"].astype(str)))
        raise ValueError(f"canonical HCL matrix lacks marker genes: {missing}")
    if (marker_rows[value_cols] <= 0).any().any():
        raise ValueError("canonical HCL matrix has a non-positive HCL marker value")

    source = GeoMatrixSource(
        cancer_code="HCL",
        source_cohort=str(entry["source_cohort"]),
        source_project=str(entry.get("source_project") or "Zenodo"),
        citation=str(entry.get("citation") or ""),
        file_name=archive.name,
        unit="TPM",
        expected_source_samples=5,
        expected_samples_by_code={"HCL": 5},
        source_scale_class="scrna_malignant_cell_pseudobulk_ntpm",
        linear_tpm_comparable=False,
        tpm_proxy=True,
        native_unit=str(entry.get("unit") or "nTPM (pseudobulk)"),
        notes=(
            "Bohn et al. primary classic-HCL scRNA-seq. Author-annotated malignant "
            "HCL cells from one pretreatment T0 profile per donor are pseudobulked "
            "and library-size scaled to nTPM. P1 is excluded because no T0 pseudobulk "
            "is published. No per-donor BRAF status is inferred. This single-cell "
            "pseudobulk is a direct HCL reference but is not bulk-TPM comparable or, "
            "by itself, classification-ready."
        ),
        processing_pipeline=(
            "zenodo_14917813_hcl_t0_malignant_cell_pseudobulk_ntpm_ensembl112_clean_tpm_16_9_75"
        ),
        tumor_origin="primary",
        source_type="zenodo-scrna-pseudobulk",
    )
    out_dir = Path(output_dir) if output_dir is not None else cache / "derived"
    result = build_canonical_source_matrices(
        source,
        matrix,
        routed_samples={"HCL": value_cols},
        output_dir=out_dir,
        mapping_audit=audit,
        parse_diagnostics=diagnostics,
    )

    manifest_path = out_dir / "zenodo_14917813_bohn_2026_hcl_sample_manifest.csv"
    marker_path = out_dir / "zenodo_14917813_bohn_2026_hcl_marker_qc.csv"
    manifest_temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    marker_temp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    manifest.to_csv(manifest_temp, index=False)
    manifest_temp.replace(manifest_path)
    marker_qc = marker_rows[["Ensembl_Gene_ID", "Symbol", *value_cols]].copy()
    marker_qc["minimum_ntpm"] = marker_qc[value_cols].min(axis=1)
    marker_qc["median_ntpm"] = marker_qc[value_cols].median(axis=1)
    marker_qc.to_csv(marker_temp, index=False)
    marker_temp.replace(marker_path)
    return SourceMatrixBuildResult(
        source=result.source,
        matrices=result.matrices,
        matrix_paths=result.matrix_paths,
        summary_rows=result.summary_rows,
        mapping_audit=result.mapping_audit,
        parse_diagnostics=result.parse_diagnostics,
        sample_qc=result.sample_qc,
        sidecar_paths={
            **result.sidecar_paths,
            "sample_manifest": manifest_path,
            "marker_qc": marker_path,
        },
    )


def geo_platform_url(platform_id: str) -> str:
    """GEO text view containing one platform's probe annotations."""
    return (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?targ=self&form=text&view=full&acc={platform_id}"
    )


def geo_series_matrix_url(accession: str) -> str:
    """FTP URL for an accession's primary GEO series matrix."""
    prefix = f"{accession[:-3]}nnn"
    return (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/matrix/"
        f"{accession}_series_matrix.txt.gz"
    )


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _read_geo_platform_table(path: str | Path) -> pd.DataFrame:
    """Read the delimited table inside a GEO platform text response."""
    table_started = False
    header: list[str] = []
    rows: list[list[str]] = []
    with _open_text(Path(path)) as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                table_started = True
                header = handle.readline().rstrip("\n").split("\t")
                continue
            if not table_started:
                continue
            if line.startswith("!platform_table_end"):
                break
            values = line.rstrip("\n").split("\t")
            rows.append((values + [""] * len(header))[: len(header)])
    if not table_started or not header:
        raise ValueError(f"{path} has no GEO platform table")
    return pd.DataFrame(rows, columns=header, dtype=str)


def _parallel_symbol_entrez_pairs(symbols: str, entrez_ids: str) -> list[tuple[str, str]]:
    symbol_values = [value.strip() for value in str(symbols).split("///")]
    entrez_values = [value.strip() for value in str(entrez_ids).split("///")]
    entrez_values.extend([""] * (len(symbol_values) - len(entrez_values)))
    return list(zip(symbol_values, entrez_values[: len(symbol_values)]))


def parse_geo_platform_annotations(path: str | Path) -> pd.DataFrame:
    """Return probe, symbol, and optional Entrez assignments from GEO."""
    table = _read_geo_platform_table(path)
    probe_col = next(
        (
            column
            for column in ("ID", "ID_REF", "Probe Set ID", "ProbeName", "ProbeID")
            if column in table.columns
        ),
        None,
    )
    symbol_col = next(
        (
            column
            for column in (
                "Gene symbol",
                "Gene Symbol",
                "GeneSymbol",
                "Symbol",
                "GENE_SYMBOL",
                "ILMN_Gene",
            )
            if column in table.columns
        ),
        None,
    )
    entrez_col = next(
        (
            column
            for column in (
                "ENTREZ_GENE_ID",
                "Entrez Gene ID",
                "Entrez_Gene_ID",
                "EntrezGeneID",
                "ENTREZID",
                "GENE",
            )
            if column in table.columns
        ),
        None,
    )
    if probe_col is None:
        raise ValueError(f"{path} has no recognized GEO probe column")

    out = pd.DataFrame({"probe_id": table[probe_col].astype(str)})
    out["gene_symbol"] = (
        table[symbol_col].astype(str) if symbol_col is not None else out["probe_id"]
    )
    out["entrez_id"] = table[entrez_col].astype(str) if entrez_col is not None else ""
    out["symbol_entrez_pairs"] = out.apply(
        lambda row: _parallel_symbol_entrez_pairs(row["gene_symbol"], row["entrez_id"]),
        axis=1,
    )
    out = out.explode("symbol_entrez_pairs")
    out[["gene_symbol", "entrez_id"]] = pd.DataFrame(
        out["symbol_entrez_pairs"].tolist(),
        index=out.index,
    )
    out = out.drop(columns="symbol_entrez_pairs")
    valid = (
        out["probe_id"].str.strip().ne("")
        & out["gene_symbol"].str.strip().ne("")
        & out["gene_symbol"].str.strip().ne("---")
    )
    return out.loc[valid].reset_index(drop=True)


def _unversioned_accession(value: str) -> str:
    return str(value).split(".", 1)[0].strip()


def parse_geo_platform_annotation_bridge(
    primary_path: str | Path,
    symbol_path: str | Path,
) -> pd.DataFrame:
    """Join a symbol-less platform to a same-design symbol platform by GB_ACC."""
    primary = _read_geo_platform_table(primary_path)
    symbol_table = _read_geo_platform_table(symbol_path)
    probe_col = next(
        (column for column in ("ID", "ID_REF", "ProbeName") if column in primary.columns),
        None,
    )
    primary_accession_col = next(
        (column for column in ("GB_ACC", "SystematicName", "ID") if column in primary.columns),
        None,
    )
    symbol_accession_col = next(
        (column for column in ("GB_ACC", "SystematicName") if column in symbol_table.columns),
        None,
    )
    symbol_col = next(
        (
            column
            for column in ("GENE_SYMBOL", "Gene Symbol", "GeneSymbol")
            if column in symbol_table.columns
        ),
        None,
    )
    if not all((probe_col, primary_accession_col, symbol_accession_col, symbol_col)):
        raise ValueError("GEO platform bridge lacks probe, accession, or symbol columns")
    entrez_col = next(
        (
            column
            for column in ("GENE", "ENTREZ_GENE_ID", "Entrez Gene ID")
            if column in symbol_table.columns
        ),
        None,
    )

    lookup = pd.DataFrame(
        {
            "join_key": symbol_table[symbol_accession_col].map(_unversioned_accession),
            "gene_symbol": symbol_table[symbol_col].astype(str).str.strip(),
            "entrez_id": (
                symbol_table[entrez_col].astype(str).str.strip() if entrez_col is not None else ""
            ),
        }
    )
    lookup = lookup.loc[lookup["gene_symbol"].ne("")].drop_duplicates("join_key")
    primary_rows = pd.DataFrame(
        {
            "probe_id": primary[probe_col].astype(str),
            "join_key": primary[primary_accession_col].map(_unversioned_accession),
        }
    )
    out = primary_rows.merge(lookup, on="join_key", how="inner").drop(columns="join_key")
    return out.loc[out["probe_id"].str.strip().ne("")].reset_index(drop=True)


def parse_geo_series_matrix(
    path: str | Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Parse probe intensities and per-sample metadata from a GEO series matrix."""
    metadata_rows: dict[str, list[str]] = {}
    characteristic_rows: list[list[str]] = []
    table_started = False
    with gzip.open(Path(path), "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                table_started = True
                break
            if not line.startswith("!Sample_"):
                continue
            key, *values = line.split("\t")
            cleaned = [value.strip().strip('"') for value in values]
            if key == "!Sample_characteristics_ch1":
                characteristic_rows.append(cleaned)
            else:
                metadata_rows[key.removeprefix("!")] = cleaned
        if not table_started:
            raise ValueError(f"{path} has no series-matrix table")
        header = [value.strip().strip('"') for value in next(handle).rstrip("\n").split("\t")]
        sample_ids = header[1:]
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{path} has empty or duplicate sample columns")
        probe_ids: list[str] = []
        values: list[list[float]] = []
        for line in handle:
            if line.startswith("!series_matrix_table_end"):
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                raise ValueError(f"{path} contains a malformed expression row")
            probe_ids.append(fields[0].strip().strip('"'))
            values.append(
                [pd.to_numeric(field.strip().strip('"'), errors="coerce") for field in fields[1:]]
            )

    matrix = pd.DataFrame(values, index=probe_ids, columns=sample_ids, dtype=float)
    matrix.index.name = "probe_id"
    sample_metadata = {sample: {} for sample in sample_ids}
    for key, row in metadata_rows.items():
        if len(row) == len(sample_ids):
            for sample, value in zip(sample_ids, row):
                sample_metadata[sample][key] = value
    for row in characteristic_rows:
        if len(row) != len(sample_ids):
            raise ValueError(f"{path} has a malformed sample-characteristics row")
        for sample, value in zip(sample_ids, row):
            if not value:
                continue
            key, separator, field_value = value.partition(":")
            metadata_key = f"char_{key.strip().lower().replace(' ', '_')}" if separator else "char"
            sample_metadata[sample][metadata_key] = (
                field_value.strip() if separator else key.strip()
            )
    return matrix, sample_metadata


def microarray_tpm_proxy(
    intensities: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    log2_transformed: bool,
) -> pd.DataFrame:
    """Collapse probes by symbol and scale each sample to a TPM-like proxy."""
    required = {"probe_id", "gene_symbol"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"microarray annotations lack columns: {missing}")
    if intensities.empty or intensities.shape[1] == 0:
        raise ValueError("microarray intensity matrix is empty")
    if not np.isfinite(intensities.to_numpy(dtype=float)).all():
        raise ValueError("microarray intensity matrix contains non-finite values")

    joined = intensities.join(
        annotations[["probe_id", "gene_symbol"]].set_index("probe_id"),
        how="inner",
    )
    if joined.empty:
        raise ValueError("microarray probes do not overlap the platform annotations")
    by_symbol = joined.groupby("gene_symbol", sort=False).max()
    linear = np.power(2.0, by_symbol) if log2_transformed else by_symbol.clip(lower=0.0)
    totals = linear.sum(axis=0)
    if (totals <= 0).any():
        failed = totals.index[totals <= 0].astype(str).tolist()
        raise ValueError(f"microarray samples have no positive signal: {failed}")
    return linear.div(totals, axis=1) * 1_000_000.0


def microarray_routed_samples(
    sample_metadata: Mapping[str, Mapping[str, str]],
    sample_patterns: Mapping[str, str | None],
) -> dict[str, list[str]]:
    """Apply mutually exclusive cancer-code regexes to sample metadata."""
    routed = {str(code): [] for code in sample_patterns}
    for sample_id, fields in sample_metadata.items():
        matched_codes = [
            str(code)
            for code, pattern in sample_patterns.items()
            if pattern is None or any(re.search(pattern, str(value)) for value in fields.values())
        ]
        if len(matched_codes) > 1:
            raise ValueError(
                f"microarray sample {sample_id!r} matches multiple codes: {matched_codes}"
            )
        if matched_codes:
            routed[matched_codes[0]].append(str(sample_id))
    return routed


def build_geo_microarray_group(
    group: GeoMicroarrayGroup,
    *,
    cache_dir: str | Path,
    output_dir: str | Path | None = None,
    series_matrix_path: str | Path | None = None,
    platform_path: str | Path | None = None,
    symbol_platform_path: str | Path | None = None,
    force_download: bool = False,
    ensembl_release: int = 112,
    high_expression_threshold: float = 1.0,
) -> SourceMatrixBuildResult:
    """Build one GEO microarray group through the shared canonical pipeline."""
    entry = _registry_entry(group.source_id)
    cache = Path(cache_dir)
    series = (
        Path(series_matrix_path)
        if series_matrix_path is not None
        else _download(
            geo_series_matrix_url(group.accession),
            cache / f"{group.accession}_series_matrix.txt.gz",
            force=force_download,
        )
    )
    platform = (
        Path(platform_path)
        if platform_path is not None
        else _download(
            geo_platform_url(group.platform_id),
            cache / f"{group.platform_id}.platform_table.txt",
            force=force_download,
        )
    )
    symbol_platform = None
    if group.symbol_platform_id is not None:
        symbol_platform = (
            Path(symbol_platform_path)
            if symbol_platform_path is not None
            else _download(
                geo_platform_url(group.symbol_platform_id),
                cache / f"{group.symbol_platform_id}.platform_table.txt",
                force=force_download,
            )
        )

    intensities, metadata = parse_geo_series_matrix(series)
    annotations = (
        parse_geo_platform_annotation_bridge(platform, symbol_platform)
        if symbol_platform is not None
        else parse_geo_platform_annotations(platform)
    )
    proxy = microarray_tpm_proxy(
        intensities,
        annotations,
        log2_transformed=group.log2_transformed,
    )
    raw = proxy.reset_index().rename(columns={"gene_symbol": "source_symbol"})
    value_cols = [str(column) for column in proxy.columns]
    _, diagnostics = coerce_source_expression_values(
        raw,
        value_cols=value_cols,
        row_id_col="source_symbol",
    )
    matrix, audit = canonicalize_source_gene_matrix(
        raw,
        row_id_col="source_symbol",
        value_cols=value_cols,
        high_expression_threshold=high_expression_threshold,
    )
    routed = microarray_routed_samples(metadata, group.sample_patterns)
    source = GeoMatrixSource(
        cancer_code=list(group.sample_patterns),
        source_cohort=str(entry["source_cohort"]),
        source_project=str(entry.get("source_project") or "GEO"),
        citation=str(entry.get("citation") or ""),
        file_name=series.name,
        unit="TPM",
        expected_source_samples=len(intensities.columns),
        expected_samples_by_code=group.expected_samples,
        source_scale_class="microarray_tpm_proxy",
        linear_tpm_comparable=False,
        tpm_proxy=True,
        native_unit=str(entry.get("unit") or "TPM proxy"),
        notes=(
            f"{group.platform_name} ({group.platform_id}) microarray-derived TPM proxy. "
            "Preserves within-sample gene rank but is not directly comparable to RNA-seq TPM. "
            f"{entry.get('special_handling') or ''}"
        ).strip(),
        processing_pipeline=(
            f"{group.platform_id.lower()}_microarray_tpm_proxy_ensembl"
            f"{ensembl_release}_clean_tpm_16_9_75"
        ),
        tumor_origin=str(entry.get("tumor_origin") or "primary"),
        source_type=str(entry.get("source_type") or "geo-microarray"),
    )
    return build_canonical_source_matrices(
        source,
        matrix,
        routed_samples=routed,
        output_dir=Path(output_dir) if output_dir is not None else cache / "derived",
        mapping_audit=audit,
        parse_diagnostics=diagnostics,
    )
