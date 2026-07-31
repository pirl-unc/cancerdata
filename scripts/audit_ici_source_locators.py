#!/usr/bin/env python3
"""Anchor ICI estimate rows to locations in their cited source documents.

The audit is intentionally conservative. It downloads one document at a time,
extracts labeled tables, figures, result sections, abstracts, or
ClinicalTrials.gov outcome modules, and scores those blocks against the
structured endpoint row. A row is only marked ``verified`` when its existing
``source_verified`` flag is true and the source block contains the endpoint
plus row-specific numeric evidence.

Downloaded source documents are gzip-compressed in the cache and are never
loaded as a corpus. This keeps peak memory bounded by one article.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import time
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESTIMATES = ROOT / "oncoref" / "data" / "cancer-ici-response-estimates.csv"
DEFAULT_AUDIT = ROOT / "oncoref" / "data" / "cancer-ici-source-locator-audit.csv"
DEFAULT_CACHE = Path.home() / ".cache" / "oncoref" / "ici-source-locator-audit"

USER_AGENT = "oncoref-ici-source-audit/1.0 (https://github.com/pirl-unc/oncoref)"
AUDIT_COLUMNS = (
    "estimate_id",
    "ref",
    "source_document_url",
    "source_document_kind",
    "source_locator",
    "source_locator_status",
    "locator_match_basis",
    "match_score",
    "audited_on",
)

METRIC_LABELS = {
    "ORR": (
        "Overall response rate (ORR)",
        ("overall response rate", "objective response rate", "orr"),
    ),
    "IRORR": (
        "Immune-related overall response rate (iORR)",
        ("immune-related overall response", "immune response rate", "iorr"),
    ),
    "UNCONFIRMED_ORR": (
        "Best response rate including unconfirmed responses",
        ("best response", "unconfirmed response", "tumor regression", "tumour regression"),
    ),
    "CRR": (
        "Complete response rate (CRR)",
        ("complete response rate", "complete response", "crr"),
    ),
    "PR": ("Partial response rate (PR)", ("partial response rate", "partial response", "pr")),
    "PRR": (
        "Partial response rate (PRR)",
        ("partial response rate", "partial response", "prr"),
    ),
    "PR_RATE": (
        "Partial response rate",
        ("partial response rate", "partial response", "pr"),
    ),
    "DCR": (
        "Disease control rate (DCR)",
        ("disease control rate", "disease control", "dcr"),
    ),
    "CBR": (
        "Clinical benefit rate (CBR)",
        ("clinical benefit rate", "clinical benefit", "cbr"),
    ),
    "IRCBR": (
        "Immune-related clinical benefit rate (iCBR)",
        ("immune-related clinical benefit", "immune clinical benefit", "icbr"),
    ),
    "CLINICAL_BENEFIT_RATE": (
        "Clinical benefit rate",
        ("clinical benefit rate", "clinical benefit"),
    ),
    "DURABLE DISEASE CONTROL": (
        "Durable disease control",
        ("durable disease control",),
    ),
    "NON_PROGRESSION_RATE": (
        "Non-progression rate",
        ("non-progression rate", "nonprogression rate"),
    ),
    "PFS": ("Median progression-free survival (PFS)", ("progression-free survival", "pfs")),
    "PFS_HR": (
        "Progression-free survival hazard ratio (PFS HR)",
        ("progression-free survival", "hazard ratio", "pfs", "hr"),
    ),
    "PFS_RATE": ("Progression-free survival rate", ("progression-free survival", "pfs")),
    "OS": ("Median overall survival (OS)", ("overall survival", "os")),
    "OS_RATE": ("Overall survival rate", ("overall survival", "os")),
    "DOR": ("Duration of response (DOR)", ("duration of response", "dor")),
    "DOR_RATE": ("Duration-of-response rate", ("duration of response", "dor")),
    "TTR": ("Time to response (TTR)", ("time to response", "ttr")),
    "TIME_TO_RESPONSE": ("Time to response", ("time to response",)),
    "MEDIAN_FOLLOW_UP": ("Median follow-up", ("median follow-up", "median follow up")),
    "MARROW_CR": ("Bone-marrow complete response", ("marrow complete response",)),
    "HEMATOLOGIC_IMPROVEMENT": (
        "Hematologic improvement",
        ("hematologic improvement", "haematological improvement"),
    ),
    "CYTOGENETIC_RESPONSE": ("Cytogenetic response", ("cytogenetic response",)),
    "STABLE_DISEASE": ("Stable disease", ("stable disease",)),
    "SD": ("Stable disease", ("stable disease",)),
    "PROGRESSIVE_DISEASE": ("Progressive disease", ("progressive disease",)),
    "PR_COUNT": ("Partial-response count", ("partial response",)),
    "TUMOR_SHRINKAGE": ("Tumor shrinkage", ("tumor shrinkage", "tumour shrinkage")),
}

PROPORTION_METRICS = frozenset(
    {
        "CBR",
        "CLINICAL_BENEFIT_RATE",
        "CRR",
        "DCR",
        "DURABLE DISEASE CONTROL",
        "IRCBR",
        "IRORR",
        "NON_PROGRESSION_RATE",
        "ORR",
        "PR",
        "PR_RATE",
        "PRR",
        "UNCONFIRMED_ORR",
    }
)

# Corrections found while checking the legacy extraction against public source
# records. Keeping them in one table makes every exception reviewable and keeps
# the uniform provenance pass below free of row-specific branches.
ESTIMATE_CORRECTIONS: dict[str, dict[str, object]] = {
    "ICI-4398a82d1e-01": {
        "value": 3.7,
        "value_status": "numeric",
        "note": "2/54 complete responses; rate computed from the reported count.",
    },
    "ICI-143d70f5e9-01": {
        "value": 18.5,
        "value_status": "numeric",
        "note": "10/54 partial responses; rate computed from the reported count.",
    },
    "ICI-7be940d63d-01": {
        "metric": "PFS_HR",
        "value": 1.03,
        "value_status": "numeric",
        "unit": "hazard_ratio",
        "ci_low": 0.63,
        "ci_low_status": "numeric",
        "ci_high": 1.67,
        "ci_high_status": "numeric",
        "ci_basis": "reported",
        "note": "PD-L1-positive population; atezolizumab monotherapy versus sunitinib.",
    },
    "ICI-b71d41ed18-01": {
        "metric": "PFS_HR",
        "value": 0.64,
        "value_status": "numeric",
        "unit": "hazard_ratio",
        "ci_low": 0.38,
        "ci_low_status": "numeric",
        "ci_high": 1.08,
        "ci_high_status": "numeric",
        "ci_basis": "reported",
        "note": "PD-L1-positive population; atezolizumab plus bevacizumab versus sunitinib.",
    },
    "ICI-25b2af6aac-01": {
        "value_status": "not_verified",
        "source_verified": False,
        "note": (
            "The legacy note approximated an OAK all-comer ORR as 14%, but the "
            "numeric value and denominator were not re-confirmed in the cited source."
        ),
    },
    "ICI-784f9b8f57-01": {
        "value": 20.2,
        "value_status": "numeric",
        "ci_low": 14.8,
        "ci_low_status": "numeric",
        "ci_high": 25.8,
        "ci_high_status": "numeric",
        "ci_basis": "reported",
        "note": (
            "Median OS 20.2 months (90% CI 14.8-25.8), reported in the efficacy "
            "results and Supplementary Figure 2."
        ),
    },
    "ICI-de9e5a30cc-01": {
        "value_status": "not_reported",
        "note": (
            "The source reports individual durable outcomes but does not report a "
            "median duration of response."
        ),
    },
    "ICI-2c3303896a-01": {
        "value_status": "not_verified",
        "note": (
            "The legacy median DOR could not be confirmed in the primary source; "
            "earlier conference and press materials report different values."
        ),
    },
    "ICI-ca26aa02c1-01": {
        "value_status": "not_reported",
        "note": (
            "The source reports individual response durations of 5, 7, 12+, and "
            "13+ months, not a cohort median DOR."
        ),
    },
    "ICI-7c1d85c64f-01": {
        "value_status": "not_estimable",
        "source_verified": False,
        "note": (
            "KEYNOTE-051 reports one partial response in epithelioid sarcoma but "
            "does not give a histology-specific denominator, so ORR is not estimable."
        ),
    },
    "ICI-cb72c7e2fd-01": {
        "metric": "CBR",
        "note": (
            "The conference report describes approximately 50% clinical benefit; "
            "the legacy DCR label was not the named endpoint."
        ),
    },
    "ICI-c8a3a23b71-01": {
        "metric": "CBR",
        "note": (
            "Clinical benefit (CR, PR, or stable disease over 6 months) was "
            "reported for 10/32 patients (31%)."
        ),
    },
    "ICI-727d3fda5e-01": {
        "metric": "UNCONFIRMED_ORR",
        "note": (
            "Best response in 4/7 patients (2 confirmed and 2 unconfirmed partial "
            "responses); this is not an immune-related response criterion endpoint."
        ),
    },
    "ICI-2e11ae750c-01": {
        "responders": 30,
        "note": (
            "Disease control is computed as 12 objective responses plus 18 "
            "patients with stable disease among 98 treated patients."
        ),
    },
    "ICI-8e49d3b5f6-01": {
        "metric": "DURABLE DISEASE CONTROL",
        "source_endpoint_label": "Durable disease control at 6 months",
        "note": (
            "Stable disease for at least 6 months in 5/56 patients (9%); this is "
            "durable disease control, not conventional CR+PR+SD disease control."
        ),
    },
}

VALUE_BASIS_CORRECTIONS = {
    # The source gives outcomes from which the value can be calculated, but does
    # not report the named endpoint for this cohort.
    "ICI-9459107c19-01": "inferred_from_outcomes",
    "ICI-d9f97a062f-01": "inferred_from_outcomes",
    "ICI-9374924e16-01": "inferred_from_outcomes",
    # This combines two CheckMate 908 treatment arms and is audit context, not a
    # directly reported single-cohort endpoint.
    "ICI-e045c0d841-01": "derived_cross_cohort",
    # These standard rates are calculated from source-reported response counts.
    "ICI-49ee9f3ea2-01": "computed_from_counts",
    "ICI-244928197d-01": "computed_from_counts",
    "ICI-0643a1207d-01": "computed_from_counts",
    "ICI-2e11ae750c-01": "computed_from_counts",
    "ICI-e2003093ee-01": "computed_from_counts",
}

SOURCE_VERIFICATION_CORRECTIONS = {
    # These values came from secondary coverage and were not found in the cited
    # primary article during this audit.
    "ICI-14ea45d388-01": False,
    "ICI-0c6031d882-01": False,
}

CURATED_LOCATOR_OVERRIDES = {
    "ICI-784f9b8f57-01": "Supplementary Figure 2",
}

STOPWORDS = frozenset(
    {
        "advanced",
        "all",
        "analysis",
        "cancer",
        "cohort",
        "disease",
        "metastatic",
        "patients",
        "phase",
        "previously",
        "primary",
        "recurrent",
        "source",
        "treated",
        "trial",
    }
)


@dataclass(frozen=True)
class SourceBlock:
    locator: str
    text: str
    kind: str


@dataclass(frozen=True)
class SourceDocument:
    url: str
    kind: str
    blocks: tuple[SourceBlock, ...]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _normalized(value: object) -> str:
    text = str(value or "").lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return " ".join(text.split())


def _cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _fetch_cached(url: str, cache_path: Path) -> bytes:
    if cache_path.is_file():
        with gzip.open(cache_path, "rb") as f:
            return f.read()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    time.sleep(0.12)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".part")
    with gzip.open(temporary, "wb") as f:
        f.write(payload)
    temporary.replace(cache_path)
    return payload


def _id_conversion(refs: list[str], cache_dir: Path) -> dict[str, dict]:
    records = {}
    for prefix in ("PMID:", "DOI:"):
        identifiers = [ref.split(":", 1)[1] for ref in refs if ref.startswith(prefix)]
        for start in range(0, len(identifiers), 75):
            chunk = identifiers[start : start + 75]
            query = urllib.parse.urlencode(
                {
                    "format": "json",
                    "tool": "oncoref",
                    "email": "alex.rubinsteyn@unc.edu",
                    "ids": ",".join(chunk),
                }
            )
            url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?{query}"
            payload = _fetch_cached(
                url,
                cache_dir / f"idconv-{_cache_key(','.join(chunk))}.json.gz",
            )
            for record in json.loads(payload).get("records", ()):
                requested = str(record.get("requested-id", "")).lower()
                if requested:
                    records[requested] = record
    return records


def _deduplicate_blocks(blocks: list[SourceBlock]) -> tuple[SourceBlock, ...]:
    unique = []
    seen = set()
    for block in blocks:
        key = (block.locator, block.text)
        if not block.text or key in seen:
            continue
        seen.add(key)
        unique.append(block)
    return tuple(unique)


def _jats_section_blocks(section: ET.Element, path: tuple[str, ...]) -> list[SourceBlock]:
    title = _text(next((child for child in section if _tag(child) == "title"), None))
    current_path = path + ((title,) if title else ())
    blocks = []
    paragraphs = [_text(child) for child in section if _tag(child) == "p"]
    paragraph_text = " ".join(text for text in paragraphs if text)
    if paragraph_text:
        locator = " > ".join(current_path) or "Article body"
        blocks.append(SourceBlock(locator, paragraph_text, "section"))
    for child in section:
        if _tag(child) == "sec":
            blocks.extend(_jats_section_blocks(child, current_path))
    return blocks


def _pmc_document(pmcid: str, cache_dir: Path) -> SourceDocument:
    url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    query = urllib.parse.urlencode({"db": "pmc", "id": pmcid, "retmode": "xml"})
    xml_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}"
    payload = _fetch_cached(xml_url, cache_dir / f"{pmcid.lower()}.xml.gz")
    root = ET.fromstring(payload)
    blocks = []

    for abstract in root.findall(".//abstract"):
        for section in abstract.findall("./sec"):
            title = _text(section.find("./title"))
            locator = f"Abstract, {title}" if title else "Abstract"
            blocks.append(SourceBlock(locator, _text(section), "abstract"))
        if not abstract.findall("./sec"):
            blocks.append(SourceBlock("Abstract", _text(abstract), "abstract"))

    for table in root.findall(".//table-wrap"):
        label = _text(table.find("./label")) or "Table"
        blocks.append(SourceBlock(label, _text(table), "table"))

    for figure in root.findall(".//fig"):
        label = _text(figure.find("./label")) or "Figure"
        caption = _text(figure.find("./caption"))
        blocks.append(SourceBlock(label, caption, "figure"))

    body = root.find(".//body")
    if body is not None:
        for section in body:
            if _tag(section) == "sec":
                blocks.extend(_jats_section_blocks(section, ()))

    kind = "pmc_full_text" if body is not None else "pmc_abstract"
    return SourceDocument(url, kind, _deduplicate_blocks(blocks))


def _pubmed_documents(pmids: list[str], cache_dir: Path) -> dict[str, SourceDocument]:
    documents = {}
    for start in range(0, len(pmids), 100):
        chunk = pmids[start : start + 100]
        query = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"})
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}"
        payload = _fetch_cached(
            url,
            cache_dir / f"pubmed-{_cache_key(','.join(chunk))}.xml.gz",
        )
        root = ET.fromstring(payload)
        for article in root.findall(".//PubmedArticle"):
            pmid = _text(article.find(".//PMID"))
            blocks = []
            for abstract in article.findall(".//Abstract/AbstractText"):
                label = abstract.attrib.get("Label") or abstract.attrib.get("NlmCategory")
                locator = f"Abstract, {label.title()}" if label else "Abstract"
                blocks.append(SourceBlock(locator, _text(abstract), "abstract"))
            documents[pmid] = SourceDocument(
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "pubmed_abstract",
                _deduplicate_blocks(blocks),
            )
    return documents


def _europe_pmc_doi_document(doi: str, cache_dir: Path) -> SourceDocument:
    query = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json", "pageSize": 1})
    api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{query}"
    payload = _fetch_cached(
        api_url,
        cache_dir / f"doi-{_cache_key(doi)}.json.gz",
    )
    results = json.loads(payload).get("resultList", {}).get("result", ())
    blocks = []
    if results:
        abstract = str(results[0].get("abstractText") or "").strip()
        if abstract:
            blocks.append(SourceBlock("Abstract", abstract, "abstract"))
    return SourceDocument(
        f"https://doi.org/{doi}",
        "doi_record",
        _deduplicate_blocks(blocks),
    )


def _flatten_json_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_json_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_json_text(item) for item in value)
    return str(value or "")


def _clinical_trial_document(nct: str, cache_dir: Path) -> SourceDocument:
    api_url = f"https://clinicaltrials.gov/api/v2/studies/{nct}"
    payload = _fetch_cached(api_url, cache_dir / f"{nct.lower()}.json.gz")
    record = json.loads(payload)
    results = record.get("resultsSection", {})
    outcome_module = results.get("outcomeMeasuresModule", {})
    blocks = []
    for outcome in outcome_module.get("outcomeMeasures", ()):
        title = str(outcome.get("title") or "Unnamed outcome").strip()
        blocks.append(
            SourceBlock(
                f"Results, Outcome Measures: {title}",
                _flatten_json_text(outcome),
                "trial_results",
            )
        )
    if results:
        blocks.append(SourceBlock("Results", _flatten_json_text(results), "trial_results"))
    return SourceDocument(
        f"https://clinicaltrials.gov/study/{nct}",
        "clinicaltrials_results",
        _deduplicate_blocks(blocks),
    )


def _source_document(
    ref: str,
    *,
    conversions: dict[str, dict],
    pubmed: dict[str, SourceDocument],
    cache_dir: Path,
) -> SourceDocument | None:
    """Load one source, preferring PMC full text over its smaller public record."""
    identifier = ref.split(":", 1)[1] if ":" in ref else ""
    conversion = conversions.get(identifier.lower(), {})
    pmcid = str(conversion.get("pmcid") or "")
    if pmcid:
        try:
            return _pmc_document(pmcid, cache_dir)
        except (OSError, ET.ParseError) as error:
            warnings.warn(
                f"could not load {pmcid} full text for {ref}; using source record: {error}",
                stacklevel=2,
            )
    if ref.startswith("PMID:"):
        return pubmed.get(
            identifier,
            SourceDocument(
                f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/",
                "pubmed_record",
                (),
            ),
        )
    if ref.startswith("DOI:"):
        return _europe_pmc_doi_document(identifier, cache_dir)
    if ref.startswith("NCT"):
        return _clinical_trial_document(ref, cache_dir)
    return None


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _number_pattern(value: object, unit: str = "") -> re.Pattern[str] | None:
    number = _number(value)
    if number is None:
        return None
    forms = {f"{number:g}"}
    if number.is_integer():
        forms.add(str(int(number)))
    alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    suffix = r"\s*%" if unit in {"percent", "rate_percent"} else ""
    return re.compile(
        rf"(?<![\d.])(?:{alternatives})(?!\d)(?!\.\d){suffix}",
        re.IGNORECASE,
    )


def _ratio_pattern(responders: object, denominator: object) -> re.Pattern[str] | None:
    numerator = _number(responders)
    total = _number(denominator)
    if numerator is None or total is None:
        return None
    n = f"{numerator:g}"
    d = f"{total:g}"
    return re.compile(
        rf"(?<!\d){re.escape(n)}\s*(?:/|of|out of)\s*{re.escape(d)}(?!\d)",
        re.IGNORECASE,
    )


def _denominator_pattern(value: object) -> re.Pattern[str] | None:
    number = _number(value)
    if number is None:
        return None
    text = re.escape(f"{number:g}")
    return re.compile(
        rf"(?:\bn\s*=\s*{text}\b|\b{re.escape(text)}\s+(?:patients|participants|cases)\b)",
        re.IGNORECASE,
    )


def _population_terms(value: object) -> tuple[str, ...]:
    words = re.findall(r"[a-z][a-z0-9+-]{3,}", _normalized(value))
    terms = []
    for word in words:
        if word in STOPWORDS or word in terms:
            continue
        terms.append(word)
    return tuple(terms[:8])


def _endpoint_label(row: pd.Series) -> str:
    metric = str(row["metric"]).strip().upper()
    label = METRIC_LABELS.get(metric, (metric.replace("_", " ").title(), ()))[0]
    timepoint = str(row.get("timepoint") or "").strip()
    if timepoint and timepoint.lower() not in {"nan", "median"}:
        return f"{label}; {timepoint}"
    return label


def _contains_alias(text: str, alias: str) -> bool:
    if re.fullmatch(r"[a-z0-9]+", alias) and len(alias) <= 5:
        return bool(re.search(rf"\b{re.escape(alias)}\b", text))
    return alias in text


def _block_is_candidate(block: SourceBlock) -> bool:
    locator = _normalized(block.locator)
    if block.kind == "abstract":
        return not any(
            part in locator
            for part in ("background", "introduction", "method", "patients and methods")
        )
    if block.kind != "section":
        return True
    excluded = (
        "background",
        "discussion",
        "introduction",
        "method",
        "statistical",
        "study design",
    )
    return not any(part in locator and "result" not in locator for part in excluded)


def _match_block(row: pd.Series, block: SourceBlock) -> tuple[int, tuple[str, ...]]:
    if not _block_is_candidate(block):
        return 0, ()
    metric = str(row["metric"]).strip().upper()
    aliases = METRIC_LABELS.get(metric, ("", (metric.lower().replace("_", " "),)))[1]
    text = _normalized(block.text)
    signals = []
    score = 0

    if any(_contains_alias(text, alias) for alias in aliases):
        signals.append("endpoint")
        score += 7

    ratio = _ratio_pattern(row.get("responders"), row.get("metric_n"))
    if ratio and ratio.search(text):
        signals.append("numerator/denominator")
        score += 8

    value = _number_pattern(row.get("value"), str(row.get("unit") or ""))
    if value and value.search(text):
        signals.append("value")
        score += 4

    value_status = str(row.get("value_status") or "")
    if value_status in {"not_reached", "not_estimable"} and re.search(
        r"\bnot (?:reached|estimable)\b|\bNR\b|\bNE\b",
        block.text,
    ):
        signals.append("value_status")
        score += 4

    denominator = _denominator_pattern(row.get("metric_n") or row.get("source_n"))
    if denominator and denominator.search(text):
        signals.append("denominator")
        score += 2

    if str(row.get("ci_basis") or "") == "reported":
        ci_low = _number_pattern(row.get("ci_low"))
        ci_high = _number_pattern(row.get("ci_high"))
        if ci_low and ci_high and ci_low.search(text) and ci_high.search(text):
            signals.append("ci")
            score += 4

    timepoint = _normalized(row.get("timepoint"))
    if timepoint and timepoint != "nan" and timepoint in text:
        signals.append("timepoint")
        score += 3

    population_hits = sum(term in text for term in _population_terms(row.get("setting")))
    if population_hits:
        signals.append(f"population:{population_hits}")
        score += min(population_hits, 3)

    if block.kind == "table":
        score += 2
    elif block.kind in {"abstract", "trial_results"}:
        score += 1

    specific = {
        "numerator/denominator",
        "value",
        "value_status",
        "ci",
        "timepoint",
    }
    if "endpoint" not in signals or not specific.intersection(signals):
        return 0, ()
    return score, tuple(signals)


def _best_match(row: pd.Series, document: SourceDocument) -> tuple[SourceBlock, int, str] | None:
    matches = []
    for position, block in enumerate(document.blocks):
        score, signals = _match_block(row, block)
        if score >= 11:
            matches.append((score, -position, block, ",".join(signals)))
    if not matches:
        return None
    score, _, block, basis = max(matches, key=lambda item: (item[0], item[1]))
    return block, score, basis


def _curated_note_locator(row: pd.Series) -> str | None:
    note = str(row.get("note") or "")
    match = re.search(
        r"(Supplementary\s+)?(Table|Figure|Fig\.)\s+([A-Z]?\d+[A-Za-z]?)(?!\s*%)",
        note,
    )
    if not match:
        return None
    prefix = "Supplementary " if match.group(1) else ""
    kind = "Figure" if match.group(2) in {"Figure", "Fig."} else "Table"
    return f"{prefix}{kind} {match.group(3)}"


def _source_section_fallback(document: SourceDocument | None) -> str:
    if document is None:
        return ""
    preferred = (
        block
        for block in document.blocks
        if _block_is_candidate(block)
        and (
            any(label in _normalized(block.locator) for label in ("finding", "outcome", "result"))
            or block.kind == "trial_results"
            or _normalized(block.locator) == "abstract"
        )
    )
    block = next(preferred, None)
    return block.locator if block else ""


def build_audit(estimates: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    refs = sorted(ref for ref in estimates["ref"].astype(str).unique() if ref)
    conversions = _id_conversion(refs, cache_dir)
    pmids = sorted(ref.split(":", 1)[1] for ref in refs if ref.startswith("PMID:"))
    pubmed = _pubmed_documents(pmids, cache_dir)
    rows_by_position = {}
    today = date.today().isoformat()
    positioned = estimates.copy()
    positioned["_audit_position"] = range(len(positioned))
    for ref, ref_rows in positioned.groupby("ref", sort=False):
        ref = str(ref or "")
        document = _source_document(
            ref,
            conversions=conversions,
            pubmed=pubmed,
            cache_dir=cache_dir,
        )
        for _, estimate in ref_rows.iterrows():
            value_basis = str(estimate["value_basis"] or "")
            source_verified = _truthy(estimate["source_verified"])
            match = _best_match(estimate, document) if document else None
            curated_locator = CURATED_LOCATOR_OVERRIDES.get(
                str(estimate["estimate_id"])
            ) or _curated_note_locator(estimate)

            if value_basis == "derived_blend":
                locator = ""
                status = "not_applicable"
                match_basis = "curator-derived value"
                score = 0
            elif curated_locator and source_verified:
                locator = curated_locator
                status = "verified"
                match_basis = "curated estimate note names the exact source table or figure"
                score = 0
            elif match:
                block, score, match_basis = match
                locator = block.locator
                status = "verified" if source_verified else "located_unverified"
            elif source_verified:
                locator = _source_section_fallback(document)
                status = "source_section" if locator else "citation_only"
                match_basis = (
                    "source citation verified; only a coarse public source section was available"
                    if locator
                    else "source citation verified; no public text block was available"
                )
                score = 0
            else:
                locator = ""
                status = "not_verified"
                match_basis = "no source block matched the endpoint and row-specific evidence"
                score = 0

            rows_by_position[int(estimate["_audit_position"])] = {
                "estimate_id": estimate["estimate_id"],
                "ref": ref,
                "source_document_url": document.url if document else "",
                "source_document_kind": document.kind if document else "missing_citation",
                "source_locator": locator,
                "source_locator_status": status,
                "locator_match_basis": match_basis,
                "match_score": score,
                "audited_on": today,
            }
        del document
    rows = [rows_by_position[position] for position in range(len(positioned))]
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def apply_estimate_corrections(estimates: pd.DataFrame) -> pd.DataFrame:
    """Apply reviewed row corrections found during the source audit."""
    output = estimates.copy()
    by_id = output.set_index("estimate_id")
    unknown = (
        set(ESTIMATE_CORRECTIONS)
        | set(VALUE_BASIS_CORRECTIONS)
        | set(SOURCE_VERIFICATION_CORRECTIONS)
        | set(CURATED_LOCATOR_OVERRIDES)
    ) - set(by_id.index)
    if unknown:
        raise ValueError(f"estimate corrections reference unknown IDs: {sorted(unknown)}")

    for estimate_id, changes in ESTIMATE_CORRECTIONS.items():
        for column, value in changes.items():
            output.loc[output["estimate_id"] == estimate_id, column] = value
    for estimate_id, value_basis in VALUE_BASIS_CORRECTIONS.items():
        output.loc[output["estimate_id"] == estimate_id, "value_basis"] = value_basis
    for estimate_id, source_verified in SOURCE_VERIFICATION_CORRECTIONS.items():
        output.loc[output["estimate_id"] == estimate_id, "source_verified"] = source_verified
    return output


def _wilson_ci(responders: object, denominator: object) -> tuple[float, float] | None:
    numerator = _number(responders)
    total = _number(denominator)
    if numerator is None or total is None or total <= 0 or not 0 <= numerator <= total:
        return None
    z = 1.96
    proportion = numerator / total
    scale = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / scale
    half_width = (
        z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / scale
    )
    low = max(0.0, 100 * (center - half_width))
    high = min(100.0, 100 * (center + half_width))
    return round(low, 1), round(high, 1)


def _explicitly_missing_ci(note: object) -> bool:
    return bool(
        re.search(
            r"\b(?:CI|confidence interval) (?:was |is )?not reported\b|"
            r"\bno source-reported (?:ORR )?CI\b",
            str(note or ""),
            flags=re.IGNORECASE,
        )
    )


def _unavailable_ci_status(row: pd.Series) -> str:
    if not _truthy(row["source_verified"]):
        return "not_verified"
    if _explicitly_missing_ci(row.get("note")) or row["source_locator_status"] == "verified":
        return "not_reported"
    return "source_unavailable"


def complete_value_and_ci_provenance(estimates: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy extraction placeholders with explicit value and CI states."""
    output = estimates.copy()
    for index, row in output.iterrows():
        value = _number(row["value"])
        value_status = str(row["value_status"] or "")
        if value is not None:
            output.at[index, "value_status"] = "numeric"
        elif value_status == "not_extracted":
            raise ValueError(f"{row['estimate_id']} still has an unresolved value")

        low = _number(row["ci_low"])
        high = _number(row["ci_high"])
        if low is not None:
            output.at[index, "ci_low_status"] = "numeric"
        if high is not None:
            output.at[index, "ci_high_status"] = "numeric"

        metric = str(row["metric"]).upper()
        can_compute = (
            metric in PROPORTION_METRICS
            and value is not None
            and ((low is None and high is None) or str(row["ci_basis"]) == "computed_wilson")
        )
        computed = _wilson_ci(row["responders"], row["metric_n"]) if can_compute else None
        if computed:
            output.at[index, "ci_low"] = computed[0]
            output.at[index, "ci_low_status"] = "numeric"
            output.at[index, "ci_high"] = computed[1]
            output.at[index, "ci_high_status"] = "numeric"
            output.at[index, "ci_basis"] = "computed_wilson"
            continue

        has_structured_ci = (
            low is not None
            or high is not None
            or str(row["ci_low_status"]) in {"NR", "NE"}
            or str(row["ci_high_status"]) in {"NR", "NE"}
        )
        if has_structured_ci:
            output.at[index, "ci_basis"] = (
                "computed_wilson" if row["ci_basis"] == "computed_wilson" else "reported"
            )

        missing_status = (
            "not_applicable"
            if str(output.at[index, "value_status"]) != "numeric"
            or str(output.at[index, "value_basis"]) == "derived_blend"
            else _unavailable_ci_status(output.loc[index])
        )
        if low is None and str(row["ci_low_status"]) not in {"NR", "NE"}:
            output.at[index, "ci_low_status"] = missing_status
        if high is None and str(row["ci_high_status"]) not in {"NR", "NE"}:
            output.at[index, "ci_high_status"] = missing_status
        if not has_structured_ci:
            output.at[index, "ci_basis"] = missing_status
    return output


def apply_audit(estimates: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if not audit["estimate_id"].is_unique:
        raise ValueError("source-locator audit estimate_id values must be unique")
    indexed = audit.set_index("estimate_id")
    missing = set(estimates["estimate_id"]) - set(indexed.index)
    if missing:
        raise ValueError(f"source-locator audit is missing {len(missing)} estimate rows")

    output = estimates.copy()
    for column in ("source_locator", "source_locator_status"):
        output[column] = output["estimate_id"].map(indexed[column])
    output["source_endpoint_label"] = output.apply(_endpoint_label, axis=1)
    output["source_population_label"] = output["setting"].fillna("").astype(str).str.strip()
    return complete_value_and_ci_provenance(output)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, default=DEFAULT_ESTIMATES)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--write-estimates",
        action="store_true",
        help="replace locator/endpoint/population fields in the estimates table",
    )
    args = parser.parse_args()

    estimates = apply_estimate_corrections(pd.read_csv(args.estimates, keep_default_na=False))
    # Compute count-derived intervals before matching so a clean first run has
    # the same numeric match signals as every later run. Locator-dependent
    # unavailable/not-reported states are finalized again in apply_audit().
    estimates = complete_value_and_ci_provenance(estimates)
    audit = build_audit(estimates, args.cache_dir)
    _write_csv(audit, args.audit_output)
    if args.write_estimates:
        _write_csv(apply_audit(estimates, audit), args.estimates)

    print(audit["source_locator_status"].value_counts().to_string())
    cache_bytes = sum(path.stat().st_size for path in args.cache_dir.glob("*.gz"))
    print(
        f"cache: {cache_bytes / 1024**2:.1f} MiB across {len(list(args.cache_dir.glob('*.gz')))} files"
    )


if __name__ == "__main__":
    main()
