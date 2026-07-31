# oncoref

[![Tests](https://github.com/pirl-unc/oncoref/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/oncoref/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/oncoref.svg)](https://pypi.org/project/oncoref/)

Curated cancer reference data — cancer-type ontology, tumor mutational burden
(TMB), incidence/mortality, checkpoint-inhibitor (ICI) response, per-cohort
RNA-seq expression, Human Protein Atlas (HPA) normal-tissue expression, and
HPA-derived cancer-testis antigen references — behind one small Python API, a
data fetch/cache CLI, and a set of reference plots.

## Role in the stack

The openvax/PIRL tools are split by ownership boundary, not by file format.
`oncoref` is the base layer: downstream packages can depend on it, but it never
imports its consumers.

- **oncoref** is the empirical base: what is true, measured, or canonically named
  about cancer types, cohorts, genes, and reference datasets. It owns gene
  identity and canonicalization, the cancer-type ontology/registry, expression
  reference data and normalization, epidemiology, TMB, ICI/anti-PD-1 response, and
  source-anchored cancer-testis antigen (CTA) facts. If a row has an `n`, a
  confidence interval, a source cohort, or a PMID/DOI anchoring a measurement,
  it is usually an oncoref fact.
- [**pirlygenes**](https://github.com/pirl-unc/pirlygenes) owns curated gene
  sets and panels: which genes are useful for a purpose. That includes
  lineage/family/compartment/supertype panels, discriminators, surfaceome, tumor
  microenvironment (TME) and stem-cell markers, response-signature panels,
  target-to-therapy registries, and other opinionated selections keyed to
  oncoref cancer codes and gene IDs. An empty set can be a valid pirlygenes
  answer.
- [**trufflepig**](https://github.com/pirl-unc/trufflepig) owns per-sample
  interpretation: quality-control (QC) narration, library-prep/source warnings,
  deconvolution, scoring, and rule tables that fire against one tumor sample.

Adoption is staged: consumers can delegate parity-clean primitives while
retaining their own curated artifacts and compatibility APIs. Shared facts and
identifiers should be fixed here; purpose-specific panels and per-sample rules
stay downstream.

## Core model

- **Canonical identities first.** Cancer facts key on the cancer-type registry;
  expression cohorts and evidence scopes are explicit rather than inferred from
  names. Gene APIs resolve to a canonical Ensembl space.
- **Small facts in the wheel, large matrices on demand.** Ontology, gene,
  burden, response, and provenance tables install with the package. The large
  expression bundle and HPA datasets use versioned download caches.
- **Provenance is part of the result.** Reference rows expose source scope,
  sample counts, versions, and review status. Missing or ineligible data stays
  distinguishable from a valid empty result.
- **Consumer policy stays downstream.** oncoref exposes HPA-derived CTA calls
  and empirical expression facts; it does not turn them into therapy panels or
  one-sample decisions.

## Install

```bash
pip install oncoref
```

Optional integrations are explicit extras:

```bash
pip install 'oncoref[genome]'  # pyensembl-backed transcript and gene lookup
pip install 'oncoref[plots]'   # reference plotting
```

## Quick start

The flat `oncoref` namespace remains available for compatibility and quick
interactive use. For new code, prefer the semantic submodules in
the [API guide](https://pirl-unc.github.io/oncoref/api/); they make it clearer
whether you are working with the cancer ontology, cohorts, ICI response, CTA
coverage, generic antigen-panel coverage, or CTA-specific peptides.

```python
import oncoref as od

od.resolve_cancer_type("prostate")        # -> "PRAD"
od.cancer_type_info("SARC_RMS_ARMS")      # full registry record + burden + tmb
od.cancer_tmb("LUAD_EGFR")                # 6.9  (inherited from LUAD)
od.cancer_burden("pancreas", metric="us_mortality_pct")
od.burden_category("SARC_OS")             # -> "bone_and_joint" (incidence/mortality bucket)
od.cancer_ici_response("SKCM")            # 42% objective response rate
od.cancer_ici_response("SKCM", regimen="PD-1+CTLA-4")   # 57.6  (pin a regimen)

# Cancer-testis antigens (HPA-derived tissue-restriction):
od.cta_gene_names()                       # expressed CTA symbols (MAGEA4, CT83, …)
od.cta_evidence()                         # full HPA restriction table
od.cta_clinical_target_evidence()         # explicit clinical/canonical tier + leak flags

# Per-cohort expression percentiles (downloads the data bundle on first use):
od.cohort_gene_percentiles("PRAD")        # per-gene p0…p100 vector (within-cohort)
od.within_sample_top_fraction("PRAD")     # per-gene frac of samples top-5% (within-sample)
```

## Domain map

| Question | Preferred modules |
| --- | --- |
| What does this cancer code mean? | `oncoref.cancer_ontology`, `oncoref.cohorts` |
| What expression reference is available? | `oncoref.expression`, `oncoref.source_matrices` |
| How is expression normalized or filtered? | `oncoref.normalization`, `oncoref.gene_families` |
| What is the canonical gene identity? | `oncoref.gene_ids`, `oncoref.genome`, `oncoref.proteoforms` |
| What is the TMB, burden, or ICI response evidence? | `oncoref.tmb`, `oncoref.incidence`, `oncoref.ici_response` |
| Which genes meet the HPA-derived CTA definition? | `oncoref.cta`, `oncoref.cta_coverage`, `oncoref.cta_peptides` |
| Where is a dataset cached or downloaded? | `oncoref.catalog`, `oncoref.data_bundle`, `oncoref.hpa` |

The [API guide](https://pirl-unc.github.io/oncoref/api/) explains each domain,
its provenance contract, and the distinction between reader, builder, and
compatibility APIs.

## Command line

```bash
oncoref cancer-type prostate     # registry info as JSON
oncoref tmb LUAD_EGFR            # 6.9
oncoref ici SKCM                # 42  (--regimen to pin, --all-regimens to compare)
oncoref burden pancreas --metric us_mortality_pct
oncoref cta --count             # number of expressed CTAs
oncoref plot apd1-vs-tmb --out apd1_vs_tmb.png
oncoref plot patient-coverage --gene-set cta --out coverage_out
oncoref plot cta-curation --out cta_curation_out
```

### Data cache

```bash
oncoref data list               # every wheel/bundle/HPA/source dataset
oncoref data status bundle      # expression-bundle cache state (no download)
oncoref data metadata           # package/data/cache/release contract JSON
oncoref data fetch bundle       # download the large expression bundle
oncoref data fetch hpa          # HPA RNA / immunohistochemistry / single-cell data
oncoref data dir bundle         # where the data bundle is cached
oncoref data prune --yes        # delete stale bundle version caches
oncoref version
```

## Development

```bash
./develop.sh   # editable install with dev extras
./format.sh    # ruff format
./lint.sh      # ruff check + format --check
./test.sh      # lint + the full pytest suite with coverage and two workers
```

## License

Apache 2.0.
