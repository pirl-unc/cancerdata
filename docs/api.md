# oncoref API Guide

`oncoref` keeps its historical flat top-level imports for compatibility, but new
code should prefer the semantic submodules below. They make the domain boundary
clear and avoid guessing whether a broad name such as `coverage` or `peptides` is
general or specific to cancer-testis antigens (CTAs).

Package boundary: oncoref is the upstream home for empirical base facts and
canonical identifiers that are ready to be reused across the PIRL stack.
pirlygenes owns purpose-specific gene sets and panels; trufflepig owns
per-sample interpretation, QC narration, and rule firing. As a rule of thumb,
source-anchored measurements with denominators, confidence intervals, cohorts,
PMIDs/DOIs, or shared ontology implications belong in oncoref. Opinionated gene
selections and target-to-therapy registries belong in pirlygenes. One-sample
rules belong in trufflepig. When a missing data field, gene universe,
bundle-integrity rule, or source-QC decision affects shared reference artifacts,
the durable fix should live or be exposed here rather than only in a downstream
compatibility layer.

## Guide Map

Read the guide from concepts to operations:

| Layer | Start here |
| --- | --- |
| Canonical cancer and gene identities | [Cancer Vocabulary](#cancer-vocabulary), [Gene Identity](#gene-identity) |
| Expression reads, artifacts, and normalization | [Expression And Normalization](#expression-and-normalization) |
| Clinical and epidemiological reference facts | [ICI Response](#ici-response), [Burden, TMB, Fusions, and Signatures](#burden-tmb-fusions-and-signatures) |
| Cancer-testis antigen and panel calculations | [CTA Antigens](#cta-antigens), [Generic Antigen Panels](#generic-antigen-panels) |
| Downloads, caches, and release metadata | [Data Management](#data-management) |
| Historical import paths | [Compatibility Modules](#compatibility-modules) |

Within each section, the intended use and primary modules come first. Detailed
schema, provenance, fallback, and migration contracts follow.

## Cancer Vocabulary

- `oncoref.cancer_ontology` — cancer-type registry, aliases, parent/child tree,
  lineage/family groupings, molecular subtype axes, mismatch repair (MMR) and
  microsatellite instability (MSI) classifier-status semantics, matched normal
  tissues, source-scoped evidence resolution, and display helpers.
- `oncoref.cohorts` — expression/source cohort IDs, computed aggregate cohorts,
  source versions, and mixture-cohort flags.

Use these when asking "what cancer type or cohort does this code mean?"
Prefer the DataFrame-returning query helpers when code will be passed into
other oncoref domains; they keep the result type and columns stable.

### Ontology and category model

The registry separates hierarchy from taxonomic level. `parent_code` is the
tree edge, while `ontology_level` (`grouping`, `type`,
`molecular_subtype`, `evidence_scope`) and `ontology_kind`
(`computed_union`, `source_scope`, `anatomic_type`,
`molecular_status_subtype`, etc.) say what kind of node a row is. Do not infer
semantic level from `mixture_cohort`; that legacy flag only says the reference
cohort/source is pooled or source-scoped. For example `CRC_MSI` is a
`molecular_subtype` under `CRC` but remains a source-scope clinical evidence
row, while `OV` is an anatomical grouping and `FTC` / `PPC` are anatomical
cancer types. Pure clinical fact scopes such as `NET_NONPANCREATIC` and
`NEN_EXTRAPULMONARY_HG` use `ontology_level="evidence_scope"` so they do not
look like groupings with missing children. Differentiation and grade are
orthogonal sparse axes: use `differentiation="NEC"` for native neuroendocrine
lineage labels, or `grade_tier="high"` for normalized high-grade rows, without
treating either one as a parentless cancer type.

### Expression and classification backing

Expression/classification backing is explicit. Use `reference_source`,
`cancer_type_reference_source()`, `cancer_type_reference_code()`, or
`cancer_type_records(reference_source=...)` instead of inferring from
`mixture_cohort` or from whether a code is anatomical or molecular. The enum is
data-driven:

- `own_cohort` — this code has its own separable expression cohort.
- `member_union` — this code is backed by a union of expression-bearing member
  cohorts; reportability is controlled separately by `is_classification_target`.
- `parent` — this code carries an annotation/slice but should be reported at
  its nearest reportable ancestor.
- `none` — pure provenance or unsupported scope; walk up the tree if a coarser
  call is needed.

Classification eligibility and reference availability are separate gates.
`is_classification_target`, `classification_target_codes`, and
`cancer_type_records(classification_target=True)` preserve the reviewed owner
registry policy and additionally require `reference_source` to be `own_cohort`
or `member_union`. A reference can therefore make a reviewed target unavailable,
but adding comparison data cannot promote a validation-only cohort into a
diagnosis. CMN, for example, remains non-classifying even though its microarray
TPM proxy is returnable for marker/rank validation. `COAD_MSI`, `COAD_MSS`,
`READ_MSI`, and `READ_MSS` are `own_cohort` because the TCGA COAD/READ MSI
partitions have separable expression; `CRC_MSI` is a reviewed classification
target backed by a `member_union` over `COAD_MSI ∪ READ_MSI`, not merely an
annotation. A molecular slice falls to `parent` when oncoref has not measured a
separable cohort, as with the current STAD/UCEC molecular subtype rows.

Computed expression pools are also explicit. Use `computed_union_codes()` for
registry rows whose `expression_source="computed"` and
`reference_source_codes("member_union")` for all member-union
references, including source-scope unions such as `CRC_MSI`, `NSCLC`, and
`SGC`. SGC remains a reference-only union because its reviewed
`is_classification_target` policy is false.

Source-scope union membership is all-or-nothing. BTC declares `CHOL ∪ GBC` and
now reports a `member_union` only because both members have selected expression
matrices. Before the direct GSE139682 GBC reference was published, BTC reported
`reference_source="none"` rather than returning CHOL alone as pan-BTC data.

### Category queries

For category-aware downstream code, start with
`cancer_type_category_schema()` and `cancer_type_category_summary()`. The schema
is the compact public vocabulary for `ontology_level`, the observed
`ontology_kind` values, and `reference_source`; the summary reports counts and
example codes for every observed level/kind/reference-source combination. This
is the intended replacement for ad hoc tests like "has children", "is
mixture_cohort", or "does the code name contain MSI".

### Examples

```python
from oncoref import cancer_ontology, cohorts, expression

cancer_ontology.resolve_cancer_type("prostate")
cancer_ontology.cancer_type_tree("CRC")
cancer_ontology.cancer_type_path("COAD_MSI")

# CRC plus anatomical children and molecular leaves.
crc = cancer_ontology.cancer_type_records(under="CRC")
crc["code"].tolist()
crc[["code", "parent_code", "ontology_level", "ontology_kind"]]

# Cross-cutting molecular axes can be intersected with hierarchy or lineage.
msi_crc = cancer_ontology.cancer_type_records(subtype_group="MSI", under="CRC")
epithelial_msi = cancer_ontology.cancer_type_records(
    subtype_group="MSI", lineage_group="Epithelial"
)
source_scope_msi = cancer_ontology.cancer_type_records(
    under="CRC", ontology_level="molecular_subtype", ontology_kind="molecular_source_scope"
)
classification_targets = cancer_ontology.cancer_type_records(classification_target=True)
clinical_fact_scopes = cancer_ontology.cancer_type_records(classification_target=False)
computed_pools = cancer_ontology.computed_union_codes()
member_union_refs = cancer_ontology.reference_source_codes("member_union")
cancer_ontology.cancer_type_category_schema()
cancer_ontology.cancer_type_category_summary()
cancer_ontology.cancer_type_reference_source("CRC_MSI")
cancer_ontology.cancer_type_reference_code("STAD_MSI")

# The MMR/MSI classifier axis keeps positive, negative, and confounder classes
# explicit. STAD_MSI exists as an ontology code, but expression_only=True
# excludes it until split STAD subtype expression shards are built.
cancer_ontology.mmrd_cancer_codes()
cancer_ontology.pmmr_cancer_codes(under="CRC")
cancer_ontology.mmr_confounder_cancer_codes()
cancer_ontology.mmr_hypermutated_confounder_codes()
cancer_ontology.mmrd_cancer_codes(expression_only=True)
cancer_ontology.cancer_mismatch_repair_status("UCEC_POLE")

# COAD_MSI / READ_MSI keep anatomical expression context but resolve evidence
# rows through CRC_MSI when published sources are colorectal-level.
msi_crc[["code", "evidence_source_code", "normal_tissue_code", "hpa_tissues"]]

# Join scalar references for the returned codes.
cancer_ontology.cancer_type_reference_data(msi_crc)

# Ask whether each ontology node has a direct expression reference, a computed
# member-union reference, parent fallback, or no expression backing.
cancer_ontology.expression_reference_coverage(subtype_group="MSI", under="CRC")
cancer_ontology.coverage_for_cancer_type("ASTB")

# Use codes directly with expression accessors.
codes = cancer_ontology.cancer_type_codes(subtype_group="MSI", under="CRC")
expression.cancer_reference_expression(codes)

# Matched normal RNA expression is an explicit HPA read.
cancer_ontology.matched_normal_tissue_expression("COAD", genes=["ENSG00000141510"])

cohorts.cohort_registry_df()
```

### Cohort sample counts

`cohort_registry_df()` describes physical source cohorts. For a non-computed
cohort, `n_samples` is the number of matrix rows before diagnosis or histology
routing, while `n_codes` is the number of canonical cancer codes receiving
samples from that source. Per-code routed counts are available from
`cancer_reference_expression_availability(..., reference_source="summary_rows_all",
sample_qc="all", all_sources=True)`.

For example, `GSE294016_BARTL_2025_SGC` contains 95 physical matrix rows and
feeds two cancer codes. Its released histology-specific references contain 57
ADCC samples and 3 ACINIC samples; the other 35 source rows are different
histologies and are deliberately excluded from those two references.

### Consumer readiness

`expression_reference_coverage()` is the ontology-wide readiness table for
classifier consumers. It distinguishes direct observed-bulk source matrices
from single-cell donor pseudobulks and microarray proxies,
computed member-union references for curated grouping/source-scope codes such as
`NET`, `CRC`, `CRC_MSI`, `NSCLC`, and `SGC`, parent fallback via
`classification_reference_code`, explicit `is_classification_target` eligibility,
matched normal tissue availability,
molecular/fusion-only definitions, canonical gene/proteoform space, data/source
matrix versions, and a conservative `consumer_recommendation`:
`direct_reference`, `computed_reference`, `reference_only`, `parent_reference`,
`molecular_only`, or `unsupported`. SGC is `reference_only`: its histology-member
union remains available for comparison, but the source/therapy grouping is not a
valid final classification label.
`has_direct_expression_reference` remains literal; computed groupings use
`expression_reference_kind="computed_union"` and expose their pooled member codes
in `computed_expression_member_codes`. HCL is a concrete example of the
difference between availability and classification eligibility: its five-donor
T0 malignant-cell pseudobulk is a direct reference with
`expression_reference_kind="single_cell_pseudobulk"`, but the non-comparable
nTPM proxy remains `reference_only` and has no `classification_reference_code`.
EPN follows the same consumer contract: its 11 diagnosis-stage, patient-level
malignant-cell mean-TPM pseudobulks are direct reference-only profiles, not
bulk-TPM-comparable classifier targets.
CRANIO is likewise reference-only: OpenPBTA release v23 contributes 29
independent pediatric primary tumors from its stranded RSEM-TPM matrix, including
20 harmonized adamantinomatous tumors and nine tumors not molecularly classified.
The cohort contains no papillary tumors, so oncoref does not infer CTNNB1, BRAF,
or papillary status and does not treat it as independently classification-ready.
DIPG adds 32 independent initial solid-tumor donors from the OpenPBTA poly-A and
stranded RSEM-TPM matrices. Every included profile has both integrated and
harmonized H3 K28-mutant source diagnoses—the OpenPBTA label corresponding to the
canonical H3 K27-altered entity. Unannotated, H3-wild-type, IDH-mutant,
non-initial, and duplicate-donor profiles remain explicit exclusions; the mixed
library preparation keeps the cohort reference-only.
The table intentionally does not
synthesize marker-program or discriminator fallbacks; those remain consumer-layer
choices in packages such as trufflepig.

## Gene Identity

- `oncoref.gene_ids` — canonical ENSG space, alt-haplotype / retired Ensembl ID
  migration, symbol/synonym resolution, and report-facing gene labels.
- `oncoref.genome` — optional (`pip install 'oncoref[genome]'`) pyensembl-backed
  transcript/gene lookup and transcript-to-gene aggregation for source matrices.

### Resolution contract

Use the gene-id helpers before building expression artifacts or joining
downstream gene sets to oncoref references. `canonical_gene_id()` is the primary
any-identifier entry point for the shipped ENSG + symbol/synonym space: it
normalizes versioned or case-varied Ensembl gene IDs, follows retired/alt ENSG
aliases into the canonical space, resolves symbols and synonyms, and returns
`None` for inputs that cannot be mapped to a canonical oncoref gene.
`canonical_gene_symbol()`, `display_gene_name()`, and `short_gene_name()` use
the same resolver so report code does not invent a separate symbol mapping.
`entrez_gene_mappings()` and `resolve_entrez_id()` expose the filtered NCBI
Entrez/GeneID table used by the resolver; it covers live IDs from NCBI dbXrefs
or current symbols plus discontinued IDs redirected through NCBI gene_history.
`gene_identifier_mapping_coverage()` and
`gene_identifier_mapping_summary()` make the shipped ENSG, symbol/synonym, and
Ensembl-alias coverage explicit for migration audits, including non-unique
symbols and missing-symbol rows. They do not claim that RefSeq or UniProt
coverage is complete.

### Examples

```python
from oncoref import canonical_gene_id, canonical_gene_symbol, display_gene_name, gene_ids

canonical_gene_id("GNB2L1")        # previous symbol -> ENSG00000204628
canonical_gene_id("7157")          # Entrez/GeneID -> ENSG00000141510
canonical_gene_symbol("GNB2L1")    # previous symbol -> RACK1
display_gene_name("ENSG00000005955")  # retired Ensembl id -> GGNBP2
gene_ids.gene_identifier_mapping_summary()
```

## ICI Response

- `oncoref.ici_response` — checkpoint-inhibitor response anchors, anti-PD-1
  shortcuts, regimen-aware lookups, extracted objective response rate (ORR)
  estimates, and pooled response summaries.

### Regimen selection

`DEFAULT_ICI_REGIMEN_PRIORITY` is the unpinned regimen priority
(`PD-1`, then `PD-L1`, then `PD-1+CTLA-4`). The older
`REGIMEN_FALLBACK` name remains available in `oncoref.ici` for compatibility.

### Examples

```python
from oncoref import ici_response

ici_response.apd1_response("SKCM")
ici_response.best_available_ici_response("SARC_ASPS")
ici_response.ici_response_by_regimen("SKCM")
ici_response.ici_response_estimates_df()
ici_response.ici_source_locator_audit_df()
```

### Evidence and source audit

`ici_response_estimates_df()` is the auditable long table behind the compact
ORR anchors. Each row has a stable `estimate_id`; compact
`ici_response_record(...)` / `apd1_response_df()` rows expose that pointer as
`source_estimate_id`. `ici_source_locator_audit_df()` has exactly one row per
`estimate_id` and records the public source URL, document kind, table/figure/section
locator, match evidence, and audit date.

Use the estimates table for analysis and the locator table to inspect how an
estimate was checked. `source_endpoint_label` and `source_population_label` are
normalized oncoref labels for comparison; they are not represented as verbatim
quotes from the paper.

`source_locator_status` distinguishes:

- `verified`: the endpoint and row-specific numeric evidence matched a source block,
  or the extraction note named the exact table or figure.
- `source_section`: the cited evidence was previously verified and a real public
  results/abstract section is available, but the audit did not recover a more exact
  numeric block.
- `citation_only`: the cited evidence was previously verified, but the public source
  record exposed no usable text block. The locator is intentionally blank.
- `located_unverified`: a block matched, but the estimate remains explicitly
  unverified because its population or value could not be confirmed.
- `not_verified`: no supporting source block was confirmed.
- `not_applicable`: a curator-derived value has no single source location.

`ci_basis` distinguishes source-reported intervals from 95% `computed_wilson`
intervals, `not_reported`, `source_unavailable`, `not_verified`, and
`not_applicable`. Reported intervals retain source-specific levels such as 80% or
90% in their extraction notes; they must not be assumed to be uniformly 95%.
`ci_low_status` and `ci_high_status` preserve numeric, `NR`, and `NE` bounds
independently. `value_status` likewise distinguishes numeric, not-reached,
not-estimable, not-reported, and unverified values. There are no remaining legacy
`not_extracted` states in these fields.

`value_basis` controls interpretation and pooling:

- `reported` is a value reported for the named source population.
- `computed_from_counts` is calculated from source-reported response counts.
- `inferred_from_outcomes` is implied by reported outcomes but was not a named
  endpoint; it is excluded from pooling.
- `derived_cross_cohort` combines source cohorts or treatment arms and is excluded
  from pooling.
- `reported_context` preserves an overlapping subgroup or comparator and is excluded
  from pooling.
- `derived_blend` is a curator-modeled value without a single trial estimate and is
  excluded from pooling.

The audit can be reproduced with:

```bash
python scripts/audit_ici_source_locators.py --write-estimates
```

The script retrieves one public source document at a time and stores compressed
cache entries under `~/.cache/oncoref/ici-source-locator-audit`, keeping the source
corpus out of memory.

## CTA Antigens

A cancer-testis antigen (CTA) is encoded by a gene that is normally restricted
to reproductive tissues but can be reactivated in tumors. In oncoref, a
candidate is called a CTA by an explicit Human Protein Atlas (HPA) normal-tissue
expression rule; the call is not evidence that its antigen is presented by the
major histocompatibility complex (MHC) or that it is a validated therapy target.

CTA identity and tumor coverage answer different questions. The HPA-derived CTA
definition determines which genes are in the reference set. Tumor expression
then asks how often those CTAs are active in a cohort. An absolute threshold such
as 50 clean TPM compares expression magnitudes. A within-sample p90 or p95
threshold instead ranks the complete biological transcriptome inside each tumor:
p90 means the CTA is in that tumor's top 10% of expression values, and p95 means
the top 5%. Rank thresholds are appropriate when source scales are not directly
comparable.

Patient coverage requires the joint per-sample matrix. It is the fraction of
patients with at least one positive CTA, counting a patient once even when
several CTAs are positive. It cannot be recovered by adding per-gene prevalence
or taking the largest prevalence value. `cta_within_sample_percentile_coverage()`
retains this co-occurrence and returns a deterministic greedy antigen order for
p90/p95; `cta_within_sample_percentile_addressable_fraction_by_cohort()` returns
the final true patient union. Both use biological clean TPM and collapse
identical-protein CTA loci before ranking by default. They use QC-passing samples
by default; pass `sample_qc="all"` only for an explicit forensic view.

### Restriction synthesis

`synthesize_restriction()` prefers the HPA protein restriction when protein data
exist and otherwise uses the RNA restriction. Confidence increases when the
modalities agree. The broad RNA call `REPRODUCTIVE` supports `TESTIS`,
`PLACENTAL`, or `REPRODUCTIVE` protein calls; it never supports a `SOMATIC`
protein call.

- `oncoref.cta` — CTA definition, HPA restriction tiers, axes, aliases, and gene
  ID/name sets. Strict helpers such as `cta_gene_names()` and
  `cta_filtered_gene_names()` preserve the HPA reproductive-restriction default;
  `cta_clinical_target_evidence()` exposes a separate clinical/canonical tier for
  source-anchored CTA targets that may be strict-pass, HPA-excluded, or
  candidate-only. `cta_specificity_audit()` exposes machine-readable specificity
  demotion and candidate-only decisions for genes whose normal-tissue evidence
  makes strict-default inclusion unsafe or unresolved.
- `oncoref.cta_coverage` — CTA patient coverage over per-sample expression
  matrices.
- `oncoref.cta_peptides` — CTA-specific 9-mer counts and load.

`cta_specific_9mer_count_map()` returns a map from a join key to
`n_specific_9mers`; those counts are used as weights when computing
`cta_specific_9mer_load()`.

Broader therapy-target curation, mass-spectrometry evidence, and downstream
prioritization rules can live in consumer packages while they remain
package-specific.

```python
from oncoref import cta, cta_coverage, cta_peptides

cta.cta_gene_names()
cta.cta_clinical_target_evidence()
cta.cta_specificity_audit()
cta_coverage.cta_addressable_fraction("LUAD")
coverage = cta_coverage.cta_within_sample_percentile_coverage(
    ["LUAD", "SKCM"], percentiles=(0.90, 0.95)
)
cta_coverage.cta_within_sample_percentile_addressable_fraction_by_cohort(
    ["LUAD", "SKCM"], percentile=0.95, coverage=coverage
)
cta_peptides.cta_specific_9mer_count_map(by="proteoform_key")
```

When `cohorts` is omitted, the percentile-coverage APIs inspect only cached
per-sample matrices and do not download data. Use
`locally_available_percentile_cohorts()` and
`locally_available_within_sample_cohorts()` to plan local work across both
package/artifact data and a partial bundle cache. Set `include_recomputable=False`
when only already-built shards should count.

## Generic Antigen Panels

- `oncoref.antigen_coverage` — coverage helpers for caller-supplied gene lists.

Use this when the panel is not necessarily CTA. The function names require
`gene_ids=` so a caller cannot accidentally rely on the CTA default. This module
computes coverage for a supplied list; it does not make oncoref the owner of
downstream panel curation.

```python
from oncoref import antigen_coverage

antigen_coverage.addressable_antigen_fraction("LUAD", gene_ids={"ENSG00000141510"})
antigen_coverage.greedy_antigen_coverage("LUAD", gene_ids={"ENSG00000141510"})
```

## Expression And Normalization

Expression readers are the stable downstream surface. Builder, registry, and
engine modules produce and audit those artifacts; they are separated below so
read-time choices do not get mixed with source-ingestion details.
Expression values use transcripts per million (TPM) unless a section states a
different unit.

### Acquisition sources and selected matrices

The expression-source registry describes datasets that can be acquired or built.
The source-matrix registry describes the matrix currently selected for each
cancer code. They are related planning surfaces, not interchangeable provenance:
the selected ACC matrix, for example, currently comes from Treehouse even though
`tcga-acc` is a registered acquisition source.

An acquisition source with a known physical cohort also carries a nonempty
`source_project`. This makes `expression_sources()` self-contained for display
provenance. Its label may be more specific than the cohort registry's shared
project label, so consumers should preserve the source-owned value instead of
replacing it with a cohort-level fallback.

When a GEO dataset publication has been verified, `source_pmid` records it as a
structured `PMID:<digits>` value. Selected cancer-registry rows for that same
physical source must carry the same PMID; biological background citations remain
separate and should not be inferred from citation prose.

Use `oncoref.source_matrices.codes_for_source(source_id)` for the selected code
list. Use `oncoref.source_matrices.resolution_for_source(source_id)` when
provenance matters. Its `resolution_method` is `physical_source` only when the
registered and selected `source_cohort` values match; `declared_cancer_code`
means the source's declared codes route to matrices built from other physical
sources. Each returned `SelectedSourceMatrix` always retains the selected
matrix's real `source_cohort`. A registered source without a published matrix
returns `unavailable` plus a machine-readable `availability_reason`.

`expression_registry.expression_source_candidates()` remains an acquisition
planning table. A row is marked `direct_reference_available` only when it names
the same physical source as the selected matrix; its `reference_code` is then
the exact `cancer_code`. A selected matrix for the same cancer code but a
different cohort does not overwrite that candidate's accession, URL, or
processing plan.

### Reader APIs

- `oncoref.expression` — read-time accessors for per-sample expression,
  percentile vectors, representative samples, within-sample top fractions, and
  pan-cancer reference tables. `sample_expression_qc` reports per-sample
  detected-gene counts, literal-zero fraction, top-gene/top-10 concentration,
  biological-housekeeping detection, source-scale class, and source-type caveats
  so sparse source-matrix artifacts can be audited before using absolute TPM
  floors or housekeeping normalization. `per_sample_expression(...,
  sample_qc="pass" | "pass_or_warn" | "all")` filters sample columns at read time; the raw per-sample accessor
  defaults to `"all"` for forensic access, while live summaries such as
  `cohort_stats` and `pooled_cohort_stats` default to QC-passing samples.
  `source_matrix_sample_qc_manifest`, `expression_artifact_build_metadata`, and
  `expression_artifact_build_summary` read the optional QC/build metadata emitted
  by regenerated expression bundles. Until a regenerated heavy bundle ships those
  files, they return schema-stable empty metadata by default; use
  `on_missing="raise"` when a downstream migration requires the manifests.
  `housekeeping_cancer_expression_coverage(...)` is the reusable #202 audit
  surface for evaluating clean-TPM biological housekeeping candidates across
  cancer cohorts. Pass its result to
  `housekeeping_cancer_expression_coverage_summary(...)` for one row per candidate
  with a source-aware linear-floor status and the worst comparable cohort. Treat
  absolute TPM floors as hard evidence only where
  `recommended_for_absolute_tpm_floor` is true; microarray/proxy or otherwise
  non-linear sources stay visible as warning/rank calibration inputs, not vetoes.

### Diagnosis and molecular evidence

A diagnosis-labelled reference sample is not automatically positive for a common
driver. Oncoref represents three separate facts: the canonical cancer entity, the
entity-level published driver spectrum, and any sample-level molecular observation.
This matters for infantile fibrosarcoma (`SARC_IFS`) and congenital mesoblastic
nephroma (`CMN`): they share an infantile MAPK-rearranged spindle-cell spectrum, but
are distinct diagnoses with different sites and driver distributions. `ETV6-NTRK3`
is common in IFS, not required for the diagnosis and never inferred for a
diagnosis-only Treehouse sample.

- `oncoref.drivers.cancer_driver_spectrum(code)` returns the structured observed
  fusion, intragenic-rearrangement, and unresolved states for an entity.
- `oncoref.samples.molecular_provenance_for_cancer_code(code)` returns public
  sample/library evidence with donor identity, diagnosis, driver event, assay,
  confirmation status, expression availability, and access level.
- `oncoref.samples.molecular_sample_counts(code)` reports libraries and distinct
  donors separately for each physical source cohort.

Public Treehouse PolyA and RiboD cohorts remain separate even after clean TPM, and
the GSE11482 CMN array cohort remains an explicitly non-comparable TPM proxy.
Applying the common censored-gene composition to that proxy does not turn array
intensities into absolute TPM; use it only for marker patterns and ranks.
Controlled EGA, St. Jude, and CCDI sources expose acquisition state and public
molecular annotations without claiming that their expression is currently loadable.

### Builder APIs

Every published source matrix has one regeneration path owned by oncoref. The
path may use a generic builder or a small source-specific adapter, but it always
ends at the same canonical matrix, mapping-audit, parse-diagnostic, sample-QC,
and summary-row contract. Source-scale caveats remain data: microarray TPM
proxies, GSE125285 BCC/cSCC nCPM proxies, and CTCL/HCL single-cell pseudobulk
nTPM are retained for within-sample rank uses while explicitly marked
unsuitable for absolute comparison with bulk RNA-seq TPM.

- `oncoref.expression_builders` — build-time ingestion and artifact cores used by
  data-bundle generation scripts. `GeoMatrixSource` /
  `build_source_matrices` own the generic supplementary-matrix path from raw
  source file to canonical per-code per-sample TPM parquet, mapping audit, parse
  diagnostics, sample-QC sidecars, and `SourceMatrixBuildResult.summary_rows`.
  Raw-count inputs are canonicalized first and use Ensembl gene lengths from
  the requested pyensembl release in an `oncoref[genome]` build environment;
  `--gene-lengths-kb` remains available as an explicit build-input override.
  `summarize_source_matrix` is the standalone producer for those
  per-gene-per-cohort reference-expression rows: raw TPM stats, clean-TPM
  16/9/75 stats, `n_samples`, `n_detected`, and source provenance in one schema.
  `geo_matrix_source_from_registry` and
  `scripts/build_geo_matrix.py` make `source_type: geo-matrix` entries in the
  packaged source registry directly buildable; `GeoMatrixSource` also preserves
  summary-row provenance (`notes`, `pipeline_stem`, `tumor_origin`,
  `metastasis_site`) so downstream shard writers do not need a parallel source
  registry. `tumor_origin` is validated against
  `TUMOR_ORIGIN_VALUES` (`primary`, `metastasis`, `recurrence`, `cell_line`,
  `pdx`, `normal_tissue`, `mixed`). `GdcSource`,
  `query_gdc_star_count_manifest`, `build_gdc_sample_manifest`,
  `read_gdc_star_counts_tpm`, `build_gdc_source_matrices`, and
  `scripts/build_gdc_source.py` own the common GDC STAR-counts path: open
  RNA-seq file discovery, deterministic sample-per-case selection, per-sample
  TPM matrix assembly, canonicalization, sample QC, and summary-row sidecars.
  Source-specific GDC lineage routing can now attach to this shared contract
  instead of carrying separate BL/MM/TARGET-style builders. `Recount3Source`,
  `recount3_gene_sums_to_tpm`, `build_recount3_source_matrices`, and
  `scripts/build_recount3_source.py` do the same for `source_type: recount3`
  entries, including run-to-sample aggregation and metadata-based routing before
  writing the standard source-matrix artifact set. `SraNcbiCountSource`,
  `sra_ncbi_count_source_from_registry`,
  `build_sra_ncbi_count_source_matrices`, and
  `scripts/build_sra_ncbi_counts_source.py` own the preferred processed-count
  path for SRA studies with NCBI Gene Feature RNA-seq counts. The registry pins
  NCBI analysis and run accessions, count-file checksums, sample roles, and a
  RefSeq GFF; only explicitly routed tumor runs enter reference matrices.
  `SraSalmonSource`,
  `sra_salmon_source_from_registry`, `build_sra_salmon_source_matrices`, and
  `scripts/build_sra_salmon_source.py` provide the raw-read fallback: the
  registry pins run roles, read checksums, and the Ensembl transcriptome; all
  declared runs are audited while only explicitly routed tumor runs enter
  reference matrices. See
  [SRA Expression Sources](sra-expression-integration.md). `TreehouseSource`,
  `treehouse_source_from_registry`, `treehouse_cohorts_for_group`, and
  `scripts/build_treehouse_source.py` own the direct Treehouse-compendium path:
  clinical disease-label routing, log2(TPM+1) inverse transform, symbol
  canonicalization, sample QC, and summary-row sidecars. Treehouse selectors are
  registry-native for direct clinical routing (`""` and `tcga`) plus selected
  side-table-backed routes: GDC project membership, cBioPortal patient/sample
  clinical attributes, and cBioPortal mutation-positive case sets.
  `scripts/rebuild_expression_artifacts.py`
  then applies the same sample-QC policy to derived shards by default
  (`--sample-qc pass`) and emits `source-matrix-sample-qc.csv` plus
  `expression-artifact-build-metadata.*` in the staging directory so bundle
  releases record which source samples fed percentiles, representatives,
  proteoform summaries, and within-sample summaries. The rebuild also assigns
  every representative source group to the released train/validation partition
  and records the policy, role counts, and per-cohort validation coverage in the
  same metadata. Representative sample selection uses
  `representative_sample_columns` / `cohort_medoids` on the biological clean-TPM
  view, then stores the selected samples' full clean_tpm_16_9_75 vectors. Release
  builds retain curated cohorts that have no strict QC-pass samples only through
  explicit source-aware fallbacks recorded in the build metadata, and clip
  invalid negative source expression values to zero with per-cohort counts.
- `oncoref.expression_source_adapters` — narrow parsers and routers for public
  sources whose sample labels live outside the primary expression matrix. These
  adapters cover TARGET ALL phase-matrix B/T lineage, TARGET NBL cBioPortal MYCN
  status, GSE75885 histology titles, DRMetrics histology attributes, audited GEO
  microarrays, GSE171811 CTCL TCR-beta-selected case pseudobulks, and the
  checksum-pinned Zenodo 14917813 HCL T0 donor pseudobulks, checksum-pinned
  GSE141460 EPN diagnosis-stage malignant-cell mean-TPM pseudobulks, checksum-pinned
  GSE125285 BCC/cSCC tumor routing, and checksum-pinned GSE139682 GBC tumor
  routing, plus checksum-pinned OpenPBTA release-v23 primary CRANIO and
  H3 K27-altered DIPG routing. The OpenPBTA adapter exposes
  `openpbta_cranio_matrix(...)` and `openpbta_dipg_matrix(...)` as public
  data-frame transforms, with corresponding public canonical builders; only the
  upstream RDS deserializer remains private. The CRANIO route
  retains three recurrent and four progressive craniopharyngiomas as explicit
  exclusions and routes 29 independent primary donors. The GSE125285 adapter
  retains all matched normals as exclusions and labels its inverse-transformed
  author CPM as an nCPM proxy; the GSE139682 adapter retains its matched normals
  as exclusions and renormalizes tumor RPKM to TPM. Both emit stable GSM sample
  IDs and complete sample manifests.
  The HCL adapter
  renames artifact-local columns to stable donor IDs, retains the sixth study
  donor as an explicit no-T0 exclusion, requires positive ANXA1/MS4A1/CD22/
  IL2RA/ITGAE/ITGAX marker values in every selected donor, and never infers a
  per-donor BRAF call. They
  delegate normalization, canonicalization, QC, summaries, and artifact writing
  to `expression_builders` rather than defining parallel data contracts.
  For GSE141460, `gse141460_epn_pseudobulk(...)` is the public, deterministic
  source transform and `build_gse141460_source_matrices(...)` is the public
  checksum-verified canonical builder. The transform reads each specimen's
  sequencing protocol from the pinned clinical Table S1; neither sample naming
  nor cancer type supplies a protocol assumption. It retains all 28 patient
  specimens in the audit manifest while routing only the 11 diagnosis-stage
  specimens with author-labeled malignant cells.
  `scripts/merge_expression_artifact_update.py` merges a targeted rebuild into
  a complete prior bundle, recomputes representative partitions globally, and
  preserves unrelated cohort shards byte-for-byte.

### Registry and low-level APIs

- `oncoref.expression_registry` — source-registry inspection helpers over the
  bundled `expression_sources.yaml`. Use `expression_source_registry_entries()`
  for the full raw YAML dictionaries, `expression_source_registry_entries(source_type="geo-matrix")`
  for generic GEO build configs, or `expression_source_registry_path()` only when
  a subprocess needs the packaged registry path. Downstream packages should use
  these helpers instead of shipping a second copy of the registry. GEO
  accessions named anywhere in a source's citation or file metadata are also
  stored in the structured `accession` field.
- `oncoref.expression_engine` — reusable low-level builder primitives for
  expression tables: identity/value column detection, transcript-to-gene
  aggregation, source row ID-type detection, source gene-row mapping audits,
  missing-vs-non-parsing numeric diagnostics, and canonical ENSG aggregation in
  linear expression space. It is an explicit public module, so downstream
  builders can import `oncoref.expression_engine.map_source_gene_rows`,
  `canonicalize_source_gene_matrix`, and `coerce_source_expression_values`
  without reaching into scripts. Use these in builders before committing a
  source matrix so unresolved high-expression rows and duplicate canonical IDs
  are explicit artifacts rather than hidden cleanup. The source audit frames are
  intentionally unversioned public API objects: provenance belongs in build
  metadata, while the frames themselves use stable canonical columns such as
  `sample_qc_status`, `sample_qc_reasons`,
  `source_expression_nonzero_samples`, and
  `source_expression_sample_with_max`.
- `oncoref.source_matrices` — raw per-cohort source-matrix cache/fetch helpers.
  `source_matrix_version(code)` reports the exact release pinned for that cohort;
  `release_url(code)` and `local_path(code)` use the same per-cohort pin. This
  lets a corrected or newly added matrix move to a new release without copying
  every unchanged source asset or invalidating its existing cache.
  `source_matrix_regeneration_audit()` matches each selected matrix by the exact
  `(cancer_code, source_cohort)` physical-source pair. A pair must have exactly
  one registry owner declaring either an existing repository-relative builder
  script or an `external_build_exemption`; `validate_source_matrix_regeneration()`
  raises for missing, ambiguous, invalid, or nonexistent builder ownership. The
  sole current exemption is the controlled UNC NUTM1 case series. This audit is
  a source-checkout/release-build check because builder scripts are not installed
  in the runtime wheel.
  Use `source_matrices.sample_qc(code)` for the live source-matrix QC audit and
  `source_matrices.sample_qc_manifest(...)` for the optional generated-bundle QC
  manifest that records which samples fed derived artifacts.

### Normalization API

- `oncoref.normalization` — TPM conversion, clean TPM, technical-RNA filtering,
  log transforms, percentile ranks, and housekeeping normalization.

The normalization helpers are intended to be reusable directly. Expression
accessors and bundles are also reusable, but downstream packages may keep their
own packaged expression artifacts until row-set, value, provenance, and QC
contracts are parity-clean for the specific accessor they want to replace.

### Tumor-reference summaries

Oncoref exposes two related products with different biological meanings. The
TCGA table is tumor-attributed TPM produced by Trufflepig's existing per-sample
TME decomposition. Oncoref migrates the pinned output and does not implement a
second decomposition algorithm. The subtype/cohort table contains passthrough
aggregations on source-declared scales; its historical filename says
"deconvolved", but callers must not infer that method from the name.

- `tumor_references.tcga_deconvolved_expression()` returns TCGA tumor-attributed
  median, Q1, Q3, and contributing-sample counts by cancer code.
- `tumor_references.subtype_tumor_reference_expression()` is the preferred name
  for source-separated subtype/cohort summaries. The historical
  `subtype_deconvolved_expression()` name remains as a compatibility alias.
- `tumor_references.tumor_reference_expression_provenance()` states the
  derivation for each physical source: `tme_deconvolution`,
  `high_purity_passthrough`, or `observed_tpm_passthrough`.

Both expression accessors validate finite nonnegative values, quantile order,
positive sample counts, and logical-key uniqueness. They canonicalize cancer and
cohort identities without pooling physical sources. Known subtype aliases are
canonicalized; source-defined subgroup labels that are not ontology nodes remain
explicit source labels. `scale="classifier_tpm"`
is the default comparable analysis view: technical RNA is removed within each
source group and median mass is scaled to one million. `scale="native"` returns
the validated migrated values without that read-time transform. Subtype rows
may remain symbol-only when the legacy source did not provide an unambiguous
Ensembl gene ID; BeatAML rows are rebuilt from Oncoref's ID-bearing raw source
matrices using only samples marked `sample_qc_status="pass"`, normalized through
the canonical 16/9/75 `clean_tpm` API before aggregation, because the legacy
table contained invalid negative Q1 values.

Each expression result has the same `DataFrame.attrs["oncoref"]` metadata
shape. TCGA records its dataset-wide derivation method directly. Because the
subtype artifact mixes source-level derivations, it records
`derivation_method=None`, `derivation_scope="source"`, and the provenance
dataset to query instead of inventing a fourth derivation-method label.

### Pan-cancer table

`expression.pan_cancer_expression()` defaults to oncoref's entity-first schema:
HPA normal tissue columns are `<tissue>_nTPM_raw`, TCGA source/provenance
columns are `<CODE>_FPKM_raw`, deterministic TCGA TPM companions are
`<CODE>_TPM_raw`, and analysis columns append `_clean`, `_hk`, `_percentile`, or
`_log1p`. For migration code that needs pirlygenes' unsuffixed column names, use
`column_style="pirlygenes"`; the legacy `to_tpm=True` keyword is accepted as a
compatibility alias for that view and maps the default call to `normalize="tpm"`.
The pan-cancer view also emits raw-TPM companion columns for member-backed
grouping/source-scope references (`NET`, `CRC`, `NSCLC`, `SGC`) by pooling
the selected `cancer-reference-expression` summary rows with n-sample weights.
Incomplete closed unions are omitted. BTC is now emitted because CHOL and GBC
are both reference-backed; this does not make the small direct GBC cohort an
independent classification target.
Existing directly sourced columns, including `SARC` and `OV`, keep their current
source-table behavior.

### Cohort reference expression

`expression.cancer_reference_expression()` returns cohort-level tumor reference
expression with stable long or wide output. It accepts canonical cancer codes,
aliases, and aggregate cohorts, resolves gene filters by ENSG or symbol, and can
return one or more normalization modes in one call:

- `normalize="tpm_clean"` / `"clean_tpm"` — shipped biological clean-TPM
  percentiles.
- `normalize="tpm_clean_biological"` — explicit name for that biological-only
  reference artifact.
- `normalize="tpm_clean_log1p"` — stored log1p biological clean-TPM percentiles.
- `normalize="tpm_raw"` / `"tpm"` — source-matrix raw TPM summaries recomputed
  through `cohort_stats`.

Long output includes source/provenance columns by default, including source
cohort, source project/version/PMID, tumor origin, source type/unit, source scale
class, reference method, selected source gene/sample counts, `DATA_VERSION`, and
`SOURCE_MATRIX_VERSION`. Use
`expression.cancer_reference_expression_source_metadata(cancer_type,
source_cohort=...)` for the same structured provenance without loading expression
rows. Omitting `source_cohort` resolves the selected source; an explicit physical
cohort never borrows metadata from another source registered for the cancer type.
This accessor is the compatibility surface for reference-expression reads;
expression artifact row-set/value parity is tracked separately in the upstream
parity issues.

`reference_source="artifact"` is the historical default: clean/log clean TPM
comes from shipped percentile shards, and raw TPM is recomputed from source
matrices. `reference_source="summary_rows"` uses the shipped
`cancer-reference-expression` per-source sidecars when `sample_qc="all"` and
uses the one physical source explicitly selected by `source-matrices.csv` and the
availability manifest. Gene and sample counts describe each source; they never
promote an alternative source or pool it into the selected reference. For
`sample_qc="pass"` or `"pass_or_warn"`, the summary-row source selector
intentionally recomputes via `cohort_stats(..., sample_qc=...)` so
QC-filtered reference-expression views are shaped at read time rather than by a
build-time drop. This keeps the source sidecars as all-sample evidence while
allowing downstream code to ask for QC-passing summaries without maintaining a
private filtered bundle.

Use `reference_source="summary_rows_all", sample_qc="all"` when downstream code
needs the full source-union table rather than one selected source per cancer
code. This long-only mode returns one row per gene, cancer code, normalization,
and source cohort; even with `include_provenance=False`, it keeps
`source_cohort` and sample-count columns because they are part of the source-row
identity. With provenance enabled, it also preserves sidecar fields such as
`processing_pipeline` and `notes`. It accepts `source_kind=...`,
`source_cohort=...`, `exclude_microarray_proxy=True`, and `pool=True` for an
explicit n-sample-weighted pooled view. Because these sidecars are all-sample
artifacts, this source-union mode intentionally rejects `format="wide"`,
`sample_qc="pass"`, and `sample_qc="pass_or_warn"`; QC-filtered all-source
reference artifacts remain part of the expression-artifact rebuild work. Pooling
groups after the requested gene-ID projection by output `Ensembl_Gene_ID` (not
display symbol), so each cancer/normalization result has one row per projected
gene and each physical source contributes its sample count at most once.

For compatibility with pirlygenes reference-expression consumers, the accessor
also exposes the gene-to-proteoform bridge columns on every long-form row:
`Proteoform_ID` is the cDNA/read-recovery identity that the row maps to, and
`Member_Ensembl_Gene_IDs` is the row's member ENSG list. Without a collapse flag
this is only an annotation; it does not fold rows. Use
`collapse_cdna_identical=True` for the read-recovery space: byte-identical CDS
groups plus the curated proteoform-collapse overrides. Use
`collapse_protein_identical=True` for the genome-wide identical-protein space.
Set at most one. These modes sum `expression`, `q1`, and `q3` in linear TPM
space inside each source context and leave wide output in the historical
`Ensembl_Gene_ID`, `Symbol`, value-column shape.

### Availability and missing data

Use `expression.cancer_reference_expression_availability()` before delegating a
downstream reference-expression accessor that must distinguish unavailable
oncoref artifacts from empty gene filters. It returns one row per requested
code/mode with `requested_code`, expanded `cancer_code`, `request_kind`,
`available`, `missing_reason`, provenance fields, and the reference-expression
schema/data versions. Provenance includes the structured `source_pmid`,
`source_scale_class`, and `linear_tpm_comparable` fields; proxy exclusion uses
those fields before compatibility fallbacks. `expression.cancer_reference_expression(...,
on_missing="empty")` returns a schema-stable empty frame and stores the same
missing rows in `df.attrs["missing_requests"]`; `on_missing="raise"` fails fast
for required cohorts. `include_request_metadata=True` adds request/availability
columns to long expression output, which is useful when a requested aggregate
expands to child expression cohorts.

### Representative evaluation partitions

Representative vectors are real source samples. A model trained on all of them
must not report accuracy on those same vectors as an estimate of generalization.
Use `expression.representative_partition_manifest()` to obtain the released
training and evaluation assignment before fitting or benchmarking:

```python
from oncoref import representative_partition_manifest

partition = representative_partition_manifest()
train_ids = partition.loc[partition["partition_role"] == "train", "representative_id"]
validation_ids = partition.loc[
    partition["partition_role"].isin(["validation", "validation_external"]),
    "representative_id",
]
```

The physical `source_group_id`, not the displayed representative ID or cancer
label, is the partition unit. Parent labels, subtypes, and compatibility aliases
backed by the same sample therefore always receive one shared role:

- `train` — eligible for model fitting.
- `validation` — deterministic within-cohort holdout.
- `validation_external` — a complete independent source project held out when a
  cohort spans projects.
- `audit_only` — retained for inspection but excluded from fitting and evaluation
  because its source group is not benchmark-eligible.

The ordinary within-cohort policy holds out up to two independent groups, never
more than half of a cohort. A usual five-representative cohort therefore has
three training and two validation groups. When a cohort spans independent source
projects, the smallest viable project is instead held out whole and labeled
`validation_external`. A cohort with only one eligible group remains train-only
and reports `partition_status="insufficient_independent_groups"` instead of
claiming validation coverage. `partition_policy_version` makes the exact
assignment contract release-visible.

### Derived artifact fields

Representative and percentile artifact readers have explicit downstream-facing
contracts:

- `expression.representative_cohort_samples(..., format="long",
  include_provenance=True)` includes the representative id, source cohort/project,
  source sample id and stable source-group id, source diagnosis/morphology when a
  sample has been reviewed, effective QC status/reasons, source scale class,
  linear-TPM and absolute-floor comparability flags, representative role and
  benchmark eligibility, partition role/status/policy, review evidence, cohort
  sample count, deterministic selection rank/method/basis, artifact schema
  version, `DATA_VERSION`, and `SOURCE_MATRIX_VERSION`.
  Treehouse PolyA parent, subset, and annotation-derived cohorts share one physical
  sample namespace, so aliases of the same source vector receive the same
  `source_group_id` even when their displayed source cohorts differ.
  Public representative ids default to pirlygenes-compatible `CODE_rep01`
  columns/values. Pass `representative_id_style="internal"` to expose the
  underlying bundle/provenance ids (`CODE__rep1`).
  Representatives are selected by central-medoid plus farthest-first traversal in
  log1p biological clean-TPM space, with stable sample-id tie-breaking; the
  persisted vectors remain full clean_tpm_16_9_75.
- `expression.representative_cohort_availability()` returns one row per shipped
  cohort with the same QC/scale qualification and a machine-readable availability
  reason. `available_representative_cohorts(linear_tpm_comparable=True,
  benchmark_eligible=True)` gives a fail-closed classifier-ready cohort list while
  retaining proxy cohorts such as MTC for rank/percentile workflows.
- `expression.cohort_gene_percentiles(..., include_provenance=True)` appends the
  cohort code, normalization, expression unit, percentile basis, artifact schema
  version, `DATA_VERSION`, and `SOURCE_MATRIX_VERSION`.
- Gene-level representative and percentile readers default to canonical oncoref
  ENSG IDs. For pirlygenes migration wrappers, pass
  `gene_id_style="pirlygenes"` to present known one-to-one
  `remapped_to_oncoref` rows with their legacy pirlygenes ENSG IDs. This is
  intentionally a presentation shim: it does not synthesize missing rows or alter
  expression values.
- Gene-level reference, representative, and percentile readers default to
  `gene_universe="artifact"`, which preserves the exact shipped row set. Pass
  `gene_universe="tumor_signal"` to drop rows explicitly audited as
  oncoref-only filterable extras for the requested artifact/cohort: strict
  technical extras plus biotype-resolved non-signal extras such as pseudogene,
  small-RNA, and immune-receptor segment rows. Protein-coding and lncRNA
  oncoref-only rows are retained as biological extras. Pass
  `gene_universe="pirlygenes"` only for migration parity: it starts from the
  tumor-signal policy, then also drops audited oncoref-only biological or
  unresolved extras unless the row is a documented remap target for a pirlygenes
  legacy ENSG ID. Combined with `gene_id_style="pirlygenes"`, this can
  alias-expand a documented remap row when current pirlygenes exposes both the
  legacy and canonical ENSG IDs for the same measured vector. This reproduces
  pirlygenes row-universe expectations without inventing missing expression
  measurements. Pass
  `include_gene_universe_flags=True` for long reference output or any
  representative/percentile output to append row-level `artifact_row_class`,
  `is_filterable_extra`, `is_technical_extra`, `is_missing_biological`, and
  `recommended_consumer_action` columns. These options filter or label known
  artifact row classes; they never invent missing biological expression rows
  and only the explicit `pirlygenes` mode drops biological oncoref-only extras.
- Representative and percentile readers default to `sample_qc="pass"` and
  validate any shipped `expression-artifact-build-metadata.csv` rows before
  returning a precomputed shard. If the metadata says a shard was built with
  `sample_qc="all"` or another policy, the reader raises rather than silently
  treating the shard as QC-pass. Use `sample_qc="artifact"` only for explicit
  legacy/audit reads where the caller wants exactly whatever policy the bundle
  used. Metadata-missing legacy bundles remain readable but expose
  `df.attrs["artifact_sample_qc_verified"] = False`.
- Gene-level reference, representative, and percentile readers attach
  `df.attrs["gene_universe_delta_summary"]` and
  `df.attrs["gene_universe_delta_n"]` for the requested cohort/product. These
  attrs summarize the known pirlygenes/oncoref row-universe deltas that still
  apply to the returned artifact, so migration wrappers can separate remapped
  rows, missing upstream data, and intentional oncoref-only rows without
  reimplementing the audit-table matching logic.
- Missing percentile shards still raise by default. Use
  `on_missing="empty"` to return an empty but schema-stable frame with
  `df.attrs["missing_reason"]`, which is useful for compatibility adapters that
  need to distinguish unavailable upstream data from private downstream fallback
  data.

### Bundle contents and gene-universe parity

The QC-policy expression bundle ships representative, percentile,
within-sample, CTA-scope proteoform percentile, CTA-scope proteoform
within-sample, sample-QC, and build-metadata artifacts. Non-shipped proteoform
scopes can still recompute from cached source matrices. Row-set and value parity
with pirlygenes is still governed by the gene-universe and expression-artifact
parity issues.

`expression.expression_artifact_gene_universe_deltas()` exposes the known
pirlygenes/oncoref row-universe deltas from the current parity audit: canonical
remaps such as legacy `PAXX` to its oncoref ENSG, sequence-identical
representative-sample remaps where the measured oncoref artifact row can be
presented under the pirlygenes legacy ID, and the full current set of oncoref-only
representative extras. The prior broad
`unresolved_oncoref_extra` bucket is resolved where possible by current oncoref
gene metadata into strict technical extras, `non_signal_oncoref_extra` rows that
`gene_universe="tumor_signal"` filters, `biological_oncoref_extra` rows that stay
visible in the default and tumor-signal views, `sequence_identical_remapped_to_oncoref`
rows that `gene_id_style="pirlygenes"` can present or alias-expand without
inventing missing expression measurements, or a small
remaining unresolved set with no current biotype. In the current audit table, no
rows remain flagged as missing biological; only 29 oncoref-only rows remain truly
`unresolved_oncoref_extra`. The resolved status labels are deliberately explicit
so downstream wrappers do not need to infer policy from biotypes.
Use `expression.expression_artifact_gene_universe_delta_summary()` for counts by
product/cohort/status, or
`expression.expression_artifact_gene_universe_delta_report(product, cancer_types)`
for the compact request-scoped report used by accessor attrs. These tables include
`gene_biotype`, `artifact_row_class`, `is_filterable_extra`,
`is_technical_extra`, `is_missing_biological`, and
`recommended_consumer_action` so current-bundle row classes do not have to be
inferred from prose. Use
`expression.expression_artifact_technical_extra_gene_ids(...)` to get the
oncoref-only technical-extra ENSG IDs for a product/cohort filter. This surface is
intentionally provenance: it makes differences explicit for migration code, but
does not synthesize missing expression rows or alter artifact values.

### Clean TPM compartments

Clean TPM has one assay-independent censored-gene table and one public
compartment contract. `clean-tpm-censored-genes.csv` carries each gene's
`category`, `reference_tpm`, `reference_source`, and `reference_profile_version`.
The transform replaces measured censored-gene composition with this Treehouse
25.01 PolyA median profile for every assay:

- `clean-tpm-censored-genes.csv:category == "ribosomal_protein"` — 16%
  ribosomal compartment.
- `clean-tpm-censored-genes.csv:category == "technical"` — 9% other-technical
  compartment. This includes mitochondrial/rRNA artifacts, nuclear-retained
  polyA-bias lncRNAs, and structural noncoding RNA biotypes whose measured abundance
  can change artifactually by 10-fold or more with library preparation
  (`snRNA`, `snoRNA`, `scaRNA`, `misc_RNA`, `ribozyme`, `sRNA`, `vault_RNA`). It
  deliberately does not classify all small ncRNAs or miRNAs as technical.
- genes absent from the censored table — 75% biological compartment.

The fixed compartment budgets and within-compartment PolyA weights make the
biological 75% comparable across library preparations; they do not make the assays
physically equivalent. `clean_tpm`, `filter_technical_rna`, gene-QC classification,
and `technical_rna_gene_ids()` use this same global membership, never an assay-specific
list. The structural ncRNAs added for ribo-depleted data are therefore also technical
in PolyA and every other assay; only their raw measured abundance differs. Keep
polyA-selected, ribo-depleted, and microarray sources distinct in source
selection and pooling. Sample QC reports raw top-gene fractions for audit, but applies
its concentration gates to the clean-TPM fractions so depletion-sensitive structural
RNA cannot fail an otherwise usable ribo-depleted library.

The category-specific helper sets are available from `oncoref.gene_families`:

```python
from oncoref import gene_families

gene_families.clean_tpm_ribosomal_gene_ids()
gene_families.clean_tpm_other_technical_gene_ids()
gene_families.clean_tpm_censored_gene_ids()
gene_families.clean_tpm_censored_genes()
```

### Housekeeping normalization

Housekeeping normalization is an explicit, consumer-specific expression space, not
the general oncoref default. Prefer clean TPM when additive abundance matters,
`log1p(clean TPM)` when magnitude needs compression, and percentile ranks when only
within-sample ordering matters. Use an HK-derived size factor only after the consumer
has shown that it improves its own calibrated task.

The active 30-gene biological panel is deliberately stable. Cancer-side low-tail
coverage can identify warnings and candidate replacements, but it does not
automatically promote genes: changing the panel changes every normalized value and
requires downstream recalibration. Use the raw and summarized source-aware audits:

```python
from oncoref import expression

coverage = expression.housekeeping_cancer_expression_coverage(
    ["LUAD", "SKCM"], auto_fetch=False, on_missing="raise"
)
summary = expression.housekeeping_cancer_expression_coverage_summary(coverage)
summary[["Symbol", "linear_floor_status", "worst_linear_p5_cancer_code"]]
```

Only rows marked `recommended_for_absolute_tpm_floor` participate in the summary's
linear TPM pass/fail decision. Proxy or non-linear sources remain counted but cannot
pass or veto an absolute clean-TPM floor. The raw audit records requested, audited,
and unavailable cancer codes in DataFrame attributes. The summary rejects a partial
audit, or one whose completeness is unknown, by default;
`require_complete=False` is an explicit exploratory-cache opt-out, not a
release-quality panel decision.

For clean-TPM housekeeping denominators, use the biological HPA-stable panel:

```python
gene_families.clean_tpm_biological_housekeeping_gene_ids()
gene_families.clean_tpm_biological_housekeeping_genes()
gene_families.clean_tpm_biological_housekeeping_genes(primary_only=False)
```

Housekeeping normalization is defined as a median-of-ratios size factor against a
fixed, versioned per-gene reference profile:

```python
from oncoref import normalization

normalization.housekeeping_reference_profile()
normalization.tpm_to_housekeeping_normalized(matrix)
```

For each sample, oncoref computes:

```text
size_factor = median(housekeeping_clean_tpm[g] / reference_tpm[g])
normalized_expression[g] = clean_tpm[g] / size_factor
```

The default reference is the HPA v23-derived clean-TPM biological housekeeping
panel (`HOUSEKEEPING_REFERENCE_PROFILE_VERSION`). This is a sample-scale estimate
relative to a fixed biological HK profile, not the old "divide by the panel's
geometric mean" ratio. Prefer log1p clean TPM or percentile-rank clean TPM unless
the analysis specifically needs an HK-derived size factor.

The old geNorm-style denominator is deliberately buried behind
`method="legacy_geomean"` for explicit audits of historical outputs. The shorter
`method="geomean"` spelling is not accepted.

The legacy qPCR/reference-gene panel remains available as
`legacy_qpcr_housekeeping_*` and through the historical `housekeeping_*` helpers,
but it is not the clean-TPM biological denominator.

## Genes and Proteoforms

Gene-symbol aliases are loaded as lossless strings, including literal aliases
such as `NA` and `NaN`. `gene_ids.resolve_symbol()` checks the NCBI alias's exact
case first. It falls back case-insensitively only when the folded alias is
unambiguous or the pinned NCBI snapshot supplies an exact uppercase row;
otherwise the input is returned unchanged rather than selecting a gene by row
order.

- `oncoref.gene_ids` — bundled canonical Ensembl gene space, cross-release alias
  resolution, symbol synonyms, and biotype checks.
- `oncoref.genome` — optional (`pip install 'oncoref[genome]'`) pyensembl-backed
  gene/transcript lookup against installed Ensembl releases.
- `oncoref.proteoforms` — identical-protein paralog grouping and expression
  collapse helpers.
- `oncoref.gene_qc` / `oncoref.gene_families` — technical-RNA and gene-family
  classification used by normalization. These are normalization/QC reference
  families, not the general home for pirlygenes marker panels.

## Burden, TMB, Fusions, and Signatures

- `oncoref.tmb` — tumor mutational burden reference values.
  `tmb.cancer_tmb_df()` includes evidence-schema columns (`estimate_type`,
  `source_scope`, `missing_reason`), and `tmb.cancer_tmb_record()` /
  `tmb.resolve_tmb_source()` preserve requested-code metadata for source-scoped
  lookups such as `COAD_MSI` or `READ_MSI` resolving through `CRC_MSI`. Direct
  audited gaps use `inheritance_kind="direct_missing"` so callers can distinguish
  “known no supported site-specific estimate” from an unmapped cancer code.
- `oncoref.incidence` — incidence/mortality burden and burden categories.
  `incidence.cancer_burden_df()` is the auditable burden table: percentages are
  the public lookup values, and raw-count, source-locator, source-site,
  derivation, and rounding columns are preserved as provenance fields. The
  `aggregation` and `source_anchor` columns are the lossless pirlygenes
  compatibility contract: every burden category has an explicit site
  composition or residual formula and one or more resolvable PMID/DOI anchors.
  Locator status values such as `not_extracted` remain explicit until exact
  per-region table/export locators and raw counts are filled in.
- `oncoref.fusions` — defining fusions and partner-family lookups.
- `oncoref.response_signatures` — legacy/compatibility response-signature
  surface used by oncoref plots. Treat it as transitional: new or extended
  therapy-response signature panels belong in pirlygenes unless they are recast
  as source-anchored empirical fact/provenance rows.

## Data Management

### Dataset catalog

- `oncoref.catalog` — unified dataset inventory and fetch/status/path operations.

### Expression bundle

- `oncoref.data_bundle` — heavy expression bundle cache. Use
  `data_bundle.bundle_contract()` to inspect the downstream-stable package/data
  version linkage, release asset URLs, cache environment variables, completion
  marker policy, and expected artifact inventory for the active bundle. The
  inventory includes the generated sample-QC manifest, per-cohort build metadata,
  within-sample prevalence shards, and CTA-scope proteoform percentile/prevalence
  shards, not just the legacy pirlygenes expression tables.
  `data_bundle.bundle_is_local()` reports whether the entire downloadable cache is
  populated. For a side-effect-free check of one required artifact across both an
  in-repository/package data directory and a partial cache, use
  `data_bundle.item_is_local(path)` or `data_bundle.find_local_item(path)`. When
  package data and the cache can each hold different shards of the same artifact,
  use `data_bundle.local_item_paths(path)` to inspect both non-empty roots in read
  precedence order. These item-level probes never fetch data and reject empty
  files or directories. Use
  `data_bundle.bundle_release_manifest()` to fetch and validate only the small
  release manifest/checksum for the active `DATA_VERSION`, including tarball
  sha256 plus any artifact inventory, builder commit, source-matrix version, and
  sample-QC policy metadata published with the release. Manifest version 2 can
  describe a compact overlay: the active archive pins a complete earlier
  oncoref bundle by version, size, and SHA-256 and contains only added, changed,
  or deleted files. Fetch reuses an already verified base cache when available,
  otherwise downloads that base once, verifies the overlay, and materializes a
  normal complete cache for the active version. Release builders create these
  assets with `scripts/build_data_overlay.py COMPLETE_DIR BASE_DIR
  BASE_MANIFEST OUTPUT_DIR`; full bundle releases remain supported. Use
  `data_bundle.bundle_metadata()` when a downstream package needs one
  no-heavy-download JSON object containing the static contract, local cache path
  and completeness state, local artifact inventory, and validated release
  manifest.
  CLI equivalents are available for CI/notebooks: `oncoref data contract` prints
  the static bundle contract, `oncoref data metadata [oncoref|pirlygenes]`
  prints the composed dependency state, and `oncoref data release-manifest
  [oncoref|pirlygenes]` prints only the validated release manifest/checksum
  metadata.

### HPA data

- `oncoref.reference_data` / `oncoref.hpa` — HPA reference-data cache and HPA
  tissue/cell-type accessors. Use
  `reference_data.provenance(name, version, verify_content=True)` for a
  defensive provenance snapshot containing the concrete source URL, local
  path and size, recorded SHA-256 and download time, existence state, and
  checksum result. `reference_data.status(verify_content=True)` exposes the
  same fields for every default-version HPA source; omit `verify_content` to
  avoid hashing large cached files.

## Compatibility Modules

These modules remain importable but are less discoverable than the organized
facades above:

- `oncoref.apd1` — legacy anti-PD-1 response slice; prefer
  `oncoref.ici_response`.
- `oncoref.ici` — core ICI implementation; prefer `oncoref.ici_response` for the
  organized public surface.
- `oncoref.coverage` — original mixed CTA/generic antigen-panel coverage module;
  prefer `oncoref.cta_coverage` or `oncoref.antigen_coverage`.
- `oncoref.peptides` — original CTA-specific 9-mer module; prefer
  `oncoref.cta_peptides`.
