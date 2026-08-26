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

"""The complete, classified inventory of oncoref-domain data.

This is the single declarative source of truth for *what data oncoref owns*
and how each dataset is held. ``catalog.inventory()`` exposes that current
self-owned inventory. A separately named, version-pinned pirlygenes migration
snapshot preserves the June 2026 takeover audit without pretending to follow
later additions or removals in another repository.

Ownership rule:
  oncoref owns empirical base facts and reference mechanics: canonical gene IDs,
  cancer ontology, reference expression/normalization, epidemiology, TMB, ICI/aPD1,
  source-anchored CTA facts, and measurement/provenance tables. pirlygenes owns
  purpose-specific gene sets/panels and target-to-therapy registries. trufflepig
  owns per-sample rules and QC interpretation.

Held buckets (oncoref domain):
  WHEEL     — small base-layer tables shipped in the wheel (no fetch). A
              ``legacy-compat`` category marks shipped historical surfaces that
              remain importable but are not the intended future ownership boundary.
  BUNDLE    — heavy expression artifacts in the version-pinned release tarball.
  HPA       — Human Protein Atlas reference tables fetched per-source on demand.
  SOURCE    — the raw per-sample TPM matrices (a separate large optional bundle).
  PLANNED   — oncoref-domain fact tables still to port.
  SUPERSEDED— a pirlygenes table oncoref replaced with its own regenerated one.

OUT_OF_SCOPE — pirlygenes or trufflepig data that is NOT oncoref's domain:
  target-selection panels, marker/signature gene sets, therapy modality catalogs,
  sample-rule tables, and artifact/QC interpretation rules.

The catalog (:mod:`oncoref.catalog`) manages the fetchable subset; this module
is the inventory + classification.
"""

from __future__ import annotations

#: {name: (category, description)} — small curated tables shipped in the wheel.
WHEEL: dict[str, tuple[str, str]] = {
    "cancer-type-registry": ("ontology", "cancer-type codes, hierarchy, family/tissue"),
    "cancer-subtype-groupings": ("ontology", "cross-cutting MSI/POLE/HPV/MYCN axes"),
    "cancer-cohort-aggregates": ("ontology", "computed histology rollups (SARC_RMS, …)"),
    "cancer-lineage-groups": ("ontology", "registry family -> coarse histogenesis lineage group"),
    "cancer-lineage-group-overrides": ("ontology", "per-code lineage-group overrides (NBL)"),
    "cohort-registry": ("ontology", "first-class cohort vocabulary"),
    "cancer-fusions": ("ontology", "characteristic driver fusions per cancer type"),
    "cancer-code-burden-map": ("ontology", "anatomic burden-category overrides"),
    "cancer-incidence-mortality": ("epidemiology", "incidence/mortality by burden category"),
    "cancer-tmb": ("genomics", "median tumor mutational burden per type"),
    "cancer-apd1-response": ("response", "anti-PD-1 monotherapy ORR per type"),
    "therapy-benefit-toxicity-evidence": (
        "therapy-evidence",
        "source-anchored clinical benefit/toxicity fact rows",
    ),
    "cancer-reference-expression-samples": ("expression", "per-sample curation manifest"),
    "expression_sources": ("expression", "cohort expression-source registry"),
    # normalization references (R-norm; consumed by the clean TPM engine)
    "housekeeping-genes": ("normalization", "housekeeping panel for normalization"),
    "censored-gene-reference-tpm": (
        "normalization",
        "legacy symbol projection of the global clean-TPM censored profile",
    ),
    "clean-tpm-censored-genes": (
        "normalization",
        "global censored-gene membership and clean-PolyA reference composition",
    ),
    "histone-genes": ("normalization", "histone-cluster genes"),
    "ribosomal-protein-genes": ("normalization", "ribosomal-protein genes"),
    "ribosomal-protein-pseudogenes": ("normalization", "ribosomal-protein pseudogenes"),
    "mitochondrial-genes": ("normalization", "mtDNA-encoded genes"),
    "rrna-and-pseudogenes": ("normalization", "rRNA + rRNA pseudogenes"),
    "numt-pseudogenes": ("normalization", "NUMT-like nuclear-mito pseudogenes"),
    "small-noncoding-rnas": ("normalization", "small non-coding RNA loci"),
    "nuclear-retained-lncrnas": ("normalization", "polyA-bias nuclear-retained lncRNAs"),
    "hemoglobin-genes": ("normalization", "hemoglobin genes"),
    # gene-id / symbol resolution (R-resolve)
    "ensembl-id-aliases": ("gene-id", "alt-haplotype → primary-contig Ensembl gene ids"),
    "ncbi-symbol-synonyms": ("gene-id", "NCBI gene symbol synonyms → official symbol"),
    "extra-tx-mappings": ("gene-id", "supplemental transcript→gene mappings"),
    "cdna-identical-gene-groups": ("gene-id", "cDNA-identical gene groups"),
    "proteoform-collapse-overrides": ("gene-id", "manual proteoform-collapse overrides"),
    # Legacy compatibility tables imported before the stack boundary was clarified.
    # Keep shipped/readable for old callers, but do not treat them as the future
    # oncoref ownership surface for marker panels, target selection, or sample rules.
    "cancer-key-genes": (
        "legacy-compat",
        "legacy per-type biomarker/target rows; future purpose-specific panels live downstream",
    ),
    "cancer-driver-genes": (
        "legacy-compat",
        "legacy per-type driver-gene rows; source-anchored fact model still unresolved",
    ),
    "cancer-driver-variants": (
        "legacy-compat",
        "legacy per-type driver-variant rows; source-anchored fact model still unresolved",
    ),
    "cancer-type-genes": (
        "legacy-compat",
        "legacy role-stratified per-type gene rows; not a general panel-ownership surface",
    ),
    "cancer-viral-antigens": (
        "legacy-compat",
        "legacy per-oncovirus targetable-antigen rows; target-selection curation lives downstream",
    ),
    "disease-state-rules": (
        "legacy-compat",
        "legacy narrative disease-state rules; per-sample rule ownership belongs downstream",
    ),
    "narrative-gene-sets": (
        "legacy-compat",
        "legacy named narrative gene sets; purpose-specific gene-set ownership belongs to pirlygenes",
    ),
    "degenerate-subtype-pairs": ("ontology", "expression-degenerate subtype pairs"),
    "rare-cancer-fusion-rules": (
        "legacy-compat",
        "legacy direct fusion rules for rare cancers; per-sample interpretation belongs downstream",
    ),
    "fusion-surrogate-expression": (
        "legacy-compat",
        "legacy fusion-expression surrogate rows; sample-rule use belongs downstream",
    ),
    "fusion-expression-effects": (
        "legacy-compat",
        "legacy downstream-expression rules per fusion; sample-rule use belongs downstream",
    ),
    # expression-source metadata + genomics (R-exprmeta)
    "cancer-expression-source-candidates": ("expression", "candidate expression sources per type"),
    "cancer-frameshift-burden": ("genomics", "per-type frameshift-indel burden"),
}

#: {name: (category, description)} — pirlygenes-originated heavy artifacts in
#: the release tarball.
BUNDLE_PIRLYGENES: dict[str, tuple[str, str]] = {
    "cancer-reference-expression": ("expression", "per-cohort RNA-seq summary shards"),
    "cancer-reference-expression-percentiles": ("expression", "per-gene percentile vectors"),
    "cancer-reference-expression-representatives": ("expression", "per-cohort medoid samples"),
    "pan-cancer-expression": ("expression", "pan-cancer HPA-tissue + TCGA matrix"),
    "hpa-cell-type-expression": ("hpa", "HPA cell-type nTPM matrix"),
}

#: {name: (category, description)} — oncoref-originated bundle artifacts produced
#: by the rebuilt expression-artifact pipeline. These are not in the frozen
#: pirlygenes data inventory, but they are part of oncoref's downstream bundle
#: contract.
BUNDLE_ORIGINATED: dict[str, tuple[str, str]] = {
    "cancer-reference-expression-within-sample-top5": (
        "expression",
        "within-sample high-expression prevalence by cohort",
    ),
    "cancer-reference-expression-percentiles-proteoform-cta": (
        "expression",
        "CTA-scope proteoform-summed percentile vectors",
    ),
    "cancer-reference-expression-within-sample-top5-proteoform-cta": (
        "expression",
        "CTA-scope proteoform-summed within-sample prevalence",
    ),
    "source-matrix-sample-qc": (
        "expression",
        "per-sample QC manifest for rebuilt expression artifacts",
    ),
    "expression-artifact-build-metadata": (
        "expression",
        "per-cohort build/QC provenance for rebuilt expression artifacts",
    ),
    "expression-artifact-build-metadata.json": (
        "expression",
        "bundle-level build/QC provenance for rebuilt expression artifacts",
    ),
}

#: {name: (category, description)} — artifacts migrated from Trufflepig. The
#: importer and release provenance live in oncoref, but the pinned TCGA
#: decomposition output and legacy subtype source table originated downstream.
BUNDLE_TRUFFLEPIG: dict[str, tuple[str, str]] = {
    "tcga-deconvolved-expression.csv.gz": (
        "expression",
        "TCGA tumor-attributed TPM summaries migrated from pinned Trufflepig output",
    ),
    "subtype-deconvolved-expression.csv.gz": (
        "expression",
        "source-separated tumor-reference subtype and cohort summaries",
    ),
    "tumor-reference-expression-provenance": (
        "expression",
        "source-level derivation metadata for tumor-reference summary artifacts",
    ),
}

#: {name: (category, description)} — every heavy artifact in the release tarball.
#: Names match ``data_bundle.DOWNLOADABLE_PATHS`` (sans extension for ``.csv``);
#: kept in sync by a test.
BUNDLE: dict[str, tuple[str, str]] = {
    **BUNDLE_PIRLYGENES,
    **BUNDLE_ORIGINATED,
    **BUNDLE_TRUFFLEPIG,
}

#: {name: (category, description)} — HPA tables fetched per-source (not pirlygenes files).
HPA: dict[str, tuple[str, str]] = {
    "hpa_rna_consensus": ("hpa", "HPA RNA consensus per-tissue nTPM"),
    "hpa_normal_tissue": ("hpa", "HPA IHC protein detection per tissue"),
    "hpa_single_cell": ("hpa", "HPA single-cell-type RNA nTPM"),
}

#: The raw per-sample TPM matrices — every derived artifact is built from these.
SOURCE: dict[str, tuple[str, str]] = {
    "per-sample-tpm-matrices": ("expression", "raw per-sample cohort TPM (build inputs)"),
}

#: {name: (category, description)} — oncoref-domain fact tables still to port.
PLANNED: dict[str, tuple[str, str]] = {}

#: pirlygenes tables oncoref replaced with its own regenerated equivalent.
SUPERSEDED: dict[str, str] = {
    "protein-identical-gene-groups": "proteoform-groups-genome (byte-identical, regenerated)",
    "cta-protein-groups": "proteoform-groups (byte-identical; pirlygenes' is ≥90% identity)",
}

#: {name: (category, description)} — oncoref-ORIGINATED wheel tables: derived or
#: regenerated here rather than copied from pirlygenes, so they aren't in the
#: pirlygenes snapshot but DO ship in the wheel and belong in the inventory.
CANCERDATA_ORIGINATED: dict[str, tuple[str, str]] = {
    "hpa-safety-tissue-map": (
        "hpa",
        "source/version-aware conceptual-to-native safety-tissue mappings",
    ),
    "cancer-entity-driver-spectrum": (
        "genomics",
        "structured observed driver distributions for heterogeneous cancer entities",
    ),
    "cancer-reference-sample-molecular-provenance": (
        "genomics",
        "sample-level diagnosis and orthogonal molecular evidence provenance",
    ),
    "cancer-reference-expression-availability": (
        "expression",
        "one-row-per-source reference-expression availability manifest",
    ),
    "cancer-testis-antigens": ("cta", "CTA definition (HPA tissue-restriction over candidates)"),
    "cta-clinical-targets": (
        "cta",
        "explicit clinical/canonical CTA target tier joined to HPA evidence",
    ),
    "cta-specificity-audit": (
        "cta",
        "machine-readable CTA specificity demotion and candidate-only audit decisions",
    ),
    "proteoform-groups": ("gene-id", "byte-identical CTA proteoform groups"),
    "proteoform-groups-genome": ("gene-id", "byte-identical proteoform groups (genome-wide)"),
    "source-matrices": (
        "expression",
        "per-cohort raw-matrix registry (code/source/n_samples/release)",
    ),
    "expression-artifact-gene-universe-deltas": (
        "expression",
        "known pirlygenes/oncoref expression artifact row-universe deltas",
    ),
    "representative-source-adjudications": (
        "expression",
        "reviewed source-sample roles and benchmark eligibility for representative artifacts",
    ),
    "ncbi-entrez-gene-mappings": (
        "gene-id",
        "filtered NCBI Entrez/GeneID -> canonical Ensembl gene mappings",
    ),
    "tissue-burden-map": ("ontology", "primary_tissue -> anatomic burden category"),
    "family-burden-map": ("ontology", "registry family -> burden category (fallback)"),
    "cancer-mismatch-repair-statuses": (
        "ontology",
        "cross-cancer MMRd/MSI-H vs pMMR/MSS-like classifier state semantics",
    ),
    "cancer-who-soft-tissue-bone-audit": (
        "ontology",
        "WHO soft-tissue/bone category and behavior parity audit",
    ),
    "cta-ihc-unreliable": ("cta", "CTAs whose HPA IHC is cross-reactive (RNA-only fallback)"),
    "clean-tpm-biological-housekeeping-genes": (
        "normalization",
        "HPA-stable biological housekeeping denominator candidates for clean TPM",
    ),
    "cancer-response-signatures": (
        "legacy-compat",
        "transitional response-signature surface; new/extended therapy panels belong in pirlygenes",
    ),
}

#: pirlygenes/trufflepig data that is NOT oncoref's domain — target selection /
#: therapy modality catalogs, marker/signature panels, surfaceome, and per-sample
#: interpretation/rule tables. These stay downstream and key to oncoref IDs.
OUT_OF_SCOPE: frozenset[str] = frozenset(
    {
        # therapeutic modality / drug data
        "ADC-approved",
        "ADC-trials",
        "ADC-withdrawn",
        "bispecific-antibodies-approved",
        "CAR-T-approved",
        "TCR-T-approved",
        "TCR-T-trials",
        "multispecific-tcell-engager-trials",
        "radioligand-targets",
        # surfaceome / targetability
        "surface-proteins",
        "cancer-surfaceome",
        # therapy signatures / evidence
        "therapy-response-signatures",
        "estimate-signatures",
        "immune-receptor-segments",
        # analysis / QC gene sets
        "tme-markers",
        "stem-cell-marker-panels",
        "culture-stress-genes",
        "ffpe-sensitive-markers",
        "degradation-gene-pairs",
        "mutation-expression-effects",
        "rare-cancer-rna-surrogates",
        # differential-expression signatures
        "tumor-up-vs-matched-normal",
        "heme-tumor-up-vs-matched-normal",
        # gene-set panels (analysis / target selection)
        "cancer-lineage-panels",
        "cancer-family-panels",
        "lineage-genes",
        "gene-sets",
        # build QC
        "artifact-expectations",
    }
)

#: Provenance for :data:`PIRLYGENES_MIGRATION_SNAPSHOT`. This exact commit's
#: ``pirlygenes/data`` tree normalizes to the snapshot below. It is historical
#: migration evidence, not a live cross-repository completeness promise.
PIRLYGENES_MIGRATION_SNAPSHOT_METADATA: dict[str, str] = {
    "repository": "pirl-unc/pirlygenes",
    "ref": "v5.22.80",
    "commit": "bb6679e39029218e93acb2e54317a8d4ff98f424",
    "data_path": "pirlygenes/data",
    "captured_at": "2026-06-12",
}

#: Exact normalized inventory of ``pirlygenes/data`` at the pinned migration
#: commit above. Use ``scripts/audit_pirlygenes_inventory.py`` to reproduce this
#: snapshot or compare it with a newer pirlygenes ref.
PIRLYGENES_MIGRATION_SNAPSHOT: frozenset[str] = frozenset(
    {
        "ADC-approved",
        "ADC-trials",
        "ADC-withdrawn",
        "artifact-expectations",
        "bispecific-antibodies-approved",
        "cancer-apd1-response",
        "cancer-code-burden-map",
        "cancer-cohort-aggregates",
        "cancer-driver-genes",
        "cancer-driver-variants",
        "cancer-expression-source-candidates",
        "cancer-family-panels",
        "cancer-frameshift-burden",
        "cancer-fusions",
        "cancer-incidence-mortality",
        "cancer-key-genes",
        "cancer-lineage-group-overrides",
        "cancer-lineage-groups",
        "cancer-lineage-panels",
        "cancer-reference-expression",
        "cancer-reference-expression-percentiles",
        "cancer-reference-expression-representatives",
        "cancer-reference-expression-samples",
        "cancer-subtype-groupings",
        "cancer-surfaceome",
        "cancer-tmb",
        "cancer-type-genes",
        "cancer-type-registry",
        "cancer-viral-antigens",
        "CAR-T-approved",
        "cdna-identical-gene-groups",
        "censored-gene-reference-tpm",
        "clean-tpm-censored-genes",
        "cohort-registry",
        "cta-protein-groups",
        "culture-stress-genes",
        "degenerate-subtype-pairs",
        "degradation-gene-pairs",
        "disease-state-rules",
        "ensembl-id-aliases",
        "estimate-signatures",
        "expression_sources",
        "extra-tx-mappings",
        "ffpe-sensitive-markers",
        "fusion-expression-effects",
        "fusion-surrogate-expression",
        "gene-sets",
        "heme-tumor-up-vs-matched-normal",
        "hemoglobin-genes",
        "histone-genes",
        "housekeeping-genes",
        "hpa-cell-type-expression",
        "immune-receptor-segments",
        "lineage-genes",
        "mitochondrial-genes",
        "multispecific-tcell-engager-trials",
        "mutation-expression-effects",
        "narrative-gene-sets",
        "ncbi-symbol-synonyms",
        "nuclear-retained-lncrnas",
        "numt-pseudogenes",
        "pan-cancer-expression",
        "protein-identical-gene-groups",
        "proteoform-collapse-overrides",
        "radioligand-targets",
        "rare-cancer-fusion-rules",
        "rare-cancer-rna-surrogates",
        "ribosomal-protein-genes",
        "ribosomal-protein-pseudogenes",
        "rrna-and-pseudogenes",
        "small-noncoding-rnas",
        "stem-cell-marker-panels",
        "surface-proteins",
        "TCR-T-approved",
        "TCR-T-trials",
        "therapy-benefit-toxicity-evidence",
        "therapy-response-signatures",
        "tme-markers",
        "tumor-up-vs-matched-normal",
    }
)

# Historical compatibility alias. New code should use the explicitly named
# snapshot so it cannot be mistaken for pirlygenes' current inventory.
PIRLYGENES_DATA = PIRLYGENES_MIGRATION_SNAPSHOT


def captured() -> set[str]:
    """Datasets already held or shipped for compatibility."""
    return set(WHEEL) | set(BUNDLE) | set(SUPERSEDED)


def in_scope() -> set[str]:
    """Base-layer inventory: captured datasets plus still-planned fact tables."""
    return captured() | set(PLANNED)


def owned_dataset_names() -> frozenset[str]:
    """Current oncoref-owned dataset names across every storage backend."""

    return frozenset(
        set(WHEEL)
        | set(BUNDLE)
        | set(HPA)
        | set(SOURCE)
        | set(PLANNED)
        | set(CANCERDATA_ORIGINATED)
    )
