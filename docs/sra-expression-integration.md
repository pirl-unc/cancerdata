# SRA Expression Sources

Oncoref prefers an authoritative processed expression product when one is
available for a public SRA study. A processed source avoids reproducing a large
alignment or quantification workflow, while registry-pinned checksums preserve a
reviewable connection to the exact upstream files.

The `sra-ncbi-counts` builder consumes NCBI's public Gene Feature RNA-seq count
tables. It does not download raw reads and does not run Salmon or another
aligner or quantifier. Raw-read `sra-salmon` support remains available for
studies that have no suitable processed product, but it is a separate fallback
contract.

## MMNST Cohort

`prjna1083972-mmnst` provides the `SARC_MMNST` reference cohort. BioProject
`PRJNA1083972` / SRA study `SRP493407` contains three primary malignant melanotic
nerve sheath tumors and three normal-nerve controls. The controls are independent
donors, not patient-matched specimens.

All six NCBI analysis accessions and count-file MD5 digests are pinned in
`expression_sources.yaml`. All six runs appear in the run manifest and mapping
coverage audit. Only records with both `role: tumor` and
`cancer_code: SARC_MMNST` enter the released tumor matrix. A normal-control
record cannot declare a cancer code, so an invalid route fails when the registry
is parsed.

The cohort is described in [PMID 41995777](https://pubmed.ncbi.nlm.nih.gov/41995777/)
and deposited under
[PRJNA1083972](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1083972). The paper
reports oligo(dT) mRNA enrichment, while the deposited INSDC records say
`RANDOM`; the registry preserves this discrepancy instead of choosing one
description silently.

## VSCC Cohort

`prjna994918-vscc` provides the direct `VSCC` reference from BioProject
`PRJNA994918` / SRA study `SRP449588`. The paper reports 13
pathology-confirmed invasive vulvar squamous cell carcinomas; nine tumors had
RIN above 6 and received poly(A), paired-end bulk RNA sequencing. The selected
RNA libraries represent six tumors reported as primary and three as recurrent,
and the Tumor 2 specimen was taken from an unspecified metastatic site.

Every NCBI analysis uses `scf_rnaseq_gene_counts` 0.5.6 against
`GCF_000001405.40`, with paired RF libraries and reported alignment rates from
55.21% to 70.15%. The registry pins all nine run/analysis pairs and count-file
MD5 digests. All nine profiles pass the expression QC policy and map 95.7% to
98.3% of source counts into the reference input. The resulting 44,446-gene
matrix is a direct expression reference, but its small mixed-origin composition
keeps `VSCC` non-classifying.

The source's public molecular-provenance rows retain all 13 study tumors. They
record direct hybridization-capture evidence for three PCR-confirmed HPV16
integrations, two HPV coinfections without detected human-virus junctions, and
eight HPV-DNA-negative tumors. HPV status is never inferred from diagnosis or
expression. The cohort is described in
[PMID 38898085](https://pubmed.ncbi.nlm.nih.gov/38898085/) and deposited under
[PRJNA994918](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA994918).

## Processed-Count Contract

NCBI publishes one unnormalized gene-level count table per SRA run in its
[SRA Gene Feature RNA-seq counts archive](https://registry.opendata.aws/ncbi-sra-rnaseq/).
For each run, the oncoref registry pins:

- the run and NCBI analysis accessions;
- the BioSample, sample title, and tumor/control role;
- the immutable count-table path and MD5 digest.

The generic [NCBI GEO pipeline
page](https://www.ncbi.nlm.nih.gov/geo/info/rnaseqcounts.html) still names
GRCh38.p13 and Human Annotation Release `109.20190905`. These S3 analyses were
generated in June 2026 and contain 6,574 GeneIDs absent from that release,
including expressed rows. Their 59,068-row universe has length parity for all
but 15 rows with `GCF_000001405.40-RS_2025_08`, so the registry pins that GFF and
assembly report by URL and SHA-256 and records the discrepancy rather than
silently dropping the newer genes.

The builder streams the compressed GFF and retains intervals only for GeneIDs
present in the count tables. Gene length is the exon union across RefSeq
accessions assigned to `Primary Assembly`, plus the mitochondrial `non-nuclear`
accession. This selection reproduces NCBI's published raw-count-to-TPM output;
alternate loci and patch scaffolds would incorrectly multiply some lengths. A
gene or pseudogene without exon records uses its gene span. Retired NCBI GeneIDs
use the current live GeneID's length when NCBI Gene history provides an exact
replacement.

Only rows with both a canonical Ensembl mapping and a resolved gene length enter
TPM normalization. The mapping audit retains every source row, and the per-run
coverage sidecar records the source-count fraction that entered the reference.
Reviewed translations reconcile NCBI mitochondrial symbols such as `COX1` and
`RNR2` with canonical `MT-CO1` and `MT-RNR2`; aggregate loci such as `IGH`,
`IGK`, and `IGL` remain unresolved rather than being assigned to an arbitrary
member gene.

## Build

Run the processed-count builder:

```bash
python scripts/build_sra_ncbi_counts_source.py prjna1083972-mmnst
python scripts/build_sra_ncbi_counts_source.py prjna994918-vscc
```

The default cache is
`~/.cache/oncoref/expression/<source-id>/`. Each build downloads approximately
80 MB when the shared RefSeq annotation is not already cached; the count tables
themselves are small.

Reviewed local inputs can be supplied without network access:

```bash
python scripts/build_sra_ncbi_counts_source.py prjna1083972-mmnst \
  --counts-dir /path/to/ncbi-counts \
  --annotation /path/to/GCF_000001405.40_GRCh38.p14_genomic.gff.gz \
  --assembly-report /path/to/GCF_000001405.40_GRCh38.p14_assembly_report.txt
```

`--counts-dir` must contain
`<run-accession>.ncbi-gene-counts.tsv` for every declared run. Local inputs are
still verified against the registry checksums.

## Outputs

The derived directory contains:

- `SARC_MMNST_per_sample_tpm.parquet`: the three tumor runs only.
- `SRP493407_MMNST_2024_run_manifest.csv`: all six analyses, roles, routes,
  URLs, and checksums.
- `SRP493407_MMNST_2024_ncbi_count_mapping_coverage.csv`: raw-count mapping and
  reference-input coverage for every run.
- `SRP493407_MMNST_2024_ncbi_gene_length_audit.csv`: the length and fallback
  method for every NCBI GeneID.
- The standard mapping, parse, sample-QC, and summary-row sidecars used by the
  expression artifact rebuild.

This is an ultra-rare cohort (`n=3`). Its dispersion and tail percentiles are
necessarily thin even though all three tumor samples pass the expression QC
policy.

For `prjna994918-vscc`, the analogous matrix and sidecars are named
`VSCC_per_sample_tpm.parquet` and `SRP449588_VSCC_2024_*`. The matrix contains
nine tumors, while the separate molecular-provenance API retains all 13 study
tumors.

## Raw-Read Fallback

`SraSalmonSource` and `scripts/build_sra_salmon_source.py` remain the generic
fallback for an SRA study that has no reviewed processed matrix or NCBI count
product. That contract pins paired-read checksums, a transcriptome checksum,
Salmon arguments, and the Salmon version used by each index and quantification
cache. Salmon is not imported by oncoref and is not a runtime dependency.

The MMNST source does not use this fallback.
