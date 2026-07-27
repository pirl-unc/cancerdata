# Raw SRA Expression Sources

Some public RNA-seq studies deposit raw reads without a processed expression
matrix. Oncoref handles these as explicit build-time sources: the registry pins
the complete run inventory, read checksums, sample roles, and transcriptome
reference; `scripts/build_sra_salmon_source.py` turns those inputs into the same
canonical source-matrix artifacts used by the other expression builders.

Salmon is required only to regenerate a raw-read source. It is not imported by
oncoref and is not a runtime package dependency.

## MMNST Source

`prjna1083972-mmnst` is the first `source_type: sra-salmon` entry. BioProject
`PRJNA1083972` / study `SRP493407` contains three malignant melanotic nerve
sheath tumors and three matched normal-nerve controls.

The build always quantifies and audits all six runs. Only the three runs whose
registry records declare both `role: tumor` and `cancer_code: SARC_MMNST` enter
the tumor matrix. Matched-normal records cannot declare a cancer code, so a
configuration error fails during registry parsing instead of contaminating the
reference cohort.

## Build

Install Salmon, then run:

```bash
python scripts/build_sra_salmon_source.py prjna1083972-mmnst --threads 8
```

The default cache is
`~/.cache/oncoref/expression/prjna1083972-mmnst/`. Allow roughly 18 GB for the
paired FASTQs, Ensembl transcriptome, Salmon index, quantifications, and derived
artifacts.

Every FASTQ is downloaded from its pinned ENA INSDC mirror URL and verified
against the deposited MD5. The Ensembl release 112 GRCh38 cDNA and ncRNA FASTAs
are each checksum-pinned, then deterministically combined and verified against a
third SHA-256. Salmon runs with automatic library detection, mapping validation,
sequence bias correction, and GC bias correction. Transcript TPM is summed by
the `gene:` identifier in that combined Ensembl FASTA before oncoref
canonicalizes gene IDs and computes sample QC. Interrupted downloads retain
their partial file and resume with an HTTP range request when the mirror supports
it.

Reviewed Salmon output can be reused without rerunning quantification:

```bash
python scripts/build_sra_salmon_source.py prjna1083972-mmnst \
  --quant-dir /path/to/quant \
  --transcriptome /path/to/Homo_sapiens.GRCh38.cdna_ncrna.fa.gz
```

`--quant-dir` must contain `<run-accession>/quant.sf` for every declared tumor
and control run. The transcriptome checksum is still enforced.

## Outputs

The derived directory contains:

- `SARC_MMNST_per_sample_tpm.parquet`: the three tumor runs only.
- `SRP493407_MMNST_2024_run_manifest.csv`: all six roles, routes, URLs, and
  checksums.
- `SRP493407_MMNST_2024_salmon_quantification_audit.csv`: transcript mapping
  coverage, Salmon version, inferred library type, and TPM conservation for
  every run.
- The standard mapping, parse, sample-QC, and summary-row sidecars consumed by
  the expression artifact rebuild.

The source is an ultra-rare cohort (`n=3`), so its dispersion and tail
percentiles are intrinsically thin. The registry preserves its
`salmon_gene_tpm` source-scale class so consumers can distinguish it from
alignment/count-based cohorts when absolute cross-source comparisons matter.
