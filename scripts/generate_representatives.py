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

"""Build the per-cohort *representative-samples* artifact (n=5 medoids).

The packaged cohort references are aggregates (median/percentile vectors); some
consumers need a few *real* joint per-sample vectors per cohort — e.g. to show
co-expression that an aggregate washes out. For each cohort this keeps ``k``
medoid samples: the cohort medoid (most central tumor) first, then farthest-first
picks that span the within-cohort variation (see ``expression_builders.cohort_medoids``).

Computed from the full per-sample matrices, which are never shipped (see
``source_matrices`` for the per-cohort fetch). The medoid distance geometry uses
the biological clean-TPM view with technical/ribosomal rows removed, but the kept
columns store the **original** full clean TPM vectors (the reader optionally
``log1p``-transforms).

Input
-----
A directory of per-cohort parquet files, one per cohort code
(``<INPUT>/<CODE>.parquet``), each with ``Ensembl_Gene_ID`` + ``Symbol`` columns
and one column per sample (clean TPM).

Output
------
``oncoref/data/cancer-reference-expression-representatives/<CODE>.parquet``
with ``Ensembl_Gene_ID, Symbol`` and up to ``k`` representative columns named
``<CODE>__rep{i}`` (medoid first), plus a shared ``_provenance.csv`` mapping each
``representative_id`` to its source sample, cohort, and cohort size — the columns
``expression.representative_cohort_samples(include_provenance=True)`` reads back.

Run:
    python scripts/generate_representatives.py --input <per-sample-dir> [--k 5]

After building, ensure ``cancer-reference-expression-representatives`` is in
``data_bundle.DOWNLOADABLE_PATHS``, rebuild + upload the data tarball, and bump
``DATA_VERSION`` (never bump it before the tarball is uploaded — a 404 hangs fetch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.expression_builders import cohort_medoids, sample_columns
from oncoref.gene_families import clean_tpm_censored_gene_ids
from oncoref.representative_partitions import assign_representative_partitions
from oncoref.source_matrices import source_sample_namespace

_DATA_DIR = Path(__file__).resolve().parents[1] / "oncoref" / "data"
OUT_DIR = _DATA_DIR / "cancer-reference-expression-representatives"
_BASE = ["Ensembl_Gene_ID", "Symbol"]
#: Columns representative_cohort_samples(include_provenance=True) merges back in.
_PROVENANCE_COLUMNS = [
    "cancer_code",
    "representative_id",
    "source_cohort",
    "source_project",
    "source_sample",
    "source_group_id",
    "n_cohort_samples",
    "representative_role",
    "benchmark_eligible",
]


def _drop_technical(df: pd.DataFrame) -> pd.DataFrame:
    censored = clean_tpm_censored_gene_ids()
    unversioned = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    return df[~unversioned.isin(censored)].reset_index(drop=True)


def _cohort_provenance(code: str) -> tuple[str, str]:
    """Best-effort ``(source_cohort, source_project)`` for a cancer code.

    Looks the code up in the source-matrices registry (``cancer_code ->
    source_cohort``) and that cohort up in the cohort registry (``-> source_project``).
    Either lookup failing — an unregistered code, or running this generator on an
    arbitrary input dir — falls back gracefully so the column is always present.
    """
    source_cohort, source_project = code, ""
    try:
        from oncoref.source_matrices import cohort_info

        info = cohort_info(code)
        source_cohort = str(info.get("source_cohort") or code)
    except Exception:
        return source_cohort, source_project
    try:
        from oncoref.cancer_types import cohort_registry

        entry = cohort_registry().get(source_cohort, {})
        source_project = str(entry.get("source_project") or "")
    except Exception:
        pass
    return source_cohort, source_project


def build(input_dir: Path, *, k: int = 5, out_dir: Path = OUT_DIR) -> None:
    """Build the representative-samples artifact for each cohort under ``input_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(input_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"no per-sample parquet files under {input_dir}")
    n = 0
    provenance: list[dict] = []
    for shard in shards:
        code = shard.stem
        df = pd.read_parquet(shard)
        n_cohort = len(sample_columns(df))
        if n_cohort == 0:
            print(f"  {code}: no sample columns, skipped", flush=True)
            continue
        sample_cols = sample_columns(df)
        selection_df = _drop_technical(df)
        reps = cohort_medoids(df, sample_cols=sample_cols, k=k, selection_df=selection_df)
        source_cols = [c for c in reps.columns if c not in _BASE]
        rep_ids = [f"{code}__rep{i}" for i in range(1, len(source_cols) + 1)]
        reps = reps.rename(columns=dict(zip(source_cols, rep_ids)))
        reps.to_parquet(out_dir / f"{code}.parquet", index=False, compression="zstd")
        source_cohort, source_project = _cohort_provenance(code)
        for rep_id, src in zip(rep_ids, source_cols):
            provenance.append(
                {
                    "cancer_code": code,
                    "representative_id": rep_id,
                    "source_cohort": source_cohort,
                    "source_project": source_project,
                    "source_sample": src,
                    "source_group_id": f"{source_sample_namespace(source_cohort)}:{src}",
                    "n_cohort_samples": n_cohort,
                    "representative_role": "standard",
                    "benchmark_eligible": True,
                }
            )
        n += 1
        print(f"  {code}: {len(rep_ids)} reps of {n_cohort} samples", flush=True)
    prov_df = assign_representative_partitions(
        pd.DataFrame(provenance, columns=_PROVENANCE_COLUMNS)
    )
    prov_df.to_csv(out_dir / "_provenance.csv", index=False)
    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.parquet")) / 1e6
    print(f"\ndone: {n} cohorts, {total_mb:.1f} MB -> {out_dir}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", required=True, type=Path, help="Dir of per-cohort per-sample parquet files"
    )
    p.add_argument(
        "--k", type=int, default=5, help="Representatives to keep per cohort (default 5)"
    )
    args = p.parse_args(argv)
    build(args.input, k=args.k)


if __name__ == "__main__":
    main()
