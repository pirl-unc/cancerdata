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

"""Stage the per-cohort source matrices for ``oncoref.source_matrices``.

For each cancer code in ``source-matrices.csv``, find its single selected source
cohort's raw per-sample matrix in a local matrix cache and copy it to:

  - the ``source_matrices`` on-disk cache (``<CODE>.parquet``) so the accessor +
    per-patient coverage analyses work locally without a release; and
  - (with ``--release-dir``) an upload staging dir as ``<CODE>_per_sample_tpm.parquet``,
    the exact asset names ``source_matrices.release_url`` expects on the
    ``source-v<SOURCE_MATRIX_VERSION>`` GitHub release.

The matrix cache is laid out ``<cache>/<cohort>/derived/<NAME>_per_sample_tpm.parquet``.
The cohort directory is matched to the registry's ``source_cohort`` by GSE accession
or normalized name (the same single-source choice the rebuild driver makes), so a
multi-source code stages exactly the one matrix the shipped artifacts were built on.

Run:
    python scripts/stage_source_matrices.py --cache ~/.cache/pirlygenes/expression \
        [--existing-cache ~/.cache/oncoref/source-matrices/v<previous-version>] \
        [--release-dir ~/oncoref-source-upload] [--codes LUAD,SKCM] [--limit N]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref import source_matrices as sm


def _source_key(name: str) -> str:
    m = re.search(r"GSE\d+", name.upper())
    return m.group() if m else re.sub(r"[^A-Z0-9]", "", name.upper())


def _code_key(name: str) -> str:
    stem = name.replace("_per_sample_tpm.parquet", "")
    stem = stem[5:] if stem.lower().startswith("tcga_") else stem
    return stem.replace("_", "").lower()


def _matrices_by_source(cache: Path) -> dict[str, list[Path]]:
    """``{source_key -> [matrix paths]}`` for every per-sample matrix in the cache
    (sorted, so candidate order is deterministic across machines)."""
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(cache.glob("*/derived/*_per_sample_tpm.parquet")):
        out[_source_key(p.parent.parent.name)].append(p)
    return out


def _matrices_by_code(cache: Path) -> dict[str, list[Path]]:
    """Candidate matrices indexed by normalized filename cancer code."""
    out: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(cache.glob("*/derived/*_per_sample_tpm.parquet")):
        out[_code_key(path.name)].append(path)
    return out


def _sample_count(path: Path) -> int:
    """Return the number of sample columns without loading the matrix."""
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    return len([name for name in names if name not in {"Ensembl_Gene_ID", "Symbol"}])


def _select_matrix(
    code: str,
    source_key: str,
    *,
    by_source: dict[str, list[Path]],
    by_code: dict[str, list[Path]],
    source_code_counts: dict[str, int],
    existing_cache: Path | None,
) -> Path | None:
    """Select one matrix using source match, exact code match, then prior cache."""
    source_candidates = by_source.get(source_key, [])
    exact_source_match = _match_code(code, source_candidates)
    if exact_source_match is not None:
        return exact_source_match

    if len(source_candidates) == 1 and source_code_counts[source_key] == 1:
        # A single-code source may use a source-specific filename. Shared sources
        # require an exact filename match so one cohort cannot stand in for another.
        return source_candidates[0]

    code_candidates = by_code.get(_code_key(code), [])
    if len(code_candidates) == 1:
        return code_candidates[0]

    prior = existing_cache / f"{code}.parquet" if existing_cache is not None else None
    return prior if prior is not None and prior.exists() else None


def stage(
    cache: Path,
    *,
    release_dir: Path | None,
    codes: list[str] | None,
    limit: int | None,
    existing_cache: Path | None = None,
) -> None:
    reg = sm.registry()
    by_source = _matrices_by_source(cache)
    by_code = _matrices_by_code(cache)
    source_code_counts = (
        reg.assign(_source_key=reg["source_cohort"].astype(str).map(_source_key))
        .groupby("_source_key")["cancer_code"]
        .nunique()
        .to_dict()
    )
    rows = reg.to_dict("records")
    if codes:
        wanted = {c.upper() for c in codes}
        rows = [r for r in rows if str(r["cancer_code"]).upper() in wanted]
    if limit:
        rows = rows[:limit]

    if release_dir is not None:
        release_dir.mkdir(parents=True, exist_ok=True)

    selections: list[tuple[str, Path]] = []
    missing = []
    for r in rows:
        code = str(r["cancer_code"])
        src_key = _source_key(str(r["source_cohort"]))
        chosen = _select_matrix(
            code,
            src_key,
            by_source=by_source,
            by_code=by_code,
            source_code_counts=source_code_counts,
            existing_cache=existing_cache,
        )
        if chosen is None:
            missing.append((code, r["source_cohort"]))
            continue
        expected_samples = r.get("n_samples")
        if expected_samples is not None:
            try:
                expected_samples = int(expected_samples)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{code}: registry n_samples is invalid") from exc
            actual_samples = _sample_count(chosen)
            if actual_samples != expected_samples:
                raise ValueError(
                    f"{code}: selected matrix has {actual_samples} samples; "
                    f"registry expects {expected_samples}: {chosen}"
                )
        selections.append((code, chosen))

    if missing:
        print(f"MISSING ({len(missing)}): {missing}", flush=True)
        raise FileNotFoundError(f"source matrices are missing for {len(missing)} cohort(s)")

    for code, chosen in selections:
        cache_path = sm.local_path(code)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chosen, cache_path)
        if release_dir is not None:
            shutil.copyfile(chosen, release_dir / f"{code}_per_sample_tpm.parquet")
        print(f"  {code}: <- {chosen.parent.parent.name}/{chosen.name}", flush=True)

    print(f"\nstaged {len(selections)}/{len(rows)} cohorts into versioned caches", flush=True)
    if release_dir is not None:
        print(f"release assets -> {release_dir}", flush=True)


def _match_code(code: str, candidates: list[Path]) -> Path | None:
    """The candidate matrix whose filename stem maps to ``code`` (a source dir with
    several code-matrices, e.g. treehouse). Returns ``None`` if the match isn't
    unique — the caller reports that as missing rather than staging an arbitrary
    matrix (silently shipping the wrong cohort would corrupt every analysis)."""

    want = _code_key(code)
    hits = [p for p in candidates if _code_key(p.name) == want]
    return hits[0] if len(hits) == 1 else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, type=Path, help="Local per-sample matrix cache root")
    p.add_argument("--release-dir", type=Path, default=None, help="Also write upload-named assets")
    p.add_argument(
        "--existing-cache",
        type=Path,
        default=None,
        help="Prior version cache used only when a builder-cache matrix is absent",
    )
    p.add_argument("--codes", type=str, default=None, help="Comma-separated codes (default all)")
    p.add_argument("--limit", type=int, default=None, help="Only the first N registry rows")
    args = p.parse_args(argv)
    stage(
        args.cache.expanduser(),
        release_dir=args.release_dir.expanduser() if args.release_dir else None,
        codes=args.codes.split(",") if args.codes else None,
        limit=args.limit,
        existing_cache=args.existing_cache.expanduser() if args.existing_cache else None,
    )


if __name__ == "__main__":
    main()
