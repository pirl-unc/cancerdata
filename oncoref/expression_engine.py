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

"""Transcript→gene expression aggregation (the pandas-only grouping core).

Sums transcript-level TPM to gene level given a transcript→gene mapping. This is
the part of the operation that is oncoref's domain — the deterministic grouping
and TPM summation — independent of which transcript reference produced the quant.

**Dependency boundary.** Resolving an *arbitrary, unknown* transcript ID to a gene
(and a gene symbol to its Ensembl ID) is a reference-genome operation that needs
``pyensembl`` — out of oncoref's pandas-only base layer. So this function maps
transcripts via the supplied ``tx_to_gene_name`` dict (default: oncoref's curated
``extra-tx-mappings`` back-compat set) and reports the unresolved TPM fraction;
transcripts not in the map are summed into an ``unresolved`` bucket rather than
silently dropped. A consumer that needs full Ensembl-reference resolution passes a
complete ``tx_to_gene_name`` (e.g. built from pyensembl on its side).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import pandas as pd

#: Canonical identity columns of a oncoref expression frame. Everything else is a
#: per-sample / per-representative value column. One definition shared by every
#: "value columns = all columns except the id columns" consumer (the build
#: generators, the normalization helpers, the coverage hit-matrix) so a
#: proteoform-collapsed frame's identity columns (``proteoform_key`` / ``Symbol`` /
#: ``proteoform_members``) are never mistaken for samples. (The *named*-TPM rule for
#: curated frames is :func:`is_expression_value_col`, a deliberately distinct concept.)
ID_COLUMNS = ("proteoform_key", "Ensembl_Gene_ID", "Symbol", "proteoform_members")


def id_columns(df: pd.DataFrame) -> list[str]:
    """The identity columns of an expression frame, in canonical :data:`ID_COLUMNS`
    order, restricted to those actually present (a gene-level frame lacks the
    ``proteoform_*`` columns). The single definition of "which columns are identity",
    so no consumer has to re-list them."""
    return [c for c in ID_COLUMNS if c in df.columns]


def sample_columns(df: pd.DataFrame) -> list[str]:
    """The per-sample / per-representative **value** columns of an expression frame:
    every column that is not one of the :data:`ID_COLUMNS`. The single definition of
    "which columns hold expression values" — the partner of :func:`id_columns`, and
    the one place the value/identity boundary is decided. (For the distinct *named*-TPM
    rule on curated long tables, see :func:`is_expression_value_col`.)"""
    return [c for c in df.columns if c not in ID_COLUMNS]


_DEFAULT_TX_COLUMN_CANDIDATES = (
    "transcript",
    "transcript_id",
    "transcriptid",
    "target",
    "target_id",
    "targetid",
    "name",
)
_DEFAULT_TPM_COLUMN_CANDIDATES = ("tpm",)
_DEFAULT_GENE_ROW_ID_COLUMN_CANDIDATES = (
    "Ensembl_Gene_ID",
    "ensembl_gene_id",
    "gene_id",
    "gene",
    "Gene",
    "id",
    "ID",
    "target_id",
    "name",
)
_DEFAULT_GENE_SYMBOL_COLUMN_CANDIDATES = (
    "Symbol",
    "symbol",
    "gene_symbol",
    "gene_name",
    "external_gene_name",
)

SOURCE_GENE_MAPPING_AUDIT_COLUMNS = (
    "source_row_index",
    "source_row_id",
    "source_row_id_type",
    "source_symbol",
    "mapping_status",
    "mapping_method",
    "canonical_ensembl_gene_id",
    "canonical_symbol",
    "unresolved_reason",
    "source_value_sum",
    "source_expression_max",
    "source_expression_sample_with_max",
    "source_expression_nonzero_samples",
    "high_expression_unresolved",
)

SOURCE_VALUE_PARSE_DIAGNOSTIC_COLUMNS = (
    "value_col",
    "n_values",
    "n_input_missing",
    "n_parse_missing",
    "n_literal_zero",
    "parse_missing_fraction",
    "literal_zero_fraction",
)


def find_column(df: pd.DataFrame, candidates, column_name: str) -> str:
    """The column of ``df`` matching the highest-priority ``candidates`` entry —
    absorbs naming variation across upstream quantifiers (``transcript_id`` vs ``tx``
    …). ``candidates`` is consulted **in order**, so a frame carrying both
    ``transcript_id`` and ``name`` resolves by candidate priority, not by column
    order. Matching is case-insensitive. Raises ``ValueError`` listing the available
    columns if nothing matches."""
    by_lower = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        col = by_lower.get(cand.lower())
        if col is not None:
            return col
    raise ValueError(
        f"no column for {column_name} in expression data; available: {list(df.columns)}"
    )


def _find_optional_column(df: pd.DataFrame, candidates) -> str | None:
    by_lower = {str(col).lower(): col for col in df.columns}
    for cand in candidates:
        col = by_lower.get(str(cand).lower())
        if col is not None:
            return col
    return None


def _source_row_columns(
    df: pd.DataFrame,
    row_id_col: str | None,
    symbol_col: str | None,
    *,
    require_row_id: bool,
) -> tuple[str | None, str | None]:
    if row_id_col is not None and row_id_col not in df.columns:
        raise ValueError(f"source row ID column {row_id_col!r} is not in expression data")
    if symbol_col is not None and symbol_col not in df.columns:
        raise ValueError(f"source symbol column {symbol_col!r} is not in expression data")

    detected_symbol = symbol_col or _find_optional_column(
        df, _DEFAULT_GENE_SYMBOL_COLUMN_CANDIDATES
    )
    detected_id = row_id_col or _find_optional_column(df, _DEFAULT_GENE_ROW_ID_COLUMN_CANDIDATES)
    if detected_id is None:
        detected_id = detected_symbol
    if detected_id is None and require_row_id:
        raise ValueError(
            "no source gene row ID or symbol column in expression data; "
            f"available: {list(df.columns)}"
        )
    if detected_symbol == detected_id:
        detected_symbol = None
    return detected_id, detected_symbol


def _nonempty_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def detect_source_row_id_type(values: Iterable) -> str:
    """Classify source expression row identifiers.

    Returns one of ``"ensembl_gene_id"``, ``"ensembl_transcript_id"``,
    ``"entrez_id"``, ``"symbol"``, ``"mixed"``, or ``"unknown"``. The detector is
    deliberately conservative: it is used to annotate builder audit rows, not to
    silently decide a lossy mapping path.
    """
    s = pd.Series(list(values), dtype="object").map(_nonempty_text).dropna()
    if s.empty:
        return "unknown"
    upper = s.str.upper()
    ensg = upper.str.match(r"^ENSG\d+(?:\.\d+)?(?:_PAR_[XY])?$")
    enst = upper.str.match(r"^ENST\d+(?:\.\d+)?$")
    entrez = upper.str.match(r"^\d+$")
    leading_digit_ncrna = upper.str.match(r"^(?:\d+SK|\d+SL|\d+(?:_\d+)?S(?:_RRNA)?)$")
    symbol = (upper.str.match(r"^[A-Z][A-Z0-9_.-]*$") | leading_digit_ncrna) & ~(
        ensg | enst | entrez
    )
    patterns = {
        "ensembl_gene_id": ensg,
        "ensembl_transcript_id": enst,
        "entrez_id": entrez,
        "symbol": symbol,
    }
    fractions = {name: float(mask.mean()) for name, mask in patterns.items()}
    best, frac = max(fractions.items(), key=lambda kv: kv[1])
    return best if frac >= 0.8 else "mixed"


@lru_cache(maxsize=1)
def _canonical_gene_index() -> dict[str, tuple[str, str]]:
    from .gene_ids import canonical_gene_space, unversioned

    df = canonical_gene_space()
    return {
        unversioned(gid): (unversioned(gid), str(sym))
        for gid, sym in zip(df["ensembl_gene_id"], df["symbol"])
    }


@lru_cache(maxsize=1)
def _symbol_to_gene_index() -> tuple[dict[str, tuple[str, str]], set[str]]:
    from .gene_ids import canonical_gene_space, unversioned
    from .load_dataset import get_data

    df = canonical_gene_space()
    by_symbol: dict[str, list[tuple[str, str]]] = {}
    for gid, sym in zip(df["ensembl_gene_id"], df["symbol"]):
        key = str(sym).upper()
        by_symbol.setdefault(key, []).append((unversioned(gid), str(sym)))
    unique: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    for key, hits in by_symbol.items():
        ids = {gid for gid, _ in hits}
        if len(ids) == 1:
            unique[key] = hits[0]
        else:
            ambiguous.add(key)

    synonyms = get_data("ncbi-symbol-synonyms", copy=False)
    alias_to_officials: dict[str, set[str]] = {}
    for alias, official in zip(synonyms["alias"], synonyms["official_symbol"]):
        alias_key = _nonempty_text(alias)
        official_key = _nonempty_text(official)
        if alias_key is None or official_key is None:
            continue
        alias_to_officials.setdefault(alias_key.upper(), set()).add(official_key.upper())
    for alias_key, official_keys in alias_to_officials.items():
        if len(official_keys) != 1:
            ambiguous.add(alias_key)
            unique.pop(alias_key, None)
            continue
        official_key = next(iter(official_keys))
        if official_key in ambiguous:
            ambiguous.add(alias_key)
            unique.pop(alias_key, None)
            continue
        hit = unique.get(official_key)
        if hit is None:
            continue
        existing = unique.get(alias_key)
        if existing is not None and existing[0] != hit[0]:
            ambiguous.add(alias_key)
            unique.pop(alias_key, None)
        else:
            unique.setdefault(alias_key, hit)
    return unique, ambiguous


@lru_cache(maxsize=1)
def _extra_transcript_gene_index() -> dict[str, tuple[str, str]]:
    from .gene_ids import extra_transcript_mappings, resolve_ensembl_id, unversioned

    df = extra_transcript_mappings()
    out: dict[str, tuple[str, str]] = {}
    for tx, gid, sym in zip(df["transcript_id"], df["ensembl_gene_id"], df["gene_symbol"]):
        tx_key = str(tx).split(".", 1)[0].upper()
        raw_gid = _nonempty_text(gid)
        if raw_gid is not None:
            gene_id = resolve_ensembl_id(raw_gid.upper())
            hit = _canonical_gene_index().get(unversioned(gene_id))
            if hit is not None:
                out.setdefault(tx_key, hit)
                continue
        hit_id, hit_symbol, _, _ = _resolve_symbol(str(sym))
        if hit_id is not None and hit_symbol is not None:
            out.setdefault(tx_key, (hit_id, hit_symbol))
    return out


def _source_value_columns(
    df: pd.DataFrame, row_id_col: str | None, symbol_col: str | None, value_cols
) -> list[str]:
    if value_cols is not None:
        cols = [str(c) for c in value_cols]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"requested source expression value columns are missing: {missing}")
        return cols
    excluded = {c for c in (row_id_col, symbol_col) if c is not None}
    return [c for c in df.columns if c not in excluded]


def _resolve_gene_id(raw_id: str) -> tuple[str | None, str | None, str, str | None]:
    from .gene_ids import resolve_ensembl_id, unversioned

    raw = unversioned(str(raw_id).upper().removesuffix("_PAR_X").removesuffix("_PAR_Y"))
    canonical = resolve_ensembl_id(raw)
    hit = _canonical_gene_index().get(canonical)
    if hit is None:
        return None, None, "unresolved", "unknown_ensembl_gene_id"
    method = "ensembl_gene_id" if canonical == raw else "ensembl_gene_id_alias"
    return hit[0], hit[1], method, None


def _resolve_symbol(symbol: str) -> tuple[str | None, str | None, str, str | None]:
    from .gene_ids import resolve_symbol

    symbol_index, ambiguous = _symbol_to_gene_index()
    raw = str(symbol).strip()
    key = raw.upper()
    official = resolve_symbol(raw)
    official_key = str(official).upper()
    if key in ambiguous or official_key in ambiguous:
        return None, None, "ambiguous", "ambiguous_symbol"
    hit = symbol_index.get(key) or symbol_index.get(official_key)
    if hit is None:
        return None, None, "unresolved", "unknown_symbol"
    method = "symbol_synonym" if official_key != key else "symbol"
    return hit[0], hit[1], method, None


def map_source_gene_rows(
    df: pd.DataFrame,
    *,
    row_id_col: str | None = None,
    symbol_col: str | None = None,
    value_cols=None,
    high_expression_threshold: float = 1.0,
) -> pd.DataFrame:
    """Return a row-level mapping audit for a source gene-expression matrix.

    Builders can use this before aggregation to make identifier handling explicit:
    versioned/alt Ensembl IDs are migrated to the canonical gene space, known
    transcript IDs use oncoref's supplemental transcript map, Entrez/NCBI GeneIDs use
    the shipped NCBI-derived Entrez map, symbols/synonyms map through the bundled
    canonical gene space, and unresolved/ambiguous rows remain visible in the audit
    rather than disappearing.
    """
    row_id_col, symbol_col = _source_row_columns(df, row_id_col, symbol_col, require_row_id=True)
    cols = _source_value_columns(df, row_id_col, symbol_col, value_cols)
    numeric = (
        df[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=df.index)
    )
    row_sums = numeric.sum(axis=1, skipna=True) if cols else pd.Series(0.0, index=df.index)
    row_sum_values = row_sums.to_numpy(dtype=float)
    id_type = detect_source_row_id_type(df[row_id_col])
    transcript_index = _extra_transcript_gene_index()
    from .gene_ids import resolve_entrez_id

    rows: list[dict] = []
    for pos, (idx, raw_value) in enumerate(df[row_id_col].items()):
        raw_id = _nonempty_text(raw_value)
        source_symbol = _nonempty_text(df[symbol_col].iloc[pos]) if symbol_col is not None else None
        row_type = detect_source_row_id_type([raw_id])
        canonical_id = canonical_symbol = method = reason = None
        status = "unresolved"

        if raw_id is not None and row_type == "ensembl_gene_id":
            canonical_id, canonical_symbol, method, reason = _resolve_gene_id(raw_id)
        elif raw_id is not None and row_type == "ensembl_transcript_id":
            tx = transcript_index.get(str(raw_id).split(".", 1)[0].upper())
            if tx is not None and tx[0] in _canonical_gene_index():
                canonical_id, canonical_symbol = tx
                method = "extra_transcript_mapping"
            else:
                method = "unresolved"
                reason = "unknown_ensembl_transcript_id"
        elif raw_id is not None and row_type == "entrez_id":
            hit = resolve_entrez_id(raw_id)
            if hit is not None:
                canonical_id, canonical_symbol, method = hit
            else:
                method = "unresolved"
                reason = "unknown_entrez_id"

        if canonical_id is None:
            symbol_candidate = source_symbol
            if symbol_candidate is None and row_type == "symbol":
                symbol_candidate = raw_id
            if symbol_candidate is not None:
                canonical_id, canonical_symbol, method, reason = _resolve_symbol(symbol_candidate)

        if canonical_id is not None:
            status = "resolved"
            reason = None
        elif method == "ambiguous":
            status = "ambiguous"
        else:
            status = "unresolved"
            method = method or "unresolved"
            reason = reason or "missing_row_identifier"

        source_value_sum = float(row_sum_values[pos]) if pos < len(row_sum_values) else 0.0
        if cols:
            row_values = numeric.iloc[pos]
            source_expression_nonzero_samples = int((row_values > 0).sum())
            source_expression_max = row_values.max(skipna=True)
            if pd.isna(source_expression_max):
                source_expression_max = 0.0
            source_expression_max = float(source_expression_max)
            if source_expression_max > 0:
                source_expression_sample_with_max = str(row_values.idxmax(skipna=True))
            else:
                source_expression_sample_with_max = None
        else:
            source_expression_max = 0.0
            source_expression_sample_with_max = None
            source_expression_nonzero_samples = 0
        rows.append(
            {
                "source_row_index": idx,
                "source_row_id": raw_id,
                "source_row_id_type": row_type if row_type != "unknown" else id_type,
                "source_symbol": source_symbol,
                "mapping_status": status,
                "mapping_method": method,
                "canonical_ensembl_gene_id": canonical_id,
                "canonical_symbol": canonical_symbol,
                "unresolved_reason": reason,
                "source_value_sum": source_value_sum,
                "source_expression_max": source_expression_max,
                "source_expression_sample_with_max": source_expression_sample_with_max,
                "source_expression_nonzero_samples": source_expression_nonzero_samples,
                "high_expression_unresolved": bool(
                    status != "resolved" and source_value_sum >= high_expression_threshold
                ),
            }
        )
    out = pd.DataFrame(rows, columns=SOURCE_GENE_MAPPING_AUDIT_COLUMNS)
    return out


def coerce_source_expression_values(
    df: pd.DataFrame,
    value_cols=None,
    *,
    row_id_col: str | None = None,
    symbol_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce source expression values to numeric and report parse/missing diagnostics.

    Literal zeros, input missing values, and non-parsing cells are counted separately so
    builders do not have to collapse missingness into measured zero before QC.
    """
    if value_cols is None:
        row_id_col, symbol_col = _source_row_columns(
            df, row_id_col, symbol_col, require_row_id=False
        )
        cols = _source_value_columns(df, row_id_col, symbol_col, None)
    else:
        cols = _source_value_columns(df, None, None, value_cols)
    out = df.copy()
    rows: list[dict] = []
    for col in cols:
        if col not in out.columns:
            continue
        raw = out[col]
        input_missing = raw.isna()
        coerced = pd.to_numeric(raw, errors="coerce")
        parse_missing = coerced.isna() & ~input_missing
        literal_zero = coerced == 0
        n = len(raw)
        rows.append(
            {
                "value_col": col,
                "n_values": n,
                "n_input_missing": int(input_missing.sum()),
                "n_parse_missing": int(parse_missing.sum()),
                "n_literal_zero": int(literal_zero.sum()),
                "parse_missing_fraction": float(parse_missing.sum() / n) if n else 0.0,
                "literal_zero_fraction": float(literal_zero.sum() / n) if n else 0.0,
            }
        )
        out[col] = coerced
    diagnostics = pd.DataFrame(rows, columns=SOURCE_VALUE_PARSE_DIAGNOSTIC_COLUMNS)
    return out, diagnostics


def _pseudoautosomal_symbol_alias_positions(
    df: pd.DataFrame,
    *,
    row_id_col: str,
    symbol_col: str | None,
    value_cols: list[str],
) -> set[int]:
    """Positions of redundant ``PAR_Y_<symbol>`` aliases for the same source ID.

    This is distinct from a row ID ending in ``_PAR_Y``: that convention denotes
    a separately quantified pseudoautosomal feature and must be summed into the
    canonical gene. Some matrices instead repeat the exact same Ensembl ID and
    values under both ``<symbol>`` and ``PAR_Y_<symbol>`` labels. Those are aliases,
    so counting both would double expression.
    """
    if symbol_col is None:
        return set()
    positions_by_id: dict[str, list[int]] = {}
    for position, value in enumerate(df[row_id_col]):
        row_id = _nonempty_text(value)
        if row_id is not None:
            positions_by_id.setdefault(row_id.upper(), []).append(position)

    aliases: set[int] = set()
    for row_id, positions in positions_by_id.items():
        if len(positions) < 2:
            continue
        symbols = {
            position: _nonempty_text(df[symbol_col].iloc[position]) for position in positions
        }
        par_y = {
            position: symbol
            for position, symbol in symbols.items()
            if symbol is not None and symbol.upper().startswith("PAR_Y_")
        }
        if not par_y:
            continue
        ordinary = {
            position: symbol
            for position, symbol in symbols.items()
            if symbol is not None and not symbol.upper().startswith("PAR_Y_")
        }
        matches = [
            (ordinary_position, par_y_position)
            for par_y_position, par_y_symbol in par_y.items()
            for ordinary_position, ordinary_symbol in ordinary.items()
            if par_y_symbol.upper().removeprefix("PAR_Y_") == ordinary_symbol.upper()
        ]
        if len(positions) != 2 or len(matches) != 1:
            raise ValueError(
                "source expression has an unsupported pseudoautosomal symbol alias "
                f"group for {row_id}"
            )
        ordinary_position, par_y_position = matches[0]
        if not df[value_cols].iloc[ordinary_position].equals(df[value_cols].iloc[par_y_position]):
            raise ValueError(
                "source expression has unequal values for pseudoautosomal symbol aliases "
                f"of {row_id}"
            )
        aliases.add(par_y_position)
    return aliases


def canonicalize_source_gene_matrix(
    df: pd.DataFrame,
    *,
    row_id_col: str | None = None,
    symbol_col: str | None = None,
    value_cols=None,
    high_expression_threshold: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map source rows to canonical ENSG IDs and combine duplicates in linear space.

    Separately quantified Ensembl ``_PAR_Y`` rows are folded into the base gene
    and summed. A distinct upstream convention repeats the exact same Ensembl ID
    and values under ``<symbol>`` and ``PAR_Y_<symbol>`` labels; the latter is a
    redundant alias and is counted once. Unequal or structurally ambiguous
    pseudoautosomal symbol aliases raise rather than being silently combined.

    Returns ``(matrix, audit)``. ``matrix`` has stable ``Ensembl_Gene_ID`` /
    ``Symbol`` columns plus the requested value columns. ``audit`` is the full
    row-level mapping table from :func:`map_source_gene_rows`, including unresolved
    high-expression rows for builder QC reports.
    """
    row_id_col, symbol_col = _source_row_columns(df, row_id_col, symbol_col, require_row_id=True)
    cols = _source_value_columns(df, row_id_col, symbol_col, value_cols)
    coerced, parse_diagnostics = coerce_source_expression_values(df, cols)
    audit = map_source_gene_rows(
        coerced,
        row_id_col=row_id_col,
        symbol_col=symbol_col,
        value_cols=cols,
        high_expression_threshold=high_expression_threshold,
    )
    resolved = audit["mapping_status"] == "resolved"
    if not resolved.any():
        empty = pd.DataFrame(columns=["Ensembl_Gene_ID", "Symbol", *cols])
        empty.attrs["source_gene_mapping_stats"] = source_gene_mapping_stats(audit)
        empty.attrs["source_value_parse_diagnostics"] = parse_diagnostics
        return empty, audit

    par_y_alias_positions = _pseudoautosomal_symbol_alias_positions(
        coerced,
        row_id_col=row_id_col,
        symbol_col=symbol_col,
        value_cols=cols,
    )
    alias_mask = pd.Series(
        [position in par_y_alias_positions for position in range(len(coerced))],
        index=audit.index,
    )
    audit.loc[alias_mask, "mapping_method"] = "ensembl_gene_id_par_y_symbol_alias"
    retained = resolved & ~alias_mask
    work = coerced.iloc[retained.to_numpy()][cols].copy()
    work.insert(0, "Symbol", audit.loc[retained, "canonical_symbol"].to_numpy())
    work.insert(
        0,
        "Ensembl_Gene_ID",
        audit.loc[retained, "canonical_ensembl_gene_id"].to_numpy(),
    )
    grouped = work.groupby("Ensembl_Gene_ID", sort=False)
    ids = grouped[["Symbol"]].first()
    values = grouped[cols].sum(min_count=1)
    out = ids.join(values).reset_index()
    out.attrs["source_gene_mapping_stats"] = source_gene_mapping_stats(audit)
    out.attrs["source_value_parse_diagnostics"] = parse_diagnostics
    return out[["Ensembl_Gene_ID", "Symbol", *cols]], audit


def source_gene_mapping_stats(audit: pd.DataFrame) -> dict:
    """Compact counts from a :func:`map_source_gene_rows` audit table."""
    if audit.empty:
        return {
            "n_source_rows": 0,
            "n_resolved_rows": 0,
            "n_unresolved_rows": 0,
            "n_ambiguous_rows": 0,
            "n_high_expression_unresolved_rows": 0,
        }
    status = audit["mapping_status"].value_counts().to_dict()
    return {
        "n_source_rows": len(audit),
        "n_resolved_rows": int(status.get("resolved", 0)),
        "n_unresolved_rows": int(status.get("unresolved", 0)),
        "n_ambiguous_rows": int(status.get("ambiguous", 0)),
        "n_high_expression_unresolved_rows": int(audit["high_expression_unresolved"].sum()),
    }


def expanded_tx_map(tx_to_gene_name: dict) -> dict:
    """Expand a ``{transcript_id: gene}`` map to also key the versionless id
    (``ENST….5`` → ``ENST…``), first-seen value winning. So a versioned input id
    matches a versionless map entry and vice versa."""
    out: dict = {}
    for k, v in tx_to_gene_name.items():
        out.setdefault(str(k), v)
        out.setdefault(str(k).split(".", 1)[0], v)
    return out


@lru_cache(maxsize=1)
def _default_tx_to_gene() -> dict:
    """oncoref's curated ``extra-tx-mappings`` (transcript_id → gene_symbol) —
    the back-compat known set; not a full Ensembl reference."""
    from .gene_ids import extra_transcript_mappings

    df = extra_transcript_mappings()
    return dict(zip(df["transcript_id"].astype(str), df["gene_symbol"].astype(str)))


def aggregate_transcripts_to_genes(
    df: pd.DataFrame,
    tx_to_gene_name: dict | None = None,
    *,
    transcript_id_column_candidates=_DEFAULT_TX_COLUMN_CANDIDATES,
    tpm_column_candidates=_DEFAULT_TPM_COLUMN_CANDIDATES,
    unresolved_label: str = "unresolved",
) -> pd.DataFrame:
    """Aggregate transcript-level TPM to gene level.

    Finds the transcript-id and TPM columns (by the candidate name lists), maps each
    transcript to a gene via ``tx_to_gene_name`` (default :func:`_default_tx_to_gene`,
    matched version-insensitively), and sums TPM per gene. Transcripts not in the map
    are summed into one ``unresolved_label`` row (never dropped — a gene whose every
    transcript is unknown must stay accounted for, not vanish from the quant).

    Returns a DataFrame with ``gene`` and ``TPM`` (one row per gene, plus the
    ``unresolved`` row if any), sorted by TPM. ``df.attrs["aggregation_stats"]``
    carries the known/unresolved TPM split so a caller can gate on resolution
    quality. See the module docstring for the pyensembl boundary."""
    tx_col = find_column(df, transcript_id_column_candidates, "transcript ID")
    tpm_col = find_column(df, tpm_column_candidates, "TPM")

    tx0 = df[tx_col].astype(str).str.split(".", n=1).str[0]
    tpm = pd.to_numeric(df[tpm_col], errors="coerce").fillna(0.0)
    tx_map = expanded_tx_map(
        tx_to_gene_name if tx_to_gene_name is not None else _default_tx_to_gene()
    )
    gene = tx0.map(tx_map)

    unknown = gene.isna()
    unknown_tpm = float(tpm[unknown].sum())
    gene = gene.where(~unknown, unresolved_label)

    agg = (
        pd.DataFrame({"gene": gene.astype(str), "TPM": tpm.to_numpy()})
        .groupby("gene", as_index=False, sort=False)["TPM"]
        .sum()
        .sort_values("TPM")
        .reset_index(drop=True)
    )
    total = float(tpm.sum())
    agg.attrs["aggregation_stats"] = {
        "total_tpm": total,
        "unresolved_tpm": unknown_tpm,
        "unresolved_fraction": (unknown_tpm / total) if total > 0 else 0.0,
        "unresolved_transcript_count": int(unknown.sum()),
        "n_genes": int((agg["gene"] != unresolved_label).sum()),
    }
    return agg


__all__ = [
    "SOURCE_GENE_MAPPING_AUDIT_COLUMNS",
    "SOURCE_VALUE_PARSE_DIAGNOSTIC_COLUMNS",
    "aggregate_transcripts_to_genes",
    "canonicalize_source_gene_matrix",
    "coerce_source_expression_values",
    "detect_source_row_id_type",
    "expanded_tx_map",
    "find_column",
    "map_source_gene_rows",
    "source_gene_mapping_stats",
]
