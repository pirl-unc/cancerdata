#!/usr/bin/env python
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

"""Build the small oncoref Entrez GeneID -> canonical ENSG mapping table.

Inputs are the pinned NCBI files mirrored by pirlygenes:

    Homo_sapiens.gene_info.gz
    gene_history.gz

The generated wheel table is deliberately much smaller than those raw sources:
it only keeps live or discontinued Entrez IDs that resolve into oncoref's
canonical gene space.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oncoref.gene_ids import canonical_gene_space, ensembl_id_aliases, unversioned
from oncoref.load_dataset import get_data

HUMAN_TAX_ID = "9606"
DEFAULT_GENE_INFO = (
    Path.home()
    / ".cache"
    / "pirlygenes"
    / "ncbi_gene_info"
    / "ncbi-gene-data-20260601"
    / "Homo_sapiens.gene_info.gz"
)
DEFAULT_GENE_HISTORY = (
    Path.home()
    / ".cache"
    / "pirlygenes"
    / "ncbi_gene_info"
    / "ncbi-gene-data-20260601"
    / "gene_history.gz"
)
DEFAULT_OUTPUT = Path("oncoref/data/ncbi-entrez-gene-mappings.csv.gz")

# NCBI Gene uses traditional mitochondrial symbols while Ensembl/HGNC uses the
# explicit MT- prefix. Keep this translation narrow and reviewable; aggregate
# loci such as IGH/IGK/IGL must not be assigned to an arbitrary canonical gene.
NCBI_MITOCHONDRIAL_SYMBOLS: dict[str, str] = {
    "ATP6": "MT-ATP6",
    "ATP8": "MT-ATP8",
    "COX1": "MT-CO1",
    "COX2": "MT-CO2",
    "COX3": "MT-CO3",
    "CYTB": "MT-CYB",
    "ND1": "MT-ND1",
    "ND2": "MT-ND2",
    "ND3": "MT-ND3",
    "ND4": "MT-ND4",
    "ND4L": "MT-ND4L",
    "ND5": "MT-ND5",
    "ND6": "MT-ND6",
    "RNR1": "MT-RNR1",
    "RNR2": "MT-RNR2",
    "TRNA": "MT-TA",
    "TRNC": "MT-TC",
    "TRND": "MT-TD",
    "TRNE": "MT-TE",
    "TRNF": "MT-TF",
    "TRNG": "MT-TG",
    "TRNH": "MT-TH",
    "TRNI": "MT-TI",
    "TRNK": "MT-TK",
    "TRNL1": "MT-TL1",
    "TRNL2": "MT-TL2",
    "TRNM": "MT-TM",
    "TRNN": "MT-TN",
    "TRNP": "MT-TP",
    "TRNQ": "MT-TQ",
    "TRNR": "MT-TR",
    "TRNS1": "MT-TS1",
    "TRNS2": "MT-TS2",
    "TRNT": "MT-TT",
    "TRNV": "MT-TV",
    "TRNW": "MT-TW",
    "TRNY": "MT-TY",
}


def _canonical_indexes() -> tuple[dict[str, str], dict[str, str], set[str]]:
    genes = canonical_gene_space()
    canonical_by_id = {
        unversioned(str(gid)): str(symbol)
        for gid, symbol in zip(genes["ensembl_gene_id"], genes["symbol"])
    }

    by_symbol: dict[str, list[tuple[str, str]]] = {}
    for gid, symbol in zip(genes["ensembl_gene_id"], genes["symbol"]):
        sym = str(symbol).strip()
        if not sym:
            continue
        by_symbol.setdefault(sym.upper(), []).append((unversioned(str(gid)), sym))

    unique: dict[str, str] = {}
    ambiguous: set[str] = set()
    for key, hits in by_symbol.items():
        ids = {gid for gid, _ in hits}
        if len(ids) == 1:
            unique[key] = next(iter(ids))
        else:
            ambiguous.add(key)

    synonyms = get_data("ncbi-symbol-synonyms", copy=False)
    alias_to_officials: dict[str, set[str]] = {}
    for alias, official in zip(synonyms["alias"], synonyms["official_symbol"]):
        if pd.isna(alias) or pd.isna(official):
            continue
        alias_key = str(alias).strip().upper()
        official_key = str(official).strip().upper()
        if not alias_key or not official_key:
            continue
        alias_to_officials.setdefault(alias_key, set()).add(official_key)

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
        if existing is not None and existing != hit:
            ambiguous.add(alias_key)
            unique.pop(alias_key, None)
        else:
            unique.setdefault(alias_key, hit)

    return canonical_by_id, unique, ambiguous


def _resolve_ensg(
    value: str | None,
    *,
    canonical_by_id: dict[str, str],
    aliases: dict[str, str],
) -> tuple[str, str] | None:
    if not value:
        return None
    key = unversioned(str(value)).upper()
    key = aliases.get(key, key)
    symbol = canonical_by_id.get(key)
    if symbol is None:
        return None
    return key, symbol


def _resolve_symbol(
    value: str | None,
    *,
    canonical_by_id: dict[str, str],
    symbol_index: dict[str, str],
    ambiguous_symbols: set[str],
) -> tuple[str, str] | None:
    if value is None or pd.isna(value):
        return None
    key = str(value).strip().upper()
    if not key or key in ambiguous_symbols:
        return None
    gene_id = symbol_index.get(key)
    if gene_id is None:
        return None
    symbol = canonical_by_id.get(gene_id)
    if symbol is None:
        return None
    return gene_id, symbol


def _resolve_exact_canonical_symbol(
    value: str,
    *,
    canonical_by_id: dict[str, str],
) -> tuple[str, str] | None:
    """Resolve a reviewed canonical symbol without applying alias expansion."""
    key = value.upper()
    hits = [
        (gene_id, symbol) for gene_id, symbol in canonical_by_id.items() if symbol.upper() == key
    ]
    return hits[0] if len(hits) == 1 else None


def build(gene_info: Path, gene_history: Path) -> pd.DataFrame:
    canonical_by_id, symbol_index, ambiguous_symbols = _canonical_indexes()
    aliases = ensembl_id_aliases()

    gene_info_df = pd.read_csv(
        gene_info,
        sep="\t",
        usecols=["GeneID", "Symbol", "dbXrefs"],
        dtype={"GeneID": "string", "Symbol": "string", "dbXrefs": "string"},
        low_memory=False,
    )
    gene_info_df = gene_info_df.dropna(subset=["GeneID"])

    current_rows: dict[str, dict[str, str]] = {}
    for row in gene_info_df.itertuples(index=False):
        entrez_id = str(row.GeneID).strip()
        if not entrez_id or entrez_id == "-":
            continue

        hit = None
        dbxrefs = "" if pd.isna(row.dbXrefs) else str(row.dbXrefs)
        for ensg in re.findall(r"Ensembl:(ENSG\d+)", dbxrefs):
            hit = _resolve_ensg(ensg, canonical_by_id=canonical_by_id, aliases=aliases)
            if hit is not None:
                method = "entrez_dbxrefs"
                break
        else:
            ncbi_symbol = str(row.Symbol).strip().upper()
            canonical_symbol = NCBI_MITOCHONDRIAL_SYMBOLS.get(ncbi_symbol)
            if canonical_symbol is not None:
                hit = _resolve_exact_canonical_symbol(
                    canonical_symbol,
                    canonical_by_id=canonical_by_id,
                )
                method = "entrez_mitochondrial_symbol"
            else:
                hit = _resolve_symbol(
                    row.Symbol,
                    canonical_by_id=canonical_by_id,
                    symbol_index=symbol_index,
                    ambiguous_symbols=ambiguous_symbols,
                )
                method = "entrez_current_symbol"

        if hit is None:
            continue
        gene_id, symbol = hit
        current_rows[entrez_id] = {
            "entrez_id": entrez_id,
            "live_entrez_id": entrez_id,
            "canonical_ensembl_gene_id": gene_id,
            "canonical_symbol": symbol,
            "mapping_method": method,
            "current_mapping_method": method,
        }

    history = pd.read_csv(
        gene_history,
        sep="\t",
        usecols=["#tax_id", "GeneID", "Discontinued_GeneID"],
        dtype={"#tax_id": "string", "GeneID": "string", "Discontinued_GeneID": "string"},
        low_memory=False,
    )
    history = history[history["#tax_id"].astype(str).eq(HUMAN_TAX_ID)]
    history = history[
        history["GeneID"].notna()
        & history["Discontinued_GeneID"].notna()
        & history["GeneID"].ne("-")
        & history["Discontinued_GeneID"].ne("-")
    ]

    rows = list(current_rows.values())
    for row in history.itertuples(index=False):
        live_id = str(row.GeneID).strip()
        discontinued_id = str(row.Discontinued_GeneID).strip()
        if not live_id or not discontinued_id or discontinued_id in current_rows:
            continue
        live = current_rows.get(live_id)
        if live is None:
            continue
        rows.append(
            {
                "entrez_id": discontinued_id,
                "live_entrez_id": live_id,
                "canonical_ensembl_gene_id": live["canonical_ensembl_gene_id"],
                "canonical_symbol": live["canonical_symbol"],
                "mapping_method": "entrez_gene_history",
                "current_mapping_method": live["current_mapping_method"],
            }
        )

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["entrez_id"], keep="first")
    out = out.sort_values("entrez_id", key=lambda s: s.astype(int), ignore_index=True)
    return out[
        [
            "entrez_id",
            "live_entrez_id",
            "canonical_ensembl_gene_id",
            "canonical_symbol",
            "mapping_method",
            "current_mapping_method",
        ]
    ]


def _write_deterministic_gzip_csv(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_csv(index=False, lineterminator="\n")
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle,
    ):
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-info", type=Path, default=DEFAULT_GENE_INFO)
    parser.add_argument("--gene-history", type=Path, default=DEFAULT_GENE_HISTORY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = build(args.gene_info, args.gene_history)
    _write_deterministic_gzip_csv(df, args.out)
    print(f"wrote {len(df):,} Entrez mappings to {args.out}")


if __name__ == "__main__":
    main()
