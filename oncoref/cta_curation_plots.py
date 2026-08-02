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

"""CTA curation documentation figures over oncoref's packaged CTA table.

This mirrors the shipped ``pirlygenes plot cta-curation`` surface while using
only oncoref-owned data. The figures summarize the CTA source overlap, filter
funnel/outcome, deflated reproductive-fraction distribution, and
protein-reliability-vs-RNA thresholds from ``cancer-testis-antigens.csv``.

The standard oncoref installation includes ``matplotlib_venn`` for the
source-overlap figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .load_dataset import get_data

# Primary gene-contributing sources. The broad da Silva cross-reference tag and
# tiny paralog/candidate tags are deliberately excluded from the primary-source
# overlap, matching the pirlygenes plot. Some oncoref rows carry the older
# ``daSilva2017`` tag without the protein suffix, so accept both forms.
PRIMARY_SOURCES = {
    "CTpedia": lambda tags: "CTpedia" in tags,
    "CTexploreR": lambda tags: "CTexploreR_CT" in tags or "CTexploreR_CTP" in tags,
    "daSilva2017_protein": lambda tags: "daSilva2017_protein" in tags or "daSilva2017" in tags,
    "placental_antigen": lambda tags: "placental_antigen" in tags,
}

# Deflated-RNA-fraction threshold each protein-reliability tier must clear.
RELIABILITY_THRESHOLD = {
    "Enhanced": 0.80,
    "Supported": 0.90,
    "Approved": 0.95,
    "Uncertain": 0.98,
    "no data": 0.98,
}
RELIABILITY_ORDER = ["no data", "Uncertain", "Approved", "Supported", "Enhanced"]

KEPT = "#2a7f4f"
DROP = "#b0b0b0"
WEAK = "#f0c419"

# Hyphenated filenames match pirlygenes/docs references.
FILENAMES = {
    "source_venn": "cta-source-venn.png",
    "filter_funnel": "cta-filter-funnel.png",
    "filter_outcome": "cta-filter-outcome.png",
    "deflated_dist": "cta-deflated-frac-dist.png",
    "protein_vs_rna": "cta-protein-vs-rna.png",
}


def _evidence():
    """Raw packaged CTA evidence table, including rows that fail curation filters."""
    return get_data("cancer-testis-antigens").copy()


def _bool_series(series):
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _tag_sets(df):
    """Return ``{source_label: set(Ensembl_Gene_ID)}`` for primary sources."""
    out = {name: set() for name in PRIMARY_SOURCES}
    for ensg, raw in zip(df["Ensembl_Gene_ID"], df["source_databases"].fillna("")):
        tags = {t.strip() for t in str(raw).split(";") if t.strip()}
        for name, pred in PRIMARY_SOURCES.items():
            if pred(tags):
                out[name].add(str(ensg))
    return out


def _per_source_counts(df):
    sets = _tag_sets(df)
    rows = []
    for name, members in sets.items():
        sub = df[df["Ensembl_Gene_ID"].astype(str).isin(members)]
        passes = _bool_series(sub["passes_filters"])
        weak = _bool_series(sub["never_expressed"])
        rows.append(
            {
                "source": name,
                "total": len(sub),
                "kept_confident": int((passes & ~weak).sum()),
                "kept_weak": int((passes & weak).sum()),
                "excluded": int((~passes).sum()),
            }
        )
    return sorted(rows, key=lambda r: r["total"], reverse=True)


def _save(fig, path, plt):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_source_venn(df, path, plt):
    from matplotlib_venn import venn3

    sets = _tag_sets(df)
    fig, ax = plt.subplots(figsize=(7, 6))
    keys = ("CTpedia", "CTexploreR", "daSilva2017_protein")
    placental = len(sets["placental_antigen"])
    venn3(
        [sets[k] for k in keys],
        set_labels=("CTpedia", "CTexploreR", "da Silva 2017\n(protein)"),
        ax=ax,
    )
    ax.set_title(
        f"CTA source overlap (primary databases)\n+{placental} placental-antigen genes folded in"
    )
    _save(fig, path, plt)


def _fig_filter_funnel(df, path, plt):
    rows = _per_source_counts(df)
    labels = [r["source"] for r in rows]
    kept = np.array([r["kept_confident"] + r["kept_weak"] for r in rows])
    dropped = np.array([r["excluded"] for r in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 2))
    ax.barh(y, kept, color=KEPT, label="passes filter")
    ax.barh(y, dropped, left=kept, color=DROP, label="excluded")
    for i, r in enumerate(rows):
        ax.text(r["total"] + 1, i, f"{kept[i]}/{r['total']}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("genes in source")
    ax.set_title("CTA filter funnel by source (kept vs excluded)")
    ax.legend(loc="lower right")
    _save(fig, path, plt)


def _fig_filter_outcome(df, path, plt):
    rows = _per_source_counts(df)
    labels = [r["source"] for r in rows]
    conf = np.array([r["kept_confident"] for r in rows])
    weak = np.array([r["kept_weak"] for r in rows])
    excl = np.array([r["excluded"] for r in rows])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 2))
    ax.barh(y, conf, color=KEPT, label="kept (HPA-confident)")
    ax.barh(y, weak, left=conf, color=WEAK, label="kept (weak evidence)")
    ax.barh(y, excl, left=conf + weak, color=DROP, label="excluded")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("genes")
    ax.set_title("CTA filter outcome by source")
    ax.legend(loc="lower right")
    _save(fig, path, plt)


def _fig_deflated_dist(df, path, plt):
    frac = df["rna_deflated_reproductive_frac"].astype(float).to_numpy()
    passes = _bool_series(df["passes_filters"]).to_numpy()
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        [frac[passes], frac[~passes]],
        bins=bins,
        stacked=True,
        color=[KEPT, DROP],
        label=["passes filter", "excluded"],
    )
    for thr in (0.80, 0.90, 0.95, 0.98):
        ax.axvline(thr, color="#555", ls="--", lw=0.8)
        ax.text(
            thr,
            ax.get_ylim()[1] * 0.97,
            f"{thr:.2f}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#555",
        )
    ax.set_xlabel("deflated reproductive fraction")
    ax.set_ylabel("CTA genes")
    ax.set_title("Deflated reproductive-fraction distribution")
    ax.legend()
    _save(fig, path, plt)


def _fig_protein_vs_rna(df, path, plt):
    frac = df["rna_deflated_reproductive_frac"].astype(float)
    rel = df["protein_reliability"].fillna("no data").astype(str)
    passes = _bool_series(df["passes_filters"])
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for i, tier in enumerate(RELIABILITY_ORDER):
        m = rel == tier
        if not m.any():
            continue
        x = i + (rng.random(int(m.sum())) - 0.5) * 0.6
        ax.scatter(x, frac[m], s=14, alpha=0.6, c=np.where(passes[m], KEPT, DROP))
        thr = RELIABILITY_THRESHOLD.get(tier)
        if thr is not None:
            ax.plot([i - 0.4, i + 0.4], [thr, thr], color="#c0392b", lw=2)
    ax.set_xticks(range(len(RELIABILITY_ORDER)), RELIABILITY_ORDER)
    ax.set_xlabel("protein reliability (HPA IHC)")
    ax.set_ylabel("deflated reproductive fraction")
    ax.set_title(
        "Protein reliability vs RNA fraction\n(red line = required RNA threshold for that tier)"
    )
    _save(fig, path, plt)


_BUILDERS = {
    "source_venn": _fig_source_venn,
    "filter_funnel": _fig_filter_funnel,
    "filter_outcome": _fig_filter_outcome,
    "deflated_dist": _fig_deflated_dist,
    "protein_vs_rna": _fig_protein_vs_rna,
}


def render(out_dir="cta_curation_out") -> dict:
    """Write the five CTA-curation figures into ``out_dir``.

    Returns ``{"n_genes": int, "paths": {key: Path}}``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _evidence()
    paths = {}
    for key, builder in _BUILDERS.items():
        path = out / FILENAMES[key]
        builder(df, path, plt)
        paths[key] = path
    return {"n_genes": len(df), "paths": paths}
