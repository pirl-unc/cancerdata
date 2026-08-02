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

"""Cancer-type-level reference plots built from oncoref's own data.

These need only oncoref-owned data (TMB, anti-PD-1 ORR, incidence/mortality,
the cancer-type registry for lineage colors) — no CTA or peptide data — so they
have no dependency on the target-selection libraries.

The standard oncoref installation includes matplotlib. It is imported lazily to
keep ordinary data access lightweight. Every function returns the matplotlib
``Figure`` and optionally writes a PNG when ``save`` is given.
"""

from __future__ import annotations

from functools import lru_cache

from .apd1 import cancer_apd1_response, cancer_apd1_response_df
from .cancer_types import (
    cancer_evidence_source_code,
    cancer_lineage_group,
    cancer_type_registry,
    format_cancer_code_label,
    resolve_cancer_type,
)
from .cta import cta_gene_id_to_name, cta_gene_ids
from .expression import (
    available_percentile_cohorts,
    cohort_gene_percentiles,
    locally_available_within_sample_cohorts,
    within_sample_top_fraction,
)
from .ici import cancer_ici_response
from .incidence import burden_category, cancer_burden
from .tmb import cancer_tmb

_PLT = None

#: stat name -> the percentile breakpoint column in the shipped percentile vector.
_STAT_PERCENTILE_COL = {"q1": "p25", "median": "p50", "q3": "p75"}


def _plt():
    """Lazy, one-time matplotlib import (headless-safe Agg default; honors a
    backend the caller already set)."""
    global _PLT
    if _PLT is not None:
        return _PLT
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    _PLT = plt
    return _PLT


@lru_cache(maxsize=1)
def _family_by_code() -> dict[str, str]:
    """``{code -> coarse lineage group}`` — the ~8 cell-of-origin groups (Epithelial,
    Sarcoma, Heme, CNS, Melanoma, Germ cell, Neuroendocrine, Embryonal; ``other`` if
    unmapped) used for plot colouring, matching the pirlygenes lineage palette. This is
    a deliberate coarsening of the registry's fine ``family`` field via
    :func:`cancer_lineage_group` so plots have a small, legible, collision-free legend.
    Cached (read-only); callers must not mutate the returned dict."""
    reg = cancer_type_registry()
    return {str(c): (cancer_lineage_group(str(c)) or "other") for c in reg["code"]}


@lru_cache(maxsize=1)
def _stable_palette() -> tuple:
    """A long, fixed list of distinct RGBA colors (tab20 + tab20b + tab20c = 60),
    so a family/group keeps the SAME color across every plot."""
    plt = _plt()
    out: list = []
    for name in ("tab20", "tab20b", "tab20c"):
        out.extend(plt.get_cmap(name).colors)
    return tuple(out)


@lru_cache(maxsize=1)
def _family_color_map() -> dict:
    """Stable ``{lineage group -> color}`` over the full set of groups (``tab10``,
    alphabetical, ``other`` = neutral grey) — deterministic and identical in every plot,
    and few enough groups (~8) for a clean, collision-free legend. Mirrors the pirlygenes
    lineage palette."""
    plt = _plt()
    groups = sorted(set(_family_by_code().values()) - {"other"})
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
    colors["other"] = (0.6, 0.6, 0.6, 1.0)
    return colors


def _family_colors(codes):
    """``({code -> color}, {lineage group -> color})`` by coarse lineage group. Colors
    are a stable, deterministic per-group assignment (see :func:`_family_color_map`); the
    returned group map carries only the groups present in ``codes`` (for legends)."""
    fam_by_code = _family_by_code()
    full = _family_color_map()
    present = sorted({fam_by_code.get(c, "other") for c in codes})
    fam_color = {f: full.get(f, full["other"]) for f in present}
    return {c: full.get(fam_by_code.get(c, "other"), full["other"]) for c in codes}, fam_color


def _plot_evidence_code(code):
    return cancer_evidence_source_code(code, strict=False) or str(code)


def _cohort_sample_weight(code) -> float:
    try:
        from . import source_matrices

        weight = float(source_matrices.cohort_info(code).get("n_samples") or 0)
    except Exception:
        return 1.0
    return weight if weight > 0 else 1.0


def _metric_code_for_cohort(cohort, metric_map):
    code = str(cohort)
    if code in metric_map:
        return code
    source_code = _plot_evidence_code(code)
    if source_code in metric_map:
        return source_code
    return None


def _collapse_metric_points(cohorts, metric_map, value_fn):
    grouped: dict[str, list[tuple[float, float]]] = {}
    for cohort in cohorts:
        code = _metric_code_for_cohort(cohort, metric_map)
        if code is None:
            continue
        value = float(value_fn(cohort))
        if value != value:  # NaN
            continue
        grouped.setdefault(code, []).append((value, _cohort_sample_weight(cohort)))

    points = []
    for code, values in grouped.items():
        total_weight = sum(w for _, w in values)
        if total_weight:
            x = sum(v * w for v, w in values) / total_weight
        else:
            x = sum(v for v, _ in values) / len(values)
        points.append((code, x, float(metric_map[code])))
    return points


def _apd1_axis(*, strict_pd1=False):
    if not strict_pd1:
        return (
            cancer_ici_response(),
            "Immune-checkpoint (ICI) ORR (%)",
            "immune-checkpoint (ICI)",
        )

    mapping = cancer_apd1_response()
    df = cancer_apd1_response_df()
    strict_codes = set(df.loc[df["drug_target"] == "PD-1", "cancer_code"].astype(str))
    return (
        {code: value for code, value in mapping.items() if code in strict_codes},
        "Anti-PD-1 monotherapy ORR (%)",
        "anti-PD-1 monotherapy",
    )


def _burden_metric_axis(against):
    pieces = str(against).split("_", 1)
    if (
        len(pieces) != 2
        or pieces[0] not in {"us", "world"}
        or pieces[1]
        not in {
            "incidence",
            "mortality",
        }
    ):
        raise ValueError("against must be 'apd1', 'tmb', or a burden metric")
    region, metric = pieces
    burden = cancer_burden(metric=f"{region}_{metric}_pct")
    mapping = {}
    for code in cancer_type_registry()["code"].astype(str):
        category = burden_category(code)
        if category in burden:
            mapping[code] = float(burden[category])
    return mapping, f"{region.upper()} {metric} share (%)"


def _reference_metric_axis(against):
    if against == "apd1":
        ymap, ylabel, _ = _apd1_axis(strict_pd1=True)
        return ymap, ylabel
    if against == "ici":
        ymap, ylabel, _ = _apd1_axis(strict_pd1=False)
        return ymap, ylabel
    if against == "tmb":
        return cancer_tmb(), "Median tumor mutational burden (mut/Mb)"
    if against in {"us_incidence", "us_mortality", "world_incidence", "world_mortality"}:
        return _burden_metric_axis(against)
    raise ValueError(
        "against must be 'apd1', 'ici', 'tmb', 'us_incidence', 'us_mortality', "
        "'world_incidence', or 'world_mortality'"
    )


def _burden_metric_label(metric):
    base = str(metric)
    if base.endswith("_pct"):
        base = base[: -len("_pct")]
    pieces = base.split("_", 1)
    if (
        len(pieces) == 2
        and pieces[0] in {"us", "world"}
        and pieces[1]
        in {
            "incidence",
            "mortality",
        }
    ):
        region, measure = pieces
        return f"{region.upper()} {measure} share"
    return "burden share"


def _save(fig, save):
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ---------- shared plot primitives (the centralized rendering layer) ----------
#
# Every cancer-type plot reduces to one of three shapes: a per-cancer scatter, a
# cohort×gene heatmap, or a ranked horizontal bar — all coloured by registry family
# and saved the same way. The plot functions below are thin: they assemble the data
# (points / grid / pairs) and hand it to one primitive, so adding a plot is "shape
# the data, pick a primitive", not "re-derive the matplotlib boilerplate".


def _family_legend_handles(plt, fam_color):
    return [
        plt.Line2D([], [], marker="o", linestyle="", color=col, label=fam)
        for fam, col in sorted(fam_color.items())
    ]


def _repel_labels(ax, texts):
    """Nudge point labels apart so they don't overlap, drawing thin leader lines back to
    the points (uses ``adjustText`` when installed; a no-op fallback otherwise). Best-effort
    and cosmetic: work is capped at one second, and any failure degrades to un-repelled
    labels rather than breaking figure generation."""
    if not texts:
        return
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            ax=ax,
            expand=(1.05, 1.2),
            time_lim=1.0,
            arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.4},
        )
    except Exception:  # cosmetic label placement — never fatal to figure generation
        return


def _family_scatter(
    points,
    *,
    xlabel,
    ylabel,
    title,
    logx=False,
    annotate=True,
    figsize=(12, 8),
    legend_loc="best",
    save=None,
):
    """Scatter of ``(code, x, y)`` points coloured by lineage group, with optional point
    annotations (de-overlapped via :func:`_repel_labels`) + a legend. The shared
    per-cancer scatter scaffold."""
    plt = _plt()
    codes = [p[0] for p in points]
    colors, fam_color = _family_colors(codes)
    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    for code, x, y in points:
        ax.scatter(x, y, color=colors[code], s=70, edgecolor="white", linewidth=0.6, zorder=3)
        if annotate:
            texts.append(ax.text(x, y, format_cancer_code_label(code), fontsize=6, zorder=4))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both" if logx else "major", alpha=0.3)
    ax.legend(
        handles=_family_legend_handles(plt, fam_color),
        fontsize=6,
        ncol=2,
        loc=legend_loc,
        framealpha=0.9,
    )
    _repel_labels(ax, texts)  # keep labels from overlapping (no-op without adjustText)
    fig.tight_layout()
    return _save(fig, save)


def _ranked_family_barh(pairs, *, xlabel, title, legend=False, save=None):
    """Horizontal bars of ``(code, value)`` in the given top-to-bottom order,
    coloured by registry family. The shared ranked-bar scaffold."""
    import numpy as np

    plt = _plt()
    codes = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors, fam_color = _family_colors(codes)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.40 * len(codes))))
    y = np.arange(len(codes))
    ax.barh(y, values, color=[colors[c] for c in codes])
    ax.set_yticks(y)
    ax.set_yticklabels([format_cancer_code_label(c) for c in codes], fontsize=8)
    ax.invert_yaxis()  # first pair at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
        ax.legend(handles, list(fam_color), fontsize=6, title="family", loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def _cohort_gene_heatmap(grid, *, title, cbar_label, cmap, lognorm=False, floor=0.01, save=None):
    """Heatmap of a cohorts (rows) × genes (cols) DataFrame — rows labelled by
    cancer code, columns by gene symbol. The shared cohort×gene heatmap scaffold."""
    import numpy as np

    plt = _plt()
    cols = list(grid.columns)
    rows = list(grid.index)
    data = grid.to_numpy(dtype=float)
    norm = None
    if lognorm:
        from matplotlib.colors import LogNorm

        data = np.clip(data, floor, None)
        norm = LogNorm(vmin=floor, vmax=max(1000.0, float(np.nanmax(data))))
    else:
        data = np.nan_to_num(data, nan=0.0)
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(cols)), max(6, 0.34 * len(rows))))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([format_cancer_code_label(c) for c in rows], fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return _save(fig, save)


# CTA antigen families, by symbol prefix (longest match wins). oncoref has no
# curated antigen-family table; this presentational heuristic mirrors pirlygenes and
# only drives colour grouping in the gene-level CTA plots — never analysis.
_ANTIGEN_FAMILY_PREFIXES = (
    ("MAGEA", "MAGE-A"),
    ("MAGEB", "MAGE-B"),
    ("MAGEC", "MAGE-C"),
    ("CSAG", "CSAG"),
    ("GAGE", "GAGE"),
    ("XAGE", "XAGE"),
    ("PAGE", "PAGE"),
    ("SSX", "SSX"),
    ("CT45", "CT45"),
    ("CTAG", "CTAG/NY-ESO"),
    ("LAGE", "CTAG/NY-ESO"),
    ("NY-ESO", "CTAG/NY-ESO"),  # the curated proteoform alias for CTAG1A/CTAG1B
    ("SPANX", "SPANX"),
    ("PRAME", "PRAME"),
    ("DPPA2", "DPPA"),
    ("TFDP3", "TFDP3"),
)


def _antigen_family(symbol) -> str:
    """CTA antigen family for a gene symbol via longest matching prefix; ``"other"``
    when nothing matches (e.g. CT83, CTCFL)."""
    s = str(symbol).upper()
    best = None
    for prefix, fam in _ANTIGEN_FAMILY_PREFIXES:
        if s.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, fam)
    return best[1] if best else "other"


@lru_cache(maxsize=1)
def _antigen_family_color_map() -> dict:
    """Stable ``{antigen family -> color}`` over the full antigen-family vocabulary,
    so e.g. MAGE-A is the same color in every plot."""
    palette = _stable_palette()
    families = sorted({fam for _, fam in _ANTIGEN_FAMILY_PREFIXES} | {"other"})
    return {f: palette[i % len(palette)] for i, f in enumerate(families)}


def _antigen_family_colors(symbols):
    """``({symbol -> color}, {family -> color})`` keyed by antigen family, with a
    stable per-family color assignment (see :func:`_antigen_family_color_map`). The
    returned family map carries only the families present in ``symbols``."""
    fam_by_sym = {s: _antigen_family(s) for s in symbols}
    full = _antigen_family_color_map()
    present = sorted(set(fam_by_sym.values()))
    fam_color = {f: full[f] for f in present}
    return {s: full[f] for s, f in fam_by_sym.items()}, fam_color


def _stacked_barh(rows, *, xlabel, title, legend=None, annotate=True, save=None):
    """Horizontal stacked bars. ``rows`` is ``[(row_label, [(seg_label, value,
    color), ...]), ...]`` in top-to-bottom order; each row's segments stack along x.
    ``legend`` is an optional ``{label: color}`` shown as a colour key. Segments wide
    enough are annotated with their ``seg_label``. The shared stacked-bar scaffold."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(rows))))
    total = max((sum(v for _, v, _ in segs) for _, segs in rows), default=1.0) or 1.0
    for i, (_, segs) in enumerate(rows):
        left = 0.0
        for seg_label, value, color in segs:
            ax.barh(i, value, left=left, color=color, edgecolor="white", linewidth=0.4)
            # Label every segment wide enough to fit text — denser than a fixed cut so
            # the carrying antigens are readable (the first segment is always labeled).
            if annotate and (value > 0.018 * total or left == 0.0):
                ax.text(
                    left + value / 2,
                    i,
                    seg_label,
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    rotation=90 if value < 0.04 * total else 0,
                    color="white",
                    clip_on=True,
                )
            left += value
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()  # first row at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend.values()]
        ax.legend(handles, list(legend), fontsize=6, title="antigen family", loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def _grouped_barh(categories, series, *, xlabel, title, save=None):
    """Grouped (paired) horizontal bars. ``categories`` are the row labels (top-to-
    bottom); ``series`` is ``[(name, [value per category], color), ...]`` — one bar
    per series within each category group. The shared grouped-bar scaffold."""
    import numpy as np

    plt = _plt()
    n_series = max(1, len(series))
    base = np.arange(len(categories))
    height = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(categories))))
    for k, (name, values, color) in enumerate(series):
        offset = (k - (n_series - 1) / 2) * height
        ax.barh(base + offset, values, height=height, label=name, color=color)
    ax.set_yticks(base)
    ax.set_yticklabels(categories, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def apd1_vs_tmb(*, save=None, annotate=True, strict_pd1=False):
    """Scatter of anti-PD-1 ORR (%) vs median TMB (log x), one point per cancer
    type with a curated value for both, colored by lineage family. The classic
    "more mutations -> more neoantigens -> better checkpoint response" view."""
    tmb = cancer_tmb()
    orr, ylabel, scope = _apd1_axis(strict_pd1=strict_pd1)
    codes = sorted(set(tmb) & set(orr))
    if not codes:
        raise ValueError("no cancer types with both a TMB and a checkpoint-response value")
    return _family_scatter(
        [(c, tmb[c], orr[c]) for c in codes],
        xlabel="Median tumor mutational burden (mut/Mb, log scale)",
        ylabel=ylabel,
        title=f"{scope} response vs TMB ({len(codes)} cancer types)",
        logx=True,
        annotate=annotate,
        figsize=(11, 7),
        legend_loc="upper left",
        save=save,
    )


def apd1_orr_bars(*, save=None, strict_pd1=False):
    """Horizontal bar chart of anti-PD-1 ORR by cancer type, highest at the top,
    colored by lineage family."""
    orr, xlabel, scope = _apd1_axis(strict_pd1=strict_pd1)
    codes = sorted(orr, key=lambda c: orr[c], reverse=True)  # highest at top
    return _ranked_family_barh(
        [(c, orr[c]) for c in codes],
        xlabel=xlabel,
        title=f"{scope} response by cancer type ({len(codes)} types)",
        save=save,
    )


def ici_response_by_regimen(*, save=None, only_multi=True):
    """Grouped horizontal bars of ICI ORR by cancer type, one bar per **regimen**
    (anti-PD-1 / anti-PD-L1 / anti-PD-1+anti-CTLA-4) — so the regimens are shown as
    distinct response sources and can be compared within a cancer. With ``only_multi``
    (default) only cancers with more than one regimen are shown (where the comparison
    is meaningful); set ``False`` to include every cancer with any ICI value."""
    from .ici import REGIMEN_FALLBACK, REGIMEN_LABELS, cancer_ici_response

    by_regimen = {r: cancer_ici_response(regimen=r) for r in REGIMEN_FALLBACK}
    codes = {c for m in by_regimen.values() for c in m}
    if only_multi:
        codes = {c for c in codes if sum(c in by_regimen[r] for r in REGIMEN_FALLBACK) > 1}
    if not codes:
        raise ValueError("no cancer types to plot")
    # Order rows by best available ORR (descending), top at the top.
    best = {c: max(by_regimen[r][c] for r in REGIMEN_FALLBACK if c in by_regimen[r]) for c in codes}
    ordered = sorted(codes, key=lambda c: best[c], reverse=True)

    palette = _stable_palette()
    reg_color = {r: palette[i] for i, r in enumerate(REGIMEN_FALLBACK)}
    # Absent regimen -> NaN (no bar drawn), so a missing estimate is not confused
    # with a true 0% ORR (matplotlib draws nothing for a NaN-width bar).
    series = [
        (REGIMEN_LABELS[r], [by_regimen[r].get(c, float("nan")) for c in ordered], reg_color[r])
        for r in REGIMEN_FALLBACK
    ]
    return _grouped_barh(
        [format_cancer_code_label(c) for c in ordered],
        series,
        xlabel="Objective response rate (%)",
        title=f"ICI response by regimen ({len(ordered)} cancer types)",
        save=save,
    )


def ici_regimen_comparison(*, save=None, min_regimens=1):
    """Dumbbell/dot plot of ICI ORR — **every cancer type on its own row, one colored
    dot per available regimen** (anti-PD-1 / anti-PD-L1 / anti-PD-1+anti-CTLA-4),
    connected by a line, so all three estimates can be compared side by side and
    scanned by cancer type. Rows are sorted by best available ORR (highest at top).
    ``min_regimens`` filters to cancers with at least that many regimens (1 = show
    every cancer with any ICI value; 2 = only where regimens actually differ)."""
    from .ici import REGIMEN_FALLBACK, REGIMEN_LABELS, cancer_ici_response

    by_regimen = {r: cancer_ici_response(regimen=r) for r in REGIMEN_FALLBACK}
    codes = {c for m in by_regimen.values() for c in m}
    codes = {c for c in codes if sum(c in by_regimen[r] for r in REGIMEN_FALLBACK) >= min_regimens}
    if not codes:
        raise ValueError("no cancer types to plot")
    best = {c: max(by_regimen[r][c] for r in REGIMEN_FALLBACK if c in by_regimen[r]) for c in codes}
    ordered = sorted(codes, key=lambda c: best[c])  # ascending; highest ends up on top

    plt = _plt()
    palette = _stable_palette()
    reg_color = {r: palette[i] for i, r in enumerate(REGIMEN_FALLBACK)}
    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(ordered))))
    for y, c in enumerate(ordered):
        present = [(r, by_regimen[r][c]) for r in REGIMEN_FALLBACK if c in by_regimen[r]]
        xs = [v for _, v in present]
        if len(xs) > 1:  # connect the estimates for this cancer
            ax.plot([min(xs), max(xs)], [y, y], color="#cccccc", lw=1.5, zorder=1)
        for r, v in present:
            ax.scatter(v, y, color=reg_color[r], s=48, edgecolor="white", linewidth=0.5, zorder=2)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([format_cancer_code_label(c) for c in ordered], fontsize=8)
    ax.set_xlabel("Objective response rate (%)")
    ax.set_title(f"ICI response by regimen and cancer type ({len(ordered)} types)")
    ax.grid(True, axis="x", alpha=0.3)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=reg_color[r], label=REGIMEN_LABELS[r])
        for r in REGIMEN_FALLBACK
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="regimen")
    fig.tight_layout()
    return _save(fig, save)


def ici_orr_pooled_forest(*, regimen=None, save=None):
    """Forest plot of ICI objective response rate, one cancer type per row:

    - **small grey dots** = each individual reported trial estimate (with a thin 95% CI
      whisker where the trial reported one; dot size ∝ √n),
    - **colored diamond** = the pooled responder-weighted estimate (Σresponders/Σn) with
      its 95% Wilson CI as a thick bar, colored by lineage family.

    Where no trial reports responders+n (so no pool is possible) the row shows the curated
    representative anchor as the diamond, without a CI. With ``regimen=None`` each cancer
    is shown at its fallback-resolved regimen (anti-PD-1 → anti-PD-L1 → combo); pass a
    regimen to pin one. Rows are sorted by the estimate (highest at top)."""
    from .ici import (
        cancer_ici_regimen,
        cancer_ici_response,
        pooled_ici_response,
    )

    plt = _plt()
    anchor = cancer_ici_response() if regimen is None else cancer_ici_response(regimen=regimen)
    cells = {c: (cancer_ici_regimen(c) if regimen is None else regimen) for c in anchor}

    rows = []  # (code, regimen, est, lo, hi, [(value, ci_lo, ci_hi, n), ...])
    for code, reg in cells.items():
        if reg is None:
            continue
        # Dots = every reported source; diamond = the *primary-only* pool so it never
        # double-counts a trial's overlapping subgroups (include_alternates=False).
        allsrc = pooled_ici_response(code, regimen=reg, metric="ORR", verified_only=False)
        clean = pooled_ici_response(
            code, regimen=reg, metric="ORR", verified_only=False, include_alternates=False
        )
        pts = [
            (s["value"], s["ci_low"], s["ci_high"], s["n"])
            for s in allsrc["sources"]
            if s["value"] is not None
        ]
        if clean["pooled_pct"] is not None:
            est, lo, hi = clean["pooled_pct"], clean["ci_low"], clean["ci_high"]
        else:
            est, lo, hi = anchor.get(code), None, None
        if est is None:
            continue
        rows.append((code, reg, est, lo, hi, pts))

    if not rows:
        raise ValueError("no cancer types to plot")
    rows.sort(key=lambda r: r[2])  # ascending; invert_yaxis puts the highest on top
    code_color, _ = _family_colors([r[0] for r in rows])

    fig, ax = plt.subplots(figsize=(11, max(6, 0.42 * len(rows))))
    for y in range(0, len(rows), 2):  # alternating row bands to trace label -> point
        ax.axhspan(y - 0.5, y + 0.5, color="#f4f4f4", zorder=0)
    for y, (code, _reg, est, lo, hi, pts) in enumerate(rows):
        # individual trial estimates (grey), with CI whiskers + √n sizing
        for v, clo, chi, n in pts:
            if clo is not None and chi is not None:
                ax.plot([clo, chi], [y, y], color="#bbbbbb", lw=1.0, zorder=1)
            size = 18 + 6 * (n**0.5) if n else 22
            ax.scatter(
                v,
                y,
                s=min(size, 90),
                facecolor="none",
                edgecolor="#888888",
                linewidth=0.8,
                zorder=2,
            )
        # pooled / anchor estimate (family-colored diamond + Wilson CI bar)
        col = code_color[code]
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=col, lw=2.6, alpha=0.85, zorder=3)
        ax.scatter(est, y, marker="D", s=46, color=col, edgecolor="black", linewidth=0.5, zorder=4)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{format_cancer_code_label(c)} [{reg}]" for c, reg, *_ in rows], fontsize=8
    )
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Objective response rate (%)")
    ax.set_xlim(left=-2)
    scope = "fallback-resolved regimen" if regimen is None else regimen
    ax.set_title(
        f"ICI ORR by cancer type — pooled estimate vs individual trials "
        f"({len(rows)} types, {scope})"
    )
    ax.grid(True, axis="x", alpha=0.3)
    # rows are sorted ascending and y grows upward, so the highest ORR is already on top
    handles = [
        plt.Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            color="#444444",
            label="pooled estimate (Wilson 95% CI)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="#888888",
            label="individual trial (95% CI)",
        ),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, save)


def incidence_vs_mortality(*, region="us", save=None):
    """Scatter of mortality-share vs incidence-share (%) per burden category for
    a region (``"us"`` or ``"world"``). The diagonal separates high-lethality
    (pancreas, lung) from low-lethality (thyroid, prostate) categories."""
    plt = _plt()
    if region not in ("us", "world"):
        raise ValueError("region must be 'us' or 'world'")
    inc_metric, mort_metric = f"{region}_incidence_pct", f"{region}_mortality_pct"
    inc = cancer_burden(metric=inc_metric)
    mort = cancer_burden(metric=mort_metric)
    cats = sorted(set(inc) & set(mort))

    fig, ax = plt.subplots(figsize=(9, 8))
    lim = max(max(inc.values(), default=1), max(mort.values(), default=1)) * 1.1
    ax.plot([0, lim], [0, lim], color="0.7", linestyle="--", linewidth=1, zorder=1)
    for cat in cats:
        ax.scatter(inc[cat], mort[cat], s=60, color="tab:red", alpha=0.8, zorder=3)
        ax.annotate(
            cat, (inc[cat], mort[cat]), fontsize=6, xytext=(3, 3), textcoords="offset points"
        )
    ax.set_xlabel(f"{region.upper()} share of cancer incidence (%)")
    ax.set_ylabel(f"{region.upper()} share of cancer mortality (%)")
    ax.set_title(f"Incidence vs mortality by burden category ({region.upper()})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, save)


def _cta_expression_matrix(stat, cohorts, *, proteoform=False):
    """cohorts × CTA matrix of the requested ``stat`` TPM (NaN where a cohort lacks a
    gene). Rows are cohort codes, columns are CTA symbols. With ``proteoform=True``,
    reads the proteoform-summed percentile vectors and labels columns by the proteoform
    symbol (NY-ESO-1, XAGE1A/B). Cohorts without the vector are skipped with a warning
    rather than aborting the whole plot."""
    import warnings

    import pandas as pd

    col = _STAT_PERCENTILE_COL[stat]
    id_to_name = cta_gene_id_to_name()
    cta_ids = set(cta_gene_ids())
    # In the collapsed space, a CTA group's row is keyed by its proteoform_key, and the
    # row's canonical Ensembl_Gene_ID may be an unexpressed member outside cta_gene_ids
    # — so match on the CTA *proteoform keys*, not the canonical ENSG (which would drop
    # CGB3/5/8, CT45A2/8/9, …). The keys are derived from the CTA set itself.
    cta_keys = None
    if proteoform:
        from .proteoforms import gene_to_proteoform_id

        cta_keys = set(gene_to_proteoform_id(sorted(cta_ids)).values())
    rows = {}
    skipped = []
    for code in cohorts:
        try:
            df = cohort_gene_percentiles(code, as_tpm=True, proteoform=proteoform)
        except ValueError:
            skipped.append(str(code))
            continue
        if proteoform:
            mask = df["proteoform_key"].astype(str).isin(cta_keys)
            sub = df.loc[mask]
            labels = sub["Symbol"].to_numpy()  # the proteoform symbol (NY-ESO-1, XAGE1A/B)
        else:
            ids = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
            sub = df.loc[ids.isin(cta_ids)]
            ids_kept = sub["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
            labels = ids_kept.map(id_to_name).to_numpy()
        rows[code] = pd.Series(sub[col].to_numpy(), index=labels)
    if skipped:
        warnings.warn(
            f"skipped {len(skipped)} cohort(s) without a percentile vector: {', '.join(skipped)}",
            stacklevel=2,
        )
    matrix = pd.DataFrame(rows).T  # cohorts (rows) × CTA symbols (cols)
    return matrix.loc[:, ~matrix.columns.duplicated()]


def cta_expression_heatmap(
    *,
    stat="median",
    cohorts=None,
    n_cohorts=30,
    n_ctas=30,
    high_tpm=30.0,
    proteoform=True,
    save=None,
):
    """Heatmap of cancer-testis-antigen expression — rows are cohorts, columns are
    CTAs, cells are the cohort's per-gene ``stat`` TPM (``"median"``/``"q1"``/``"q3"``
    = p50/p25/p75 of the shipped percentile vector).

    Rows are the ``n_cohorts`` cohorts with the highest single-CTA ``stat``; columns
    are the ``n_ctas`` CTAs with the highest peak across cohorts, ordered by how many
    cohorts express them above ``high_tpm`` (the actionable threshold) then by peak.
    Colour is a log scale (dark = silent, hot = high) anchored in the clinically
    meaningful 1–1000 TPM range.

    ``cohorts`` restricts the cohort pool (default: every cohort with a percentile
    vector). Needs the expression bundle present (percentile artifacts).

    ``proteoform=True`` reads the **proteoform-summed** percentile vectors: identical-
    protein paralogs were summed per sample *before* the percentiles were computed, so
    a duplicated antigen appears once under its proteoform symbol (NY-ESO-1, XAGE1A/B)
    rather than as several diluted member rows. Needs the proteoform percentile shards
    (``generate_cohort_percentiles.py --proteoform``).
    """
    if stat not in _STAT_PERCENTILE_COL:
        raise ValueError(f"stat must be one of {sorted(_STAT_PERCENTILE_COL)}")
    if cohorts is None:
        cohorts = available_percentile_cohorts(proteoform=proteoform)
        if not cohorts and proteoform:
            # No proteoform percentile shards present: recompute on the fly for cohorts
            # whose per-sample matrix is cached; if none are, fall back to per-gene.
            cohorts = _cached_per_sample_cohorts()
            if not cohorts:
                proteoform = False
                cohorts = available_percentile_cohorts(proteoform=False)
    if not cohorts:
        variant = "proteoform-summed " if proteoform else ""
        raise ValueError(
            f"no cohorts with a {variant}percentile vector — is the expression bundle present?"
        )

    matrix = _cta_expression_matrix(stat, cohorts, proteoform=proteoform)
    if matrix.empty or matrix.shape[1] == 0:
        raise ValueError(
            "no CTA expression data for the selected cohorts — none had a percentile "
            "vector, or none expressed any CTA gene."
        )
    # Columns: top CTAs by peak, ordered by breadth (#cohorts > high_tpm) then peak.
    peak = matrix.max(axis=0, skipna=True)
    breadth = (matrix > high_tpm).sum(axis=0)
    top_ctas = (
        peak.to_frame("peak")
        .assign(breadth=breadth)
        .sort_values(["breadth", "peak"], ascending=False)
        .head(n_ctas)
        .index
    )
    # Rows: cohorts with the highest single-CTA value, sorted descending.
    row_score = matrix[top_ctas].max(axis=1, skipna=True)
    top_cohorts = row_score.sort_values(ascending=False).head(n_cohorts).index
    grid = matrix.loc[top_cohorts, top_ctas]

    return _cohort_gene_heatmap(
        grid,
        title=f"CTA expression ({stat} TPM) — top {len(top_cohorts)} cohorts × {len(top_ctas)} CTAs",
        cbar_label=f"{stat} TPM (log)",
        cmap="magma",
        lognorm=True,
        save=save,
    )


def _cta_prevalence_by_cohort(threshold):
    """``{cohort code: fraction of samples where the single most-prevalent CTA is a
    top-expressed gene}`` from the within-sample artifact.

    This is the best **shipped** proxy for "fraction of patients with a targetable
    CTA": the within-sample artifact gives, per gene, the fraction of a cohort's
    samples in which it ranks in the top ``(1-threshold)`` *within that sample*; we
    take the max over the CTA set. It is a lower bound on the true "≥1 CTA expressed"
    union (different patients may express different CTAs), which needs the per-sample
    joint matrices — see :func:`oncoref.expression.proteoform_representative_samples`
    and #13. Cohorts without a within-sample shard are skipped."""
    import pandas as pd

    cta_ids = set(cta_gene_ids())
    out = {}
    for code in locally_available_within_sample_cohorts(include_recomputable=False):
        df = within_sample_top_fraction(code, threshold=threshold)
        frac_col = next((c for c in df.columns if c.startswith("frac_samples_top")), None)
        if frac_col is None:
            continue
        ids = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
        cta_frac = pd.to_numeric(df.loc[ids.isin(cta_ids), frac_col], errors="coerce")
        out[str(code)] = float(cta_frac.max()) if len(cta_frac) and cta_frac.notna().any() else 0.0
    return out


def _addressable_prevalence(source, *, threshold, threshold_tpm):
    """``({code: prevalence}, prevalence label)`` for the two prevalence sources."""
    if source == "per_sample":
        # Faithful: fraction of a cohort's patients expressing >=1 CTA above
        # threshold_tpm (the union), from the per-sample matrices.
        from .coverage import addressable_fraction_by_cohort

        prev = addressable_fraction_by_cohort(threshold_tpm=threshold_tpm).to_dict()
        return prev, f"P(≥1 CTA > {threshold_tpm:g} TPM), per-patient"
    if source == "within_sample":
        prev = _cta_prevalence_by_cohort(threshold)
        return prev, f"top-CTA prevalence, thr={threshold}"
    raise ValueError("source must be 'within_sample' or 'per_sample'")


def _cta_addressable_burden_from_prevalence(
    prevalence,
    *,
    prevalence_label,
    metric,
    n,
    save,
):
    """Render addressable burden from an already-audited cohort prevalence map."""
    if not prevalence:
        raise ValueError("no CTA prevalence available for the requested data source")
    burden = cancer_burden(metric=metric)
    burden_label = _burden_metric_label(metric)
    xlabel = f"Addressable burden ({burden_label} × {prevalence_label})"

    rows = []
    for code, fraction in prevalence.items():
        category = burden_category(code)
        burden_share = burden.get(category) if category else None
        if burden_share:
            rows.append((code, float(burden_share) * float(fraction)))
    if not rows:
        raise ValueError("no cohort mapped to both a burden category and CTA prevalence")

    rows.sort(key=lambda row: row[1], reverse=True)
    rows = rows[:n]
    return _ranked_family_barh(
        rows,
        xlabel=xlabel,
        title=f"CTA-addressable {burden_label} — top {len(rows)} cancers",
        legend=True,
        save=save,
    )


def cta_addressable_burden(
    *,
    source="within_sample",
    threshold=0.95,
    threshold_tpm=50.0,
    metric="us_incidence_pct",
    n=25,
    save=None,
):
    """Bar chart of **CTA-addressable cancer burden** per cancer type — ``incidence
    share × CTA prevalence``, ranking cancers by how many patients a CTA-directed
    therapy could address.

    ``source`` selects the prevalence basis:
      - ``"within_sample"`` (default, portable) — the single-best-CTA within-sample
        prevalence from the shipped within-sample bundle (a *proxy*; ``threshold`` is
        the within-sample rank cut 0.99/0.95/0.90);
      - ``"per_sample"`` — the **faithful** fraction of patients expressing ≥1 CTA
        above ``threshold_tpm`` (the per-patient union), from the per-sample matrices
        (:func:`oncoref.coverage.addressable_fraction_by_cohort`); needs the
        cohorts' per-sample matrices cached.

    ``metric`` selects the burden basis (for example ``"us_incidence_pct"`` or
    ``"world_mortality_pct"``); ``n`` caps the bars; bars are coloured by registry
    family. Burden is at burden-category granularity (several subtypes share a
    category), so read the bar as a relative prioritization."""
    prevalence, prevalence_label = _addressable_prevalence(
        source, threshold=threshold, threshold_tpm=threshold_tpm
    )
    return _cta_addressable_burden_from_prevalence(
        prevalence,
        prevalence_label=prevalence_label,
        metric=metric,
        n=n,
        save=save,
    )


def cta_within_sample_percentile_addressable_burden(
    *,
    percentile=0.95,
    cohorts=None,
    coverage_table=None,
    metric="us_incidence_pct",
    n=25,
    save=None,
):
    """CTA-addressable burden using the true patient union at p90/p95.

    A CTA is positive when its biological clean-TPM rank within that individual
    tumor is at least ``percentile``.  Unlike ``source='within_sample'`` in
    :func:`cta_addressable_burden`, this calculation retains CTA co-occurrence and
    counts each patient once when any CTA is positive.  It therefore needs cached
    per-sample matrices.  ``coverage_table`` can reuse one
    :func:`oncoref.coverage.within_sample_percentile_coverage_sweep` across a plot
    batch.
    """
    from .coverage import within_sample_percentile_addressable_fraction_by_cohort

    fractions = within_sample_percentile_addressable_fraction_by_cohort(
        cohorts,
        percentile=percentile,
        coverage=coverage_table,
        on_missing="record",
    )
    return _cta_addressable_burden_from_prevalence(
        fractions.to_dict(),
        prevalence_label=f"P(≥1 CTA at within-sample p{round(percentile * 100)})",
        metric=metric,
        n=n,
        save=save,
    )


def _cached_per_sample_cohorts():
    """Cohort codes whose per-sample matrix is cached locally (sorted)."""
    from . import source_matrices

    return sorted(c for c in source_matrices.available_cohorts() if source_matrices.is_cached(c))


def cta_patient_count_heatmap(
    *,
    threshold=0.95,
    threshold_tpm=None,
    cohorts=None,
    n_cohorts=30,
    n_ctas=30,
    save=None,
):
    """Heatmap of **per-patient CTA prevalence**, proteoform-aggregated (CTAG1A/B =
    NY-ESO-1 collapse once) — rows are cohorts, columns are CTAs.

    By default each cell is the fraction of a cohort's patients in whom the antigen
    ranks in the top ``(1 - threshold)`` *within that patient* — ``threshold=0.95`` →
    the within-sample **p95** (top 5%) convention pirlygenes uses, far more stringent
    than a flat low-TPM cut. Pass ``threshold_tpm=<TPM>`` instead to use the absolute
    "fraction of patients with clean TPM > threshold" from the per-sample matrices.

    Columns are ordered by **mean prevalence across the shown cohorts** (most broadly
    expressed antigen first); rows by their single best CTA. ``cohorts`` defaults
    to locally readable within-sample data in rank mode and cached per-sample
    matrices in absolute-TPM mode. Neither default triggers a download."""
    import pandas as pd

    if cohorts is not None:
        cohorts = list(cohorts)
    elif threshold_tpm is not None:
        cohorts = _cached_per_sample_cohorts()
    else:
        cohorts = locally_available_within_sample_cohorts(proteoform=True, scope="cta")
    if not cohorts:
        if threshold_tpm is not None:
            raise ValueError(
                "no cached per-sample matrices for absolute-TPM prevalence — "
                "stage source matrices first."
            )
        raise ValueError(
            "no local proteoform within-sample shards or cached per-sample matrices — "
            "fetch the bundle or stage source matrices first."
        )
    rows = {}
    key_to_symbol = {}
    if threshold_tpm is not None:
        from .coverage import cta_patient_fractions

        for code in cohorts:
            pf = cta_patient_fractions(code, threshold_tpm=threshold_tpm)
            rows[code] = pf.groupby("proteoform_key")["fraction_expressing"].max()
            key_to_symbol.update(zip(pf["proteoform_key"].astype(str), pf["Symbol"].astype(str)))
        thr_label = f"> {threshold_tpm:g} TPM"
    else:
        from .proteoforms import gene_to_proteoform_id

        # within-sample top-fraction is genome-wide ranking; restrict to the CTA panel by
        # its proteoform keys (the canonical ENSG of a collapsed group may sit outside the
        # CTA id set, so match keys, not ENSGs — same trick as _cta_expression_matrix).
        cta_keys = set(gene_to_proteoform_id(sorted(cta_gene_ids())).values())
        for code in cohorts:
            df = within_sample_top_fraction(code, threshold=threshold, proteoform=True, scope="cta")
            frac_col = next(c for c in df.columns if c.startswith("frac_samples_top"))
            key = df["proteoform_key"].astype(str)
            mask = key.isin(cta_keys)
            rows[code] = pd.Series(df.loc[mask, frac_col].to_numpy(), index=key[mask])
            key_to_symbol.update(zip(key[mask], df.loc[mask, "Symbol"].astype(str)))
        thr_label = f"within-sample p{round(threshold * 100)}"
    matrix = pd.DataFrame(rows).T  # cohorts × CTA proteoform keys
    if matrix.empty or matrix.shape[1] == 0:
        raise ValueError("no CTA prevalence in the selected cohorts")

    # Columns: most broadly-prevalent CTAs first (mean prevalence ACROSS cohorts).
    top_ctas = matrix.mean(axis=0, skipna=True).sort_values(ascending=False).head(n_ctas).index
    row_score = matrix[top_ctas].max(axis=1)
    top_cohorts = row_score.sort_values(ascending=False).head(n_cohorts).index
    grid = matrix.loc[top_cohorts, top_ctas].fillna(0.0)
    # Label columns by the proteoform symbol (NY-ESO-1, CT83), not the ENSG-or-symbol key.
    grid = grid.rename(columns=lambda key: key_to_symbol.get(str(key), str(key)))

    return _cohort_gene_heatmap(
        grid,
        title=(
            f"CTA per-patient prevalence ({thr_label}) — "
            f"{len(top_cohorts)} cohorts × {len(top_ctas)} CTAs"
        ),
        cbar_label="fraction of patients",
        cmap="viridis",
        save=save,
    )


def _render_cta_coverage_curves(
    coverage_by_code,
    *,
    max_genes,
    top_n,
    threshold_label,
    save,
):
    """Render comparable greedy curves from precomputed per-cohort tables."""
    import numpy as np

    plt = _plt()
    curves = []
    for code, coverage in coverage_by_code.items():
        if coverage.empty:
            continue
        x = np.concatenate([[0], coverage["rank"].to_numpy()])
        y = np.concatenate([[0.0], coverage["cumulative_fraction"].to_numpy()])
        curves.append((code, x, y))
    if not curves:
        raise ValueError("no coverage curve could be drawn from the available cohort data")

    curves.sort(key=lambda item: item[2][-1], reverse=True)
    total = len(curves)
    shown = curves[:top_n] if top_n else curves
    cmap = plt.get_cmap("tab10" if len(shown) <= 10 else "tab20")
    code_color = {code: cmap(i % cmap.N) for i, (code, _, _) in enumerate(shown)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for code, x, y in shown:
        ax.step(
            x,
            y,
            where="post",
            color=code_color[code],
            lw=2.0,
            label=f"{format_cancer_code_label(code)} ({y[-1]:.0%})",
        )
    ax.set_xlabel("number of CTAs in panel (greedy set cover)")
    ax.set_ylabel(f"fraction of patients covered ({threshold_label})")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_genes)
    ax.grid(True, alpha=0.3)
    scope = f"top {len(shown)} of {total} cohorts by coverage" if top_n else f"{total} cohorts"
    ax.set_title(f"CTA antigen-coverage curves — {scope}")
    ax.legend(fontsize=7, loc="lower right", ncol=2, title="cohort (final coverage)")
    fig.tight_layout()
    return _save(fig, save)


def _render_cta_coverage_stacked_bars(
    coverage_by_code,
    *,
    total_cohorts,
    top_n,
    threshold_label,
    save,
):
    """Render marginal antigen contributions from precomputed greedy tables."""
    if not coverage_by_code:
        raise ValueError("no coverage to plot from the available cohort data")
    ordered = sorted(
        coverage_by_code.items(),
        key=lambda item: float(item[1]["cumulative_fraction"].iloc[-1]),
        reverse=True,
    )
    if top_n:
        ordered = ordered[:top_n]
    all_symbols = {str(symbol) for _, table in ordered for symbol in table["Symbol"]}
    symbol_colors, family_colors = _antigen_family_colors(all_symbols)

    rows = []
    for code, table in ordered:
        covered = float(table["cumulative_fraction"].iloc[-1])
        segments = [
            (
                str(row.Symbol),
                float(row.marginal_fraction),
                symbol_colors[str(row.Symbol)],
            )
            for row in table.itertuples()
        ]
        rows.append((f"{format_cancer_code_label(code)}  ({covered:.0%})", segments))
    suffix = (
        f"top {len(rows)} of {total_cohorts} cohorts by coverage"
        if top_n and total_cohorts > len(rows)
        else f"{len(rows)} cohorts"
    )
    return _stacked_barh(
        rows,
        xlabel=f"fraction of patients covered ({threshold_label})",
        title=f"CTA greedy coverage by antigen — {suffix}",
        legend=family_colors,
        save=save,
    )


def _percentile_coverage_by_code(
    cancer_types,
    *,
    percentile,
    max_genes,
    coverage_table,
):
    """Select one p90/p95 greedy table per requested cohort."""
    import pandas as pd

    from .coverage import _coverage_at_percentile, within_sample_percentile_coverage_sweep

    requested = [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
    codes = [resolve_cancer_type(code, strict=False) or str(code) for code in requested]
    if coverage_table is None:
        coverage_table = within_sample_percentile_coverage_sweep(
            codes,
            percentiles=(percentile,),
            on_missing="record",
        )
    selected = _coverage_at_percentile(coverage_table, percentile)
    selected = selected[selected["cancer_code"].astype(str).isin(set(codes))]
    selected = selected[pd.to_numeric(selected["rank"], errors="coerce") > 0]

    by_code = {}
    for code, table in selected.groupby("cancer_code", sort=False):
        ordered = table.sort_values("rank", kind="stable")
        by_code[str(code)] = ordered.head(max_genes).reset_index(drop=True)
    return codes, by_code


def _absolute_coverage_by_code(codes, *, threshold_tpm, max_genes):
    """Compute one absolute-TPM greedy table per cohort, omitting empty results."""
    from .coverage import greedy_coverage

    by_code = {}
    for code in codes:
        table = greedy_coverage(code, threshold_tpm=threshold_tpm, max_genes=max_genes)
        if not table.empty:
            by_code[code] = table
    return by_code


def cta_coverage_curves(
    cancer_types,
    *,
    threshold_tpm=50.0,
    max_genes=20,
    top_n=10,
    save=None,
):
    """Greedy CTA coverage curves at one absolute clean-TPM threshold."""
    codes = [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
    by_code = _absolute_coverage_by_code(codes, threshold_tpm=threshold_tpm, max_genes=max_genes)
    return _render_cta_coverage_curves(
        by_code,
        max_genes=max_genes,
        top_n=top_n,
        threshold_label=f"≥1 CTA > {threshold_tpm:g} TPM",
        save=save,
    )


def cta_within_sample_percentile_coverage_curves(
    cancer_types,
    *,
    percentile=0.95,
    coverage_table=None,
    max_genes=20,
    top_n=10,
    save=None,
):
    """Greedy CTA coverage curves at a tumor-specific p90/p95 rank cutoff."""
    _, by_code = _percentile_coverage_by_code(
        cancer_types,
        percentile=percentile,
        max_genes=max_genes,
        coverage_table=coverage_table,
    )
    return _render_cta_coverage_curves(
        by_code,
        max_genes=max_genes,
        top_n=top_n,
        threshold_label=f"≥1 CTA at within-sample p{round(percentile * 100)}",
        save=save,
    )


def cta_coverage_stacked_bars(
    cancer_types,
    *,
    threshold_tpm=50.0,
    max_genes=12,
    top_n=40,
    save=None,
):
    """Marginal CTA coverage per cohort at one absolute clean-TPM threshold."""
    codes = [cancer_types] if isinstance(cancer_types, str) else list(cancer_types)
    by_code = _absolute_coverage_by_code(codes, threshold_tpm=threshold_tpm, max_genes=max_genes)
    return _render_cta_coverage_stacked_bars(
        by_code,
        total_cohorts=len(codes),
        top_n=top_n,
        threshold_label=f"≥1 CTA > {threshold_tpm:g} TPM",
        save=save,
    )


def cta_within_sample_percentile_coverage_stacked_bars(
    cancer_types,
    *,
    percentile=0.95,
    coverage_table=None,
    max_genes=12,
    top_n=40,
    save=None,
):
    """Marginal CTA coverage per cohort at a tumor-specific p90/p95 cutoff."""
    codes, by_code = _percentile_coverage_by_code(
        cancer_types,
        percentile=percentile,
        max_genes=max_genes,
        coverage_table=coverage_table,
    )
    return _render_cta_coverage_stacked_bars(
        by_code,
        total_cohorts=len(codes),
        top_n=top_n,
        threshold_label=f"≥1 CTA at within-sample p{round(percentile * 100)}",
        save=save,
    )


def burden_category_bars(*, region="us", n=None, save=None):
    """Grouped horizontal bars of cancer **incidence vs mortality** share per burden
    category for a region (``"us"``/``"world"``), categories ordered by incidence. The
    reference view behind :func:`incidence_vs_mortality` — read the gap between a
    category's two bars as its lethality."""
    if region not in ("us", "world"):
        raise ValueError("region must be 'us' or 'world'")
    plt = _plt()
    inc = cancer_burden(metric=f"{region}_incidence_pct")
    mort = cancer_burden(metric=f"{region}_mortality_pct")
    cats = sorted(set(inc) & set(mort), key=lambda c: inc.get(c, 0.0), reverse=True)
    if not cats:
        raise ValueError("no burden category with both an incidence and a mortality share")
    if n is not None:
        cats = cats[:n]
    cmap = plt.get_cmap("tab10")
    return _grouped_barh(
        cats,
        [
            ("incidence", [inc[c] for c in cats], cmap(0)),
            ("mortality", [mort[c] for c in cats], cmap(3)),
        ],
        xlabel=f"{region.upper()} share of cancer cases / deaths (%)",
        title=f"Cancer burden by category ({region.upper()}) — incidence vs mortality",
        save=save,
    )


def cta_burden_vs_response(*, against="apd1", threshold_tpm=50.0, cohorts=None, save=None):
    """Scatter of a cohort's **mean CTA antigen load** (mean number of CTAs a patient
    expresses above ``threshold_tpm``, from
    :func:`oncoref.coverage.mean_antigens_per_patient`) vs its anti-PD-1 ORR
    (``against="apd1"``), broader ICI response (``"ici"``), median TMB
    (``"tmb"``), or a burden axis (``"us_incidence"``, ``"us_mortality"``,
    ``"world_incidence"``, ``"world_mortality"``), one point per cancer type,
    coloured by lineage family.

    The per-patient counterpart to :func:`apd1_response_signature_scatter`: does a
    cohort's CTA antigen load track checkpoint response / mutational burden? Points are
    cohorts with BOTH a cached per-sample matrix and the chosen response metric. Needs
    the per-sample matrices cached."""
    from .coverage import mean_antigens_per_patient

    ymap, ylabel = _reference_metric_axis(against)

    cohorts = list(cohorts) if cohorts is not None else _cached_per_sample_cohorts()
    points = _collapse_metric_points(
        cohorts,
        ymap,
        lambda code: mean_antigens_per_patient(code, threshold_tpm=threshold_tpm),
    )
    if not points:
        raise ValueError(
            f"no cohort with both a cached per-sample matrix and a {against} value — "
            "fetch the matrices (source_matrices.fetch) for those cohorts first."
        )
    return _family_scatter(
        points,
        xlabel=f"mean CTAs expressed per patient (> {threshold_tpm:g} TPM)",
        ylabel=ylabel,
        title=f"CTA antigen load vs {against} — {len(points)} cancers",
        save=save,
    )


def apd1_response_signature_scatter(signature="t_cell_inflamed", *, cohorts=None, save=None):
    """Scatter of a cohort's **aPD1 response-signature score** vs its anti-PD-1 ORR,
    one point per cancer type — the mechanism view of *why* response varies.

    The signature score is the cohort-mean log clean-TPM of the signature's genes
    (:func:`oncoref.response_signatures.signature_score`); ``signature`` is one of
    :func:`oncoref.response_signatures.response_signature_names` (e.g.
    ``"t_cell_inflamed"`` / ``"cytotoxic"`` / ``"antigen_presentation"`` →
    response-associated, ``"tgfb_exclusion"`` → resistance-associated). Points are
    cohorts that have BOTH a cached per-sample matrix and a curated aPD1 ORR,
    coloured by lineage family. Needs the per-sample matrices cached."""
    from .response_signatures import response_signature_direction, signature_score

    direction = response_signature_direction(signature)  # validates the name
    orr, ylabel, _ = _apd1_axis(strict_pd1=True)
    cohorts = list(cohorts) if cohorts is not None else _cached_per_sample_cohorts()
    points = _collapse_metric_points(cohorts, orr, lambda code: signature_score(code, signature))
    if not points:
        raise ValueError(
            "no cohort with both a cached per-sample matrix and an aPD1 ORR — fetch "
            "the matrices (source_matrices.fetch) for aPD1 cohorts first."
        )
    return _family_scatter(
        points,
        xlabel=f"{signature} signature score (cohort-mean log clean TPM)",
        ylabel=ylabel,
        title=f"aPD1 response vs {signature} ({direction}-associated) — {len(points)} cancers",
        save=save,
    )


def cta_specific_9mer_load(*, against="tmb", threshold_tpm=50.0, cohorts=None, save=None):
    """Scatter of a cohort's **mean per-patient CTA-specific 9-mer load**
    (:func:`oncoref.peptides.cta_specific_9mer_load`) vs its median TMB
    (``against="tmb"``), anti-PD-1 ORR (``against="apd1"``), broader ICI response
    (``"ici"``), or a burden axis (``"us_incidence"``, ``"us_mortality"``,
    ``"world_incidence"``, ``"world_mortality"``), one point per cancer type,
    coloured by lineage family.

    The 9-mer load is, for the average patient, the total CTA-specific 9-mers across
    the CTAs they express above ``threshold_tpm`` — a per-patient measure of
    tumor-restricted neoepitope source breadth. CTA-specific 9-mers come from the
    reference proteome (the longest protein per gene, background-subtracted against
    all non-CTA proteins); see :mod:`oncoref.peptides`.

    Points are cohorts with BOTH a cached per-sample matrix and the chosen metric.
    Needs the per-sample matrices cached and a downloaded Ensembl release with protein
    sequences (the first call builds + caches the per-CTA 9-mer table)."""
    from .peptides import cta_specific_9mer_load as _load

    xmap, ylabel2 = _reference_metric_axis(against)

    cohorts = list(cohorts) if cohorts is not None else _cached_per_sample_cohorts()
    points = _collapse_metric_points(
        cohorts,
        xmap,
        lambda code: _load(code, threshold_tpm=threshold_tpm),
    )
    if not points:
        raise ValueError(
            f"no cohort with both a cached per-sample matrix and a {against} value — "
            "fetch the matrices (source_matrices.fetch) for those cohorts first."
        )
    return _family_scatter(
        points,
        xlabel=f"mean CTA-specific 9-mers per patient (> {threshold_tpm:g} TPM)",
        ylabel=ylabel2,
        title=f"CTA-specific 9-mer load vs {against} — {len(points)} cancers",
        save=save,
    )
