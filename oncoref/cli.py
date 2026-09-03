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

"""``oncoref`` command-line interface.

Reference lookups over the bundled tables (cancer-type / TMB / burden / ICI) plus
the ``data`` manager for downloadable expression-bundle, HPA, and per-sample sources.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import (
    apd1,
    cancer_types,
    cta,
    data_bundle,
    expression_registry,
    ici,
    incidence,
    proteoforms,
    reference_data,
    samples,
    tmb,
)
from .version import __version__


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"oncoref v{__version__}")
    return 0


def _print_bundle_prune(args: argparse.Namespace) -> int:
    deleted = data_bundle.prune_cache(keep_current=not args.include_current, dry_run=not args.yes)
    if not deleted:
        print("Nothing to prune.")
        return 0
    verb = "Would delete" if not args.yes else "Deleted"
    for entry in deleted:
        print(f"{verb} {entry['version']}  ({_fmt_bytes(entry['size_bytes'])})  {entry['path']}")
    if not args.yes:
        print("\n(dry run — pass --yes to delete)")
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    """Unified data management over the catalog (expression bundle + HPA sources)."""
    from . import catalog, source_matrices

    if args.action == "list":
        # The full inventory: every oncoref-domain dataset and how it's held.
        # `Cohorts` is the per-cohort file count held *inside* a directory dataset.
        print(
            f"{'Dataset':<46} {'Held':<8} {'Avail':<6} {'Cohorts':>8} {'Category':<14} Description"
        )
        print("-" * 120)
        rows = catalog.inventory()
        if args.name and args.name != "all":
            rows = [
                r
                for r in rows
                if r["held"] == args.name or r["category"] == args.name or r["name"] == args.name
            ]
            if not rows:
                print(f"Error: unknown data list filter {args.name!r}", file=sys.stderr)
                return 1
        for r in rows:
            avail = "yes" if r["available"] else "-"
            cohorts = str(r["cohorts"]) if r["cohorts"] is not None else "-"
            print(
                f"{r['name']:<46} {r['held']:<8} {avail:<6} {cohorts:>8} "
                f"{r['category']:<14} {r['description']}"
            )
        return 0

    if args.action == "status":
        try:
            rows = catalog.status(args.name)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(
            f"{'Dataset':<46} {'Kind':<7} {'Present':<8} {'Size':>10} {'Cohorts':>8}  Description"
        )
        print("-" * 110)
        for r in rows:
            present = "yes" if r["present"] else "no"
            size = _fmt_bytes(r["size_bytes"])
            cohorts = str(r["cohorts"]) if r["cohorts"] is not None else "-"
            print(
                f"{r['name']:<46} {r['kind']:<7} {present:<8} {size:>10} {cohorts:>8}  "
                f"{r['description']}"
            )
        return 0

    if args.action == "contract":
        print(json.dumps(data_bundle.bundle_contract(), indent=2, sort_keys=True, default=str))
        return 0

    if args.action == "metadata":
        source = args.name or "oncoref"
        try:
            metadata = data_bundle.bundle_metadata(source)
        except (ValueError, data_bundle.BundleIntegrityError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(json.dumps(metadata, indent=2, sort_keys=True, default=str))
        return 0

    if args.action == "release-manifest":
        source = args.name or "oncoref"
        try:
            manifest = data_bundle.bundle_release_manifest(source)
        except (ValueError, data_bundle.BundleIntegrityError, OSError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        return 0

    if args.action == "dir":
        target = args.name or "all"
        dirs = {
            "bundle": data_bundle.cache_dir(),
            "hpa": reference_data.cache_dir(),
            "source": source_matrices.cache_dir(),
        }
        if target == "all":
            for name, path in dirs.items():
                print(f"{name}\t{path}")
            return 0
        if target not in dirs:
            print("Error: data dir target must be all, bundle, hpa, or source", file=sys.stderr)
            return 1
        print(dirs[target])
        return 0

    if args.action == "fetch":
        try:
            downloaded = catalog.fetch(args.name or "all", force=args.force)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:  # network / extract failure
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Downloaded: {', '.join(downloaded)}" if downloaded else "Already present.")
        return 0

    if args.action == "path":
        if not args.name:
            print("Error: 'data path' requires a dataset name", file=sys.stderr)
            return 1
        try:
            print(catalog.ensure(args.name))
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:  # network / extract failure
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    if args.action == "prune":
        target = args.name or "bundle"
        if target != "bundle":
            print("Error: data prune currently supports only the bundle cache", file=sys.stderr)
            return 1
        return _print_bundle_prune(args)

    return 1


def _cmd_cancer_type(args: argparse.Namespace) -> int:
    try:
        info = cancer_types.cancer_type_info(args.query)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if info is None:
        print("Error: no cancer type given", file=sys.stderr)
        return 1
    print(json.dumps(info, indent=2, default=str))
    return 0


def _cmd_tmb(args: argparse.Namespace) -> int:
    if args.code is None:
        for code, value in sorted(tmb.cancer_tmb().items()):
            print(f"{code}\t{value:g}")
        return 0
    try:
        value = tmb.cancer_tmb(args.code, inherit=not args.no_inherit)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if value is None:
        print(f"No TMB value for {args.code!r}", file=sys.stderr)
        return 1
    print(f"{value:g}")
    return 0


def _cmd_apd1(args: argparse.Namespace) -> int:
    if args.code is None:
        for code, value in sorted(apd1.cancer_apd1_response().items()):
            print(f"{code}\t{value:g}")
        return 0
    try:
        value = apd1.cancer_apd1_response(args.code, inherit=not args.no_inherit)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if value is None:
        print(f"No anti-PD-1 ORR for {args.code!r}", file=sys.stderr)
        return 1
    print(f"{value:g}")
    return 0


def _cmd_ici(args: argparse.Namespace) -> int:
    regimen = args.regimen
    if args.code is None:
        mapping = ici.cancer_ici_response(regimen=regimen)
        for code, value in sorted(mapping.items()):
            print(f"{code}\t{value:g}")
        return 0
    try:
        if args.all_regimens:
            per = ici.cancer_ici_response(args.code, fallback=False, inherit=not args.no_inherit)
            if not per:
                print(
                    _no_ici_orr_message(args.code, inherit=not args.no_inherit),
                    file=sys.stderr,
                )
                return 1
            for reg in ici.ici_regimens():
                if reg in per:
                    print(f"{reg}\t{per[reg]:g}")
            return 0
        value = ici.cancer_ici_response(args.code, regimen=regimen, inherit=not args.no_inherit)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if value is None:
        print(_no_ici_orr_message(args.code, inherit=not args.no_inherit), file=sys.stderr)
        return 1
    print(f"{value:g}")
    return 0


def _no_ici_orr_message(code: str, *, inherit: bool) -> str:
    """Explain *why* there is no ORR when the code is a reviewed gap.

    An audited gap and an uncurated code both resolve to no value; only the gap can
    say why, and the reason is the whole point of curating it."""
    resolved = ici.resolve_ici_response_source(code, inherit=inherit)
    if resolved.get("inheritance_kind") != "direct_missing":
        return f"No ICI ORR for {code!r}"
    reason = resolved.get("missing_reason") or "no_curated_representative_orr"
    message = f"No representative ICI ORR for {code!r}: {reason}"
    note = resolved.get("notes")
    return f"{message}\n{note}" if note else message


def _cmd_cta(args: argparse.Namespace) -> int:
    if args.unfiltered:
        genes = cta.cta_unfiltered_gene_ids() if args.ids else cta.cta_unfiltered_gene_names()
    else:
        genes = cta.cta_gene_ids() if args.ids else cta.cta_gene_names()
    if args.count:
        print(len(genes))
    else:
        for g in sorted(genes):
            print(g)
    return 0


def _cmd_proteoforms(args: argparse.Namespace) -> int:
    if args.gene:
        symbol = proteoforms.proteoform_for_gene(args.gene)
        if symbol is None:
            print(f"{args.gene} is not in any proteoform group", file=sys.stderr)
            return 1
        members = proteoforms.proteoform_members_for_gene(args.gene) or ()
        print(f"{symbol}\t{', '.join(members)}")
        return 0
    symbol_map = proteoforms.proteoform_symbol_map()
    if args.count:
        print(len(symbol_map))
        return 0
    for label in sorted(symbol_map):
        print(f"{label}\t{', '.join(symbol_map[label])}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from . import plots

    fns = {
        "apd1-vs-tmb": plots.apd1_vs_tmb,
        "apd1-orr-bars": plots.apd1_orr_bars,
        "incidence-vs-mortality": plots.incidence_vs_mortality,
        "cta-expression-heatmap": plots.cta_expression_heatmap,
        "cta-addressable-burden": plots.cta_addressable_burden,
        "cta-patient-heatmap": plots.cta_patient_count_heatmap,
        "cta-coverage-curves": plots.cta_coverage_curves,
        "cta-coverage-stacked": plots.cta_coverage_stacked_bars,
        "cta-burden-vs-response": plots.cta_burden_vs_response,
        "cta-specific-9mer-load": plots.cta_specific_9mer_load,
        "burden-category-bars": plots.burden_category_bars,
        "apd1-response-signature": plots.apd1_response_signature_scatter,
    }
    # Only pass threshold_tpm when the user explicitly set it, so each plot keeps its own
    # default (50 TPM for the TPM-based plots, within-sample p95 for the patient heatmap)
    # rather than being forced onto a single CLI-wide value.
    tpm = {} if args.threshold_tpm is None else {"threshold_tpm": args.threshold_tpm}
    try:
        if args.which == "patient-coverage":
            from . import coverage

            threshold = (
                args.threshold
                if args.threshold is not None
                else (args.threshold_tpm if args.threshold_tpm is not None else 25)
            )
            result = coverage.render_patient_coverage(
                args.gene_set,
                cohorts=[c.strip() for c in args.codes.split(",")] if args.codes else None,
                threshold=threshold,
                out_dir=args.out,
            )
            if result["n_cohorts"] == 0:
                print(
                    f"Error: no cohorts with cached per-sample data and coverage for "
                    f"gene set {result['label']!r}",
                    file=sys.stderr,
                )
                return 1
            print(f"{result['label']}: {result['n_cohorts']} cohorts (> {threshold:g} TPM)")
            for kind, path in result["paths"].items():
                print(f"  {kind}: {path}")
            return 0
        if args.which == "cta-curation":
            from . import cta_curation_plots

            result = cta_curation_plots.render(out_dir=args.out)
            print(f"CTA curation figures ({result['n_genes']} evidence rows):")
            for kind, path in result["paths"].items():
                print(f"  {kind}: {path}")
            return 0
        if args.which in ("incidence-vs-mortality", "burden-category-bars"):
            kwargs = {"region": args.region}
        elif args.which == "apd1-response-signature":
            kwargs = {"signature": args.signature}
        elif args.which == "cta-expression-heatmap":
            kwargs = {"stat": args.stat}
        elif args.which == "cta-addressable-burden":
            kwargs = {"source": args.source, **tpm}
        elif args.which == "cta-patient-heatmap":
            kwargs = {**tpm}
        elif args.which in ("cta-burden-vs-response", "cta-specific-9mer-load"):
            kwargs = {"against": args.against, **tpm}
        elif args.which in ("cta-coverage-curves", "cta-coverage-stacked"):
            if not args.codes:
                print(f"Error: {args.which} needs --codes", file=sys.stderr)
                return 1
            kwargs = {
                "cancer_types": [c.strip() for c in args.codes.split(",")],
                **tpm,
            }
        else:
            kwargs = {}
        fns[args.which](save=args.out, **kwargs)
    except (ValueError, ModuleNotFoundError, NotImplementedError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {args.out}")
    return 0


def _cmd_expression_sources(args: argparse.Namespace) -> int:
    srcs = expression_registry.expression_sources()
    if args.code:
        srcs = [s for s in srcs if args.code in s.cancer_codes]
    if args.type:
        srcs = [s for s in srcs if s.source_type == args.type]
    if not srcs:
        print("No matching expression sources.", file=sys.stderr)
        return 1
    print(f"{'Source':<26} {'Type':<20} {'Unit':<7} {'Codes':<6} Cancer codes")
    print("-" * 100)
    for s in sorted(srcs, key=lambda x: (x.source_type, x.id)):
        codes = ";".join(s.cancer_codes)
        codes = codes if len(codes) <= 40 else codes[:37] + "..."
        print(
            f"{s.id:<26} {s.source_type:<20} {(s.unit or ''):<7} {len(s.cancer_codes):<6} {codes}"
        )
    print(f"\n{len(srcs)} sources. Registry: oncoref/data/expression_sources.yaml")
    return 0


def _cmd_samples(args: argparse.Namespace) -> int:
    if args.counts or (not args.code and not args.cohort):
        counts = samples.sample_counts_by_cancer_code(included_only=not args.all)
        for code, n in counts.items():
            print(f"{code}\t{n}")
        print(f"\n{int(counts.sum())} samples across {len(counts)} cancer codes", file=sys.stderr)
        return 0
    if args.code:
        df = samples.samples_for_cancer_code(args.code, included_only=not args.all)
    else:
        df = samples.samples_for_cohort(args.cohort, included_only=not args.all)
    cols = [c for c in ("sample_id", "source_cohort", "sample_type", "raw_unit") if c in df.columns]
    if df.empty:
        print("No matching samples.", file=sys.stderr)
        return 1
    print(df[cols].to_string(index=False))
    return 0


def _cmd_burden(args: argparse.Namespace) -> int:
    try:
        if args.category is None:
            for cat, pct in sorted(incidence.cancer_burden(metric=args.metric).items()):
                print(f"{cat}\t{pct:g}")
            return 0
        value = incidence.cancer_burden(args.category, metric=args.metric)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if value is None:
        print(f"No burden value for category {args.category!r}", file=sys.stderr)
        return 1
    print(f"{value:g}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oncoref",
        description="Curated cancer reference data: ontology, TMB, incidence/mortality, expression.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the installed oncoref version").set_defaults(
        func=_cmd_version
    )

    # --- reference lookups ---
    p_ct = sub.add_parser(
        "cancer-type", help="Resolve a cancer type/alias/name and print its registry info"
    )
    p_ct.add_argument("query", help="Cancer code, alias, or display name (e.g. PRAD, prostate)")
    p_ct.set_defaults(func=_cmd_cancer_type)

    p_tmb = sub.add_parser("tmb", help="Median TMB (mut/Mb) for a code, or the full map")
    p_tmb.add_argument("code", nargs="?", default=None, help="Cancer code/alias (omit for all)")
    p_tmb.add_argument("--no-inherit", action="store_true", help="Do not inherit an ancestor's TMB")
    p_tmb.set_defaults(func=_cmd_tmb)

    p_apd1 = sub.add_parser(
        "apd1", help="Anti-PD-1 monotherapy ORR (%%) for a code, or the full map"
    )
    p_apd1.add_argument("code", nargs="?", default=None, help="Cancer code/alias (omit for all)")
    p_apd1.add_argument(
        "--no-inherit", action="store_true", help="Do not inherit an ancestor's ORR"
    )
    p_apd1.set_defaults(func=_cmd_apd1)

    p_ici = sub.add_parser(
        "ici", help="ICI ORR (%%) for a code by regimen (anti-PD-1/PD-L1/combo), or the full map"
    )
    p_ici.add_argument("code", nargs="?", default=None, help="Cancer code/alias (omit for all)")
    p_ici.add_argument(
        "--regimen",
        default=None,
        choices=list(ici.REGIMEN_FALLBACK),
        help="Pin a regimen (default: PD-1 → PD-L1 → PD-1+CTLA-4 fallback)",
    )
    p_ici.add_argument(
        "--all-regimens", action="store_true", help="Show every regimen present for the code"
    )
    p_ici.add_argument("--no-inherit", action="store_true", help="Do not inherit an ancestor's ORR")
    p_ici.set_defaults(func=_cmd_ici)

    p_cta = sub.add_parser("cta", help="List cancer-testis antigens (expressed set by default)")
    p_cta.add_argument("--unfiltered", action="store_true", help="Full candidate universe")
    p_cta.add_argument("--ids", action="store_true", help="Ensembl gene IDs instead of symbols")
    p_cta.add_argument("--count", action="store_true", help="Print the count only")
    p_cta.set_defaults(func=_cmd_cta)

    p_proteoforms = sub.add_parser(
        "proteoforms",
        help="List identical-protein CGA groups whose TPM sums to proteoform level",
    )
    p_proteoforms.add_argument("--gene", help="Show the group for one gene (Ensembl ID or symbol)")
    p_proteoforms.add_argument("--count", action="store_true", help="Print the group count only")
    p_proteoforms.set_defaults(func=_cmd_proteoforms)

    p_plot = sub.add_parser("plot", help="Render a cancer-type reference plot to a PNG")
    p_plot.add_argument(
        "which",
        choices=[
            "apd1-vs-tmb",
            "apd1-orr-bars",
            "incidence-vs-mortality",
            "cta-expression-heatmap",
            "cta-addressable-burden",
            "cta-patient-heatmap",
            "cta-coverage-curves",
            "cta-coverage-stacked",
            "cta-burden-vs-response",
            "cta-specific-9mer-load",
            "burden-category-bars",
            "apd1-response-signature",
            "patient-coverage",
            "cta-curation",
        ],
        help="Which plot to render",
    )
    p_plot.add_argument(
        "--out",
        required=True,
        help="Output PNG path, or output directory for patient-coverage/cta-curation",
    )
    p_plot.add_argument(
        "--signature",
        default="t_cell_inflamed",
        help="Response signature for apd1-response-signature (e.g. t_cell_inflamed, tgfb_exclusion)",
    )
    p_plot.add_argument(
        "--region", default="us", choices=["us", "world"], help="Region for incidence-vs-mortality"
    )
    p_plot.add_argument(
        "--stat",
        default="median",
        choices=["q1", "median", "q3"],
        help="Statistic for cta-expression-heatmap",
    )
    p_plot.add_argument(
        "--source",
        default="within_sample",
        choices=["within_sample", "per_sample"],
        help="Prevalence basis for cta-addressable-burden (per_sample needs matrices)",
    )
    p_plot.add_argument(
        "--against",
        default="apd1",
        choices=["apd1", "tmb"],
        help="Response metric for cta-burden-vs-response",
    )
    p_plot.add_argument(
        "--threshold-tpm",
        type=float,
        default=None,
        help=(
            "Clean-TPM 'expressed' cut for the per-sample CTA plots. Omit to use each "
            "plot's default (50 TPM, or within-sample p95 for the patient heatmap)."
        ),
    )
    p_plot.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Patient-coverage TPM cutoff (alias-style option matching pirlygenes).",
    )
    p_plot.add_argument(
        "--gene-set",
        default="cta",
        help=(
            "Gene set for patient-coverage: cta, mito, housekeeping, family:<name>, "
            "therapy:<agent_class>, lineage:<code>, or a CSV/TSV/TXT path"
        ),
    )
    p_plot.add_argument(
        "--codes",
        default=None,
        help="Comma-separated cancer codes (for cta coverage plots or patient-coverage)",
    )
    p_plot.set_defaults(func=_cmd_plot)

    p_burden = sub.add_parser(
        "burden", help="Incidence/mortality share for a burden category, or the full map"
    )
    p_burden.add_argument(
        "category", nargs="?", default=None, help="Burden category (omit for all)"
    )
    p_burden.add_argument(
        "--metric",
        default="us_incidence_pct",
        choices=list(incidence._BURDEN_METRICS),
        help="Which share to report (default: us_incidence_pct)",
    )
    p_burden.set_defaults(func=_cmd_burden)

    p_data = sub.add_parser(
        "data",
        help="Unified data management over every managed dataset (expression bundle + HPA)",
    )
    p_data.add_argument(
        "action",
        choices=[
            "list",
            "status",
            "contract",
            "metadata",
            "release-manifest",
            "dir",
            "fetch",
            "path",
            "prune",
        ],
        help=(
            "list (catalog), status (cache state), contract (bundle contract JSON), "
            "metadata (contract + local/release state JSON), release-manifest "
            "(release metadata JSON), dir (cache roots), fetch (download), path "
            "(ensure + print), prune (bundle cache cleanup)"
        ),
    )
    p_data.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Dataset name; or a fetch target: all / hpa / bundle / source / "
        "per-sample:<CODE> (omit = all)",
    )
    p_data.add_argument("--force", action="store_true", help="Re-download even if cached")
    p_data.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete stale bundle caches for 'data prune' (default: dry run)",
    )
    p_data.add_argument(
        "--include-current",
        action="store_true",
        help="For 'data prune', also delete the current bundle cache",
    )
    p_data.set_defaults(func=_cmd_data)

    p_exprsrc = sub.add_parser(
        "expression-sources",
        help="List the cohort expression sources (Treehouse, GDC, GEO, recount3, …)",
    )
    p_exprsrc.add_argument("--code", default=None, help="Only sources feeding this cancer code")
    p_exprsrc.add_argument("--type", default=None, help="Only this source_type (e.g. gdc, geo)")
    p_exprsrc.set_defaults(func=_cmd_expression_sources)

    p_samples = sub.add_parser(
        "samples", help="Per-sample curation manifest (counts, or rows for a code/cohort)"
    )
    p_samples.add_argument("--code", default=None, help="Samples assigned to this cancer code")
    p_samples.add_argument("--cohort", default=None, help="Samples from this source cohort")
    p_samples.add_argument("--counts", action="store_true", help="Per-cancer-code sample counts")
    p_samples.add_argument(
        "--all", action="store_true", help="Include excluded samples (default: included only)"
    )
    p_samples.set_defaults(func=_cmd_samples)

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
