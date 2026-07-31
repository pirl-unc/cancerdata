# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from oncoref import cli, data_bundle, plots


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_apd1_vs_tmb_renders(tmp_path):
    out = tmp_path / "apd1_vs_tmb.png"
    fig = plots.apd1_vs_tmb(save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_apd1_vs_tmb_strict_pd1_filters_proxy_targets(tmp_path):
    fig = plots.apd1_vs_tmb(strict_pd1=True, save=str(tmp_path / "strict.png"))
    labels = {t.get_text() for t in fig.axes[0].texts}
    proxy_codes = set(
        plots.cancer_apd1_response_df().query("drug_target != 'PD-1'")["cancer_code"].astype(str)
    )
    assert labels.isdisjoint(proxy_codes)


def test_apd1_orr_bars_renders(tmp_path):
    out = tmp_path / "bars.png"
    plots.apd1_orr_bars(save=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_incidence_vs_mortality_renders(tmp_path):
    out = tmp_path / "inc.png"
    plots.incidence_vs_mortality(region="world", save=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_ici_orr_pooled_forest_renders(tmp_path):
    out = tmp_path / "forest.png"
    fig = plots.ici_orr_pooled_forest(save=str(out))
    assert out.exists() and out.stat().st_size > 0
    # one row per cancer type at its fallback regimen (every covered cancer)
    assert len(fig.axes[0].get_yticklabels()) > 50
    # pinning a regimen also works
    plots.ici_orr_pooled_forest(regimen="PD-1", save=str(tmp_path / "f2.png"))


def test_incidence_bad_region():
    with pytest.raises(ValueError, match="region must be"):
        plots.incidence_vs_mortality(region="moon")


def test_cli_plot(tmp_path):
    out = tmp_path / "cli.png"
    assert cli.main(["plot", "apd1-vs-tmb", "--out", str(out)]) == 0
    assert out.exists()


def test_cli_plot_burden_category_bars(tmp_path):
    out = tmp_path / "cats.png"
    assert cli.main(["plot", "burden-category-bars", "--region", "world", "--out", str(out)]) == 0
    assert out.exists()


def test_patient_coverage_renderer_writes_pirlygenes_style_artifacts(tmp_path, monkeypatch):
    from oncoref import coverage

    fixture = pd.DataFrame(
        {
            "Ensembl_Gene_ID": ["ENSG_A", "ENSG_B", "ENSG_C"],
            "Symbol": ["GA", "GB", "GC"],
            "p0": [10.0, 0.0, 0.0],
            "p1": [10.0, 0.0, 10.0],
            "p2": [0.0, 10.0, 0.0],
            "p3": [0.0, 0.0, 0.0],
        }
    )
    monkeypatch.setattr(coverage, "per_sample_expression", lambda *a, **k: fixture.copy())
    panel = tmp_path / "panel.csv"
    panel.write_text("Ensembl_Gene_ID\nENSG_A\nENSG_B\nENSG_C\n")

    result = coverage.render_patient_coverage(
        str(panel),
        cohorts=["LUAD"],
        threshold=5,
        thresholds=(5,),
        out_dir=tmp_path / "out",
        proteoform=False,
    )

    assert result["n_cohorts"] == 1
    for path in result["paths"].values():
        assert Path(path).exists()
    assert (tmp_path / "out" / "panel_patient_counts.csv").exists()
    assert (tmp_path / "out" / "panel_stacked_coverage_t5.png").exists()
    assert (tmp_path / "out" / "panel_coverage_curves_t5.png").exists()


def test_cli_plot_patient_coverage_delegates(monkeypatch, tmp_path, capsys):
    from oncoref import coverage

    captured = {}

    def fake(gene_set, **kwargs):
        captured["gene_set"] = gene_set
        captured.update(kwargs)
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        counts = out / "counts.csv"
        counts.write_text("ok\n")
        return {"paths": {"counts_csv": str(counts)}, "label": "CTA", "n_cohorts": 2}

    monkeypatch.setattr(coverage, "render_patient_coverage", fake)
    rc = cli.main(
        [
            "plot",
            "patient-coverage",
            "--gene-set",
            "cta",
            "--codes",
            "LUAD,SKCM",
            "--threshold",
            "25",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert captured["gene_set"] == "cta"
    assert captured["cohorts"] == ["LUAD", "SKCM"]
    assert captured["threshold"] == 25
    assert "counts_csv" in capsys.readouterr().out


def test_cta_curation_source_counts_partition():
    from oncoref import cta_curation_plots as ccp

    rows = ccp._per_source_counts(ccp._evidence())
    assert rows
    for row in rows:
        assert row["kept_confident"] + row["kept_weak"] + row["excluded"] == row["total"]
        assert row["total"] > 0
    totals = [row["total"] for row in rows]
    assert totals == sorted(totals, reverse=True)


def test_cta_curation_tag_sets_cover_primary_sources():
    from oncoref import cta_curation_plots as ccp

    sets = ccp._tag_sets(ccp._evidence())
    assert set(sets) == set(ccp.PRIMARY_SOURCES)
    assert sets["CTpedia"]
    assert sets["CTexploreR"]
    assert sets["daSilva2017_protein"]


def test_cta_curation_renderer_writes_five_figures(tmp_path):
    from oncoref import cta_curation_plots as ccp

    result = ccp.render(out_dir=tmp_path)
    assert set(result["paths"]) == set(ccp.FILENAMES)
    assert result["n_genes"] > 0
    for path in result["paths"].values():
        assert path.exists() and path.stat().st_size > 0


def test_cli_plot_cta_curation_delegates(monkeypatch, tmp_path, capsys):
    from oncoref import cta_curation_plots

    captured = {}

    def fake(*, out_dir):
        captured["out_dir"] = out_dir
        out = tmp_path / "cur"
        out.mkdir(exist_ok=True)
        fig = out / "cta-source-venn.png"
        fig.write_text("ok\n")
        return {"n_genes": 3, "paths": {"source_venn": fig}}

    monkeypatch.setattr(cta_curation_plots, "render", fake)
    rc = cli.main(["plot", "cta-curation", "--out", str(tmp_path / "cur")])
    assert rc == 0
    assert captured["out_dir"] == str(tmp_path / "cur")
    assert "source_venn" in capsys.readouterr().out


def test_cli_plot_coverage_stacked_needs_codes(capsys):
    # The coverage plots require --codes; missing it is a clean error, not a crash.
    assert cli.main(["plot", "cta-coverage-stacked", "--out", "x.png"]) == 1


def test_cli_plot_threshold_tpm_is_opt_in(monkeypatch, tmp_path):
    # Without --threshold-tpm the CLI must NOT inject a value, so the patient heatmap keeps
    # its within-sample p95 default (it never silently reverts to a flat TPM cut); when the
    # flag is given it is forwarded.
    captured = {}

    def fake(*, save, **kwargs):
        captured.clear()
        captured.update(kwargs)
        open(save, "wb").close()

    monkeypatch.setattr(plots, "cta_patient_count_heatmap", fake)
    assert cli.main(["plot", "cta-patient-heatmap", "--out", str(tmp_path / "a.png")]) == 0
    assert "threshold_tpm" not in captured  # -> p95 default preserved
    assert (
        cli.main(
            [
                "plot",
                "cta-patient-heatmap",
                "--out",
                str(tmp_path / "b.png"),
                "--threshold-tpm",
                "25",
            ]
        )
        == 0
    )
    assert captured["threshold_tpm"] == 25.0


# ---- CTA expression heatmap (needs the expression bundle / percentile data) ----


def test_cta_expression_heatmap_renders(tmp_path):
    if not data_bundle.item_is_local("cancer-reference-expression-percentiles"):
        pytest.skip("percentile artifacts not present locally")

    out = tmp_path / "cta.png"
    cohorts = __import__("oncoref").available_percentile_cohorts()[:6]
    fig = plots.cta_expression_heatmap(
        cohorts=cohorts, n_cohorts=4, n_ctas=8, proteoform=False, save=str(out)
    )
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_expression_heatmap_bad_stat():
    with pytest.raises(ValueError, match="stat must be one of"):
        plots.cta_expression_heatmap(stat="mean", cohorts=["X"])


def test_cta_expression_heatmap_proteoform_reads_collapsed_vectors(tmp_path, monkeypatch):
    # proteoform=True reads the proteoform-summed percentile vectors and labels CTA
    # columns by the proteoform symbol (NY-ESO-1). Stub the per-cohort vector.
    import pandas as pd

    from oncoref import cta
    from oncoref.proteoforms import gene_to_proteoform_id

    # Use two real CTAs and their real proteoform keys, so the heatmap's CTA-key filter
    # (derived from the same CTA set) keeps both rows. At least one is a collapsed group.
    grouped = next(g for g in sorted(cta.cta_gene_ids()) if "/" in gene_to_proteoform_id([g])[g])
    singleton = next(
        g for g in sorted(cta.cta_gene_ids()) if "/" not in gene_to_proteoform_id([g])[g]
    )
    keys = gene_to_proteoform_id([grouped, singleton])

    def fake_pct(code, *, as_tpm=True, proteoform=False):
        assert proteoform  # the heatmap must request the collapsed variant
        return pd.DataFrame(
            {
                "proteoform_key": [keys[grouped], keys[singleton]],
                "Ensembl_Gene_ID": [grouped, singleton],
                "Symbol": [keys[grouped], "OTHER_CTA"],
                "proteoform_members": ["CTAG1A/CTAG1B", "OTHER_CTA"],
                "p25": [5.0, 1.0],
                "p50": [40.0, 2.0],
                "p75": [80.0, 3.0],
            }
        )

    monkeypatch.setattr(plots, "cohort_gene_percentiles", fake_pct)
    monkeypatch.setattr(plots, "available_percentile_cohorts", lambda *, proteoform=False: ["LUAD"])
    out = tmp_path / "cta_pf.png"
    fig = plots.cta_expression_heatmap(proteoform=True, n_cohorts=2, n_ctas=4, save=str(out))
    assert out.exists() and fig is not None


def test_cta_expression_heatmap_proteoform_missing_bundle():
    with pytest.raises(ValueError, match="proteoform-summed percentile vector"):
        plots.cta_expression_heatmap(proteoform=True, cohorts=[])


def test_cta_expression_heatmap_no_cohorts():
    with pytest.raises(ValueError, match="no cohorts"):
        plots.cta_expression_heatmap(cohorts=[])


def test_cta_expression_heatmap_skips_unusable_cohorts():
    # A code with no percentile vector is skipped with a warning, not a crash; with
    # only unusable codes the matrix is empty -> clean ValueError.
    with (
        pytest.warns(UserWarning, match="without a percentile vector"),
        pytest.raises(ValueError, match="no CTA expression data"),
    ):
        plots.cta_expression_heatmap(cohorts=["NOT_A_REAL_COHORT"])


# ---- CTA addressable burden (P2) + 9mer scaffold (P3) ----


def test_cta_addressable_burden_renders(tmp_path, monkeypatch):
    # Hermetic: stub the within-sample prevalence so the test doesn't need the
    # bundle. Real codes map to a burden category + incidence, so bars render.
    monkeypatch.setattr(
        plots,
        "_cta_prevalence_by_cohort",
        lambda threshold: {"LUAD": 0.6, "SKCM": 0.4, "BRCA": 0.2},
    )
    out = tmp_path / "burden.png"
    fig = plots.cta_addressable_burden(n=10, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_addressable_burden_labels_mortality_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(
        plots,
        "_cta_prevalence_by_cohort",
        lambda threshold: {"LUAD": 0.6, "SKCM": 0.4, "BRCA": 0.2},
    )
    out = tmp_path / "burden_mortality.png"
    fig = plots.cta_addressable_burden(metric="world_mortality_pct", n=10, save=str(out))
    assert out.exists() and fig is not None
    ax = fig.axes[0]
    assert "WORLD mortality share" in ax.get_xlabel()
    assert "WORLD mortality share" in ax.get_title()


def test_cta_addressable_burden_no_within_sample(monkeypatch):
    monkeypatch.setattr(plots, "_cta_prevalence_by_cohort", lambda threshold: {})
    with pytest.raises(ValueError, match="no CTA prevalence available"):
        plots.cta_addressable_burden()


def test_cta_addressable_burden_no_mapped_cohorts(monkeypatch):
    # A code that resolves to no burden category yields no bars -> clean ValueError.
    monkeypatch.setattr(plots, "_cta_prevalence_by_cohort", lambda threshold: {"NOT_A_CODE": 0.5})
    with pytest.raises(ValueError, match="no cohort mapped"):
        plots.cta_addressable_burden()


def test_cta_specific_9mer_load_renders(tmp_path, monkeypatch):
    from oncoref import peptides

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM", "MM"])
    monkeypatch.setattr(plots, "cancer_tmb", lambda: {"LUAD": 6.9, "SKCM": 13.0, "MM": 1.5})
    monkeypatch.setattr(
        peptides,
        "cta_specific_9mer_load",
        lambda code, **k: {"LUAD": 120.0, "SKCM": 540.0, "MM": 60.0}[code],
    )
    out = tmp_path / "9mer.png"
    fig = plots.cta_specific_9mer_load(against="tmb", save=str(out))
    assert out.exists() and fig is not None


def test_cta_specific_9mer_load_collapses_crc_msi_source_scope(tmp_path, monkeypatch):
    from oncoref import peptides, source_matrices

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["COAD_MSI", "READ_MSI"])
    monkeypatch.setattr(plots, "cancer_tmb", lambda: {"CRC_MSI": 46.0})
    monkeypatch.setattr(
        peptides,
        "cta_specific_9mer_load",
        lambda code, **k: {"COAD_MSI": 10.0, "READ_MSI": 30.0}[code],
    )
    monkeypatch.setattr(
        source_matrices,
        "cohort_info",
        lambda code: {"n_samples": {"COAD_MSI": 3, "READ_MSI": 1}[code]},
    )

    out = tmp_path / "crc_msi_9mer.png"
    fig = plots.cta_specific_9mer_load(against="tmb", save=str(out))
    offsets = [tuple(coll.get_offsets()[0]) for coll in fig.axes[0].collections]
    assert len(offsets) == 1
    assert offsets[0][0] == pytest.approx(15.0)
    assert offsets[0][1] == pytest.approx(46.0)
    assert [t.get_text() for t in fig.axes[0].texts] == ["CRC_MSI"]


def test_cta_specific_9mer_load_bad_against():
    with pytest.raises(ValueError, match="against must be"):
        plots.cta_specific_9mer_load(against="nonsense")


def test_cta_specific_9mer_load_no_data(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    with pytest.raises(ValueError, match="no cohort with both"):
        plots.cta_specific_9mer_load()


# ---- per-patient plots (consume the per-sample matrices via coverage.py) ----


def test_cta_addressable_burden_per_sample_source(tmp_path, monkeypatch):
    # source="per_sample" pulls the faithful union from coverage; stub it hermetically.
    from oncoref import coverage

    monkeypatch.setattr(
        coverage,
        "addressable_fraction_by_cohort",
        lambda **k: __import__("pandas").Series({"LUAD": 0.95, "SKCM": 0.99, "BRCA": 0.5}),
    )
    out = tmp_path / "burden_faithful.png"
    fig = plots.cta_addressable_burden(source="per_sample", n=10, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_addressable_burden_bad_source():
    with pytest.raises(ValueError, match="source must be"):
        plots.cta_addressable_burden(source="nonsense")


def test_cta_patient_count_heatmap_renders(tmp_path, monkeypatch):
    import pandas as pd

    def fake_ws(code, *, threshold, proteoform, scope):
        # within-sample p95 prevalence: two cohorts, three CTA proteoforms
        base = {"LUAD": [0.7, 0.3, 0.1], "SKCM": [0.9, 0.2, 0.5]}[code]
        return pd.DataFrame(
            {
                "proteoform_key": ["E1", "E2", "E3"],
                "Symbol": ["GA", "GB", "GC"],
                "frac_samples_top5pct": base,
                "n_samples": [100, 100, 100],
            }
        )

    from oncoref import proteoforms

    monkeypatch.setattr(
        plots,
        "locally_available_within_sample_cohorts",
        lambda **kwargs: ["LUAD", "SKCM"],
    )
    monkeypatch.setattr(plots, "within_sample_top_fraction", fake_ws)
    monkeypatch.setattr(plots, "cta_gene_ids", lambda: ["E1", "E2", "E3"])
    monkeypatch.setattr(proteoforms, "gene_to_proteoform_id", lambda ids: {i: i for i in ids})
    out = tmp_path / "patient_heatmap.png"
    fig = plots.cta_patient_count_heatmap(n_ctas=3, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_patient_count_heatmap_no_cohorts(monkeypatch):
    monkeypatch.setattr(plots, "locally_available_within_sample_cohorts", lambda **kwargs: [])
    with pytest.raises(ValueError, match="no local proteoform within-sample"):
        plots.cta_patient_count_heatmap()


def test_cta_patient_count_heatmap_absolute_mode_uses_only_cached_matrices(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    monkeypatch.setattr(
        plots,
        "locally_available_within_sample_cohorts",
        lambda **kwargs: pytest.fail("absolute mode must not plan from summary shards"),
    )
    with pytest.raises(ValueError, match="no cached per-sample matrices"):
        plots.cta_patient_count_heatmap(threshold_tpm=50)


def test_cta_patient_count_heatmap_duplicate_symbols(tmp_path, monkeypatch):
    # Distinct proteoforms that happen to share a display Symbol must not crash the
    # cohort×CTA frame alignment — keying is on the unique proteoform_key.
    import pandas as pd

    def fake_ws(code, *, threshold, proteoform, scope):
        return pd.DataFrame(
            {
                "proteoform_key": ["K1", "K2", "K3"],  # unique keys
                "Symbol": ["GA", "GA", "GB"],  # GA duplicated as a display label
                "frac_samples_top5pct": [0.7, 0.3, 0.1],
                "n_samples": [100, 100, 100],
            }
        )

    from oncoref import proteoforms

    monkeypatch.setattr(
        plots,
        "locally_available_within_sample_cohorts",
        lambda **kwargs: ["LUAD", "SKCM"],
    )
    monkeypatch.setattr(plots, "within_sample_top_fraction", fake_ws)
    monkeypatch.setattr(plots, "cta_gene_ids", lambda: ["K1", "K2", "K3"])
    monkeypatch.setattr(proteoforms, "gene_to_proteoform_id", lambda ids: {i: i for i in ids})
    out = tmp_path / "dup.png"
    fig = plots.cta_patient_count_heatmap(save=str(out))  # must not raise
    assert out.exists() and fig is not None


def test_cta_coverage_curves_renders(tmp_path, monkeypatch):
    import pandas as pd

    from oncoref import coverage

    def fake_greedy(code, *, threshold_tpm, max_genes):
        return pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "Ensembl_Gene_ID": ["E1", "E2", "E3"],
                "Symbol": ["GA", "GB", "GC"],
                "marginal_patients": [60, 20, 10],
                "marginal_fraction": [0.6, 0.2, 0.1],
                "cumulative_patients": [60, 80, 90],
                "cumulative_fraction": [0.6, 0.8, 0.9],
            }
        )

    monkeypatch.setattr(coverage, "greedy_coverage", fake_greedy)
    out = tmp_path / "curves.png"
    fig = plots.cta_coverage_curves(["LUAD", "SKCM"], save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_coverage_curves_empty_raises(monkeypatch):
    import pandas as pd

    from oncoref import coverage

    monkeypatch.setattr(coverage, "greedy_coverage", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="no coverage curve"):
        plots.cta_coverage_curves(["LUAD"])


def _percentile_coverage_table():
    rows = []
    for code, final in (("LUAD", 0.8), ("SKCM", 1.0)):
        rows.extend(
            [
                {
                    "cancer_code": code,
                    "within_sample_percentile": 0.95,
                    "n_patients": 10,
                    "rank": 1,
                    "Ensembl_Gene_ID": "E1",
                    "Symbol": "MAGEA4",
                    "marginal_patients": 6,
                    "marginal_fraction": 0.6,
                    "cumulative_patients": 6,
                    "cumulative_fraction": 0.6,
                    "proteoform_key": "E1",
                    "proteoform_members": "MAGEA4",
                },
                {
                    "cancer_code": code,
                    "within_sample_percentile": 0.95,
                    "n_patients": 10,
                    "rank": 2,
                    "Ensembl_Gene_ID": "E2",
                    "Symbol": "CTAG1B",
                    "marginal_patients": round((final - 0.6) * 10),
                    "marginal_fraction": final - 0.6,
                    "cumulative_patients": round(final * 10),
                    "cumulative_fraction": final,
                    "proteoform_key": "E2",
                    "proteoform_members": "CTAG1B",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_cta_within_sample_percentile_coverage_plots_render(tmp_path):
    coverage_table = _percentile_coverage_table()
    curves = tmp_path / "percentile-curves.png"
    stacked = tmp_path / "percentile-stacked.png"

    curve_figure = plots.cta_within_sample_percentile_coverage_curves(
        ["lung adenocarcinoma", "SKCM"], coverage_table=coverage_table, save=str(curves)
    )
    stacked_figure = plots.cta_within_sample_percentile_coverage_stacked_bars(
        ["LUAD", "SKCM"], coverage_table=coverage_table, save=str(stacked)
    )

    assert curves.stat().st_size > 0 and stacked.stat().st_size > 0
    assert "within-sample p95" in curve_figure.axes[0].get_ylabel()
    assert "within-sample p95" in stacked_figure.axes[0].get_xlabel()


def test_cta_within_sample_percentile_addressable_burden_uses_joint_union(tmp_path, monkeypatch):
    from oncoref import coverage

    supplied = _percentile_coverage_table()
    captured = {}

    def fake_fraction(cohorts, **kwargs):
        captured.update(kwargs)
        return pd.Series({"LUAD": 0.8, "SKCM": 1.0})

    monkeypatch.setattr(
        coverage, "within_sample_percentile_addressable_fraction_by_cohort", fake_fraction
    )
    out = tmp_path / "percentile-burden.png"
    figure = plots.cta_within_sample_percentile_addressable_burden(
        cohorts=["LUAD", "SKCM"], coverage_table=supplied, save=str(out)
    )

    assert captured["coverage"] is supplied
    assert captured["percentile"] == 0.95
    assert out.stat().st_size > 0 and figure is not None


def test_antigen_family_grouping():
    assert plots._antigen_family("MAGEA4") == "MAGE-A"
    assert plots._antigen_family("MAGEB2") == "MAGE-B"
    assert plots._antigen_family("CTAG1B") == "CTAG/NY-ESO"
    assert plots._antigen_family("LAGE1") == "CTAG/NY-ESO"  # same family as CTAG
    assert plots._antigen_family("SSX2") == "SSX"
    assert plots._antigen_family("CT83") == "other"


def test_cta_coverage_stacked_bars_renders(tmp_path, monkeypatch):
    import pandas as pd

    from oncoref import coverage

    def fake_greedy(code, *, threshold_tpm, max_genes):
        return pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "Ensembl_Gene_ID": ["E1", "E2", "E3"],
                "Symbol": ["MAGEA4", "CTAG1B", "SSX2"],
                "marginal_patients": [60, 20, 10],
                "marginal_fraction": [0.6, 0.2, 0.1],
                "cumulative_patients": [60, 80, 90],
                "cumulative_fraction": [0.6, 0.8, 0.9],
            }
        )

    monkeypatch.setattr(coverage, "greedy_coverage", fake_greedy)
    out = tmp_path / "stacked.png"
    fig = plots.cta_coverage_stacked_bars(["LUAD", "SKCM"], save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_cta_coverage_stacked_bars_empty_raises(monkeypatch):
    import pandas as pd

    from oncoref import coverage

    monkeypatch.setattr(coverage, "greedy_coverage", lambda *a, **k: pd.DataFrame())
    with pytest.raises(ValueError, match="no coverage to plot"):
        plots.cta_coverage_stacked_bars(["LUAD"])


def test_burden_category_bars_renders(tmp_path):
    out = tmp_path / "cats.png"
    fig = plots.burden_category_bars(region="us", n=8, save=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None


def test_burden_category_bars_bad_region():
    with pytest.raises(ValueError, match="region must be"):
        plots.burden_category_bars(region="moon")


def test_cta_burden_vs_response_renders(tmp_path, monkeypatch):
    from oncoref import coverage

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM", "MM"])
    monkeypatch.setattr(
        plots, "cancer_apd1_response", lambda: {"LUAD": 19.0, "SKCM": 42.0, "MM": 3.0}
    )
    monkeypatch.setattr(
        coverage,
        "mean_antigens_per_patient",
        lambda code, **k: {"LUAD": 1.5, "SKCM": 4.0, "MM": 8.0}[code],
    )
    out = tmp_path / "load.png"
    fig = plots.cta_burden_vs_response(against="apd1", save=str(out))
    assert out.exists() and fig is not None


def test_cta_burden_vs_response_supports_burden_axis(tmp_path, monkeypatch):
    from oncoref import coverage

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM"])
    monkeypatch.setattr(
        coverage,
        "mean_antigens_per_patient",
        lambda code, **k: {"LUAD": 1.5, "SKCM": 4.0}[code],
    )
    out = tmp_path / "burden_axis.png"
    fig = plots.cta_burden_vs_response(against="us_incidence", save=str(out))
    assert out.exists() and fig is not None
    assert fig.axes[0].get_ylabel() == "US incidence share (%)"


def test_cta_burden_vs_ici_includes_ici_only_anchor(tmp_path, monkeypatch):
    from oncoref import coverage

    assert "THYM" not in plots.cancer_apd1_response()
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["THYM"])
    monkeypatch.setattr(
        coverage,
        "mean_antigens_per_patient",
        lambda code, **k: {"THYM": 2.5}[code],
    )
    out = tmp_path / "ici_only.png"
    fig = plots.cta_burden_vs_response(against="ici", save=str(out))
    assert out.exists() and fig is not None
    assert [t.get_text() for t in fig.axes[0].texts] == ["THYM"]


def test_cta_burden_vs_burden_axis_keeps_direct_msi_cohorts(tmp_path, monkeypatch):
    from oncoref import coverage

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["COAD_MSI", "READ_MSI"])
    monkeypatch.setattr(
        coverage,
        "mean_antigens_per_patient",
        lambda code, **k: {"COAD_MSI": 10.0, "READ_MSI": 30.0}[code],
    )
    out = tmp_path / "burden_msi.png"
    fig = plots.cta_burden_vs_response(against="us_incidence", save=str(out))
    assert out.exists() and fig is not None
    assert {t.get_text() for t in fig.axes[0].texts} == {"COAD_MSI", "READ_MSI"}
    xs = sorted(float(coll.get_offsets()[0][0]) for coll in fig.axes[0].collections)
    assert xs == [10.0, 30.0]


def test_cta_specific_9mer_burden_axis_keeps_direct_msi_cohorts(tmp_path, monkeypatch):
    from oncoref import peptides

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["COAD_MSI", "READ_MSI"])
    monkeypatch.setattr(
        peptides,
        "cta_specific_9mer_load",
        lambda code, **k: {"COAD_MSI": 100.0, "READ_MSI": 300.0}[code],
    )
    out = tmp_path / "9mer_burden_msi.png"
    fig = plots.cta_specific_9mer_load(against="us_incidence", save=str(out))
    assert out.exists() and fig is not None
    assert {t.get_text() for t in fig.axes[0].texts} == {"COAD_MSI", "READ_MSI"}
    xs = sorted(float(coll.get_offsets()[0][0]) for coll in fig.axes[0].collections)
    assert xs == [100.0, 300.0]


def test_cta_burden_vs_response_bad_against():
    with pytest.raises(ValueError, match="against must be"):
        plots.cta_burden_vs_response(against="nonsense")


def test_cta_burden_vs_response_no_data(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    with pytest.raises(ValueError, match="no cohort with both"):
        plots.cta_burden_vs_response()


def test_apd1_response_signature_scatter_renders(tmp_path, monkeypatch):
    from oncoref import response_signatures as rs

    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: ["LUAD", "SKCM", "MM"])
    monkeypatch.setattr(
        plots, "cancer_apd1_response", lambda: {"LUAD": 19.0, "SKCM": 42.0, "MM": 3.0}
    )
    monkeypatch.setattr(
        rs, "signature_score", lambda code, sig, **k: {"LUAD": 3.1, "SKCM": 2.9, "MM": 1.0}[code]
    )
    out = tmp_path / "sig.png"
    fig = plots.apd1_response_signature_scatter("t_cell_inflamed", save=str(out))
    assert out.exists() and fig is not None


def test_apd1_response_signature_scatter_no_data(monkeypatch):
    monkeypatch.setattr(plots, "_cached_per_sample_cohorts", lambda: [])
    with pytest.raises(ValueError, match="no cohort with both"):
        plots.apd1_response_signature_scatter("t_cell_inflamed")


def test_apd1_response_signature_bad_name():
    with pytest.raises(ValueError, match="unknown signature"):
        plots.apd1_response_signature_scatter("not_a_signature")


def test_regenerate_plots_runner_references_real_functions():
    # Guard: every (family, name, fn_attr, kwargs) job in the batch runner must
    # name a real callable on oncoref.plots — catches typos like a removed or
    # renamed figure before a full run does.
    import importlib.util
    from pathlib import Path

    from oncoref import plots

    runner = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_plots.py"
    spec = importlib.util.spec_from_file_location("_regen_plots", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    availability = {
        "per_sample": ("LUAD", "SKCM"),
        "percentile_gene": ("LUAD", "SKCM"),
        "percentile_proteoform": ("LUAD", "SKCM"),
        "within_sample": ("LUAD", "SKCM"),
    }
    jobs = mod._jobs(availability)
    assert jobs, "runner produced no jobs"
    names = {name for _, name, _, _ in jobs}
    assert {"apd1_vs_tmb_ici", "apd1_vs_tmb_strict_pd1"} <= names
    assert {
        "cta_expression_heatmap_q1",
        "cta_expression_heatmap_median",
        "cta_expression_heatmap_q3",
    } <= names
    assert {
        "cta_burden_vs_us_incidence_t25",
        "cta_burden_vs_world_mortality_t50",
        "cta_specific_9mer_load_vs_world_mortality_t50",
    } <= names
    assert {
        "cta_addressable_burden_per_sample_p90_world_mortality",
        "cta_addressable_burden_per_sample_t50_us_mortality",
    } <= names
    assert {
        "cta_coverage_curves_p90",
        "cta_coverage_curves_p95",
        "cta_coverage_stacked_bars_p90",
        "cta_coverage_stacked_bars_p95",
    } <= names
    assert {"cta_patient_count_heatmap_p90", "cta_patient_count_heatmap_t50"} <= names
    for family, name, fn_attr, kwargs in jobs:
        assert family and name and isinstance(kwargs, dict)
        assert callable(getattr(plots, fn_attr, None)), f"{fn_attr} is not a plots function"


def test_regenerate_plots_runner_computes_percentile_coverage_once(monkeypatch):
    import importlib.util

    runner = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_plots.py"
    spec = importlib.util.spec_from_file_location("_regen_plots_shared", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    availability = {
        "per_sample": ("LUAD", "SKCM"),
        "percentile_gene": (),
        "percentile_proteoform": (),
        "within_sample": (),
    }
    jobs = mod._jobs(availability)
    sentinel = _percentile_coverage_table()
    calls = []

    def fake_sweep(cohorts, **kwargs):
        calls.append((cohorts, kwargs))
        return sentinel

    monkeypatch.setattr(mod, "within_sample_percentile_coverage_sweep", fake_sweep)
    prepared, result = mod._attach_shared_percentile_coverage(jobs, availability)

    assert result is sentinel
    assert calls == [(("LUAD", "SKCM"), {"percentiles": (0.9, 0.95), "on_missing": "record"})]
    dependent = [
        kwargs for _, _, function, kwargs in prepared if function in mod._PERCENTILE_COVERAGE_PLOTS
    ]
    assert len(dependent) == 12
    assert all(kwargs["coverage_table"] is sentinel for kwargs in dependent)


def test_regenerate_plots_runner_records_exact_data_inventory():
    import importlib.util

    runner = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_plots.py"
    spec = importlib.util.spec_from_file_location("_regen_plots_inventory", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    availability = {
        "per_sample": ("LUAD", "SKCM"),
        "percentile_gene": ("LUAD",),
        "percentile_proteoform": ("SKCM",),
        "within_sample": ("LUAD", "SKCM"),
    }
    coverage = pd.DataFrame()
    coverage.attrs.update({"audited_cohorts": ["LUAD"], "missing_cohorts": {"SKCM": "not staged"}})

    text = "\n".join(mod._availability_index_lines(availability, coverage))

    assert "Cached per-sample matrices (2): `LUAD`, `SKCM`" in text
    assert "p90/p95 joint coverage cohorts audited (1): `LUAD`" in text
    assert "- `SKCM`: not staged" in text


def test_regenerate_plots_runner_writes_all_figures_pdf(tmp_path):
    import importlib.util
    from pathlib import Path

    import matplotlib.pyplot as plt

    runner = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_plots.py"
    spec = importlib.util.spec_from_file_location("_regen_plots", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rels = ["first/one.png", "second/two.png"]
    for rel in rels:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, [[[0.2, 0.4, 0.6], [0.9, 0.8, 0.1]]])

    pdf = mod._write_all_figures_pdf(tmp_path, rels)

    assert pdf == tmp_path / "all-figures.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 0


def test_regenerate_plots_runner_closes_each_returned_figure(tmp_path, monkeypatch):
    import importlib.util
    import sys

    import matplotlib.pyplot as plt

    runner = Path(__file__).resolve().parent.parent / "scripts" / "regenerate_plots.py"
    spec = importlib.util.spec_from_file_location("_regen_plots_memory", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def stub_plot(*, save, label):
        fig, ax = plt.subplots()
        ax.set_title(label)
        fig.savefig(save)
        return fig

    monkeypatch.setattr(mod.plots, "_memory_test_plot", stub_plot, raising=False)
    monkeypatch.setattr(
        mod,
        "_jobs",
        lambda availability: [
            ("memory", "first", "_memory_test_plot", {"label": "first"}),
            ("memory", "second", "_memory_test_plot", {"label": "second"}),
        ],
    )
    monkeypatch.setattr(
        mod,
        "_plot_data_availability",
        lambda: {
            "per_sample": (),
            "percentile_gene": (),
            "percentile_proteoform": (),
            "within_sample": (),
        },
    )
    monkeypatch.setattr(
        mod.cta_curation_plots,
        "render",
        lambda **kwargs: {"n_genes": 0, "paths": {}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["regenerate_plots.py", "--out-dir", str(tmp_path), "--no-timestamp"],
    )

    existing_figures = set(plt.get_fignums())
    assert mod.main() == 0
    assert set(plt.get_fignums()) == existing_figures
    assert (tmp_path / "memory" / "first.png").exists()
    assert (tmp_path / "memory" / "second.png").exists()
    assert (tmp_path / "all-figures.pdf").exists()
