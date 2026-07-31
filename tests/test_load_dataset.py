from pathlib import Path

import pandas as pd

from oncoref import load_dataset


def _reference_rows(n_rows=20_000):
    return pd.DataFrame(
        {
            "Ensembl_Gene_ID": [f"ENSG{i:011d}" for i in range(n_rows)],
            "Symbol": [f"GENE{i}" for i in range(n_rows)],
            "cancer_code": ["SARC_DDLPS"] * n_rows,
            "source_cohort": ["GSE30929_SINGER_2007_LPS"] * n_rows,
            "source_project": ["GEO"] * n_rows,
            "source_version": ["2026-07-18"] * n_rows,
            "processing_pipeline": ["source_summary_rows"] * n_rows,
            "notes": ["Repeated provenance note"] * n_rows,
            "tumor_origin": ["primary"] * n_rows,
            "metastasis_site": [None] * n_rows,
            "TPM_clean_median": [1.0] * n_rows,
        }
    )


def test_reference_expression_owning_cache_categorizes_repeated_provenance():
    raw = _reference_rows()
    raw_bytes = raw.memory_usage(index=True, deep=True).sum()

    optimized = load_dataset._optimize_cached_dataframe("cancer-reference-expression", raw)

    for column in load_dataset._CATEGORICAL_COLUMNS_BY_DATASET["cancer-reference-expression"]:
        assert isinstance(optimized[column].dtype, pd.CategoricalDtype)
    optimized_bytes = optimized.memory_usage(index=True, deep=True).sum()
    assert optimized_bytes < raw_bytes * 0.35


def test_reference_expression_parquet_cache_preserves_compact_dtypes(tmp_path, monkeypatch):
    shard_dir = tmp_path / "cancer-reference-expression"
    shard_dir.mkdir()
    _reference_rows(n_rows=20).iloc[:10].to_csv(shard_dir / "a.csv", index=False)
    _reference_rows(n_rows=20).iloc[10:].to_csv(shard_dir / "b.csv", index=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    built = load_dataset._load_shard_directory(shard_dir)
    loaded = load_dataset._load_shard_directory(shard_dir)

    pd.testing.assert_frame_equal(loaded, built)
    assert isinstance(loaded["source_cohort"].dtype, pd.CategoricalDtype)
    signature = (
        tmp_path / ".cache" / "oncoref" / "shard_cache" / "cancer-reference-expression.sig"
    ).read_text()
    assert signature.startswith(f"({load_dataset._SHARD_CACHE_SCHEMA_VERSION},")


def test_reference_expression_csv_shards_are_compacted_before_concatenation(tmp_path, monkeypatch):
    shard_dir = tmp_path / "cancer-reference-expression"
    shard_dir.mkdir()
    paths = []
    for index, cohort in enumerate(("SOURCE_A", "SOURCE_B")):
        path = shard_dir / f"{index}.csv"
        frame = _reference_rows(n_rows=10)
        frame["source_cohort"] = cohort
        frame.to_csv(path, index=False)
        paths.append(path)

    optimized_shards = []
    real_optimize = load_dataset._optimize_cached_dataframe

    def record_optimized_shard(dataset_name, frame):
        result = real_optimize(dataset_name, frame)
        optimized_shards.append(result)
        return result

    monkeypatch.setattr(load_dataset, "_optimize_cached_dataframe", record_optimized_shard)
    loaded = load_dataset._read_shards_for_cache(shard_dir, paths)

    assert loaded == optimized_shards
    assert len(loaded) == 2
    assert all(isinstance(frame["source_cohort"].dtype, pd.CategoricalDtype) for frame in loaded)


def test_ncbi_synonym_loader_preserves_na_like_aliases():
    load_dataset._clear_cache()
    synonyms = load_dataset.get_data("ncbi-symbol-synonyms")

    expected = {"NA": "XK", "NaN": "SCN11A"}
    observed = synonyms[synonyms["alias"].isin(expected)].set_index("alias")["official_symbol"]
    assert observed.to_dict() == expected
    assert isinstance(synonyms["alias"].dtype, pd.StringDtype)


def test_ncbi_synonym_loader_keeps_empty_cells_missing(tmp_path):
    path = tmp_path / "ncbi-symbol-synonyms.csv"
    path.write_text("alias,official_symbol\nNA,XK\nNaN,SCN11A\n,EMPTY\n")

    loaded = load_dataset._read_csv_for_cache(path, "ncbi-symbol-synonyms")

    assert loaded["alias"].iloc[:2].tolist() == ["NA", "NaN"]
    assert pd.isna(loaded["alias"].iloc[2])
