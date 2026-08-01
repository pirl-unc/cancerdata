from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import oncoref

REDIRECT_ROOT = Path(__file__).resolve().parents[1] / "pypi-oncodata-redirect"
LEGACY_SUBMODULES = (
    "apd1",
    "cancer_genes",
    "cancer_types",
    "catalog",
    "cli",
    "coverage",
    "cta",
    "cta_regen",
    "cta_tissues",
    "data_bundle",
    "data_manifest",
    "expression",
    "expression_builders",
    "expression_engine",
    "expression_registry",
    "fusions",
    "gene_families",
    "gene_ids",
    "gene_qc",
    "genome",
    "hpa",
    "ici",
    "incidence",
    "load_dataset",
    "normalization",
    "peptides",
    "plots",
    "proteoforms",
    "reference_data",
    "response_signatures",
    "samples",
    "source_matrices",
    "tmb",
    "version",
)


def _import_redirect():
    sys.path.insert(0, str(REDIRECT_ROOT))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return importlib.import_module("oncodata")
    finally:
        sys.path.remove(str(REDIRECT_ROOT))


def test_redirect_reexports_public_api():
    oncodata = _import_redirect()

    assert oncodata.__version__ == oncoref.__version__
    assert oncodata.__all__ == oncoref.__all__
    assert oncodata.cancer_type_codes is oncoref.cancer_type_codes


def test_all_legacy_submodules_alias_oncoref_modules():
    oncodata = _import_redirect()

    for name in LEGACY_SUBMODULES:
        legacy = importlib.import_module(f"oncodata.{name}")
        replacement = importlib.import_module(f"oncoref.{name}")
        assert legacy is replacement
        assert getattr(oncodata, name) is replacement


def test_documented_expression_submodule_import():
    _import_redirect()

    from oncodata.expression import SHARD_DATASETS

    assert SHARD_DATASETS is oncoref.expression.SHARD_DATASETS
