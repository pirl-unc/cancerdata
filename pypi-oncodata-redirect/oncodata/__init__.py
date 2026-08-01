"""Compatibility shim for the package renamed from ``oncodata`` to ``oncoref``.

The redirect preserves the former top-level API and module import paths while
warning callers to migrate. It contains no data or implementation of its own.
"""

import importlib
import sys
import warnings

import oncoref as _oncoref

warnings.warn(
    "The 'oncodata' package has been renamed to 'oncoref'. "
    "Install it with `pip install oncoref` and use `import oncoref`. "
    "This 'oncodata' package is frozen and will not be updated.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public API and preserve every Python module shipped by oncodata
# 1.5.0. Registering the original paths in sys.modules means both
# ``import oncodata.expression`` and ``from oncodata.expression import ...``
# resolve to the one corresponding oncoref module object.
from oncoref import *  # noqa: E402, F403
from oncoref import __version__  # noqa: E402

__all__ = _oncoref.__all__

_LEGACY_SUBMODULES = (
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

for _name in _LEGACY_SUBMODULES:
    _module = importlib.import_module(f"oncoref.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module

del _module, _name
