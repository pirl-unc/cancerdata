# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Machine-readable compatibility contract for oncoref's supported API.

The checked-in manifest is intentionally a lower bound: adding a new module,
symbol, or optional parameter is compatible, while removing a contracted name
or making an existing call signature stricter fails CI.
"""

from __future__ import annotations

import importlib
import inspect
import json
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from typing import Any

PUBLIC_API_CONTRACT_VERSION = 1
PUBLIC_API_MODULES = (
    "oncoref",
    "oncoref.antigen_coverage",
    "oncoref.cancer_ontology",
    "oncoref.cohorts",
    "oncoref.cta",
    "oncoref.cta_coverage",
    "oncoref.cta_peptides",
    "oncoref.expression",
    "oncoref.expression_builders",
    "oncoref.expression_engine",
    "oncoref.gene_families",
    "oncoref.gene_ids",
    "oncoref.genome",
    "oncoref.hpa",
    "oncoref.ici_response",
    "oncoref.incidence",
    "oncoref.normalization",
    "oncoref.proteoforms",
    "oncoref.source_matrices",
    "oncoref.therapy_evidence",
    "oncoref.tmb",
    "oncoref.tumor_references",
)

_CONTRACT_RESOURCE = "public-api-contract.json"
_POSITIONAL_KINDS = {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}
_VARIADIC_KINDS = {"VAR_POSITIONAL", "VAR_KEYWORD"}


def _public_names(module: ModuleType) -> tuple[str, ...]:
    explicit = getattr(module, "__all__", None)
    if explicit is not None:
        return tuple(sorted(set(explicit)))

    # Modules without __all__ expose only locally defined public callables.
    # Imported helpers and module constants are implementation details unless
    # explicitly re-exported by the flat oncoref namespace.
    names = []
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if (inspect.isfunction(value) or inspect.isclass(value)) and getattr(
            value, "__module__", None
        ) == module.__name__:
            names.append(name)
    return tuple(sorted(names))


def _parameter_contract(parameter: inspect.Parameter) -> dict[str, Any]:
    variadic = parameter.kind.name in _VARIADIC_KINDS
    required = parameter.default is inspect.Parameter.empty and not variadic
    return {
        "name": parameter.name,
        "kind": parameter.kind.name,
        "required": required,
        "default": (
            None
            if parameter.default is inspect.Parameter.empty
            else _stable_default_repr(parameter.default)
        ),
    }


def _stable_default_repr(value: Any) -> str:
    """Deterministic representation for defaults containing unordered containers."""

    if isinstance(value, frozenset):
        return "frozenset({" + ", ".join(sorted(map(_stable_default_repr, value))) + "})"
    if isinstance(value, set):
        return "{" + ", ".join(sorted(map(_stable_default_repr, value))) + "}"
    if isinstance(value, tuple):
        values = ", ".join(_stable_default_repr(item) for item in value)
        if len(value) == 1:
            values += ","
        return f"({values})"
    if isinstance(value, list):
        return "[" + ", ".join(_stable_default_repr(item) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(
            (_stable_default_repr(key), _stable_default_repr(item)) for key, item in value.items()
        )
        return "{" + ", ".join(f"{key}: {item}" for key, item in items) + "}"
    return repr(value)


def _signature_contract(value: Any) -> dict[str, Any] | None:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return None
    return {
        "parameters": [
            _parameter_contract(parameter) for parameter in signature.parameters.values()
        ],
    }


def _symbol_contract(value: Any) -> dict[str, Any]:
    if inspect.ismodule(value):
        return {"kind": "module"}
    if inspect.isclass(value):
        return {"kind": "class", "signature": _signature_contract(value)}
    if callable(value):
        return {"kind": "callable", "signature": _signature_contract(value)}
    return {"kind": "constant"}


def current_public_api_manifest() -> dict[str, Any]:
    """Inspect the supported modules and return their current API contract."""

    modules = {}
    for module_name in PUBLIC_API_MODULES:
        module = importlib.import_module(module_name)
        modules[module_name] = {
            "symbols": {
                name: _symbol_contract(getattr(module, name)) for name in _public_names(module)
            }
        }
    return {
        "contract_version": PUBLIC_API_CONTRACT_VERSION,
        "modules": modules,
    }


def public_api_contract_path() -> Path:
    """Filesystem path to the checked-in machine-readable API contract."""

    return Path(str(files("oncoref").joinpath("data", _CONTRACT_RESOURCE)))


def load_public_api_contract(path: str | Path | None = None) -> dict[str, Any]:
    """Load the checked-in API contract, or an explicit contract path."""

    contract_path = Path(path) if path is not None else public_api_contract_path()
    return json.loads(contract_path.read_text())


def _signature_compatibility_errors(
    label: str,
    expected: dict[str, Any] | None,
    observed: dict[str, Any] | None,
) -> list[str]:
    if expected is None:
        return []
    if observed is None:
        return [f"{label}: inspectable signature disappeared"]

    errors = []
    old_parameters = expected.get("parameters", [])
    new_parameters = observed.get("parameters", [])
    new_by_name = {parameter["name"]: parameter for parameter in new_parameters}
    old_names = {parameter["name"] for parameter in old_parameters}

    old_positional = [
        parameter["name"] for parameter in old_parameters if parameter["kind"] in _POSITIONAL_KINDS
    ]
    new_positional = [
        parameter["name"] for parameter in new_parameters if parameter["kind"] in _POSITIONAL_KINDS
    ]
    if new_positional[: len(old_positional)] != old_positional:
        errors.append(f"{label}: positional parameter order changed")

    for old in old_parameters:
        current = new_by_name.get(old["name"])
        if current is None:
            errors.append(f"{label}: removed parameter {old['name']!r}")
            continue
        if current["kind"] != old["kind"]:
            errors.append(
                f"{label}: parameter {old['name']!r} changed kind "
                f"from {old['kind']} to {current['kind']}"
            )
        if not old["required"]:
            if current["required"]:
                errors.append(f"{label}: optional parameter {old['name']!r} became required")
            elif current.get("default") != old.get("default"):
                errors.append(
                    f"{label}: parameter {old['name']!r} changed default "
                    f"from {old.get('default')} to {current.get('default')}"
                )

    for current in new_parameters:
        if current["name"] not in old_names and current["required"]:
            errors.append(f"{label}: added required parameter {current['name']!r}")
    return errors


def public_api_compatibility_errors(
    expected: dict[str, Any] | None = None,
    observed: dict[str, Any] | None = None,
) -> list[str]:
    """Return incompatible removals/changes relative to the stored contract.

    Additive modules, symbols, and optional parameters are accepted. Existing
    names, symbol kinds, positional call order, parameter kinds, and optional
    defaults remain stable.
    """

    expected = expected if expected is not None else load_public_api_contract()
    observed = observed if observed is not None else current_public_api_manifest()
    errors = []
    if expected.get("contract_version") != PUBLIC_API_CONTRACT_VERSION:
        errors.append(
            f"unsupported public API contract version: {expected.get('contract_version')!r}"
        )
        return errors

    current_modules = observed.get("modules", {})
    for module_name, module_contract in expected.get("modules", {}).items():
        current_module = current_modules.get(module_name)
        if current_module is None:
            errors.append(f"removed public module {module_name}")
            continue
        current_symbols = current_module.get("symbols", {})
        for symbol_name, symbol_contract in module_contract.get("symbols", {}).items():
            label = f"{module_name}.{symbol_name}"
            current_symbol = current_symbols.get(symbol_name)
            if current_symbol is None:
                errors.append(f"removed public symbol {label}")
                continue
            if current_symbol.get("kind") != symbol_contract.get("kind"):
                errors.append(
                    f"{label}: changed kind from {symbol_contract.get('kind')} "
                    f"to {current_symbol.get('kind')}"
                )
                continue
            if symbol_contract.get("kind") in {"callable", "class"}:
                errors.extend(
                    _signature_compatibility_errors(
                        label,
                        symbol_contract.get("signature"),
                        current_symbol.get("signature"),
                    )
                )
    return errors


def assert_public_api_compatible() -> None:
    """Raise :class:`RuntimeError` if the checked-in API contract is broken."""

    errors = public_api_compatibility_errors()
    if errors:
        raise RuntimeError("incompatible public API changes:\n- " + "\n- ".join(errors))


__all__ = [
    "PUBLIC_API_CONTRACT_VERSION",
    "PUBLIC_API_MODULES",
    "assert_public_api_compatible",
    "current_public_api_manifest",
    "load_public_api_contract",
    "public_api_compatibility_errors",
    "public_api_contract_path",
]
