#!/usr/bin/env python3
"""Check or deliberately update oncoref's public API compatibility manifest.

This developer/CI tool is intentionally outside the runtime :mod:`oncoref`
package. Public module paths are discovered when the manifest is updated, then
the checked-in manifest itself is the fixed compatibility baseline used by CI.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "oncoref"
CONTRACT_PATH = PACKAGE_ROOT / "data" / "public-api-contract.json"
PUBLIC_API_CONTRACT_VERSION = 1
_POSITIONAL_KINDS = {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}
_VARIADIC_KINDS = {"VAR_POSITIONAL", "VAR_KEYWORD"}

sys.path.insert(0, str(REPO_ROOT))

import oncoref  # noqa: E402


def discover_public_modules(package_root: Path = PACKAGE_ROOT) -> tuple[str, ...]:
    """Discover importable, non-private module paths for a manifest update.

    Discovery happens only while refreshing the baseline. Routine checks read
    the fixed paths from the checked-in manifest so deleting a module cannot
    also delete it from the set being checked.
    """

    modules = ["oncoref"]
    modules.extend(
        f"oncoref.{path.stem}"
        for path in package_root.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    )
    return tuple(sorted(modules))


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


def current_public_api_manifest(module_names) -> dict[str, Any]:
    """Inspect the requested baseline modules and return their current contract."""

    modules = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        modules[module_name] = {
            "symbols": {
                name: _symbol_contract(getattr(module, name)) for name in _public_names(module)
            }
        }
    return {
        "contract_version": PUBLIC_API_CONTRACT_VERSION,
        "modules": modules,
    }


def load_public_api_contract(path: str | Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the checked-in API contract, or an explicit contract path."""

    return json.loads(Path(path).read_text())


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
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    """Return incompatible removals/changes relative to the stored contract."""

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


def _observed_manifest(expected: dict[str, Any], *, discover: bool) -> dict[str, Any]:
    baseline_modules = set(expected.get("modules", {}))
    if discover:
        baseline_modules.update(discover_public_modules())
    return current_public_api_manifest(sorted(baseline_modules))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the current additive API")
    parser.add_argument(
        "--allow-breaking",
        metavar="REVIEW_REFERENCE",
        help="approve a deliberate incompatible update and record its issue/PR reference",
    )
    args = parser.parse_args(argv)
    if args.allow_breaking and not args.update:
        parser.error("--allow-breaking requires --update")

    expected = load_public_api_contract()
    current = _observed_manifest(expected, discover=args.update)
    errors = public_api_compatibility_errors(expected, current)
    if not args.update:
        if errors:
            print("incompatible public API changes:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print("public API contract compatible")
        return 0

    if errors and not args.allow_breaking:
        print("refusing to overwrite incompatible public API changes", file=sys.stderr)
        print("review them, then pass --allow-breaking ISSUE_OR_PR if intentional", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    current["generated_for_package"] = oncoref.__version__
    current["breaking_change_approval"] = args.allow_breaking
    CONTRACT_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    print(f"updated {CONTRACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
