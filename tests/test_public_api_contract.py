from copy import deepcopy

import pytest

from oncoref import api_contract


def _function(parameters):
    return {
        "kind": "callable",
        "signature": {"parameters": parameters},
    }


def _parameter(name, *, kind="POSITIONAL_OR_KEYWORD", required=True, default=None):
    return {"name": name, "kind": kind, "required": required, "default": default}


def _manifest(symbol):
    return {
        "contract_version": api_contract.PUBLIC_API_CONTRACT_VERSION,
        "modules": {"example": {"symbols": {"call": symbol}}},
    }


def test_checked_in_public_api_contract_matches_current_api():
    assert api_contract.public_api_compatibility_errors() == []
    api_contract.assert_public_api_compatible()


def test_public_api_contract_is_machine_readable_and_versioned():
    contract = api_contract.load_public_api_contract()
    assert contract["contract_version"] == api_contract.PUBLIC_API_CONTRACT_VERSION
    assert contract["generated_for_package"] == "1.8.185"
    assert set(api_contract.PUBLIC_API_MODULES) <= set(contract["modules"])


def test_additive_module_symbol_and_optional_parameter_are_compatible():
    expected = _manifest(_function([_parameter("value")]))
    observed = deepcopy(expected)
    observed["modules"]["example"]["symbols"]["call"]["signature"]["parameters"].append(
        _parameter("mode", kind="KEYWORD_ONLY", required=False, default="'safe'")
    )
    observed["modules"]["example"]["symbols"]["new_call"] = _function([])
    observed["modules"]["new_module"] = {"symbols": {}}

    assert api_contract.public_api_compatibility_errors(expected, observed) == []


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda manifest: manifest["modules"].clear(), "removed public module"),
        (
            lambda manifest: manifest["modules"]["example"]["symbols"].clear(),
            "removed public symbol",
        ),
        (
            lambda manifest: manifest["modules"]["example"]["symbols"]["call"].update(
                kind="constant"
            ),
            "changed kind",
        ),
        (
            lambda manifest: manifest["modules"]["example"]["symbols"]["call"]["signature"][
                "parameters"
            ].append(_parameter("required_new")),
            "added required parameter",
        ),
    ],
)
def test_incompatible_api_changes_are_rejected(mutate, match):
    expected = _manifest(_function([_parameter("value")]))
    observed = deepcopy(expected)
    mutate(observed)

    errors = api_contract.public_api_compatibility_errors(expected, observed)
    assert any(match in error for error in errors)


def test_optional_parameter_default_change_is_rejected():
    expected = _manifest(_function([_parameter("mode", required=False, default="'safe'")]))
    observed = _manifest(_function([_parameter("mode", required=False, default="'unsafe'")]))
    assert any(
        "changed default" in error
        for error in api_contract.public_api_compatibility_errors(expected, observed)
    )
