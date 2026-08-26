import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_pirlygenes_inventory.py"
_SPEC = importlib.util.spec_from_file_location("audit_pirlygenes_inventory", _SCRIPT)
audit = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(audit)


def test_inventory_entry_normalization():
    entries = [
        "small.csv",
        "compressed.csv.gz",
        "registry.yaml",
        "other.yml",
        "artifact-directory",
    ]
    assert audit.inventory_from_entries(entries) == {
        "small",
        "compressed",
        "registry",
        "other",
        "artifact-directory",
    }


def test_inventory_diff_is_directional_and_sorted():
    assert audit.inventory_diff(frozenset({"b", "a"}), frozenset({"b", "c"})) == {
        "added": ["c"],
        "removed": ["a"],
    }
