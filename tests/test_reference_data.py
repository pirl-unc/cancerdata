# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import json

import pytest

from oncoref import reference_data


def _seed_cache(name, content: bytes, version="v23"):
    """Write a cached TSV + a manifest recording its true size/sha256, merging
    into any existing manifest (so multiple versions can be seeded)."""
    dest = reference_data.local_path(name, version)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    manifest_path = reference_data.cache_dir() / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[reference_data._manifest_key(name, version)] = {
        "name": name,
        "version": version,
        "url": "http://example/test",
        "path": str(dest),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "downloaded_at": "2024-01-01T00:00:00+00:00",
    }
    manifest_path.write_text(json.dumps(manifest))
    return dest


def test_cache_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    assert reference_data.cache_dir() == tmp_path / "sources"
    assert reference_data.cache_dir().exists()


def test_resolve_version_default_and_errors():
    assert reference_data.resolve_version("hpa_rna_consensus") == "v23"
    assert reference_data.resolve_version("hpa_rna_consensus", "latest") == "latest"
    with pytest.raises(reference_data.ReferenceDataError):
        reference_data.resolve_version("hpa_normal_tissue", "v99")
    with pytest.raises(reference_data.ReferenceDataError):
        reference_data.resolve_version("not_a_source")


def test_local_path_and_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    p = reference_data.local_path("hpa_rna_consensus")
    assert p == tmp_path / "sources" / "hpa_rna_consensus" / "v23" / "rna_tissue_consensus.tsv"
    assert not reference_data.is_cached("hpa_rna_consensus")
    p.parent.mkdir(parents=True)
    p.write_text("Gene\tTissue\tnTPM\n")
    assert reference_data.is_cached("hpa_rna_consensus")


def test_status_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    rows = reference_data.status()
    names = {r["name"] for r in rows}
    assert names == {"hpa_rna_consensus", "hpa_normal_tissue", "hpa_single_cell"}
    assert all(r["cached"] is False for r in rows)  # nothing downloaded
    assert all(r["verification_state"] == "no_manifest" for r in rows)
    assert all(r["sha256"] is None for r in rows)


def test_single_cell_registered():
    # The single-cell nTPM source is part of the registry.
    spec = reference_data.REFERENCE_SOURCES["hpa_single_cell"]
    assert spec["filename"] == "rna_single_cell_type.tsv"
    assert "v23" in spec["urls"]


def test_cached_file_ok_detects_truncation(monkeypatch, tmp_path):
    # #21: a TSV left truncated by a killed extract must NOT be served as valid.
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    dest = _seed_cache("hpa_rna_consensus", b"Gene\tTissue\tnTPM\n" * 100)
    assert reference_data._cached_file_ok("hpa_rna_consensus", "v23", dest) is True
    dest.write_bytes(b"Gene\tTissue\n")  # truncated -> size no longer matches manifest
    assert reference_data._cached_file_ok("hpa_rna_consensus", "v23", dest) is False


def test_cached_file_ok_trusts_unrecorded(monkeypatch, tmp_path):
    # A file with no manifest record can't be disproven -> trusted (back-compat).
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    dest = reference_data.local_path("hpa_rna_consensus")
    dest.parent.mkdir(parents=True)
    dest.write_text("Gene\tTissue\tnTPM\n")
    assert reference_data._cached_file_ok("hpa_rna_consensus", "v23", dest) is True


def test_cached_file_ok_is_per_version(monkeypatch, tmp_path):
    # Each cached version keeps its own size record: truncating v23 must be
    # detected even after a (good) 'latest' was the last thing downloaded.
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    v23 = _seed_cache("hpa_rna_consensus", b"v23-data" * 50, version="v23")
    latest = _seed_cache("hpa_rna_consensus", b"latest-data" * 80, version="latest")
    assert reference_data._cached_file_ok("hpa_rna_consensus", "v23", v23) is True
    assert reference_data._cached_file_ok("hpa_rna_consensus", "latest", latest) is True
    v23.write_bytes(b"trunc")  # corrupt only v23
    assert reference_data._cached_file_ok("hpa_rna_consensus", "v23", v23) is False
    # 'latest' is unaffected — its own record still matches.
    assert reference_data._cached_file_ok("hpa_rna_consensus", "latest", latest) is True


def test_verify_is_per_version(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    _seed_cache("hpa_rna_consensus", b"v23-data" * 50, version="v23")
    latest = _seed_cache("hpa_rna_consensus", b"latest-data" * 80, version="latest")
    assert reference_data.verify("hpa_rna_consensus", "v23") is True
    assert reference_data.verify("hpa_rna_consensus", "latest") is True
    latest.write_bytes(b"corrupted")
    assert reference_data.verify("hpa_rna_consensus", "latest") is False
    assert reference_data.verify("hpa_rna_consensus", "v23") is True  # unaffected


def test_ensure_refetches_corrupt_cache(monkeypatch, tmp_path):
    # ensure() must re-download when the cached file fails the size check.
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    dest = _seed_cache("hpa_rna_consensus", b"x" * 500)
    dest.write_bytes(b"x" * 3)  # corrupt: size mismatch

    called = {}

    def fake_download(name, version=None, *, force=False):
        called["name"] = name
        return dest

    monkeypatch.setattr(reference_data, "download", fake_download)
    reference_data.ensure("hpa_rna_consensus")
    assert called.get("name") == "hpa_rna_consensus"


def test_verify_matches_and_mismatches(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    dest = _seed_cache("hpa_rna_consensus", b"Gene\tTissue\tnTPM\nA\tB\t1\n")
    assert reference_data.verify("hpa_rna_consensus") is True
    dest.write_bytes(b"corrupted")
    assert reference_data.verify("hpa_rna_consensus") is False


def test_verify_without_manifest_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    with pytest.raises(reference_data.ReferenceDataError, match="nothing to verify"):
        reference_data.verify("hpa_rna_consensus")


def test_provenance_selects_exact_cached_version(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    v23 = _seed_cache("hpa_rna_consensus", b"v23-data", version="v23")
    latest = _seed_cache("hpa_rna_consensus", b"latest-data", version="latest")

    v23_record = reference_data.provenance("hpa_rna_consensus", "v23")
    latest_record = reference_data.provenance("hpa_rna_consensus", "latest")

    assert v23_record["version"] == "v23"
    assert v23_record["path"] == str(v23)
    assert v23_record["sha256"] == hashlib.sha256(b"v23-data").hexdigest()
    assert latest_record["version"] == "latest"
    assert latest_record["path"] == str(latest)
    assert latest_record["sha256"] == hashlib.sha256(b"latest-data").hexdigest()


def test_provenance_distinguishes_verification_states(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    dest = _seed_cache("hpa_rna_consensus", b"good")

    unchecked = reference_data.provenance("hpa_rna_consensus")
    assert unchecked["verification_state"] == "not_checked"
    assert unchecked["checksum_matches"] is None

    verified = reference_data.provenance("hpa_rna_consensus", verify_content=True)
    assert verified["verification_state"] == "verified"
    assert verified["checksum_matches"] is True

    dest.write_bytes(b"bad")
    mismatched = reference_data.provenance("hpa_rna_consensus", verify_content=True)
    assert mismatched["verification_state"] == "checksum_mismatch"
    assert mismatched["checksum_matches"] is False

    dest.unlink()
    missing = reference_data.provenance("hpa_rna_consensus", verify_content=True)
    assert missing["verification_state"] == "missing_file"
    assert missing["checksum_matches"] is False


def test_provenance_distinguishes_no_manifest_and_missing_checksum(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    dest = reference_data.local_path("hpa_rna_consensus")
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"unrecorded")

    absent = reference_data.provenance("hpa_rna_consensus", verify_content=True)
    assert absent["manifest_recorded"] is False
    assert absent["verification_state"] == "no_manifest"
    assert absent["checksum_matches"] is None

    manifest_path = reference_data.cache_dir() / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                reference_data._manifest_key("hpa_rna_consensus", "v23"): {
                    "name": "hpa_rna_consensus",
                    "version": "v23",
                    "url": "http://example/test",
                    "path": str(dest),
                    "bytes": len(b"unrecorded"),
                }
            }
        )
    )
    missing_checksum = reference_data.provenance("hpa_rna_consensus", verify_content=True)
    assert missing_checksum["manifest_recorded"] is True
    assert missing_checksum["verification_state"] == "missing_checksum"
    assert missing_checksum["checksum_matches"] is None


def test_provenance_is_a_defensive_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    _seed_cache("hpa_rna_consensus", b"stable")

    first = reference_data.provenance("hpa_rna_consensus")
    expected_sha256 = first["sha256"]
    first["sha256"] = "mutated"
    first["url"] = "http://example/mutated"

    second = reference_data.provenance("hpa_rna_consensus")
    assert second["sha256"] == expected_sha256
    assert second["url"] == "http://example/test"


def test_status_can_verify_default_version_content(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_DATA_DIR", str(tmp_path))
    _seed_cache("hpa_rna_consensus", b"stable")

    by_name = {row["name"]: row for row in reference_data.status(verify_content=True)}
    record = by_name["hpa_rna_consensus"]
    assert record["verification_state"] == "verified"
    assert record["checksum_matches"] is True
    assert record["url"] == "http://example/test"
    assert record["recorded_bytes"] == len(b"stable")
