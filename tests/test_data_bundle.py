# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import io
import tarfile
import urllib.error

import pytest

from oncoref import data_bundle
from oncoref.version import DATA_VERSION, SOURCE_MATRIX_VERSION, __version__


def test_is_downloadable_distinguishes_bundle_from_wheel():
    assert data_bundle.is_downloadable("cancer-reference-expression")
    assert data_bundle.is_downloadable("cancer-reference-expression-percentiles")
    assert data_bundle.is_downloadable("cancer-reference-expression-within-sample-top5")
    assert data_bundle.is_downloadable("source-matrix-sample-qc.csv")
    assert data_bundle.is_downloadable("pan-cancer-expression.csv")
    # Small wheel-bundled tables are NOT downloadable items.
    assert not data_bundle.is_downloadable("cancer-type-registry")
    assert not data_bundle.is_downloadable("cancer-tmb")


def test_item_availability_checks_package_then_partial_cache(monkeypatch, tmp_path):
    package_root = tmp_path / "package"
    cache_root = tmp_path / "cache"
    item = "cancer-reference-expression-percentiles"
    monkeypatch.setattr(data_bundle, "_PACKAGE_DATA_DIR", package_root)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(cache_root))

    packaged = package_root / item
    packaged.mkdir(parents=True)
    (packaged / "LUAD.parquet").write_text("packaged")
    cached = cache_root / item
    cached.mkdir(parents=True)
    (cached / "SKCM.parquet").write_text("cached")

    assert data_bundle.item_is_local(item)
    assert data_bundle.find_local_item(item) == packaged

    (packaged / "LUAD.parquet").unlink()
    assert data_bundle.find_local_item(item) == cached


def test_local_item_paths_returns_package_and_partial_cache(monkeypatch, tmp_path):
    package_root = tmp_path / "package"
    cache_root = tmp_path / "cache"
    item = "cancer-reference-expression-percentiles"
    monkeypatch.setattr(data_bundle, "_PACKAGE_DATA_DIR", package_root)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(cache_root))

    packaged = package_root / item
    cached = cache_root / item
    packaged.mkdir(parents=True)
    cached.mkdir(parents=True)
    (packaged / "LUAD.parquet").write_text("packaged")
    (cached / "SKCM.parquet").write_text("cached")

    assert data_bundle.local_item_paths(item) == (packaged, cached)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "nested/../../escape"])
def test_local_item_paths_rejects_paths_outside_data_roots(path):
    with pytest.raises(ValueError, match="relative_path"):
        data_bundle.local_item_paths(path)


def test_item_availability_rejects_missing_and_empty_items(monkeypatch, tmp_path):
    package_root = tmp_path / "package"
    cache_root = tmp_path / "cache"
    item = "cancer-reference-expression-percentiles"
    monkeypatch.setattr(data_bundle, "_PACKAGE_DATA_DIR", package_root)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(cache_root))

    (package_root / item).mkdir(parents=True)
    (cache_root / item).mkdir(parents=True)

    assert not data_bundle.item_is_local(item)
    assert data_bundle.find_local_item(item) is None


def test_find_rejects_empty_cached_item(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    empty = tmp_path / "cancer-reference-expression-percentiles"
    empty.mkdir()

    assert data_bundle.find(empty.name) is None


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "vX"))
    assert data_bundle.cache_dir() == tmp_path / "vX"
    assert data_bundle.cache_root() == tmp_path


def test_legacy_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("CANCERDATA_BUNDLED_DATA", raising=False)
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(tmp_path / "vY"))
    assert data_bundle.cache_dir() == tmp_path / "vY"


def test_new_env_wins_over_legacy(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / "new"))
    monkeypatch.setenv("PIRLYGENES_BUNDLED_DATA", str(tmp_path / "old"))
    assert data_bundle.cache_dir() == tmp_path / "new"


def test_status_reports_missing_without_download(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path))
    snap = data_bundle.status()
    assert snap["contract_version"] == data_bundle.BUNDLE_CONTRACT_VERSION
    assert snap["package_version"] == __version__
    assert snap["all_local"] is False
    assert snap["data_version"] == DATA_VERSION
    assert snap["source_matrix_version"] == SOURCE_MATRIX_VERSION
    assert snap["completion_marker"]["present"] is False
    assert snap["completion_marker"]["valid"] is False
    assert snap["release_manifest_url"] == data_bundle.RELEASE_MANIFEST_URL
    assert snap["release_checksum_url"] == data_bundle.RELEASE_CHECKSUM_URL
    assert set(snap["items"]) == set(data_bundle.DOWNLOADABLE_PATHS)
    assert all(not v["present"] for v in snap["items"].values())
    assert snap["contract"] == data_bundle.bundle_contract()


def test_bundle_contract_links_package_data_release_and_cache_policy():
    contract = data_bundle.bundle_contract()

    assert contract["contract_version"] == data_bundle.BUNDLE_CONTRACT_VERSION
    assert contract["package_version"] == __version__
    assert contract["data_version"] == DATA_VERSION
    assert contract["source_matrix_version"] == SOURCE_MATRIX_VERSION
    assert contract["cache_dir_env_var"] == data_bundle.CACHE_DIR_ENV_VAR
    assert contract["legacy_cache_dir_env_var"] == data_bundle.LEGACY_CACHE_DIR_ENV_VAR
    assert contract["downloadable_paths"] == list(data_bundle.DOWNLOADABLE_PATHS)
    assert contract["primary_release_source"]["name"] == "oncoref"
    assert contract["primary_release_source"]["require_integrity"] is True
    assert contract["primary_release_source"]["manifest_url"] == data_bundle.RELEASE_MANIFEST_URL
    assert contract["primary_release_source"]["checksum_url"] == data_bundle.RELEASE_CHECKSUM_URL
    assert [s["name"] for s in contract["release_sources"]] == ["oncoref", "pirlygenes"]


def _write_bundle_fixture(root):
    for p in data_bundle.DOWNLOADABLE_PATHS:
        target = root / p
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{p}\n")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "shard.parquet").write_text(f"{p}\n")


def _bundle_tarball(tmp_path, source_root, *, missing=()):
    tar_path = tmp_path / "bundle.tar.gz"
    missing = set(missing)
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in data_bundle.DOWNLOADABLE_PATHS:
            if p in missing:
                continue
            tf.add(source_root / p, arcname=p)
    return tar_path


def _release_manifest(tar_path):
    return {
        "manifest_version": data_bundle.BUNDLE_MANIFEST_VERSION,
        "data_version": DATA_VERSION,
        "source": "oncoref",
        "repo": data_bundle.GITHUB_REPO,
        "manifest_url": data_bundle.RELEASE_MANIFEST_URL,
        "package_version": __version__,
        "source_matrix_version": SOURCE_MATRIX_VERSION,
        "sample_qc_policy": "pass",
        "sample_qc_policy_version": "sample_expression_qc_v2",
        "source_matrix_sample_qc": "source-matrix-sample-qc.csv",
        "artifact_build_metadata": {
            "cohort_metadata": "expression-artifact-build-metadata.csv",
            "bundle_metadata": "expression-artifact-build-metadata.json",
            "n_cohorts": 118,
        },
        "tarball": {
            "filename": data_bundle.TARBALL_FILENAME,
            "url": data_bundle.RELEASE_URL,
            "bytes": tar_path.stat().st_size,
            "sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
            "downloadable_paths": list(data_bundle.DOWNLOADABLE_PATHS),
        },
        "builder_commit": "abc123",
        "inventory": {
            path: {"path": path, "file_count": 1, "size_bytes": 10}
            for path in data_bundle.DOWNLOADABLE_PATHS
        },
    }


def test_fetch_release_manifest_accepts_sha256_sidecar(monkeypatch):
    sha = "a" * 64

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(url):
        if url == data_bundle.RELEASE_MANIFEST_URL:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if url == data_bundle.RELEASE_CHECKSUM_URL:
            return Response(f"{sha}  {data_bundle.TARBALL_FILENAME}\n".encode())
        raise AssertionError(url)

    monkeypatch.setattr(data_bundle.urllib.request, "urlopen", fake_urlopen)

    manifest = data_bundle._fetch_release_manifest(data_bundle.RELEASE_SOURCES[0])
    assert manifest["tarball"]["sha256"] == sha
    assert manifest["tarball"]["filename"] == data_bundle.TARBALL_FILENAME


def test_bundle_release_manifest_preserves_inventory_and_build_metadata(monkeypatch, tmp_path):
    src = tmp_path / "src"
    _write_bundle_fixture(src)
    tar_path = _bundle_tarball(tmp_path, src)
    release_manifest = _release_manifest(tar_path)

    def fake_read_url_text(url):
        if url == data_bundle.RELEASE_MANIFEST_URL:
            return data_bundle.json.dumps(release_manifest)
        raise AssertionError(url)

    monkeypatch.setattr(data_bundle, "_read_url_text", fake_read_url_text)

    manifest = data_bundle.bundle_release_manifest()

    assert manifest["source"] == "oncoref"
    assert manifest["tarball"]["sha256"] == release_manifest["tarball"]["sha256"]
    assert manifest["builder_commit"] == "abc123"
    assert manifest["package_version"] == __version__
    assert manifest["source_matrix_version"] == SOURCE_MATRIX_VERSION
    assert manifest["sample_qc_policy"] == "pass"
    assert manifest["sample_qc_policy_version"] == "sample_expression_qc_v2"
    assert manifest["source_matrix_sample_qc"] == "source-matrix-sample-qc.csv"
    assert manifest["artifact_build_metadata"]["n_cohorts"] == 118
    assert set(manifest["inventory"]) == set(data_bundle.DOWNLOADABLE_PATHS)
    assert manifest["inventory"]["pan-cancer-expression.csv"]["file_count"] == 1


def test_bundle_metadata_composes_contract_local_status_and_release_manifest(monkeypatch, tmp_path):
    root = tmp_path / f"v{DATA_VERSION}"
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root))
    release_manifest = {
        "source": "oncoref",
        "data_version": DATA_VERSION,
        "tarball": {"sha256": "a" * 64},
        "inventory": {
            "pan-cancer-expression.csv": {
                "path": "pan-cancer-expression.csv",
                "file_count": 1,
                "size_bytes": 10,
            }
        },
    }
    monkeypatch.setattr(
        data_bundle,
        "bundle_release_manifest",
        lambda source="oncoref": release_manifest | {"source": source},
    )

    metadata = data_bundle.bundle_metadata("oncoref")

    assert metadata["contract_version"] == data_bundle.BUNDLE_CONTRACT_VERSION
    assert metadata["package_version"] == __version__
    assert metadata["data_version"] == DATA_VERSION
    assert metadata["source_matrix_version"] == SOURCE_MATRIX_VERSION
    assert metadata["release_source"] == "oncoref"
    assert metadata["contract"] == data_bundle.bundle_contract()
    assert metadata["local_cache"]["cache_dir"] == str(root)
    assert metadata["local_cache"]["all_local"] is False
    assert set(metadata["local_cache"]["inventory"]) == set(data_bundle.DOWNLOADABLE_PATHS)
    assert metadata["release_manifest"]["tarball"]["sha256"] == "a" * 64


def test_bundle_metadata_can_skip_release_manifest(monkeypatch, tmp_path):
    root = tmp_path / f"v{DATA_VERSION}"
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root))
    monkeypatch.setattr(
        data_bundle,
        "bundle_release_manifest",
        lambda source="oncoref": pytest.fail("release manifest should not be fetched"),
    )

    metadata = data_bundle.bundle_metadata(include_release_manifest=False)

    assert metadata["release_manifest"] is None
    assert metadata["local_cache"]["cache_dir"] == str(root)


def test_bundle_release_manifest_source_validation_and_legacy_absence(monkeypatch):
    with pytest.raises(ValueError, match="unknown bundle release source"):
        data_bundle.bundle_release_manifest("other")

    def fake_read_url_text(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data_bundle, "_read_url_text", fake_read_url_text)

    assert data_bundle.bundle_release_manifest("pirlygenes") is None
    with pytest.raises(data_bundle.BundleIntegrityError, match="missing release manifest"):
        data_bundle.bundle_release_manifest("oncoref")


def test_release_manifest_rejects_partial_inventory_and_source_version_mismatch(tmp_path):
    src = tmp_path / "src"
    _write_bundle_fixture(src)
    tar_path = _bundle_tarball(tmp_path, src)
    manifest = _release_manifest(tar_path)

    partial = dict(manifest)
    partial["inventory"] = {
        "pan-cancer-expression.csv": manifest["inventory"]["pan-cancer-expression.csv"]
    }
    with pytest.raises(data_bundle.BundleIntegrityError, match="inventory lacks required"):
        data_bundle._validate_release_manifest(
            partial,
            data_bundle.RELEASE_SOURCES[0],
            manifest_url=data_bundle.RELEASE_MANIFEST_URL,
        )

    mismatch = dict(manifest)
    mismatch["source_matrix_version"] = "old"
    with pytest.raises(data_bundle.BundleIntegrityError, match="source_matrix_version"):
        data_bundle._validate_release_manifest(
            mismatch,
            data_bundle.RELEASE_SOURCES[0],
            manifest_url=data_bundle.RELEASE_MANIFEST_URL,
        )


def test_is_local_requires_nonempty_dirs(monkeypatch, tmp_path):
    # #21: an interrupted extract that created the shard directories but no
    # shards must NOT read as "local" (else ensure_local never re-fetches).
    root = tmp_path / f"v{DATA_VERSION}"
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root))
    # Create every downloadable path but leave the directories empty.
    for p in data_bundle.DOWNLOADABLE_PATHS:
        target = root / p
        if target.suffix:  # a file entry
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")
        else:  # a directory entry — created but empty
            target.mkdir(parents=True, exist_ok=True)
    assert data_bundle.is_local() is False
    # Now drop a shard into each directory entry -> complete.
    for p in data_bundle.DOWNLOADABLE_PATHS:
        target = root / p
        if not target.suffix:
            (target / "shard.parquet").write_text("data")
    assert data_bundle.is_local() is True
    with pytest.raises(RuntimeError, match="checksum-verified completion marker"):
        data_bundle.verify_local()


def test_download_and_extract_writes_completion_marker(monkeypatch, tmp_path):
    root = tmp_path / f"v{DATA_VERSION}"
    root.mkdir()
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root))
    src = tmp_path / "src"
    _write_bundle_fixture(src)
    tar_path = _bundle_tarball(tmp_path, src)
    monkeypatch.setattr(data_bundle.urllib.request, "urlopen", lambda url: tar_path.open("rb"))

    data_bundle._download_and_extract(
        "https://example.test/bundle.tar.gz",
        root,
        verbose=False,
        release_manifest=_release_manifest(tar_path),
    )

    snap = data_bundle.status()
    assert data_bundle.is_local() is True
    assert data_bundle.verify_local()["completion_marker"]["valid"] is True
    assert snap["completion_marker"]["present"] is True
    assert snap["completion_marker"]["valid"] is True
    assert snap["completion_marker"]["source_url"] == "https://example.test/bundle.tar.gz"
    assert snap["completion_marker"]["verified_sha256"] is True
    assert all(item["complete"] for item in snap["items"].values())
    assert all(item["file_count"] >= 1 for item in snap["items"].values())


def test_download_and_extract_rejects_checksum_mismatch(monkeypatch, tmp_path):
    root = tmp_path / f"v{DATA_VERSION}"
    root.mkdir()
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root))
    src = tmp_path / "src"
    _write_bundle_fixture(src)
    tar_path = _bundle_tarball(tmp_path, src)
    manifest = _release_manifest(tar_path)
    manifest["tarball"]["sha256"] = "0" * 64
    monkeypatch.setattr(data_bundle.urllib.request, "urlopen", lambda url: tar_path.open("rb"))

    with pytest.raises(data_bundle.BundleIntegrityError, match="sha256 mismatch"):
        data_bundle._download_and_extract(
            "https://example.test/bundle.tar.gz",
            root,
            verbose=False,
            release_manifest=manifest,
        )

    assert data_bundle.status()["completion_marker"]["present"] is False


def test_download_and_extract_rejects_incomplete_tarball(monkeypatch, tmp_path):
    root = tmp_path / f"v{DATA_VERSION}"
    root.mkdir()
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root))
    src = tmp_path / "src"
    _write_bundle_fixture(src)
    missing = "hpa-cell-type-expression.csv"
    tar_path = _bundle_tarball(tmp_path, src, missing={missing})
    monkeypatch.setattr(data_bundle.urllib.request, "urlopen", lambda url: tar_path.open("rb"))

    with pytest.raises(tarfile.TarError, match=missing):
        data_bundle._download_and_extract(
            "https://example.test/bad.tar.gz",
            root,
            verbose=False,
            release_manifest=_release_manifest(tar_path),
        )

    assert not (root / missing).exists()
    assert data_bundle.status()["completion_marker"]["present"] is False


def test_prune_keeps_current(monkeypatch, tmp_path):
    root = tmp_path / "bundled_data"
    (root / "v1.0.0").mkdir(parents=True)
    (root / "v2.0.0").mkdir(parents=True)
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(root / "v2.0.0"))
    planned = data_bundle.prune_cache(dry_run=True)
    versions = {e["version"] for e in planned}
    assert versions == {"v1.0.0"}  # current (v2.0.0) is kept


def test_release_urls_prefer_oncoref_then_pirlygenes():
    assert data_bundle.RELEASE_URLS == (
        data_bundle.RELEASE_URL,
        data_bundle.FALLBACK_RELEASE_URL,
    )
    assert "pirl-unc/oncoref" in data_bundle.RELEASE_URL
    assert "pirl-unc/pirlygenes" in data_bundle.FALLBACK_RELEASE_URL
    assert f"v{DATA_VERSION}" in data_bundle.RELEASE_URL
    assert data_bundle.RELEASE_MANIFEST_URL.endswith(
        f"/v{DATA_VERSION}/oncoref-data-v{DATA_VERSION}.manifest.json"
    )
    assert data_bundle.RELEASE_CHECKSUM_URL.endswith(
        f"/v{DATA_VERSION}/oncoref-data-v{DATA_VERSION}.tar.gz.sha256"
    )


def test_fetch_primary_404_fails_without_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / f"v{DATA_VERSION}"))
    attempted = []
    monkeypatch.setattr(data_bundle, "_fetch_release_manifest", lambda source: None)

    def fake_download(url, root, *, verbose, release_manifest=None):
        attempted.append(url)
        if url == data_bundle.RELEASE_URL:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        # fallback "succeeds"

    monkeypatch.setattr(data_bundle, "_download_and_extract", fake_download)
    with pytest.raises(RuntimeError, match="checksum-verified primary source"):
        data_bundle.fetch(verbose=False)
    assert attempted == [data_bundle.RELEASE_URL]


def test_fetch_corrupt_primary_tarball_fails_without_fallback(monkeypatch, tmp_path):
    # A 200 response whose body isn't a valid tar (e.g. an HTML error page) must
    # fail loudly rather than silently using a cross-project fallback.
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / f"v{DATA_VERSION}"))
    attempted = []
    monkeypatch.setattr(data_bundle, "_fetch_release_manifest", lambda source: None)

    def fake_download(url, root, *, verbose, release_manifest=None):
        attempted.append(url)
        if url == data_bundle.RELEASE_URL:
            raise tarfile.ReadError("not a gzip file")

    monkeypatch.setattr(data_bundle, "_download_and_extract", fake_download)
    with pytest.raises(RuntimeError, match="checksum-verified primary source"):
        data_bundle.fetch(verbose=False)
    assert attempted == [data_bundle.RELEASE_URL]


def test_fetch_raises_when_all_sources_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / f"v{DATA_VERSION}"))
    monkeypatch.setattr(data_bundle, "_fetch_release_manifest", lambda source: None)

    def always_404(url, root, *, verbose, release_manifest=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data_bundle, "_download_and_extract", always_404)
    with pytest.raises(RuntimeError, match="checksum-verified primary source"):
        data_bundle.fetch(verbose=False)


def test_fetch_primary_missing_manifest_fails_without_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("CANCERDATA_BUNDLED_DATA", str(tmp_path / f"v{DATA_VERSION}"))

    def missing_manifest(source):
        if source["name"] == "oncoref":
            raise data_bundle.BundleIntegrityError("missing release manifest/checksum")
        pytest.fail("must not fall back after primary integrity failure")

    monkeypatch.setattr(data_bundle, "_fetch_release_manifest", missing_manifest)
    monkeypatch.setattr(data_bundle, "_download_and_extract", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="integrity check failed"):
        data_bundle.fetch(verbose=False)


def test_cache_root_reuses_legacy_pirlygenes_dir(monkeypatch, tmp_path):
    # No env override; legacy cache already has THIS version, new root does not.
    monkeypatch.delenv("CANCERDATA_BUNDLED_DATA", raising=False)
    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA", raising=False)
    new_root = tmp_path / "oncoref" / "bundled_data"
    legacy_root = tmp_path / "pirlygenes" / "bundled_data"
    (legacy_root / f"v{DATA_VERSION}").mkdir(parents=True)
    monkeypatch.setattr(data_bundle, "_DEFAULT_CACHE_PARENT", new_root)
    monkeypatch.setattr(data_bundle, "_LEGACY_CACHE_PARENT", legacy_root)

    assert data_bundle.cache_root() == legacy_root  # reuse, no forced re-download
    # Once the new root has this version too, it wins.
    (new_root / f"v{DATA_VERSION}").mkdir(parents=True)
    assert data_bundle.cache_root() == new_root


def test_cache_root_defaults_to_oncoref_when_no_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("CANCERDATA_BUNDLED_DATA", raising=False)
    monkeypatch.delenv("PIRLYGENES_BUNDLED_DATA", raising=False)
    new_root = tmp_path / "oncoref" / "bundled_data"
    legacy_root = tmp_path / "pirlygenes" / "bundled_data"
    monkeypatch.setattr(data_bundle, "_DEFAULT_CACHE_PARENT", new_root)
    monkeypatch.setattr(data_bundle, "_LEGACY_CACHE_PARENT", legacy_root)
    assert data_bundle.cache_root() == new_root
