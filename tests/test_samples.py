"""Offline tests for the pinned sample corpus, the seeded splits, BYOD loaders and CSV export."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from conftest import synthetic_image, synthetic_records
from kandinsky_controlnet_depth_pipeline import (
    CAPTION_TEMPLATE,
    CORPUS_BASE_URL,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    SPECIES,
    build_sample_dataset,
    caption_for,
    check_split_disjoint,
    dataset_manifest,
    fetch_corpus,
    load_byod_dataset,
    read_corpus,
    sample_prompts,
    split_dataset,
    write_dataset_csv,
)
from kandinsky_controlnet_depth_pipeline import samples as sm


def test_pinned_records_are_consistent(forbid_model_imports):
    assert len(SAMPLE_RECORDS) == 60
    counts = {}
    for rid, label, photo_id, obs_id, user, size, digest, ext in SAMPLE_RECORDS:
        counts[label] = counts.get(label, 0) + 1
        assert label in SPECIES and rid.startswith(label) and ext in ("jpg", "jpeg")
        assert isinstance(photo_id, int) and isinstance(obs_id, int) and user and size > 0 and len(digest) == 64
    assert counts == {label: 10 for label in SPECIES}
    assert len({r[2] for r in SAMPLE_RECORDS}) == 60
    assert sum(r[5] for r in SAMPLE_RECORDS) == sm.CORPUS_BYTES
    assert sm.photo_url(1, "jpeg") == f"{CORPUS_BASE_URL}1/medium.jpeg"
    with pytest.raises(ValueError, match="extension"):
        sm.photo_url(1, "png")


def test_caption_template(forbid_model_imports):
    expected = CAPTION_TEMPLATE.format(common_name="House Finch", scientific_name="Haemorhous mexicanus")
    assert caption_for("house_finch") == expected
    with pytest.raises(ValueError, match="unknown species"):
        caption_for("pigeon")


def _fake_corpus(tmp_path, monkeypatch, n=12):
    """Replace the record table with `n` synthetic photos whose bytes the fake bucket serves."""
    records = []
    served = {}
    labels = list(SPECIES)
    for i in range(n):
        label = labels[i % len(labels)]
        buffer = io.BytesIO()
        synthetic_image(seed=i).save(buffer, format="JPEG")
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        records.append((f"{label}-{i:02d}", label, 1000 + i, 2000 + i, f"user{i}", len(data), digest, "jpg"))
        served[sm.photo_url(1000 + i)] = data
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(records))
    return served


def test_fetch_corpus_verifies_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    served = _fake_corpus(tmp_path, monkeypatch)
    calls = []

    def fetcher(url):
        calls.append(url)
        return served[url]

    files = fetch_corpus(cache_dir=tmp_path / "cache", fetcher=fetcher)
    assert len(files) == 12 and len(calls) == 12
    fetch_corpus(cache_dir=tmp_path / "cache", fetcher=fetcher)
    assert len(calls) == 12
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")
    records = read_corpus(files)
    assert all(r["caption"] == caption_for(r["label"]) for r in records)
    assert records[0]["inat_observation_url"].endswith("/2000") and records[0]["image"].mode == "RGB"


def test_build_sample_dataset_is_seeded_and_disjoint(tmp_path, monkeypatch, forbid_model_imports):
    served = _fake_corpus(tmp_path, monkeypatch, n=60)
    records = read_corpus(fetch_corpus(cache_dir=tmp_path / "cache", fetcher=lambda url: served[url]))
    splits = build_sample_dataset(records)
    assert {k: len(v) for k, v in splits.items()} == {k: v * 6 for k, v in SAMPLE_SPLIT.items()}
    again = build_sample_dataset(records)
    assert [r["source_id"] for r in splits["test"]] == [r["source_id"] for r in again["test"]]
    assert check_split_disjoint(splits) == {"train": 36, "validation": 12, "test": 12}
    assert len(sample_prompts(splits["test"])) == 6
    manifest = dataset_manifest(splits)
    assert manifest["splits"]["train"]["n_records"] == 36 and len(manifest["digest"]) == 64
    with pytest.raises(ValueError, match="only 10 records available"):
        build_sample_dataset(records, sizes={"train": 9, "validation": 1, "test": 1})


def test_split_dataset_groups_by_caption(forbid_model_imports):
    records = synthetic_records(20)
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 20 and len(splits["train"]) >= 4 and splits["test"]
    check_split_disjoint(splits)
    duplicated = records + [{**records[0], "id": "dup"}]
    assert sum(len(v) for v in split_dataset(duplicated).values()) == 20
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, test_fraction=0.9)


def test_byod_directory_and_zip_loaders(tmp_path, forbid_model_imports):
    records = synthetic_records(5)
    folder = tmp_path / "byod"
    folder.mkdir()
    for record in records:
        record["image"].save(folder / f"{record['id']}.jpg")
    write_dataset_csv([{**r, "id": r["id"]} for r in records], folder / "captions.csv")
    loaded = load_byod_dataset(folder)
    assert [r["id"] for r in loaded] == [r["id"] for r in records] and loaded[0]["caption"] == records[0]["caption"]
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"nested/{path.name}")
    assert len(load_byod_dataset(archive)) == 5
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "no table")
    with pytest.raises(ValueError, match="captions.csv"):
        load_byod_dataset(bad)
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "missing.tar")
    (folder / "captions.csv").write_text("id,file\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(folder)
