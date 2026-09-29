"""Offline tests for the four snapshot manifests, staging, the dataset and hint contracts and prompt seeding.
No model library is imported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import synthetic_records
from kandinsky_controlnet_depth_pipeline import (
    DEPTH_ID,
    DEPTH_REVISION,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    PRIOR_ID,
    PRIOR_REVISION,
    RESOLUTION,
    SCORER_REVISION,
    dataset_digest,
    flat_hint,
    hint_array,
    normalize_depth,
    preprocess_image,
    prompt_seed,
    stage_missing_depth_files,
    stage_missing_files,
    validate_dataset,
    validate_hint,
    validate_prompts,
    verify_depth_snapshot,
    verify_prior_snapshot,
    verify_scorer_snapshot,
    verify_snapshot,
)
from kandinsky_controlnet_depth_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_FILE = "unet/diffusion_pytorch_model.safetensors"


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": "k",
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and committed manifests -------------------------------------------------------------------


def test_identity_is_immutable_and_committed_manifests_agree():
    snapshots = (
        (pl.MODEL_KEY, MODEL_ID, MODEL_REVISION),
        (pl.PRIOR_KEY, PRIOR_ID, PRIOR_REVISION),
        (pl.DEPTH_KEY, DEPTH_ID, DEPTH_REVISION),
        (pl.SCORER_KEY, pl.SCORER_ID, SCORER_REVISION),
    )
    for key, model_id, revision in snapshots:
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
        manifest = json.loads((ROOT / "weights" / key / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert (manifest["format"], manifest["formatVersion"]) == ("dimer_hf_snapshot", 1)
        assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (model_id, revision, key)
        assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
        assert all(len(e["sha256"]) == 64 for e in manifest["files"])
        assert not any(e["path"].endswith((".bin", ".pt", ".pth", ".ckpt", ".pickle", ".py")) for e in manifest["files"])
    decoder = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {e["path"] for e in decoder["files"]} == {
        "README.md",
        "model_index.json",
        "movq/config.json",
        "movq/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    }
    depth = json.loads((ROOT / "weights" / pl.DEPTH_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {e["path"] for e in depth["files"]} == {"README.md", "config.json", "preprocessor_config.json", "model.safetensors"}


def test_movq_is_byte_identical_to_the_text_to_image_decoder_movq():
    """The MoVQ safetensors in the pinned conversion commit has the digest of the MoVQ in the pinned
    kandinsky-2-2-decoder snapshot (43a5860f..., 271,380,364 bytes)."""
    decoder = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    movq = next(e for e in decoder["files"] if e["path"] == "movq/diffusion_pytorch_model.safetensors")
    assert movq["sha256"].startswith("43a5860f") and movq["bytes"] == 271_380_364


def test_committed_small_files_match_their_manifests():
    for key in (pl.MODEL_KEY, pl.PRIOR_KEY, pl.DEPTH_KEY, pl.SCORER_KEY):
        root = ROOT / "weights" / key
        manifest = json.loads((root / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            path = root / entry["path"]
            if entry["path"].endswith(".safetensors"):
                continue
            data = path.read_bytes()
            assert len(data) == entry["bytes"], entry["path"]
            assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]


def test_unet_config_is_the_hint_conditioned_variant():
    config = json.loads((ROOT / "weights" / pl.MODEL_KEY / "unet" / "config.json").read_text(encoding="utf-8"))
    assert config["addition_embed_type"] == "image_hint"
    assert config["in_channels"] == 8 and config["out_channels"] == 8


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    manifest = _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"unet/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    assert verify_snapshot(root)["files"] == manifest["files"]
    (root / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"tensorz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)
    (root / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"tensors-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / "unet" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_snapshot_refuses_pickled_weights(tmp_path, forbid_model_imports):
    """The upstream main branch ships .bin weights; a manifest that lists one is refused."""
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"unet/diffusion_pytorch_model.bin": b"pickle"})
    with pytest.raises(ValueError, match="unexpected file type"):
        verify_snapshot(root)


def test_each_snapshot_has_its_own_identity(tmp_path, forbid_model_imports):
    prior = tmp_path / "prior"
    _write_snapshot(prior, PRIOR_ID, PRIOR_REVISION, {"prior/config.json": b"{}"})
    assert verify_prior_snapshot(prior)["modelId"] == PRIOR_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(prior)
    depth = tmp_path / "depth"
    _write_snapshot(depth, DEPTH_ID, DEPTH_REVISION, {"config.json": b"{}"})
    assert verify_depth_snapshot(depth)["modelId"] == DEPTH_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_prior_snapshot(depth)
    scorer = tmp_path / "scorer"
    _write_snapshot(scorer, pl.SCORER_ID, SCORER_REVISION, {"config.json": b"{}"})
    assert verify_scorer_snapshot(scorer)["modelId"] == pl.SCORER_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_depth_snapshot(scorer)


def test_stage_missing_files_fetches_only_absent_entries_at_the_pinned_commit(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"unet/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    (root / "unet" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"tensors")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [WEIGHTS_FILE]
    assert calls == [WEIGHTS_FILE]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_default_downloader_passes_the_full_commit_sha(tmp_path, monkeypatch, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {WEIGHTS_FILE: b"tensors"})
    (root / WEIGHTS_FILE).unlink()
    seen = []
    monkeypatch.setattr(pl, "_hub_download", lambda rel, dst, model_id, revision: seen.append((rel, model_id, revision)))
    stage_missing_files(root, allow_download=True)
    assert seen == [(WEIGHTS_FILE, MODEL_ID, "08632524a2b7e3bed39a7901d92cdd07e37544d0")]
    assert "main" not in seen[0][2] and "refs/pr" not in seen[0][2]


def test_staging_refuses_a_non_commit_revision(tmp_path, monkeypatch, forbid_model_imports):
    monkeypatch.setattr(pl, "DEPTH_REVISION", "main")
    root = tmp_path / "depth"
    _write_snapshot(root, DEPTH_ID, "main", {"config.json": b"{}"})
    with pytest.raises(ValueError, match="non-commit revision"):
        stage_missing_depth_files(root, allow_download=True, downloader=lambda rel, dst: None)


# --- dataset, prompt and hint validation ---------------------------------------------------------------------


def test_validate_dataset_enforces_limits_and_shapes(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["resolution"] == RESOLUTION
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset("not a list")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(records[:2])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(records + [records[0]])
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        validate_dataset([{**records[0], "id": ""}, *records[1:]])
    with pytest.raises(ValueError, match="caption must be a non-empty string"):
        validate_dataset([{**records[0], "caption": "   "}, *records[1:]])
    too_small = [{**records[0], "image": Image.new("RGB", (MIN_IMAGE_SIDE - 1, 512))}, *records[1:]]
    with pytest.raises(ValueError, match="image sides must be within"):
        validate_dataset(too_small)
    too_large = [{**records[0], "image": Image.new("RGB", (MAX_IMAGE_SIDE + 1, 512))}, *records[1:]]
    with pytest.raises(ValueError, match="image sides must be within"):
        validate_dataset(too_large)


def test_validate_prompts_enforces_strings_and_length(forbid_model_imports):
    assert validate_prompts([" a prompt ", "another"]) == ["a prompt", "another"]
    with pytest.raises(ValueError, match="non-empty list of strings"):
        validate_prompts([])
    with pytest.raises(ValueError, match="non-empty string"):
        validate_prompts([""])


def test_hint_contract(forbid_model_imports):
    depth = np.tile(np.arange(RESOLUTION, dtype=np.float32), (RESOLUTION, 1))
    hint = normalize_depth(depth)
    assert hint.dtype == np.uint8 and hint.shape == (RESOLUTION, RESOLUTION)
    assert hint.min() == 0 and hint.max() == 255
    assert np.array_equal(normalize_depth(np.ones((4, 4))), np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="2-D"):
        normalize_depth(np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        normalize_depth(np.array([[0.0, np.nan]]))
    array = hint_array(hint)
    assert array.shape == (3, RESOLUTION, RESOLUTION) and array.dtype == np.float32
    assert float(array.max()) == 1.0 and np.array_equal(array[0], array[2])
    assert validate_hint(flat_hint()) is not None and int(flat_hint(7)[0, 0]) == 7
    with pytest.raises(ValueError, match="shape"):
        validate_hint(np.zeros((256, 256), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        validate_hint(np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32))
    with pytest.raises(ValueError, match="numpy array"):
        validate_hint([[0]])
    with pytest.raises(ValueError, match="0..255"):
        flat_hint(300)


def test_prompt_seed_is_stable_across_processes(forbid_model_imports):
    """The prior's sampler seed must not depend on Python's per-process string-hash salt."""
    import subprocess
    import sys

    prompts = ["a photo of a blue jay", "", "étude"]
    expected = [prompt_seed(p) for p in prompts]
    assert all(0 <= s < 2**31 for s in expected)
    assert expected[0] == int.from_bytes(hashlib.sha256(prompts[0].encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF
    code = (
        "import json, sys; from kandinsky_controlnet_depth_pipeline import prompt_seed; "
        "print(json.dumps([prompt_seed(p) for p in json.loads(sys.argv[1])]))"
    )
    import os

    src_path = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"), os.environ.get("PYTHONPATH", "")])
    for hash_seed in ("1", "2"):
        out = subprocess.run(
            [sys.executable, "-c", code, json.dumps(prompts)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": src_path},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert json.loads(out) == expected


def test_no_python_hash_call_in_the_package():
    """No module calls the builtin `hash()`, whose string results change with every interpreter."""
    import ast

    for path in (ROOT / "src" / "kandinsky_controlnet_depth_pipeline").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "hash", f"{path.name}:{node.lineno} calls hash()"


def test_preprocess_image_crops_and_scales(forbid_model_imports):
    for img in (Image.new("RGB", (800, 600), color=(10, 20, 30)), Image.new("RGB", (600, 800), color=(40, 50, 60))):
        out = preprocess_image(img)
        assert out.size == (RESOLUTION, RESOLUTION) and out.mode == "RGB"


def test_dataset_digest_is_deterministic(forbid_model_imports):
    assert dataset_digest(synthetic_records(4)) == dataset_digest(synthetic_records(4))
    assert len(dataset_digest(synthetic_records(4))) == 64
