"""The isolated-environment tutorial path (NOTEBOOK_SPEC 2.2 §25.13): the kernel's `run_stage` helper and the carried
stage runner.

* The kernel-side test executes the generated notebook's own carrier and `run_stage` code (no model library needed):
  the carried files are written and hash-verified into a run directory, and an invalid BYOD zip stops the kernel with a
  RuntimeError that repeats the validator's refusal text.
* The CPU pre-flight (torch + diffusers + safetensors required) runs every stage in order against a stub UNet/MoVQ, a
  stub prior, a stub depth estimator, a stub scorer and synthetic records. Each stage builds its own pipeline, so
  everything a later stage uses crosses over through files in the run directory. It proves the stage plumbing and the
  hand-offs, not the model.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import synthetic_image, synthetic_records

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE


def _infrastructure_sources() -> tuple[str, str]:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    carrier = next(s for s in code if s.startswith("# @title Infrastructure: write and verify the carried"))
    install = next(s for s in code if s.startswith("# @title Infrastructure: install the locked runtime"))
    return carrier, install


def _kernel(tmp_path: Path) -> dict:
    """The kernel namespace after the carrier cell and the `run_stage` definition, with the current interpreter
    standing in for the isolated environment's Python."""
    carrier, install = _infrastructure_sources()
    run_root = tmp_path / "run"
    run_root.mkdir()
    namespace = {"ROOT": run_root, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": dict(os.environ)}
    exec("import hashlib\nimport json\nimport subprocess\n" + carrier, namespace)  # noqa: S102 - the notebook's own cell
    definition = install[install.index("def run_stage(") : install.index("def load_record(")]
    exec(definition, namespace)  # noqa: S102
    return namespace


def test_carrier_writes_and_verifies_every_carried_file(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    run_root = kernel["ROOT"]
    for dest, source in TEMPLATE["carried"].items():
        assert (run_root / dest).read_bytes() == (ROOT / source).read_text(encoding="utf-8").encode("utf-8"), dest
    assert kernel["NOTEBOOK_SOURCE"]["revision"] == "test-revision"


def _byod_zip(path: Path, *, columns: tuple[str, ...], image_side: int = 320) -> Path:
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(columns)
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(6):
            name = f"bird{i}.png"
            buffer = io.BytesIO()
            synthetic_image(width=image_side, height=image_side, seed=i).save(buffer, format="PNG")
            archive.writestr(name, buffer.getvalue())
            writer.writerow([f"r{i}", name, f"a photo of bird {i % 2}"][: len(columns)])
        archive.writestr("captions.csv", rows.getvalue())
    return path


@pytest.mark.parametrize(
    ("columns", "side", "refusal"),
    [
        (("id", "file"), 320, "captions.csv is missing columns ['caption']"),
        (("id", "file", "caption"), 200, "image sides must be within 256..4096 px"),
    ],
)
def test_invalid_byod_zip_raises_in_the_kernel_with_the_validator_message(tmp_path: Path, columns, side, refusal) -> None:
    kernel = _kernel(tmp_path)
    archive = _byod_zip(tmp_path / "byod.zip", columns=columns, image_side=side)
    with pytest.raises(RuntimeError) as caught:
        kernel["run_stage"]("prepare", "--byod", archive)
    assert refusal in str(caught.value)
    assert "Stage 'prepare' failed (exit 2)" in str(caught.value)
    error = json.loads((kernel["ROOT"] / "state" / "prepare.error.json").read_text(encoding="utf-8"))
    assert error["type"] == "ValueError" and refusal in error["message"]


def test_kernel_cells_install_nothing_into_the_kernel() -> None:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    kernel_code = "\n".join(
        c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")
    )
    assert not re.search(r"\bpip\b[^\n]*\binstall\b", kernel_code.replace("'pip', 'install', '--python', str(PYTHON)", ""))
    assert "sys.executable" not in kernel_code
    assert "'--require-hashes'" in kernel_code and "'--only-binary', ':all:'" in kernel_code


# ---- CPU pre-flight of the stage runner ----------------------------------------------------------------------------------


@pytest.fixture
def preflight(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("safetensors")
    import numpy as np

    from kandinsky_controlnet_depth_pipeline import (
        RESOLUTION,
        KandinskyDepthPipeline,
        lora_parameter_names,
        normalize_depth,
        split_dataset,
        validate_hint,
        validate_prompts,
    )
    from kandinsky_controlnet_depth_pipeline import pipeline as pl
    from test_adaptation import EMB, _StubMoVQ, _StubUNet

    stages = _load("tutorial_stages")
    monkeypatch.setattr(pl, "LORA_PARAMETERS", sum(p.numel() for n, p in _StubUNet().named_parameters() if n in lora_parameter_names(_StubUNet())))

    class _StubPrior:
        """Stands in for KandinskyV22PriorPipeline: seeded embeddings from the generator the pipeline passes."""

        def to(self, device):
            return self

        def set_progress_bar_config(self, **kwargs):
            return None

        def __call__(self, prompt, num_inference_steps, generator):
            return SimpleNamespace(image_embeds=torch.randn(1, EMB, generator=generator), negative_image_embeds=torch.randn(1, EMB, generator=generator))

    monkeypatch.setattr(pl, "build_prior", lambda weights_dir, *, dtype: _StubPrior())

    def fake_estimate_depth(self, images, *, batch_size=4):
        """Stands in for DPT: the brightness of the 512² centre crop, min-max normalised like the real path."""
        return [normalize_depth(np.asarray(pl.preprocess_image(im).convert("L"), dtype=np.float64)) for im in images]

    monkeypatch.setattr(KandinskyDepthPipeline, "estimate_depth", fake_estimate_depth)

    def fake_generate(self, prompts, hints, *, seed=0, steps=20, guidance_scale=4.0):
        from PIL import Image

        prompts = validate_prompts(prompts)
        assert len(hints) == len(prompts)
        state = sum(float(v.float().sum()) for k, v in self.unet.state_dict().items() if ".lora_" in k)
        images = []
        for index, (prompt, depth) in enumerate(zip(prompts, hints, strict=True)):
            self._embeds(prompt)  # refuses a prompt that is not in the cache, like the real path
            depth = validate_hint(depth)
            shade = int(abs(state * 1000 + seed + index + guidance_scale) % 60)
            # The generated layout follows the hint, so the stub estimator recovers it (depth fidelity near 1).
            pixels = np.clip(depth.astype(np.int16) // 2 + shade, 0, 255).astype(np.uint8)
            image = Image.fromarray(np.stack([pixels, pixels, 255 - pixels], axis=-1)).resize((RESOLUTION, RESOLUTION))
            images.append({"prompt": prompt, "seed": seed + index, "image": image, "hint": depth, "pixel_mean": float(pixels.mean())})
        return {"model": {"adapted": self.adapter is not None}, "steps": steps, "guidance_scale": float(guidance_scale), "images": images, "seconds": 0.0}

    monkeypatch.setattr(KandinskyDepthPipeline, "generate", fake_generate)

    def stub_pipeline(run):
        return KandinskyDepthPipeline(
            unet=_StubUNet(),
            movq=_StubMoVQ(),
            scheduler_config={"num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "beta_schedule": "linear", "prediction_type": "epsilon"},
            prior_dir=run.weights / "prior",
            depth_dir=run.weights / "depth",
            device="cpu",
            dtype=torch.float32,
            weights_dir=run.weights / "decoder",
            source="stub (CPU pre-flight)",
            use_lora=True,
        )

    def stub_from_artifact(run, artifact_dir):
        pipe = stub_pipeline(run)
        pipe.load_adapter(artifact_dir)
        return pipe

    class _StubScorer:
        identity = {"id": "stub-scorer", "revision": "0" * 40}

        def image_embeddings(self, images):
            rows = [torch.tensor([float(v) for v in im.convert("RGB").resize((2, 2)).tobytes()[:8]]) + 1.0 for im in images]
            return torch.nn.functional.normalize(torch.stack(rows), dim=-1)

        def text_embeddings(self, texts):
            rows = [torch.randn(8, generator=torch.Generator().manual_seed(len(t) * 31 + sum(map(ord, t)))) for t in texts]
            return torch.nn.functional.normalize(torch.stack(rows), dim=-1)

    captions = ("a photo of a red bird", "a photo of a blue bird")
    monkeypatch.setattr(stages, "load_pipeline", stub_pipeline)
    monkeypatch.setattr(stages, "load_from_artifact", stub_from_artifact)
    monkeypatch.setattr(stages, "load_scorer", lambda run, device: _StubScorer())
    monkeypatch.setattr(stages, "load_sample_splits", lambda run: split_dataset(synthetic_records(12, captions=captions), seed=0))
    monkeypatch.setattr(stages, "stage_snapshots", lambda run: [{"key": k, "id": k, "revision": "stub", "license": "stub", "files": 1, "fetched": []} for k in stages.SNAPSHOT_KEYS])
    run_root = tmp_path / "run"
    (run_root / "src").mkdir(parents=True)
    return stages, run_root, tmp_path / "weights"


def _run(stages, run_root: Path, weights: Path, stage: str, *options: str) -> None:
    code = stages.main(["--root", str(run_root), "--weights", str(weights), "--stage", stage, *options])
    error = run_root / "state" / f"{stage}.error.json"
    assert code == 0, error.read_text(encoding="utf-8") if error.exists() else f"{stage} exited {code}"


def test_cpu_preflight_runs_every_stage_through_files(preflight, capsys) -> None:
    stages, run_root, weights = preflight
    for stage, options in (
        ("weights", ()),
        ("prepare", ()),
        ("encode", ("--new-prompt", "a photo of a green bird on a fence")),
        ("frozen", ("--steps", "2")),
        ("activity", ("--hint", "flat")),
        ("adapt", ("--epochs", "2", "--lr", "1e-3")),
        ("evaluate", ()),
        ("reload", ()),
    ):
        _run(stages, run_root, weights, stage, *options)
    out = run_root / "outputs"
    stem = stages.STEM
    for name in (
        "weights.json",
        "prepare.json",
        "encode.json",
        "frozen.json",
        "activity.json",
        "adapt.json",
        "reload.json",
        f"{stem}_sample_captions.csv",
        f"{stem}_hints.jpg",
        f"{stem}_frozen_grid.jpg",
        f"{stem}_activity_grid.jpg",
        f"{stem}_adapted_grid.jpg",
        f"{stem}_evaluation_report.json",
        f"{stem}_adapter/adapter.safetensors",
        f"{stem}_adapter/manifest.json",
        f"{stem}_new_prompt_0.png",
        f"{stem}_new_prompt_1.png",
        f"{stem}_result.json",
    ):
        assert (out / name).is_file(), name
    assert (run_root / "state" / stages.PROMPT_CACHE).is_file() and (run_root / "state" / stages.HINT_CACHE).is_file()
    result = json.loads((out / f"{stem}_result.json").read_text(encoding="utf-8"))
    assert result["reload_parity"]["denoising_mse_diff"] < 1e-6 and result["reload_parity"]["mean_abs_pixel_diff"] < 1.0
    assert result["new_prompt"]["prompt"] == "a photo of a green bird on a fence"
    assert set(result["components"]) == {"prior", "depth_estimator"}
    assert set(result["provenance"]["snapshots"]) == {"decoder", "prior", "depth_estimator", "scorer"}
    report = json.loads((out / f"{stem}_evaluation_report.json").read_text(encoding="utf-8"))
    assert set(report["comparison"]) == {
        "denoising_mse_validation",
        "denoising_mse_test",
        "denoising_mse_test_by_timestep",
        "clip_prompt_similarity",
        "label_accuracy",
        "reference_similarity",
        "depth_correlation",
        "depth_aligned_mae",
    }
    assert set(report["comparison"]["depth_correlation"]) == {"frozen", "adapted", "mismatched_baseline"}
    assert report["adapted"]["loaded_from"] == "exported artifact, fresh process"
    assert report["activity_at_evaluation"]["hint_given"] == "flat"  # KCD-m5: labelled as the activity seen by evaluate
    assert set(report["real_photo_depth_check"]) >= {"depth_correlation", "depth_aligned_mae"}
    # The stub generator follows its hint, so the depth measures separate the own hint from the mismatched baseline.
    frozen_depth = report["frozen"]["depth_fidelity"]
    assert frozen_depth["depth_correlation"] > frozen_depth["mismatched_depth_correlation"]
    printed = capsys.readouterr().out
    assert "'prior_released': True" in printed and "'rejected'" in printed and "reload_parity" in printed


@pytest.mark.parametrize("hint", ["mismatched", "own"])
def test_cpu_preflight_activity_accepts_every_hint_choice(preflight, hint) -> None:
    stages, run_root, weights = preflight
    for stage, options in (("prepare", ()), ("encode", ()), ("frozen", ("--steps", "2")), ("activity", ("--hint", hint))):
        _run(stages, run_root, weights, stage, *options)
    summary = json.loads((run_root / "state" / "activity.json").read_text(encoding="utf-8"))
    assert summary["hint_given"] == hint
    if hint == "own":  # the control reproduces Section 6 exactly on the deterministic stub
        assert summary["depth_correlation_with_original_layout"]["this_activity"] == pytest.approx(summary["depth_correlation_with_original_layout"]["section_6_own_hint"], abs=1e-3)


def test_cpu_preflight_refuses_an_unknown_activity_hint(preflight) -> None:
    stages, run_root, weights = preflight
    assert stages.main(["--root", str(run_root), "--weights", str(weights), "--stage", "activity", "--hint", "blurred"]) == 2
    error = json.loads((run_root / "state" / "activity.error.json").read_text(encoding="utf-8"))
    assert error["message"] == "ACTIVITY_HINT must be 'flat', 'mismatched' or 'own'"


def test_cpu_preflight_refuses_a_stage_run_out_of_order(preflight) -> None:
    stages, run_root, weights = preflight
    assert stages.main(["--root", str(run_root), "--weights", str(weights), "--stage", "frozen"]) == 2
    error = json.loads((run_root / "state" / "frozen.error.json").read_text(encoding="utf-8"))
    assert "data.json is missing" in error["message"]
