"""Adaptation, evaluation and artifact tests on a stub hint-conditioned UNet/MoVQ (torch + diffusers + safetensors
required, no weights): the training loop, epoch selection, the transactional guarantee and the artifact round trip
with its refusals. Skipped where those libraries are not installed (the lightweight CI job)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("safetensors")

from conftest import synthetic_records  # noqa: E402
from kandinsky_controlnet_depth_pipeline import (  # noqa: E402
    LORA_TENSORS,
    RESOLUTION,
    KandinskyDepthPipeline,
    image_digest,
    lora_parameter_names,
)
from kandinsky_controlnet_depth_pipeline import pipeline as pl  # noqa: E402

RANK, DIM, EMB = 2, 6, 8


class _StubUNet(torch.nn.Module):
    """Names follow peft's layout; the output depends on the LoRA tensors and on the hint, so training moves them."""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        for block in range(2):
            for target in ("to_q", "to_k"):
                stem = f"down_blocks.{block}.attentions.0.{target}".replace(".", "__")
                self.register_parameter(f"{stem}__lora_A__default__weight", torch.nn.Parameter(torch.randn(RANK, DIM) * 0.1))
                self.register_parameter(f"{stem}__lora_B__default__weight", torch.nn.Parameter(torch.zeros(DIM, RANK)))
        self.seen_hints: list[tuple[int, ...]] = []

    def named_parameters(self, *args, **kwargs):  # type: ignore[override]
        for name, param in super().named_parameters(*args, **kwargs):
            yield name.replace("__", "."), param

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {k.replace("__", "."): v for k, v in super().state_dict(*args, **kwargs).items()}

    def load_state_dict(self, state, strict=True):  # type: ignore[override]
        return super().load_state_dict({k.replace(".", "__"): v for k, v in state.items()}, strict=strict)

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs, return_dict):
        hint = added_cond_kwargs["hint"]
        self.seen_hints.append(tuple(hint.shape))
        gain = 1.0
        params = dict(self.named_parameters())
        for name, param in params.items():
            if ".lora_B." in name:
                gain = gain + (param @ params[name.replace("lora_B", "lora_A")]).mean()
        pred = sample * gain + 0.01 * added_cond_kwargs["image_embeds"].mean() + 0.01 * hint.mean()
        return (torch.cat([pred, pred], dim=1),)  # 8 channels: 4 mean + 4 variance


class _StubMoVQ:
    def encode(self, pixels):
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, 4)[:, :3]
        return SimpleNamespace(latents=torch.cat([pooled, pooled[:, :1]], dim=1))


def _pipeline(monkeypatch, records) -> KandinskyDepthPipeline:
    unet = _StubUNet()
    names = lora_parameter_names(unet)
    monkeypatch.setattr(pl, "LORA_PARAMETERS", sum(dict(unet.named_parameters())[n].numel() for n in names))
    pipe = KandinskyDepthPipeline(
        unet=unet,
        movq=_StubMoVQ(),
        scheduler_config={
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
        },
        prior_dir=Path("unused"),
        depth_dir=Path("unused"),
        device="cpu",
        dtype=torch.float32,
        weights_dir=Path("unused"),
        source="stub",
        use_lora=True,
    )
    generator = torch.Generator().manual_seed(0)
    for prompt in (pl.NEGATIVE_PROMPT, "a photo of a red bird", "a photo of a blue bird"):
        pipe._prompt_cache[prompt] = {
            "image_embeds": torch.randn(EMB, generator=generator),
            "negative_image_embeds": torch.randn(EMB, generator=generator),
        }
    rng = np.random.default_rng(0)
    for record in records:
        pipe._hint_cache[image_digest(record["image"])] = rng.integers(0, 256, (RESOLUTION, RESOLUTION), dtype=np.uint8)
    return pipe


def test_stub_matches_the_contract_shape():
    names = lora_parameter_names(_StubUNet())
    assert len(names) == 8 and all(".lora_" in n for n in names) and LORA_TENSORS == 176


def test_evaluate_is_paired_seeded_and_hint_conditioned(monkeypatch):
    records = synthetic_records(4)
    pipe = _pipeline(monkeypatch, records)
    run1 = pipe.evaluate(records, seed=42)
    run2 = pipe.evaluate(records, seed=42)
    assert run1["denoising_mse"] == run2["denoising_mse"] and run1["per_record"] == run2["per_record"]
    assert set(run1["by_timestep"]) == {str(t) for t in pl.EVAL_TIMESTEPS}
    assert pipe.unet.seen_hints and pipe.unet.seen_hints[0][1:] == (3, RESOLUTION, RESOLUTION)


def test_evaluate_changes_when_the_hint_changes(monkeypatch):
    records = synthetic_records(4)
    pipe = _pipeline(monkeypatch, records)
    before = pipe.evaluate(records, seed=1)["denoising_mse"]
    for key in list(pipe._hint_cache):
        pipe._hint_cache[key] = np.full((RESOLUTION, RESOLUTION), 255, dtype=np.uint8)
    assert pipe.evaluate(records, seed=1)["denoising_mse"] != before


def test_adapt_records_history_and_keeps_the_best_epoch(monkeypatch):
    records = synthetic_records(4)
    pipe = _pipeline(monkeypatch, records)
    result = pipe.adapt(train=records, val=records[:2], epochs=2, lr=1e-3, seed=0)
    assert len(result["history"]) == 3 and result["history"][0]["train_loss"] is None
    best = result["history"][result["best_epoch"]]
    assert best["val_loss"] <= result["history"][0]["val_loss"]
    assert pipe.adapter is not None and pipe.adapter["epochs"] == 2 and pipe.adapter["seed"] == 0


def test_adapt_transactional_guarantee_on_error(monkeypatch):
    records = synthetic_records(4)
    pipe = _pipeline(monkeypatch, records)
    initial = {k: v.clone() for k, v in pipe.unet.state_dict().items()}

    def exploding_scheduler():
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(pipe, "_noise_scheduler", exploding_scheduler)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        pipe.adapt(train=records, epochs=1)
    assert pipe.adapter is None
    for k, v in pipe.unet.state_dict().items():
        assert torch.equal(v, initial[k]), f"weight {k} mutated after failed adapt()"


def test_artifact_round_trip_and_refusals(tmp_path, monkeypatch):
    records = synthetic_records(4)
    pipe = _pipeline(monkeypatch, records)
    pipe.adapt(train=records, epochs=1, seed=0)
    artifact = tmp_path / "adapter"
    manifest = pipe.save_artifact(artifact, metadata={"run_by": "test"})
    assert manifest["format"] == pl.ARTIFACT_FORMAT and manifest["metadata"]["run_by"] == "test"
    assert manifest["base_model"]["revision"] == pl.MODEL_REVISION
    assert manifest["conditioning"]["estimator"] == {"id": pl.DEPTH_ID, "revision": pl.DEPTH_REVISION}

    fresh = _pipeline(monkeypatch, records)
    loaded = fresh.load_adapter(artifact)
    assert loaded["weights"]["sha256"] == manifest["weights"]["sha256"] and fresh.adapter == manifest["adapter"]
    for k, v in pipe.unet.state_dict().items():
        assert torch.equal(v, fresh.unet.state_dict()[k])
    assert fresh.evaluate(records, seed=3)["denoising_mse"] == pipe.evaluate(records, seed=3)["denoising_mse"]

    manifest_path = artifact / pl.ARTIFACT_MANIFEST_NAME
    original = manifest_path.read_text(encoding="utf-8")
    for mutate, message in (
        (lambda m: m["base_model"].update(id="different/model"), "does not match"),
        (lambda m: m["conditioning"]["estimator"].update(revision="0" * 40), "depth estimator"),
        (lambda m: m["adapter"].update(trainable_names=["x"]), "LoRA scope"),
    ):
        edited = json.loads(original)
        mutate(edited)
        manifest_path.write_text(json.dumps(edited), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            _pipeline(monkeypatch, records).load_adapter(artifact)
    manifest_path.write_text(original, encoding="utf-8")

    weights = artifact / pl.ARTIFACT_WEIGHTS_NAME
    content = bytearray(weights.read_bytes())
    content[-1] ^= 0xFF
    weights.write_bytes(content)
    with pytest.raises(ValueError, match="weights digest mismatch"):
        _pipeline(monkeypatch, records).load_adapter(artifact)
