"""The real diffusers + peft code paths on a TINY randomly initialised hint-conditioned UNet and MoVQ (no pinned
weights, CPU): generation through `KandinskyV22ControlnetPipeline` with a hint, held-out evaluation, LoRA adaptation
with `peft`, and the artifact round trip. Skipped where torch, diffusers or peft are not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")
pytest.importorskip("safetensors")

from conftest import synthetic_records  # noqa: E402
from kandinsky_controlnet_depth_pipeline import image_digest  # noqa: E402
from kandinsky_controlnet_depth_pipeline import pipeline as pl  # noqa: E402

EMB = 16


@pytest.fixture
def tiny(monkeypatch):
    from diffusers import UNet2DConditionModel, VQModel
    from peft import inject_adapter_in_model

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=8,
        out_channels=8,
        down_block_types=("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
        up_block_types=("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
        mid_block_type="UNetMidBlock2DSimpleCrossAttn",
        block_out_channels=(32, 64),
        layers_per_block=1,
        attention_head_dim=8,
        cross_attention_dim=32,
        encoder_hid_dim=EMB,
        encoder_hid_dim_type="image_proj",
        addition_embed_type="image_hint",
        addition_embed_type_num_heads=8,
        resnet_time_scale_shift="scale_shift",
        norm_num_groups=8,
    )
    movq = VQModel(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        num_vq_embeddings=32,
        vq_embed_dim=4,
    )
    for param in unet.parameters():
        param.requires_grad_(False)
    inject_adapter_in_model(pl._lora_config(), unet, adapter_name="default")
    names = set(pl.lora_parameter_names(unet))
    for name, param in unet.named_parameters():
        if name in names:
            param.data = param.data.float()
    monkeypatch.setattr(pl, "LORA_TENSORS", len(names))
    monkeypatch.setattr(pl, "LORA_PARAMETERS", sum(p.numel() for n, p in unet.named_parameters() if n in names))
    pipe = pl.KandinskyDepthPipeline(
        unet=unet.eval(),
        movq=movq.eval(),
        scheduler_config={
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "prediction_type": "epsilon",
            "variance_type": "learned_range",
            "clip_sample": False,
        },
        prior_dir=Path("unused"),
        depth_dir=Path("unused"),
        device="cpu",
        dtype=torch.float32,
        weights_dir=Path("unused"),
        source="tiny",
        use_lora=True,
    )
    records = synthetic_records(6)
    generator = torch.Generator().manual_seed(0)
    for prompt in {pl.NEGATIVE_PROMPT, *(r["caption"] for r in records)}:
        pipe._prompt_cache[prompt] = {
            "image_embeds": torch.randn(EMB, generator=generator),
            "negative_image_embeds": torch.randn(EMB, generator=generator),
        }
    rng = np.random.default_rng(0)
    for record in records:
        pipe._hint_cache[image_digest(record["image"])] = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    return pipe, records


def _pixels(entry):
    return np.asarray(entry["image"], dtype=np.float32)


def test_generation_is_seeded_and_follows_the_hint(tiny):
    pipe, records = tiny
    hint = pipe.depth_hint(records[0])
    first = pipe.generate([records[0]["caption"]], [hint], seed=5, steps=2)["images"][0]
    again = pipe.generate([records[0]["caption"]], [hint], seed=5, steps=2)["images"][0]
    flat = pipe.generate([records[0]["caption"]], [pl.flat_hint()], seed=5, steps=2)["images"][0]
    assert first["image"].size == (512, 512)
    assert np.array_equal(_pixels(first), _pixels(again))
    assert not np.array_equal(_pixels(first), _pixels(flat))
    with pytest.raises(ValueError, match="one depth map per prompt"):
        pipe.generate([records[0]["caption"]], [], steps=2)


def test_adapt_and_artifact_round_trip_through_peft(tiny, tmp_path):
    pipe, records = tiny
    result = pipe.adapt(records[:4], records[4:], epochs=2, lr=1e-3, seed=0)
    kept = result["history"][result["best_epoch"]]["val_loss"]
    assert kept <= result["history"][0]["val_loss"]
    manifest = pipe.save_artifact(tmp_path / "adapter")
    trained = {k: v.clone() for k, v in pipe.unet.state_dict().items() if k in set(manifest["adapter"]["trainable_names"])}
    for name, param in pipe.unet.named_parameters():
        if ".lora_B." in name:
            param.data.zero_()
    pipe.load_adapter(tmp_path / "adapter")
    assert all(torch.equal(trained[k], v) for k, v in pipe.unet.state_dict().items() if k in trained)
    assert pipe.evaluate(records[4:], seed=0)["denoising_mse"] == pytest.approx(kept, abs=1e-6)
