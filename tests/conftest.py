import builtins

import numpy as np
import pytest
from PIL import Image

MODEL_LIBRARIES = {"torch", "diffusers", "transformers", "peft", "accelerate", "safetensors", "huggingface_hub", "sentencepiece"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_image(*, width: int = 640, height: int = 480, seed: int = 0) -> Image.Image:
    """A smooth, seeded RGB image (no model semantics)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([np.sin(x / 40 + seed), np.cos(y / 30 - seed), np.sin((x + y) / 60)], axis=-1)
    noise = rng.normal(0, 0.05, base.shape)
    array = ((base + noise + 1.5) / 3.0 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


CAPTIONS = ("a photo of a red bird", "a photo of a blue bird")


def synthetic_records(n: int = 8, *, captions: tuple[str, ...] = CAPTIONS, seed: int = 0):
    """`{id, image, caption}` records cycling through `captions`, each image distinct."""
    return [
        {"id": f"rec-{i:03d}", "image": synthetic_image(seed=seed + i), "caption": captions[i % len(captions)]}
        for i in range(n)
    ]
