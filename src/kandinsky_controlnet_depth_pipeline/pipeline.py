"""Kandinsky 2.2 ControlNet-depth (`kandinsky-community/kandinsky-2-2-controlnet-depth` + `kandinsky-2-2-prior`)
DIMER pipeline: verified snapshots, depth-conditioned generation (image -> depth hint -> prompt + hint -> image),
held-out denoising-loss evaluation, and bounded LoRA fine-tuning of the UNet with a portable adapter.

Kandinsky 2.2 ControlNet-depth is the Kandinsky 2.2 latent diffusion decoder retrained to accept a spatial hint:
* A **diffusion prior** (1.03 B parameters, shared with text-to-image Kandinsky 2.2) maps a text prompt to a
  predicted CLIP ViT-G/14 image embedding;
* A **UNet2DConditionModel** (1,253,429,212 parameters, `addition_embed_type="image_hint"`) denoises a 4-channel
  latent conditioned on that embedding and on a 3 x 512 x 512 depth hint, which its `input_hint_block` encodes to
  a 4-channel map concatenated with the latent (8 input channels);
* A **MoVQ VQModel** (67.8 M parameters) decodes the latent to a 512 x 512 RGB image.

The depth hint is the relative depth map of a 512 x 512 image estimated by `Intel/dpt-large` (DPT, the default
model of the `transformers` `depth-estimation` task, pinned here by revision and loaded explicitly), min-max
normalised to 0..255 as the upstream model card does, repeated over three channels and divided by 255.

Four pinned snapshots are used, each verified against its DIMER manifest before loading:
* The **decoder** (`MODEL_ID`, UNet + MoVQ, 5.29 GB safetensors). The upstream `main` branch ships only pickled
  `.bin` weights; the safetensors pinned here come from the repository's open, unmerged conversion pull request #5,
  addressed by its immutable commit (`MODEL_REVISION`). `tools/verify_conversion.py` compares them with `main`.
* The **prior** (`PRIOR_ID`, 10.57 GB safetensors);
* The **depth estimator** (`DEPTH_ID`, 1.37 GB safetensors, Apache-2.0);
* The **scorer** used for evaluation (`SCORER_ID`, an MIT-licensed CLIP ViT-B/32, 605 MB).

Every file is safetensors or plain JSON/text: nothing is unpickled and no Hub-hosted code is executed. The
adaptation contract attaches LoRA to the attention projections of the UNet (rank 8, 176 tensors, 1,646,592
parameters); the hint encoder, MoVQ, prior and depth estimator remain frozen. Everything model-related is imported
lazily so snapshot verification and input validation run before torch/diffusers/transformers are imported.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "kandinsky-community/kandinsky-2-2-controlnet-depth"
MODEL_REVISION = "08632524a2b7e3bed39a7901d92cdd07e37544d0"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "kandinsky-2-2-controlnet-depth"
ARTIFACT_FORMAT = "org.valcorza.kandinsky-controlnet-depth.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_DIR = _WEIGHTS_ROOT / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Shared prior snapshot: diffusion prior + CLIP ViT-G text/image encoders
PRIOR_ID = "kandinsky-community/kandinsky-2-2-prior"
PRIOR_REVISION = "9fc51ad5732afc5d031724219d22e6c42179c5a8"
PRIOR_LICENSE = "apache-2.0"
PRIOR_KEY = "kandinsky-2-2-prior"
PRIOR_WEIGHTS_DIR = _WEIGHTS_ROOT / PRIOR_KEY

# Depth estimator that produces the hint (the transformers `depth-estimation` default, pinned and loaded explicitly)
DEPTH_ID = "Intel/dpt-large"
DEPTH_REVISION = "bc15f29aa3a80d532f2ed650b5e16ac48d8958f9"
DEPTH_LICENSE = "apache-2.0"
DEPTH_KEY = "dpt-large"
DEPTH_WEIGHTS_DIR = _WEIGHTS_ROOT / DEPTH_KEY

# Evaluation scorer (CLIP ViT-B/32 safetensors)
SCORER_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SCORER_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
SCORER_LICENSE = "mit"
SCORER_KEY = "clip-vit-b-32-laion2b"
SCORER_WEIGHTS_DIR = _WEIGHTS_ROOT / SCORER_KEY

# Architecture and contract facts (read from the pinned safetensors headers)
UNET_PARAMETERS = 1_253_429_212
UNET_TENSORS = 740
MOVQ_PARAMETERS = 67_832_495
MOVQ_TENSORS = 431
PRIOR_PARAMETERS = 1_026_225_920
PRIOR_TENSORS = 338
DEPTH_PARAMETERS = 341_850_305
DEPTH_TENSORS = 458

RESOLUTION = 512
LATENT_CHANNELS = 4
HINT_CHANNELS = 3
MOVQ_SCALE = 8
MAX_CAPTION_CHARS = 1_000
MIN_IMAGE_SIDE = 256
MAX_IMAGE_SIDE = 4_096
MIN_TRAIN_RECORDS = 4
MAX_RECORDS = 2_000
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 4.0
MAX_STEPS = 100
MAX_GUIDANCE = 20.0
NUM_TRAIN_TIMESTEPS = 1000
EVAL_TIMESTEPS: tuple[int, ...] = (100, 300, 500, 700, 900)
LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGETS: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
LORA_TENSORS = 176
LORA_PARAMETERS = 1_646_592
NEGATIVE_PROMPT = ""


# --------------------------------------------------------------------------------------------------
# manifests and staging (four pinned snapshots)
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_seed(prompt: str) -> int:
    """Seed for the prior's sampler: the first 4 bytes of SHA-256(prompt), masked to 31 bits.

    Stable across processes, unlike ``hash(prompt)``, which Python salts per interpreter.
    """
    return int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if not entry["path"].endswith((".safetensors", ".json", ".model", ".txt", ".md")):
            raise ValueError(f"{entry['path']}: unexpected file type in a code-free snapshot")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the ControlNet-depth decoder snapshot against its DIMER manifest (size + SHA-256 of every file)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def verify_prior_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the prior snapshot against its own DIMER manifest."""
    root = Path(path) if path is not None else PRIOR_WEIGHTS_DIR
    return _verify_manifest(root, PRIOR_ID, PRIOR_REVISION)


def verify_depth_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the depth-estimator snapshot against its own DIMER manifest."""
    root = Path(path) if path is not None else DEPTH_WEIGHTS_DIR
    return _verify_manifest(root, DEPTH_ID, DEPTH_REVISION)


def verify_scorer_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the CLIP scorer snapshot against its own manifest."""
    root = Path(path) if path is not None else SCORER_WEIGHTS_DIR
    return _verify_manifest(root, SCORER_ID, SCORER_REVISION)


def _hub_download(relative_path: str, root: Path, model_id: str, revision: str) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(model_id, relative_path, revision=revision, local_dir=str(root))


def _stage_missing(
    root: Path,
    model_id: str,
    revision: str,
    allow_download: bool,
    downloader: Callable[[str, Path], None] | None,
) -> list[str]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != model_id or manifest.get("revision") != revision:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {model_id}@{revision}; refusing to stage"
        )
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"refusing to stage from a non-commit revision {revision!r}")
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {revision}"
        )
    fetch = downloader or (lambda rel, dst: _hub_download(rel, dst, model_id, revision))
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch decoder-manifest entries that are absent locally, at the pinned 40-hex commit only."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _stage_missing(root, MODEL_ID, MODEL_REVISION, allow_download, downloader)


def stage_missing_prior_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch prior-manifest entries that are absent locally."""
    root = Path(path) if path is not None else PRIOR_WEIGHTS_DIR
    return _stage_missing(root, PRIOR_ID, PRIOR_REVISION, allow_download, downloader)


def stage_missing_depth_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch depth-estimator entries that are absent locally."""
    root = Path(path) if path is not None else DEPTH_WEIGHTS_DIR
    return _stage_missing(root, DEPTH_ID, DEPTH_REVISION, allow_download, downloader)


def stage_missing_scorer_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch CLIP scorer entries that are absent locally."""
    root = Path(path) if path is not None else SCORER_WEIGHTS_DIR
    return _stage_missing(root, SCORER_ID, SCORER_REVISION, allow_download, downloader)


# --------------------------------------------------------------------------------------------------
# captioned-image records, depth hints and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": "{id, image, caption}: a PIL image (or a path to one) and the caption that describes it",
    "image_side": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "resolution": RESOLUTION,
    "preprocessing": (
        f"each image is resized so its shorter side is {RESOLUTION} px and centre-cropped to {RESOLUTION}×{RESOLUTION}; "
        "the crop is reported per record. Nothing else is changed"
    ),
    "hint": (
        f"a {RESOLUTION}×{RESOLUTION} uint8 relative depth map (larger = nearer), computed by {DEPTH_ID} on the "
        "preprocessed image and min-max normalised to 0..255; the model receives it as three identical channels / 255"
    ),
    "caption_chars": [1, MAX_CAPTION_CHARS],
    "records": [MIN_TRAIN_RECORDS, MAX_RECORDS],
    "generation": {"steps": [1, MAX_STEPS], "guidance_scale": [1.0, MAX_GUIDANCE], "size": RESOLUTION},
    "validation": (
        "record shape, image decodability and side limits, caption length, duplicate ids and hint shape/dtype only. "
        "Nothing checks that a caption describes its image, that the images are photographs, that a depth map is "
        "plausible, or that the prompt is one the model can render -- any RGB image with any string is accepted"
    ),
}


def _check_record(record: Any, index: int) -> dict[str, Any]:
    from PIL import Image

    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/caption")
    for key in ("id", "image", "caption"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid, image, caption = record["id"], record["image"], record["caption"]
    if not isinstance(rid, str) or not rid or len(rid) > 64:
        raise ValueError(f"{label}: id must be a non-empty string of at most 64 characters")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"{label}: image sides must be within {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px, got {image.size}")
    if not isinstance(caption, str) or not caption.strip() or len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"{label}: caption must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
    item = {"id": rid, "image": image.convert("RGB"), "caption": caption.strip()}
    for key in ("label", "common_name", "scientific_name", "observer", "inat_photo_id", "inat_observation_url", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def image_digest(image: Any) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same photo matches."""
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes()).hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], image_digest(r["image"]), r["caption"]] for r in records]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_TRAIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a captioned-image dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, caption} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    crops = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        width, height = item["image"].size
        if width != height:
            crops += 1
        checked.append(item)
    sides = [min(r["image"].size) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "n_captions": len({r["caption"] for r in checked}),
        "shorter_side": {"min": min(sides), "max": max(sides)},
        "centre_cropped": crops,
        "resolution": RESOLUTION,
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record; returns its id, size, the crop it will get and the caption length."""
    item = _check_record(record, 0)
    width, height = item["image"].size
    short = min(width, height)
    scale = RESOLUTION / short
    return {
        "id": item["id"],
        "size": (width, height),
        "resized_to": (round(width * scale), round(height * scale)),
        "centre_crop": (RESOLUTION, RESOLUTION),
        "caption_chars": len(item["caption"]),
    }


def validate_prompts(prompts: Sequence[str]) -> list[str]:
    """Generation prompts: non-empty strings within the caption limit; duplicates are allowed."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence) or not prompts:
        raise ValueError("prompts must be a non-empty list of strings")
    out = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_CAPTION_CHARS:
            raise ValueError(f"prompts[{index}] must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
        out.append(prompt.strip())
    return out


def validate_hint(depth: Any, *, label: str = "hint") -> Any:
    """A depth hint is a RESOLUTION × RESOLUTION uint8 array (0 = farthest, 255 = nearest in relative depth)."""
    import numpy as np

    if not isinstance(depth, np.ndarray):
        raise ValueError(f"{label} must be a numpy array, got {type(depth).__name__}")
    if depth.shape != (RESOLUTION, RESOLUTION):
        raise ValueError(f"{label} must have shape ({RESOLUTION}, {RESOLUTION}), got {depth.shape}")
    if depth.dtype != np.uint8:
        raise ValueError(f"{label} must be uint8 (0..255), got {depth.dtype}")
    return depth


def normalize_depth(depth: Any) -> Any:
    """Min-max normalise a relative depth map to uint8 0..255, as the transformers depth-estimation pipeline does
    (`(d - min) / (max - min) * 255`, truncated). A constant map becomes all zeros."""
    import numpy as np

    array = np.asarray(depth, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"depth must be a 2-D map, got shape {array.shape}")
    low, high = float(array.min()), float(array.max())
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("depth map contains non-finite values")
    if high - low <= 0.0:
        return np.zeros(array.shape, dtype=np.uint8)
    return ((array - low) / (high - low) * 255).astype(np.uint8)


def hint_array(depth: Any) -> Any:
    """The model's hint for one depth map: float32 (3, H, W) in [0, 1], the map repeated over three channels."""
    import numpy as np

    validate_hint(depth)
    return np.repeat((depth.astype(np.float32) / 255.0)[None], HINT_CHANNELS, axis=0)


def flat_hint(value: int = 128) -> Any:
    """A constant depth map (no spatial information): the 'change one thing' control of the tutorial."""
    import numpy as np

    if not 0 <= int(value) <= 255:
        raise ValueError("value must be in 0..255")
    return np.full((RESOLUTION, RESOLUTION), int(value), dtype=np.uint8)


def preprocess_image(image: Any) -> Any:
    """Resize the shorter side to RESOLUTION and centre-crop; returns a PIL RGB image of RESOLUTION²."""
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = RESOLUTION / min(width, height)
    new = (max(RESOLUTION, round(width * scale)), max(RESOLUTION, round(height * scale)))
    resized = rgb.resize(new, Image.Resampling.BICUBIC)
    left = (new[0] - RESOLUTION) // 2
    top = (new[1] - RESOLUTION) // 2
    return resized.crop((left, top, left + RESOLUTION, top + RESOLUTION))


# --------------------------------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------------------------------


def _lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, init_lora_weights="gaussian", target_modules=list(LORA_TARGETS))


def lora_parameter_names(unet: Any) -> list[str]:
    """The exact tensor set the adaptation contract may change on a UNet built with the adapter."""
    return sorted(name for name, _param in unet.named_parameters() if ".lora_A." in name or ".lora_B." in name)


def build_unet(weights_dir: Path, *, dtype: Any, use_lora: bool) -> Any:
    """Load the pinned hint-conditioned UNet from the verified snapshot; optionally attach the (untrained) LoRA."""
    import torch
    from diffusers import UNet2DConditionModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = UNet2DConditionModel.from_pretrained(str(weights_dir), subfolder="unet", torch_dtype=dtype, use_safetensors=True)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != UNET_PARAMETERS or len(model.state_dict()) != UNET_TENSORS:
        raise ValueError(
            f"unet has {n_params} parameters in {len(model.state_dict())} tensors; "
            f"expected {UNET_PARAMETERS} / {UNET_TENSORS}"
        )
    if model.config.addition_embed_type != "image_hint":
        raise ValueError(f"unet addition_embed_type is {model.config.addition_embed_type!r}, expected 'image_hint'")
    for param in model.parameters():
        param.requires_grad_(False)
    if use_lora:
        from peft import inject_adapter_in_model

        inject_adapter_in_model(_lora_config(), model, adapter_name="default")
        names = lora_parameter_names(model)
        if len(names) != LORA_TENSORS:
            raise ValueError(f"adapter attached {len(names)} LoRA tensors, expected {LORA_TENSORS}")
        name_set = set(names)
        for name, param in model.named_parameters():
            if name in name_set:
                param.data = param.data.to(torch.float32)
                param.requires_grad_(False)
    model.eval()
    return model


def build_prior(weights_dir: Path, *, dtype: Any) -> Any:
    """Load the pinned Prior pipeline from the verified prior snapshot."""
    from diffusers import KandinskyV22PriorPipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prior_pipe = KandinskyV22PriorPipeline.from_pretrained(str(weights_dir), torch_dtype=dtype, use_safetensors=True)
    return prior_pipe


def build_depth_estimator(weights_dir: Path, *, dtype: Any) -> tuple[Any, Any]:
    """Load the pinned DPT depth estimator and its image processor from the verified snapshot (never by task default)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForDepthEstimation.from_pretrained(str(weights_dir), dtype=dtype, use_safetensors=True)
        processor = AutoImageProcessor.from_pretrained(str(weights_dir))
    for param in model.parameters():
        param.requires_grad_(False)
    return model.eval(), processor


def _select_device(device: str | None) -> str:
    import torch

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    return device


@dataclass
class KandinskyDepthPipeline:
    """Depth-conditioned text-to-image generation and bounded LoRA fine-tuning for Kandinsky 2.2 ControlNet-depth."""

    unet: Any
    movq: Any
    scheduler_config: dict[str, Any]
    prior_dir: Path
    depth_dir: Path
    device: str
    dtype: Any
    weights_dir: Path
    source: str
    use_lora: bool
    adapter: dict[str, Any] | None = None
    _prior: Any = field(default=None, repr=False)
    _depth: Any = field(default=None, repr=False)
    _prompt_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _hint_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        prior_dir: str | Path | None = None,
        depth_dir: str | Path | None = None,
        allow_download: bool = False,
        use_lora: bool = False,
    ) -> KandinskyDepthPipeline:
        """Stage and verify the decoder, prior and depth snapshots, then load the UNet, MoVQ and scheduler config."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        prior = Path(prior_dir) if prior_dir is not None else PRIOR_WEIGHTS_DIR
        depth = Path(depth_dir) if depth_dir is not None else DEPTH_WEIGHTS_DIR
        stage_missing_files(root, allow_download=allow_download)
        stage_missing_prior_files(prior, allow_download=allow_download)
        stage_missing_depth_files(depth, allow_download=allow_download)
        verify_snapshot(root)
        verify_prior_snapshot(prior)
        verify_depth_snapshot(depth)

        import torch
        from diffusers import DDPMScheduler, VQModel

        chosen = _select_device(device)
        dtype = torch.float16 if chosen.startswith("cuda") else torch.float32

        unet = build_unet(root, dtype=dtype, use_lora=use_lora).to(chosen)
        movq = VQModel.from_pretrained(str(root), subfolder="movq", torch_dtype=dtype, use_safetensors=True).to(chosen).eval()
        for param in movq.parameters():
            param.requires_grad_(False)
        scheduler = DDPMScheduler.from_pretrained(str(root), subfolder="scheduler")

        return cls(
            unet=unet,
            movq=movq,
            scheduler_config=dict(scheduler.config),
            prior_dir=prior,
            depth_dir=depth,
            device=chosen,
            dtype=dtype,
            weights_dir=root,
            source="local-snapshot (verified safetensors)",
            use_lora=use_lora,
        )

    # ---- prompts and prior caching -------------------------------------------------------------------

    def encode_prompts(
        self,
        prompts: Sequence[str],
        *,
        prior_device: str | None = None,
        num_inference_steps: int = 25,
    ) -> dict[str, Any]:
        """Encode every distinct prompt into the prompt cache with the Prior pipeline (seeded by `prompt_seed`)."""
        import torch

        wanted = [p for p in dict.fromkeys([NEGATIVE_PROMPT, *validate_prompts(list(prompts))]) if p not in self._prompt_cache]
        if not wanted:
            return {"encoded": 0, "cached": len(self._prompt_cache)}

        p_device = prior_device or self.device
        p_dtype = torch.float16 if p_device.startswith("cuda") else torch.float32

        if self._prior is None:
            started = time.perf_counter()
            self._prior = build_prior(self.prior_dir, dtype=p_dtype).to(p_device)
            self._prior.set_progress_bar_config(disable=True)
            load_seconds = round(time.perf_counter() - started, 1)
        else:
            load_seconds = 0.0

        started = time.perf_counter()
        with torch.inference_mode():
            for prompt in wanted:
                gen = torch.Generator(device=p_device).manual_seed(prompt_seed(prompt))
                out = self._prior(prompt=prompt, num_inference_steps=num_inference_steps, generator=gen)
                self._prompt_cache[prompt] = {
                    "image_embeds": out.image_embeds[0].to("cpu", self.dtype),
                    "negative_image_embeds": out.negative_image_embeds[0].to("cpu", self.dtype),
                }

        return {
            "encoded": len(wanted),
            "cached": len(self._prompt_cache),
            "prior_device": p_device,
            "prior_dtype": str(p_dtype).replace("torch.", ""),
            "load_seconds": load_seconds,
            "encode_seconds": round(time.perf_counter() - started, 1),
        }

    def export_prompt_cache(self) -> dict[str, Any]:
        """Export prompt embeddings (CPU tensors keyed by prompt)."""
        return {
            k: {"image_embeds": v["image_embeds"].clone(), "negative_image_embeds": v["negative_image_embeds"].clone()}
            for k, v in self._prompt_cache.items()
        }

    def import_prompt_cache(self, cache: Mapping[str, Mapping[str, Any]]) -> int:
        """Adopt prompt embeddings exported by `export_prompt_cache`."""
        for prompt, entry in cache.items():
            if "image_embeds" not in entry or "negative_image_embeds" not in entry:
                raise ValueError(f"prompt cache entry for {prompt[:40]!r} missing required embedding keys")
            self._prompt_cache[prompt] = {
                "image_embeds": entry["image_embeds"],
                "negative_image_embeds": entry["negative_image_embeds"],
            }
        return len(self._prompt_cache)

    def release_prior(self) -> bool:
        """Drop the loaded Prior pipeline to free GPU memory."""
        had = self._prior is not None
        self._prior = None
        self._free_memory()
        return had

    def _free_memory(self) -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embeds(self, prompt: str) -> tuple[Any, Any]:
        if prompt not in self._prompt_cache:
            raise ValueError(f"prompt not encoded; call encode_prompts([...]) first: {prompt[:60]!r}")
        entry = self._prompt_cache[prompt]
        return entry["image_embeds"].to(self.device, self.dtype), entry["negative_image_embeds"].to(self.device, self.dtype)

    # ---- depth hints ---------------------------------------------------------------------------------

    def estimate_depth(self, images: Sequence[Any], *, batch_size: int = 4) -> list[Any]:
        """Relative depth of each image with the pinned DPT estimator: every image is preprocessed to RESOLUTION²,
        the predicted depth is resized back to RESOLUTION² (bicubic) and min-max normalised to uint8."""
        import numpy as np
        import torch

        if self._depth is None:
            d_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            model, processor = build_depth_estimator(self.depth_dir, dtype=d_dtype)
            self._depth = (model.to(self.device), processor, d_dtype)
        model, processor, d_dtype = self._depth
        out = []
        prepared = [preprocess_image(image) for image in images]
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            with torch.inference_mode():
                predicted = model(pixel_values=inputs["pixel_values"].to(self.device, d_dtype)).predicted_depth
                resized = torch.nn.functional.interpolate(
                    predicted[:, None].float(), size=(RESOLUTION, RESOLUTION), mode="bicubic", align_corners=False
                )[:, 0]
            out.extend(normalize_depth(np.asarray(d.cpu().numpy())) for d in resized)
        return out

    def compute_hints(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Estimate and cache the depth hint of every record image not yet cached (keyed by pixel digest)."""
        checked = validate_dataset(records, min_records=1)["records"]
        wanted = [r for r in checked if image_digest(r["image"]) not in self._hint_cache]
        started = time.perf_counter()
        if wanted:
            for record, depth in zip(wanted, self.estimate_depth([r["image"] for r in wanted]), strict=True):
                self._hint_cache[image_digest(record["image"])] = depth
        return {
            "computed": len(wanted),
            "cached": len(self._hint_cache),
            "estimator": {"id": DEPTH_ID, "revision": DEPTH_REVISION},
            "seconds": round(time.perf_counter() - started, 2),
        }

    def depth_hint(self, record: Mapping[str, Any]) -> Any:
        """The cached hint of one record (call `compute_hints` first)."""
        key = image_digest(record["image"])
        if key not in self._hint_cache:
            raise ValueError(f"no depth hint for record {record.get('id')!r}; call compute_hints([...]) first")
        return self._hint_cache[key]

    def export_hint_cache(self) -> dict[str, Any]:
        """Export depth hints (uint8 arrays keyed by record pixel digest)."""
        return {k: v.copy() for k, v in self._hint_cache.items()}

    def import_hint_cache(self, cache: Mapping[str, Any]) -> int:
        """Adopt depth hints exported by `export_hint_cache`."""
        for key, depth in cache.items():
            self._hint_cache[key] = validate_hint(depth, label=f"hint {key[:12]}")
        return len(self._hint_cache)

    def release_depth_estimator(self) -> bool:
        """Drop the loaded depth estimator."""
        had = self._depth is not None
        self._depth = None
        self._free_memory()
        return had

    def _hint_tensor(self, depths: Sequence[Any]) -> Any:
        import numpy as np
        import torch

        return torch.from_numpy(np.stack([hint_array(d) for d in depths])).to(self.device, self.dtype)

    # ---- generation ------------------------------------------------------------------------------------

    def _diffusers_pipeline(self) -> Any:
        from diffusers import DDPMScheduler, KandinskyV22ControlnetPipeline

        scheduler = DDPMScheduler.from_config(self.scheduler_config)
        return KandinskyV22ControlnetPipeline(unet=self.unet, scheduler=scheduler, movq=self.movq)

    def generate(
        self,
        prompts: Sequence[str],
        hints: Sequence[Any],
        *,
        seed: int = 0,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
    ) -> dict[str, Any]:
        """Generate one RESOLUTION² image per (prompt, hint) pair with classifier-free guidance; image i uses seed + i."""
        import numpy as np
        import torch

        prompts = validate_prompts(prompts)
        if isinstance(hints, str) or len(hints) != len(prompts):
            raise ValueError(f"hints must be a list with one depth map per prompt ({len(prompts)})")
        for index, depth in enumerate(hints):
            validate_hint(depth, label=f"hints[{index}]")
        if not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be an int in 1..{MAX_STEPS}")
        if not 1.0 <= float(guidance_scale) <= MAX_GUIDANCE:
            raise ValueError(f"guidance_scale must be in 1..{MAX_GUIDANCE}")

        if any(p not in self._prompt_cache for p in [NEGATIVE_PROMPT, *prompts]):
            self.encode_prompts(prompts)

        pipe = self._diffusers_pipeline()
        pipe.set_progress_bar_config(disable=True)
        started = time.perf_counter()
        images = []
        for index, (prompt, depth) in enumerate(zip(prompts, hints, strict=True)):
            img_emb, neg_emb = self._embeds(prompt)
            generator = torch.Generator(device=self.device).manual_seed(seed + index)
            with torch.inference_mode():
                out = pipe(
                    image_embeds=img_emb[None],
                    negative_image_embeds=neg_emb[None],
                    hint=self._hint_tensor([depth]),
                    height=RESOLUTION,
                    width=RESOLUTION,
                    num_inference_steps=steps,
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    output_type="pil",
                )
            image = out.images[0]
            pixel_mean = round(float(np.asarray(image).mean()), 3)
            images.append({"prompt": prompt, "seed": seed + index, "image": image, "hint": depth, "pixel_mean": pixel_mean})
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "steps": steps,
            "guidance_scale": float(guidance_scale),
            "size": (RESOLUTION, RESOLUTION),
            "scheduler": "DDPMScheduler (upstream config)",
            "precision": str(self.dtype).replace("torch.", ""),
            "images": images,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- latents and the denoising loss --------------------------------------------------------------

    def _latents(self, records: Sequence[Mapping[str, Any]], *, seed: int) -> Any:
        """MoVQ-encode preprocessed images to latents (B, 4, 64, 64)."""
        import numpy as np
        import torch

        arrays = [np.asarray(preprocess_image(r["image"]), dtype=np.float32) / 127.5 - 1.0 for r in records]
        pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device, self.dtype)
        with torch.no_grad():
            latents = self.movq.encode(pixels).latents
        return latents.to(self.dtype)

    def _noise_scheduler(self) -> Any:
        from diffusers import DDPMScheduler

        return DDPMScheduler.from_config(self.scheduler_config)

    def _predict_noise(self, noisy: Any, timesteps: Any, image_embeds: Any, hint: Any) -> Any:
        out = self.unet(
            sample=noisy,
            timestep=timesteps,
            encoder_hidden_states=None,
            added_cond_kwargs={"image_embeds": image_embeds, "hint": hint},
            return_dict=False,
        )[0]
        return out[:, :LATENT_CHANNELS]

    def _ensure_hints(self, records: Sequence[Mapping[str, Any]]) -> None:
        if any(image_digest(r["image"]) not in self._hint_cache for r in records):
            self.compute_hints(records)

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, seed: int = 0, batch_size: int = 4) -> dict[str, Any]:
        """Held-out denoising MSE: every record is MoVQ-encoded, noised at each of EVAL_TIMESTEPS, and the UNet's
        noise prediction -- conditioned on the record's caption embedding and its own depth hint -- is scored."""
        import torch

        checked = validate_dataset(records, min_records=1)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        self.encode_prompts([r["caption"] for r in checked])
        self._ensure_hints(checked)
        scheduler = self._noise_scheduler()
        started = time.perf_counter()
        per_timestep: dict[int, list[float]] = {t: [] for t in EVAL_TIMESTEPS}
        per_record: dict[str, float] = {}
        self.unet.eval()
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            latents = self._latents(batch, seed=seed + start)
            image_embeds = torch.stack([self._embeds(r["caption"])[0] for r in batch])
            hint = self._hint_tensor([self.depth_hint(r) for r in batch])
            record_losses = [0.0] * len(batch)
            for t in EVAL_TIMESTEPS:
                generator = torch.Generator(device="cpu").manual_seed(seed * 1_000 + t + start)
                noise = torch.randn(latents.shape, generator=generator).to(self.device, self.dtype)
                timesteps = torch.full((len(batch),), t, device=self.device, dtype=torch.long)
                noisy = scheduler.add_noise(latents.float(), noise.float(), timesteps).to(self.dtype)
                use_amp = self.dtype == torch.float16
                autocast = torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp)
                with torch.inference_mode(), autocast:
                    pred = self._predict_noise(noisy, timesteps, image_embeds, hint)
                loss = ((pred.float() - noise.float()) ** 2).mean(dim=(1, 2, 3))
                for i, value in enumerate(loss.tolist()):
                    per_timestep[t].append(value)
                    record_losses[i] += value / len(EVAL_TIMESTEPS)
            for record, value in zip(batch, record_losses, strict=True):
                per_record[record["id"]] = round(value, 6)
        by_t = {str(t): round(sum(v) / len(v), 6) for t, v in per_timestep.items()}
        mean = sum(per_record.values()) / len(per_record)
        return {
            "metric": "denoising_mse (noise-prediction MSE over the latent, mean over records and EVAL_TIMESTEPS)",
            "n_records": len(checked),
            "timesteps": list(EVAL_TIMESTEPS),
            "seed": seed,
            "denoising_mse": round(mean, 6),
            "by_timestep": by_t,
            "per_record": per_record,
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- adaptation ------------------------------------------------------------------------------------

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 4,
        lr: float = 1e-4,
        batch_size: int = 1,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded LoRA fine-tuning on the hint-conditioned noise-prediction objective: AdamW on the 176 LoRA tensors
        only, float16 autocast on CUDA. Epoch 0 records the frozen model; the lowest validation-loss epoch is kept."""
        if not self.use_lora:
            raise ValueError("adapt() needs a pipeline built with use_lora=True")
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be an int in 1..8")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None

        import torch

        self.encode_prompts([r["caption"] for r in train_checked] + ([r["caption"] for r in val_checked] if val_checked else []))
        self._ensure_hints(train_checked + (val_checked or []))
        torch.manual_seed(seed)
        started = time.perf_counter()
        model = self.unet
        names = lora_parameter_names(model)
        name_set = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in name_set)
        params = [p for n, p in model.named_parameters() if n in name_set]
        n_trainable = sum(p.numel() for p in params)
        if n_trainable != LORA_PARAMETERS:
            raise ValueError(f"{n_trainable} trainable parameters, expected {LORA_PARAMETERS}")
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        use_amp = self.dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scheduler = self._noise_scheduler()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        latents_by_id = {}
        for start in range(0, len(train_checked), 4):
            batch = train_checked[start : start + 4]
            encoded = self._latents(batch, seed=seed + 10_000 + start)
            for record, latent in zip(batch, encoded, strict=True):
                latents_by_id[record["id"]] = latent

        try:
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "note": "frozen model (LoRA at initialisation: B = 0)"}
            entry["val_loss"] = self.evaluate(val_checked, seed=seed)["denoising_mse"] if val_checked else None
            history.append(entry)
            if progress:
                progress(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            best_epoch = 0
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    latents = torch.stack([latents_by_id[r["id"]] for r in batch])
                    image_embeds = torch.stack([self._embeds(r["caption"])[0] for r in batch])
                    hint = self._hint_tensor([self.depth_hint(r) for r in batch])
                    noise = torch.randn(latents.shape, generator=generator).to(self.device, self.dtype)
                    timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (len(batch),), generator=generator).to(self.device)
                    noisy = scheduler.add_noise(latents.float(), noise.float(), timesteps).to(self.dtype)
                    with torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp):
                        pred = self._predict_noise(noisy, timesteps, image_embeds, hint)
                    loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                    optimiser.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses)}
                entry["val_loss"] = self.evaluate(val_checked, seed=seed)["denoising_mse"] if val_checked else None
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
                    best_epoch = epoch
        except BaseException:
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise

        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "method": "LoRA (peft)",
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "targets": list(LORA_TARGETS),
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "optimizer": "AdamW (weight_decay 0, grad-norm clip 1.0)",
            "precision": "float16 autocast + GradScaler" if use_amp else "float32",
            "history": history,
        }
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val if best_val != math.inf else None,
            "adapter": self.adapter,
            "steps": n_steps,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- artifact export and reload ------------------------------------------------------------------

    def save_artifact(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Save the current adapter state as a portable safetensors artifact with a manifest."""
        import safetensors.torch

        if self.adapter is None:
            raise ValueError("no adapter attached; call adapt() or load_adapter() first")
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.unet.state_dict().items() if k in names}
        weights_file = dest / ARTIFACT_WEIGHTS_NAME
        safetensors.torch.save_file(tensors, weights_file)
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY},
            "prior": {"id": PRIOR_ID, "revision": PRIOR_REVISION},
            "conditioning": {
                "type": "depth hint",
                "estimator": {"id": DEPTH_ID, "revision": DEPTH_REVISION},
                "hint": INPUT_SCHEMA["hint"],
            },
            "adapter": self.adapter,
            "weights": {
                "file": ARTIFACT_WEIGHTS_NAME,
                "bytes": weights_file.stat().st_size,
                "sha256": _sha256_file(weights_file),
                "n_tensors": len(tensors),
                "n_parameters": sum(t.numel() for t in tensors.values()),
            },
            "metadata": dict(metadata) if metadata else {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (dest / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    save_adapter = save_artifact

    def load_adapter(self, path: str | Path) -> dict[str, Any]:
        """Load an adapter saved by `save_artifact`: format, base model, estimator, tensor scope and digest are
        checked before the tensors are deserialised."""
        import safetensors.torch

        if not self.use_lora:
            raise ValueError("load_adapter() needs a pipeline built with use_lora=True")
        source = Path(path)
        manifest_file = source / ARTIFACT_MANIFEST_NAME
        weights_file = source / ARTIFACT_WEIGHTS_NAME
        if not manifest_file.is_file() or not weights_file.is_file():
            raise FileNotFoundError(f"not an adapter directory: {source}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"unexpected artifact format: {manifest.get('format')!r}")
        base = manifest.get("base_model", {})
        if base.get("id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
            raise ValueError(f"adapter base model {base} does not match {MODEL_ID}@{MODEL_REVISION}")
        estimator = manifest.get("conditioning", {}).get("estimator", {})
        if estimator.get("id") != DEPTH_ID or estimator.get("revision") != DEPTH_REVISION:
            raise ValueError(f"adapter depth estimator {estimator} does not match {DEPTH_ID}@{DEPTH_REVISION}")
        expected = set(lora_parameter_names(self.unet))
        recorded = set(manifest.get("adapter", {}).get("trainable_names", []))
        if recorded != expected:
            raise ValueError("adapter tensor names do not match the LoRA scope of this UNet")
        digest = _sha256_file(weights_file)
        if digest != manifest["weights"]["sha256"]:
            raise ValueError(f"weights digest mismatch: {digest} != manifest {manifest['weights']['sha256']}")

        tensors = safetensors.torch.load_file(weights_file)
        if set(tensors) != expected:
            raise ValueError("adapter file tensors do not match the LoRA scope of this UNet")
        current = dict(self.unet.state_dict())
        current.update({k: v.to(self.device, current[k].dtype) for k, v in tensors.items()})
        self.unet.load_state_dict(current, strict=True)
        self.adapter = manifest["adapter"]
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        weights_dir: str | Path | None = None,
        prior_dir: str | Path | None = None,
        depth_dir: str | Path | None = None,
        device: str | None = None,
    ) -> KandinskyDepthPipeline:
        """Instantiate a fresh pipeline from the verified snapshots and load an adapter from its artifact directory."""
        pipe = cls.from_pretrained(
            weights_dir=weights_dir, prior_dir=prior_dir, depth_dir=depth_dir, device=device, use_lora=True
        )
        pipe.load_adapter(artifact_path)
        return pipe
