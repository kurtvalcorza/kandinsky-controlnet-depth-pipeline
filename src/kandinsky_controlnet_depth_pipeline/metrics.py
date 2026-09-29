"""Evaluation of depth-conditioned generations: CLIP prompt alignment and similarity to held-out real photographs,
and depth fidelity -- how closely the depth of a generated image follows the hint it was generated from.

The CLIP scorer is the pinned `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` snapshot (MIT), loaded from its verified
directory; it never takes part in generation or training. Depth fidelity re-uses the pinned depth estimator that
produced the hints, so the measurement and the conditioning share one definition of depth. Every number here is
tutorial sample-sanity evidence, not a benchmark, and not a human judgement of image quality.

Depth fidelity, defined precisely (`depth_agreement`): let ``h`` be the hint and ``g`` the estimator's depth of the
generated image, both RESOLUTION x RESOLUTION maps min-max normalised to 0..255 and divided by 255.

* ``depth_correlation`` -- the Pearson correlation of the flattened ``g`` and ``h`` (0.0 when either is constant).
  It ignores any affine change of depth scale, which suits a relative depth estimator.
* ``depth_aligned_mae`` -- the mean absolute difference ``mean(|s*g + t - h|)`` after the least-squares scale ``s``
  and shift ``t`` that best align ``g`` to ``h``; in hint units (0 = identical, 1 = the full depth range).

Both are also computed against a *mismatched* hint -- the hint of the next image in the batch that has a different
hint -- which is the value an image that ignored its own hint would be expected to reach (the chance baseline).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import (
    DEPTH_ID,
    DEPTH_REVISION,
    SCORER_ID,
    SCORER_REVISION,
    SCORER_WEIGHTS_DIR,
    stage_missing_scorer_files,
    validate_hint,
    verify_scorer_snapshot,
)


def _projected(features: Any) -> Any:
    """The projected CLIP embedding as a tensor: `transformers` 5 returns the encoder's model output (the projection
    written into `pooler_output`) from `get_text_features` / `get_image_features`, earlier releases the tensor."""
    return features if hasattr(features, "shape") else features.pooler_output


class ClipScorer:
    """Frozen CLIP image/text embedder on a verified snapshot."""

    def __init__(self, *, device: str | None = None, weights_dir: str | Path | None = None, allow_download: bool = False) -> None:
        root = Path(weights_dir) if weights_dir is not None else SCORER_WEIGHTS_DIR
        stage_missing_scorer_files(root, allow_download=allow_download)
        verify_scorer_snapshot(root)
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(str(root), torch_dtype=torch.float32).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(str(root))
        self.weights_dir = root
        self.identity = {"id": SCORER_ID, "revision": SCORER_REVISION}

    def image_embeddings(self, images: Sequence[Any]) -> Any:
        import torch

        out = []
        with torch.inference_mode():
            for start in range(0, len(images), 16):
                batch = self.processor(images=[im.convert("RGB") for im in images[start : start + 16]], return_tensors="pt")
                features = _projected(self.model.get_image_features(pixel_values=batch["pixel_values"].to(self.device)))
                out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        return torch.cat(out)

    def text_embeddings(self, texts: Sequence[str]) -> Any:
        import torch

        with torch.inference_mode():
            batch = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            features = _projected(
                self.model.get_text_features(
                    input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device)
                )
            )
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()


def score_generations(
    scorer: ClipScorer,
    generated: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score `generated` records (`{prompt, image}`) with three CLIP measures:

    * `clip_prompt_similarity` -- mean cosine(image, its prompt) x 100 (prompt alignment);
    * `label_accuracy` -- the fraction of images whose nearest prompt among the distinct prompts is their own
      (zero-shot classification among the dataset's captions; argmax rule);
    * `reference_similarity` -- mean cosine(image, mean embedding of the real `references` with the same caption)
      x 100 when references are given (how close the generations sit to the held-out photographs)."""
    if not generated:
        raise ValueError("no generated records to score")
    started = time.perf_counter()
    prompts = list(dict.fromkeys(str(g["prompt"]) for g in generated))
    text = scorer.text_embeddings(prompts)
    images = scorer.image_embeddings([g["image"] for g in generated])
    index = {p: i for i, p in enumerate(prompts)}
    sims = images @ text.T  # (n_images, n_prompts)
    own = [float(sims[i, index[str(g["prompt"])]]) for i, g in enumerate(generated)]
    nearest = sims.argmax(dim=1).tolist()
    correct = [nearest[i] == index[str(g["prompt"])] for i, g in enumerate(generated)]
    per_image = []
    ref_means: dict[str, Any] = {}
    if references:
        ref_images = scorer.image_embeddings([r["image"] for r in references])
        for caption in prompts:
            rows = [i for i, r in enumerate(references) if str(r["caption"]) == caption]
            if rows:
                mean = ref_images[rows].mean(dim=0)
                ref_means[caption] = mean / mean.norm()
    ref_scores = []
    for i, g in enumerate(generated):
        entry = {
            "prompt": str(g["prompt"]),
            "seed": g.get("seed"),
            "clip_prompt_similarity": round(own[i] * 100, 3),
            "nearest_prompt": prompts[nearest[i]],
            "correct": bool(correct[i]),
        }
        if str(g["prompt"]) in ref_means:
            value = float(images[i] @ ref_means[str(g["prompt"])]) * 100
            entry["reference_similarity"] = round(value, 3)
            ref_scores.append(value)
        per_image.append(entry)
    report: dict[str, Any] = {
        "scorer": dict(scorer.identity),
        "n_images": len(generated),
        "n_prompts": len(prompts),
        "clip_prompt_similarity": round(sum(own) / len(own) * 100, 3),
        "label_accuracy": round(sum(correct) / len(correct), 4),
        "decision_rule": "argmax cosine similarity over the distinct prompts (no threshold)",
        "per_image": per_image,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if ref_scores:
        report["reference_similarity"] = round(sum(ref_scores) / len(ref_scores), 3)
        report["n_references"] = len(references or [])
    return report


def real_photo_baseline(scorer: ClipScorer, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The same three CLIP measures on real photographs (image = the record's photo, prompt = its caption,
    references = the records): the ceiling a generator could reach on these metrics."""
    generated = [{"prompt": r["caption"], "image": r["image"], "seed": None} for r in records]
    report = score_generations(scorer, generated, references=records)
    report["note"] = "real held-out photographs scored as if generated (references include each photo itself)"
    return report


# --------------------------------------------------------------------------------------------------
# depth fidelity (numpy only; the estimator runs in the pipeline)
# --------------------------------------------------------------------------------------------------


def depth_agreement(predicted: Any, target: Any) -> dict[str, float]:
    """`depth_correlation` and `depth_aligned_mae` between two uint8 depth maps (see the module docstring)."""
    import numpy as np

    g = validate_hint(predicted, label="predicted").astype(np.float64).ravel() / 255.0
    h = validate_hint(target, label="target").astype(np.float64).ravel() / 255.0
    g_std, h_std = float(g.std()), float(h.std())
    correlation = 0.0 if g_std == 0.0 or h_std == 0.0 else float(np.mean((g - g.mean()) * (h - h.mean())) / (g_std * h_std))
    if g_std == 0.0:
        aligned = np.full_like(h, h.mean())
    else:
        scale = float(np.mean((g - g.mean()) * (h - h.mean())) / (g_std**2))
        aligned = scale * (g - g.mean()) + h.mean()
    mae = float(np.mean(np.abs(aligned - h)))
    return {"depth_correlation": round(correlation, 6), "depth_aligned_mae": round(mae, 6)}


def depth_fidelity_from_maps(predicted: Sequence[Any], hints: Sequence[Any]) -> dict[str, Any]:
    """Depth fidelity of `predicted` maps against their own `hints` and against a mismatched hint (the next entry
    whose hint differs, cyclically); the mismatched value is None when every hint is identical."""
    import numpy as np

    if not predicted or len(predicted) != len(hints):
        raise ValueError("predicted and hints must be non-empty lists of equal length")
    n = len(hints)
    per_image = []
    for i, (pred, own) in enumerate(zip(predicted, hints, strict=True)):
        entry: dict[str, Any] = {"index": i, **depth_agreement(pred, own)}
        other = next((hints[(i + k) % n] for k in range(1, n) if not np.array_equal(hints[(i + k) % n], own)), None)
        if other is not None:
            mismatched = depth_agreement(pred, other)
            entry["mismatched_depth_correlation"] = mismatched["depth_correlation"]
            entry["mismatched_depth_aligned_mae"] = mismatched["depth_aligned_mae"]
        per_image.append(entry)

    def _mean(key: str) -> float | None:
        values = [e[key] for e in per_image if key in e]
        return round(sum(values) / len(values), 6) if values else None

    return {
        "n_images": n,
        "depth_correlation": _mean("depth_correlation"),
        "depth_aligned_mae": _mean("depth_aligned_mae"),
        "mismatched_depth_correlation": _mean("mismatched_depth_correlation"),
        "mismatched_depth_aligned_mae": _mean("mismatched_depth_aligned_mae"),
        "definition": (
            "Pearson correlation, and mean absolute difference after a least-squares scale-and-shift alignment, "
            "between the estimator's depth of each generated image and its hint (both min-max normalised); "
            "'mismatched' uses another image's hint as the chance baseline"
        ),
        "per_image": per_image,
    }


def score_depth_fidelity(pipe: Any, generated: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Estimate the depth of every generated image (`{image, hint}`) with the pipeline's pinned estimator and score
    it against its hint with `depth_fidelity_from_maps`."""
    if not generated:
        raise ValueError("no generated records to score")
    started = time.perf_counter()
    predicted = pipe.estimate_depth([g["image"] for g in generated])
    report = depth_fidelity_from_maps(predicted, [g["hint"] for g in generated])
    report["estimator"] = {"id": DEPTH_ID, "revision": DEPTH_REVISION}
    report["seconds"] = round(time.perf_counter() - started, 2)
    return report


def real_photo_depth_ceiling(pipe: Any, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Depth fidelity of the real photographs to their own hints, re-estimated through the same path: close to
    perfect by construction, it shows the measurement's own noise floor rather than a reachable target."""
    hints = [pipe.depth_hint(r) for r in records]
    report = depth_fidelity_from_maps(pipe.estimate_depth([r["image"] for r in records]), hints)
    report["note"] = "real photographs re-estimated against their own hints (measurement noise floor)"
    return report
