#!/usr/bin/env python3
"""Check that the pinned safetensors conversion equals the upstream `main` checkpoint, tensor by tensor.

`kandinsky-community/kandinsky-2-2-controlnet-depth` ships only pickled `.bin` weights on its `main` branch
(commit `MAIN_REVISION`). The package pins the safetensors files of the repository's open, unmerged conversion
pull request #5 by its immutable commit (`MODEL_REVISION`). This script downloads both at those exact commits,
verifies every download against its recorded SHA-256, loads the `.bin` files with `torch.load(weights_only=True)`
(no arbitrary unpickling) and the safetensors files with `safetensors`, and asserts, for the UNet and the MoVQ:

* identical key sets;
* identical shape and dtype per tensor;
* `torch.equal` per tensor (bit-for-bit values).

It prints one JSON summary and exits 0 only when every check passes. It runs on CPU and needs about 11 GB of
free disk and, with memory-mapped loading, a few GB of RAM. It is meant for a disposable CPU kernel, not for CI.

Usage:
    python tools/verify_conversion.py [--work-dir DIR] [--components unet movq]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

MODEL_ID = "kandinsky-community/kandinsky-2-2-controlnet-depth"
MAIN_REVISION = "4ecd717e8c9086cf4a16ca28b64894f70a42cd08"  # upstream `main`: pickled .bin weights only
MODEL_REVISION = "08632524a2b7e3bed39a7901d92cdd07e37544d0"  # refs/pr/5 (open conversion PR): safetensors

# (bin path, bin bytes, bin sha256) on main -> (safetensors path, bytes, sha256) at MODEL_REVISION
COMPONENTS: dict[str, dict[str, tuple[str, int, str]]] = {
    "unet": {
        "bin": (
            "unet/diffusion_pytorch_model.bin",
            5_013_996_233,
            "3418cd4f977d51ec902d095ff67482232c6097edfd91b474443148113b8f8a36",
        ),
        "safetensors": (
            "unet/diffusion_pytorch_model.safetensors",
            5_013_798_992,
            "6549f8c8471357ed8ed6b700a80ffdd8fe45bd5b54f4c486d7f81ac9fe5f343b",
        ),
    },
    "movq": {
        "bin": (
            "movq/diffusion_pytorch_model.bin",
            271_492_131,
            "772e09739d742ddee6807add2d3c2fd2a32db53896b5d07a92c729d8c879ce59",
        ),
        "safetensors": (
            "movq/diffusion_pytorch_model.safetensors",
            271_380_364,
            "43a5860fea195a7116f2471396c5cc9535fade9b63c4857d8a192ffd924b7002",
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_state_dicts(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    equal: Callable[[Any, Any], bool],
    max_listed: int = 20,
) -> dict[str, Any]:
    """Compare two name -> tensor mappings: key sets, then shape, dtype and `equal(a, b)` for every shared key.

    Library-agnostic (anything with `.shape` and `.dtype`), so it is unit-tested without torch."""
    ref_keys, cand_keys = set(reference), set(candidate)
    missing = sorted(ref_keys - cand_keys)
    extra = sorted(cand_keys - ref_keys)
    shape_mismatch, dtype_mismatch, value_mismatch = [], [], []
    n_elements = 0
    for key in sorted(ref_keys & cand_keys):
        a, b = reference[key], candidate[key]
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatch.append(key)
            continue
        if str(a.dtype) != str(b.dtype):
            dtype_mismatch.append(key)
            continue
        if not equal(a, b):
            value_mismatch.append(key)
        size = 1
        for dim in a.shape:
            size *= int(dim)
        n_elements += size
    passed = not (missing or extra or shape_mismatch or dtype_mismatch or value_mismatch)
    return {
        "passed": passed,
        "n_reference_tensors": len(ref_keys),
        "n_candidate_tensors": len(cand_keys),
        "n_elements_compared": n_elements,
        "missing_in_candidate": missing[:max_listed],
        "extra_in_candidate": extra[:max_listed],
        "shape_mismatch": shape_mismatch[:max_listed],
        "dtype_mismatch": dtype_mismatch[:max_listed],
        "value_mismatch": value_mismatch[:max_listed],
        "counts": {
            "missing": len(missing),
            "extra": len(extra),
            "shape": len(shape_mismatch),
            "dtype": len(dtype_mismatch),
            "value": len(value_mismatch),
        },
    }


def _download(relative_path: str, revision: str, work_dir: Path, size: int, digest: str) -> Path:
    from huggingface_hub import hf_hub_download

    if len(revision) != 40:
        raise ValueError(f"refusing to download from a non-commit revision {revision!r}")
    path = Path(hf_hub_download(MODEL_ID, relative_path, revision=revision, local_dir=str(work_dir / revision)))
    actual_size = path.stat().st_size
    actual_digest = sha256_file(path)
    if actual_size != size or actual_digest != digest:
        raise ValueError(f"{relative_path}@{revision[:12]}: {actual_size} bytes / {actual_digest}, expected {size} / {digest}")
    return path


def _load_bin(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialization cannot be memory-mapped
        return torch.load(path, map_location="cpu", weights_only=True)


def _load_safetensors(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def verify_component(name: str, work_dir: Path) -> dict[str, Any]:
    import torch

    spec = COMPONENTS[name]
    started = time.perf_counter()
    bin_path = _download(spec["bin"][0], MAIN_REVISION, work_dir, spec["bin"][1], spec["bin"][2])
    st_path = _download(spec["safetensors"][0], MODEL_REVISION, work_dir, spec["safetensors"][1], spec["safetensors"][2])
    report = compare_state_dicts(_load_bin(bin_path), _load_safetensors(st_path), equal=torch.equal)
    report.update(
        {
            "component": name,
            "reference": {"revision": MAIN_REVISION, "path": spec["bin"][0], "sha256": spec["bin"][2]},
            "candidate": {"revision": MODEL_REVISION, "path": spec["safetensors"][0], "sha256": spec["safetensors"][2]},
            "seconds": round(time.perf_counter() - started, 1),
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path("conversion-check"))
    parser.add_argument("--components", nargs="+", choices=sorted(COMPONENTS), default=["unet", "movq"])
    args = parser.parse_args(argv)
    import platform

    import torch

    summary: dict[str, Any] = {
        "model_id": MODEL_ID,
        "main_revision": MAIN_REVISION,
        "model_revision": MODEL_REVISION,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "components": {},
    }
    for name in args.components:
        summary["components"][name] = verify_component(name, args.work_dir)
    summary["passed"] = all(c["passed"] for c in summary["components"].values())
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
