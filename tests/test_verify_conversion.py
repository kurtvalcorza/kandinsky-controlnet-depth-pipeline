"""Offline tests for tools/verify_conversion.py: the comparison logic (numpy stand-ins, no torch), the pins it
checks against, and its import boundary. The script itself downloads 5 GB and is run on a CPU kernel, not here."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np

from kandinsky_controlnet_depth_pipeline import MANIFEST_NAME, MODEL_ID, MODEL_KEY, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_conversion.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_conversion", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_imports_no_model_library_at_module_level(forbid_model_imports):
    _load()
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert not any(n.partition(".")[0] in {"torch", "safetensors", "huggingface_hub"} for n in names)


def test_tool_pins_match_the_package_and_manifest():
    tool = _load()
    assert (tool.MODEL_ID, tool.MODEL_REVISION) == (MODEL_ID, MODEL_REVISION)
    assert len(tool.MAIN_REVISION) == 40 and tool.MAIN_REVISION != MODEL_REVISION
    manifest = json.loads((ROOT / "weights" / MODEL_KEY / MANIFEST_NAME).read_text(encoding="utf-8"))
    recorded = {e["path"]: (e["bytes"], e["sha256"]) for e in manifest["files"]}
    for spec in tool.COMPONENTS.values():
        path, size, digest = spec["safetensors"]
        assert recorded[path] == (size, digest)
        assert spec["bin"][0].endswith(".bin") and len(spec["bin"][2]) == 64


def test_compare_state_dicts_passes_identical_and_names_each_difference():
    tool = _load()
    equal = np.array_equal
    ref = {"a": np.zeros((2, 3), np.float32), "b": np.ones(4, np.float32), "c": np.ones(2, np.float32)}
    same = {k: v.copy() for k, v in ref.items()}
    report = tool.compare_state_dicts(ref, same, equal=equal)
    assert report["passed"] and report["n_elements_compared"] == 12

    changed = {
        "a": np.zeros((3, 2), np.float32),  # shape
        "b": np.ones(4, np.float16),  # dtype
        "c": np.array([1.0, 2.0], np.float32),  # value
        "d": np.ones(1, np.float32),  # extra
    }
    report = tool.compare_state_dicts({**ref, "e": np.ones(1, np.float32)}, changed, equal=equal)
    assert not report["passed"]
    assert report["shape_mismatch"] == ["a"] and report["dtype_mismatch"] == ["b"] and report["value_mismatch"] == ["c"]
    assert report["extra_in_candidate"] == ["d"] and report["missing_in_candidate"] == ["e"]
