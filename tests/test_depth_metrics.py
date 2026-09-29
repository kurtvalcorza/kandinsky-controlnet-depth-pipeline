"""Offline tests for the depth-fidelity metric (numpy only; no estimator, no model library)."""

from __future__ import annotations

import numpy as np
import pytest

from kandinsky_controlnet_depth_pipeline import RESOLUTION, depth_agreement, depth_fidelity_from_maps, flat_hint, normalize_depth


def _ramp(axis: int = 1, *, reverse: bool = False) -> np.ndarray:
    grid = np.indices((RESOLUTION, RESOLUTION))[axis].astype(np.float64)
    return normalize_depth(-grid if reverse else grid)


def _blob(cx: int, cy: int) -> np.ndarray:
    y, x = np.mgrid[0:RESOLUTION, 0:RESOLUTION]
    return normalize_depth(np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 60.0**2)))


def test_identical_maps_agree_perfectly(forbid_model_imports):
    ramp = _ramp()
    result = depth_agreement(ramp, ramp)
    assert result["depth_correlation"] == pytest.approx(1.0)
    assert result["depth_aligned_mae"] == pytest.approx(0.0, abs=1e-9)


def test_metric_is_invariant_to_affine_depth_scale(forbid_model_imports):
    """A relative estimator may compress or shift the range; correlation and the aligned MAE ignore that."""
    ramp = _ramp()
    squeezed = (ramp.astype(np.float64) * 0.5 + 40).astype(np.uint8)
    result = depth_agreement(squeezed, ramp)
    assert result["depth_correlation"] > 0.999
    assert result["depth_aligned_mae"] < 0.01


def test_inverted_depth_is_anticorrelated(forbid_model_imports):
    assert depth_agreement(_ramp(reverse=True), _ramp())["depth_correlation"] == pytest.approx(-1.0, abs=1e-3)


def test_constant_prediction_scores_zero_correlation(forbid_model_imports):
    result = depth_agreement(flat_hint(), _ramp())
    assert result["depth_correlation"] == 0.0
    assert result["depth_aligned_mae"] == pytest.approx(0.25, abs=0.01)  # mean |h - mean(h)| of a uniform ramp


def test_fidelity_reports_the_mismatched_baseline(forbid_model_imports):
    hints = [_blob(128, 128), _blob(384, 384), _ramp()]
    report = depth_fidelity_from_maps(hints, hints)
    assert report["n_images"] == 3
    assert report["depth_correlation"] == pytest.approx(1.0)
    assert report["mismatched_depth_correlation"] < report["depth_correlation"]
    assert report["mismatched_depth_aligned_mae"] > report["depth_aligned_mae"]
    assert all("mismatched_depth_correlation" in e for e in report["per_image"])


def test_mismatched_baseline_is_absent_when_every_hint_is_identical(forbid_model_imports):
    hint = _ramp()
    report = depth_fidelity_from_maps([hint, hint], [hint, hint])
    assert report["mismatched_depth_correlation"] is None


def test_fidelity_refuses_bad_inputs(forbid_model_imports):
    with pytest.raises(ValueError, match="equal length"):
        depth_fidelity_from_maps([_ramp()], [])
    with pytest.raises(ValueError, match="shape"):
        depth_agreement(np.zeros((8, 8), dtype=np.uint8), _ramp())


def test_score_depth_fidelity_uses_the_pipeline_estimator(forbid_model_imports):
    from kandinsky_controlnet_depth_pipeline import DEPTH_ID, score_depth_fidelity

    hints = [_blob(128, 128), _ramp()]

    class _FakePipe:
        def estimate_depth(self, images):
            return [hints[0], hints[1]][: len(images)]

    generated = [{"image": object(), "hint": hints[0]}, {"image": object(), "hint": hints[1]}]
    report = score_depth_fidelity(_FakePipe(), generated)
    assert report["estimator"]["id"] == DEPTH_ID
    assert report["depth_correlation"] == pytest.approx(1.0)
    assert report["mismatched_depth_correlation"] < 0.9
    with pytest.raises(ValueError, match="no generated"):
        score_depth_fidelity(_FakePipe(), [])
