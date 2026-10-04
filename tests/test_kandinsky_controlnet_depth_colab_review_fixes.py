"""Regression tests for the 2026-10-02 notebook review of `kandinsky_controlnet_depth_colab.ipynb` (KCD-M1, KCD-m1..m6).

CI installs numpy, pillow, pytest and ruff only (no torch): the metric, split, notebook-text and validator tests run there
with NumPy stand-ins. The two stage-runner tests reuse the CPU pre-flight of `test_tutorial_stages.py` (stub models) and
skip without torch + diffusers + safetensors; they prove the plumbing, not the model.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import importlib.util
import io
import itertools
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_image
from kandinsky_controlnet_depth_pipeline import (
    MIN_TRAIN_RECORDS,
    REAL_PHOTO_REFERENCE_KIND,
    load_byod_dataset,
    real_photo_baseline,
    real_photo_reference,
    split_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "kandinsky_controlnet_depth_colab.ipynb"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _runner() -> str:
    return (ROOT / "tools" / "tutorial_stages.py").read_text(encoding="utf-8")


# --- KCD-M1: the per-timestep loss falls with the noise level ------------------------------------------------------

# The strict Kaggle T4 run of the previous revision (c86e3fe, blob 10145c03): frozen held-out test loss per timestep.
RECORDED_FROZEN_BY_TIMESTEP = {"100": 0.273275, "300": 0.065541, "500": 0.017646, "700": 0.003976, "900": 0.000596}


def test_recorded_loss_falls_with_timestep_and_the_smallest_timestep_dominates_the_mean() -> None:
    stages = _load("tutorial_stages")
    values = [RECORDED_FROZEN_BY_TIMESTEP[t] for t in ("100", "300", "500", "700", "900")]
    assert values == sorted(values, reverse=True)
    shares = stages.timestep_shares(RECORDED_FROZEN_BY_TIMESTEP)
    assert shares["100"] == pytest.approx(0.757, abs=0.001)  # "about 76 % of the mean"
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
    assert round(sum(values) / len(values), 4) == 0.0722  # the recorded frozen test mean


def test_section_6_states_the_falling_loss_and_the_epsilon_reason(nb: dict) -> None:
    md = _markdown(nb)
    assert "the per-timestep loss **falling** as the noise level rises" in md
    assert "ε-prediction" in md and "`prediction_type` is `epsilon`" in md
    assert "about 76 % of the mean" in md
    for stale in ("per-timestep loss rising", "highest at large timesteps", "almost all of the latent is noise. None"):
        assert stale not in md
    assert '"share_of_test_mean_by_timestep": timestep_shares(frozen_test["by_timestep"])' in _runner()


# --- KCD-m1: the real photographs are a leave-one-out reference line, not a ceiling -------------------------------


class _T(np.ndarray):
    """The few torch-tensor methods score_generations / real_photo_reference use, on NumPy."""

    def argmax(self, dim=None, **_):  # type: ignore[override]
        return np.asarray(self).argmax(axis=dim).view(_T)

    def mean(self, dim=None, **_):  # type: ignore[override]
        return np.asarray(self).mean(axis=dim).view(_T)

    def norm(self):
        return float(np.linalg.norm(np.asarray(self)))


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class _Scorer:
    """CLIP stand-in: fixed unit vectors per image key and per caption."""

    identity = {"id": "stand-in", "revision": "0"}

    def __init__(self, images: dict[str, np.ndarray], texts: dict[str, np.ndarray]) -> None:
        self.images, self.texts = images, texts

    def image_embeddings(self, images):
        return np.stack([self.images[k] for k in images]).view(_T)

    def text_embeddings(self, texts):
        return np.stack([self.texts[t] for t in texts]).view(_T)


def test_real_photo_reference_excludes_each_photo_from_its_own_reference() -> None:
    a, b, c = _unit([1, 0.2, 0, 0]), _unit([0.6, 0.8, 0.1, 0]), _unit([0, 0, 1, 0.3])
    scorer = _Scorer({"a": a, "b": b, "c": c}, {"A": _unit([1, 0.5, 0, 0]), "C": _unit([0, 0, 1, 0])})
    records = [{"image": "a", "caption": "A"}, {"image": "b", "caption": "A"}, {"image": "c", "caption": "C"}]
    report = real_photo_reference(scorer, records)
    cos_ab = float(a @ b) * 100
    rows = report["per_image"]
    # two photos of a caption: each one's reference similarity is cos(a, b), not sqrt((1 + cos(a, b)) / 2)
    assert rows[0]["reference_similarity"] == rows[1]["reference_similarity"] == round(cos_ab, 3)
    assert rows[0]["reference_similarity"] != round(np.sqrt((1 + cos_ab / 100) / 2) * 100, 3)
    assert "reference_similarity" not in rows[2] and report["n_without_reference"] == 1
    assert report["reference_similarity"] == round(cos_ab, 3)
    assert report["reference_kind"] == REAL_PHOTO_REFERENCE_KIND == "leave-one-out real-photo reference"
    assert set(report["reading"]) == {"clip_prompt_similarity", "label_accuracy", "reference_similarity"}
    assert "not a ceiling" in report["note"] and "excludes the photo itself" in report["note"]
    assert real_photo_baseline is real_photo_reference  # the old name gives the corrected measure


def test_real_photo_reference_without_any_same_caption_pair_reports_none() -> None:
    """A BYOD test set with one photograph per caption has no leave-one-out reference; the key stays present."""
    a, c = _unit([1, 0, 0]), _unit([0, 1, 0])
    scorer = _Scorer({"a": a, "c": c}, {"A": _unit([1, 0.1, 0]), "C": _unit([0, 1, 0.1])})
    report = real_photo_reference(scorer, [{"image": "a", "caption": "A"}, {"image": "c", "caption": "C"}])
    assert report["reference_similarity"] is None and report["n_without_reference"] == 2
    assert report["label_accuracy"] == 1.0


def test_no_learner_text_calls_the_real_photographs_a_ceiling(nb: dict) -> None:
    md = _markdown(nb)
    loose = [m.start() for m in re.finditer(r"\bceiling", md, re.I) if not md[: m.start()].endswith("not a ")]
    assert not loose
    assert md.count("not a ceiling") >= 4 and "leave-one-out" in md
    assert "often it cannot" not in md and "0.75 (9 of 12)" in md and "(32.3) was above the real photographs' (30.3)" in md
    code = "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "real_reference = real_photo_reference(scorer, test_records)" in code
    assert "real_ceiling" not in code and '"real_photo_ceiling"' not in code
    for name in ("README.md", "MODEL_CARD.md", "tutorials/README.md", "STATUS.md", "docs/release-verification.md"):
        assert "real-photo ceiling" not in (ROOT / name).read_text(encoding="utf-8"), name


def test_validator_refuses_a_ceiling_or_a_removed_phrase_coming_back(nb: dict) -> None:
    va = _load("validate_release_assets")
    build = _load("build_notebook")
    code_cells, markdown = va._validate_notebook_structure(NOTEBOOK, nb)
    carrier, files = va._validate_carrier(NOTEBOOK, nb, code_cells, build)
    va._validate_notebook_content(NOTEBOOK, nb, code_cells, markdown, carrier, files)  # the committed text passes
    with pytest.raises(va.ValidationError, match="ceiling"):
        va._validate_notebook_content(NOTEBOOK, nb, code_cells, markdown + "\nCompare it with the CLIP ceiling.", carrier, files)
    for phrase in ("the per-timestep loss rising with the noise level", "Hold out by caption, not by image", "at least four images"):
        with pytest.raises(va.ValidationError, match="removed is back"):
            va._validate_notebook_content(NOTEBOOK, nb, code_cells, markdown + "\n" + phrase, carrier, files)


# --- KCD-m2 / KCD-m3: the independence assumption, the stratified split and the pretraining overlap ---------------


def test_split_assumptions_and_pretraining_overlap_are_stated(nb: dict) -> None:
    md = _markdown(nb)
    assert "**The split assumes independent photographs.**" in md and "near-duplicates" in md
    assert "stratified within each caption" in md and "new photographs of seen captions" in md
    assert "Hold out by caption" not in md and "split by caption" not in md
    assert md.count("**Pretraining overlap.**") >= 2 and "LAION-2B" in md
    cells = [_src(c) for c in nb["cells"] if c["cell_type"] == "markdown"]
    section_4 = next(i for i, text in enumerate(cells) if text.startswith("## 4."))
    assert "**The split assumes independent photographs.**" in cells[section_4]  # before the split runs
    from kandinsky_controlnet_depth_pipeline import samples

    assert "STRATIFIED WITHIN EACH CAPTION" in samples.split_dataset.__doc__ and "independent" in samples.split_dataset.__doc__


# --- KCD-m4: the measured run time is stated ------------------------------------------------------------------------


def test_run_time_is_the_recorded_measurement(nb: dict) -> None:
    md = _markdown(nb)
    assert "1,156.2 s" in md and "has not yet been recorded" not in md
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "1,156.2 s" in registry and "not yet recorded" not in registry


# --- KCD-m5: an activity rerun does not leave a stale activity unlabelled ------------------------------------------


def test_activity_rerun_scope_is_stated_and_exported_per_hint(nb: dict) -> None:
    md = _markdown(nb)
    assert "`outputs/activity_<hint>.json`" in md and "`activity_at_evaluation`" in md
    runner = _runner()
    assert 'run.write_output(f"activity_{hint_choice}.json", detail)' in runner
    assert '"activity": activity_summary' not in runner and '"activity_at_evaluation": {' in runner


# --- KCD-m6: the stated BYOD minimum is exactly what the code accepts ----------------------------------------------


def _stated_rule_accepts(sizes: tuple[int, ...]) -> bool:
    """The notebook's stated contract for captions of at most 7 images: a caption with 3..7 distinct images gives one
    test and one validation image, a caption with 1..2 goes to training only, and at least 4 training images remain."""
    train = sum(n - 2 if n >= 3 else n for n in sizes)
    return any(n >= 3 for n in sizes) and train >= MIN_TRAIN_RECORDS


def _records(sizes: tuple[int, ...]) -> list[dict]:
    out, seed = [], 0
    for caption, n in enumerate(sizes):
        for _ in range(n):
            out.append({"id": f"r{seed}", "image": synthetic_image(width=256, height=256, seed=seed), "caption": f"caption {caption}"})
            seed += 1
    return out


@pytest.mark.parametrize(
    "sizes",
    sorted({tuple(sorted(c, reverse=True)) for k in (1, 2, 3) for c in itertools.product(range(1, 8), repeat=k) if sum(c) <= 9}),
)
def test_split_accepts_exactly_the_layouts_the_stated_rule_accepts(sizes) -> None:
    if _stated_rule_accepts(sizes):
        splits = split_dataset(_records(sizes), seed=0)
        assert len(splits["train"]) >= MIN_TRAIN_RECORDS and splits["test"]
    else:
        with pytest.raises(ValueError, match=r"split leaves|records; 4..2000 are required"):
            split_dataset(_records(sizes), seed=0)


def _zip(path: Path, n: int) -> Path:
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(("id", "file", "caption"))
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n):
            buffer = io.BytesIO()
            synthetic_image(width=320, height=320, seed=50 + i).save(buffer, format="PNG")
            archive.writestr(f"bird{i}.png", buffer.getvalue())
            writer.writerow((f"r{i}", f"bird{i}.png", "a photo of a bird"))
        archive.writestr("captions.csv", rows.getvalue())
    return path


def test_stated_smallest_byod_zip_is_accepted_and_one_less_is_refused_with_the_rule(tmp_path: Path) -> None:
    splits = split_dataset(load_byod_dataset(_zip(tmp_path / "six.zip", 6)), seed=0)
    assert {k: len(v) for k, v in splits.items()} == {"test": 1, "validation": 1, "train": 4}
    with pytest.raises(ValueError, match=r"split leaves 3 training records; at least 4 are required\. .* at least 6 distinct images are needed, for example six of one caption"):
        split_dataset(load_byod_dataset(_zip(tmp_path / "five.zip", 5)), seed=0)


def test_byod_contract_text_states_the_enforced_minimum(nb: dict) -> None:
    md = _markdown(nb)
    assert "at least **6 distinct images**" in md and "at least four images" not in md
    assert "two captions of three images each are refused" in md
    assert "`split leaves N training records`" in md  # the troubleshooting row names the refusal


# --- stage runner, CPU pre-flight (stub models; skips without torch + diffusers) -----------------------------------

from test_tutorial_stages import _run  # noqa: E402
from test_tutorial_stages import preflight as _preflight_fixture  # noqa: E402,F401  (registered here as `_preflight_fixture`)


@pytest.fixture
def preflight(request):
    """The CPU pre-flight of test_tutorial_stages.py (stub models, monkeypatched with teardown); skips without torch."""
    return request.getfixturevalue("_preflight_fixture")


def test_cpu_preflight_smallest_byod_dataset_runs_every_stage(preflight, tmp_path: Path) -> None:
    """KCD-m6 end to end: six photographs of one caption split 4 / 1 / 1 and reach Section 10 (one test layout, used
    twice for the new prompt); the leave-one-out reference has no same-caption pair and says so."""
    stages, run_root, weights = preflight
    archive = _zip(tmp_path / "byod.zip", 6)
    for stage, options in (
        ("weights", ()),
        ("prepare", ("--byod", str(archive))),
        ("encode", ()),
        ("frozen", ("--steps", "2")),
        ("activity", ("--hint", "mismatched")),
        ("adapt", ("--epochs", "1", "--lr", "1e-3")),
        ("evaluate", ()),
        ("reload", ()),
    ):
        _run(stages, run_root, weights, stage, *options)
    prepare = json.loads((run_root / "outputs" / "prepare.json").read_text(encoding="utf-8"))
    assert {k: v["n_records"] for k, v in prepare["dataset"]["splits"].items()} == {"train": 4, "validation": 1, "test": 1}
    frozen = json.loads((run_root / "outputs" / "frozen.json").read_text(encoding="utf-8"))
    assert frozen["real_photo_reference"]["reference_similarity"] is None
    assert frozen["real_photo_reference"]["reference_kind"] == REAL_PHOTO_REFERENCE_KIND
    assert (run_root / "outputs" / f"{stages.STEM}_new_prompt_1.png").is_file()
    result = json.loads((run_root / "outputs" / f"{stages.STEM}_result.json").read_text(encoding="utf-8"))
    assert result["data_source"] == "BYOD (byod.zip)" and result["reload_parity"]["denoising_mse_diff"] < 1e-6


def test_cpu_preflight_activity_rerun_after_evaluate_keeps_both_hints_and_labels_the_report(preflight, capsys) -> None:
    """KCD-m5: Run all (flat), then a `mismatched` rerun of Section 7: both per-hint files exist, activity.json is the
    latest run, and the evaluation report labels the flat summary it saw as `activity_at_evaluation`."""
    stages, run_root, weights = preflight
    for stage, options in (
        ("weights", ()),
        ("prepare", ()),
        ("encode", ()),
        ("frozen", ("--steps", "2")),
        ("activity", ("--hint", "flat")),
        ("adapt", ("--epochs", "1", "--lr", "1e-3")),
        ("evaluate", ()),
        ("activity", ("--hint", "mismatched")),
    ):
        _run(stages, run_root, weights, stage, *options)
    out = run_root / "outputs"
    assert json.loads((out / "activity_flat.json").read_text(encoding="utf-8"))["hint_given"] == "flat"
    latest = json.loads((out / "activity.json").read_text(encoding="utf-8"))
    assert latest["hint_given"] == "mismatched" and latest["latest_run"] is True and latest["written_at"].endswith("Z")
    assert json.loads((out / "activity_mismatched.json").read_text(encoding="utf-8"))["hint_given"] == "mismatched"
    report = json.loads((out / f"{stages.STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert "activity" not in report
    seen = report["activity_at_evaluation"]
    assert seen["hint_given"] == "flat" and "a later rerun of Section 7" in seen["note"]
    assert report["real_photo_reference"]["reference_kind"] == REAL_PHOTO_REFERENCE_KIND
    printed = capsys.readouterr().out
    assert "share_of_test_mean_by_timestep" in printed and "'real_photo_reference'" in printed
