"""Refused user input is shown as one clean line (NOTEBOOK_SPEC 3.0 DAT19/DAT20).

Each refusal path runs the generated notebook's own kernel code (the carrier cell, the ``run_stage`` helper and the
learner's Section 4 BYOD cell) against the real stage runner in a child process, as the notebook does, with only
``google.colab``'s upload dialog stood in. For every path the cell must raise ``Refused`` (an exception, so Run all
stops at that cell), render as exactly one ``Refused: <file>: <reason>`` line through IPython's ``_render_traceback_``
hook, name the file the learner supplied, and print no ``Traceback``. The earlier run's results are removed, so none can
be read as this input's result. A genuine failure still raises ``RuntimeError`` with the stage's traceback. The content
case is the hosted refusal run's input, ``BYOD_PATH=/etc``: a folder (an accepted BYOD form) without ``captions.csv``.
CI's dependencies suffice: the BYOD checks run before any model library is imported.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from pathlib import Path

import pytest

from test_tutorial_stages import ROOT, TEMPLATE, _kernel, build

REFUSAL_CELL_ID = "kandinsky_controlnet_depth-15"  # the hosted refusal run's halting cell: Section 4's BYOD cell
CELLS = [c for c in build.render(ROOT, TEMPLATE, "t")["cells"] if c["cell_type"] == "code"]


def _byod_cell_record() -> dict:
    found = [c for c in CELLS if "USE_BYOD = False  # @param" in c["source"]]
    assert len(found) == 1
    return found[0]


def _byod_cell(path: str) -> str:
    lines = []
    for line in _byod_cell_record()["source"].split("\n"):
        if line.startswith("USE_BYOD = "):
            line = "USE_BYOD = True" + line[line.index("  #"):]
        elif line.startswith("BYOD_PATH = "):
            line = f"BYOD_PATH = {path!r}" + line[line.index("  #"):]
        lines.append(line)
    return "\n".join(lines)


def _kernel_with_cells(tmp_path: Path) -> dict:
    ns = _kernel(tmp_path)
    ns["Path"] = Path
    return ns


@contextlib.contextmanager
def _colab(*uploads: dict):
    queue = list(uploads)
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: queue.pop(0))
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _run_all(ns: dict, cells: list[str]) -> tuple[BaseException | None, str, int]:
    """Run cells in order as Run all does: the first cell that raises stops the run. Returns the exception, the cell
    output (stdout plus the exception as IPython renders it) and how many cells ran."""
    out, ran, raised = io.StringIO(), 0, None
    with contextlib.redirect_stdout(out):
        for source in cells:
            ran += 1
            try:
                exec(compile(source, f"<cell {ran}>", "exec"), ns)  # noqa: S102 - the notebook's own cells
            except BaseException as exc:  # noqa: BLE001 -- what IPython would show is asserted below
                raised = exc
                break
    shown = out.getvalue()
    if raised is not None and hasattr(raised, "_render_traceback_"):
        shown += "\n".join(raised._render_traceback_()) + "\n"
    return raised, shown, ran


def _assert_refused(raised, shown: str, ran: int, file: str) -> str:
    assert type(raised).__name__ == "Refused", (raised, shown)
    lines = raised._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith(f"Refused: {file}: "), lines
    assert [line for line in shown.splitlines() if "Refused:" in line] == lines, shown
    assert "Traceback" not in shown and "Error" not in shown.replace("Refused", ""), shown
    assert ran == 1, "Run all must stop at the refusing cell"
    return lines[0]


def _with_earlier_results(ns: dict) -> tuple[Path, Path]:
    """An earlier run's results on disk, as a learner who ran the notebook once would have them."""
    out, state = ns["ROOT"] / "outputs", ns["ROOT"] / "state"
    out.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    for name in ("weights.json", "prepare.json", "evaluation.json", "frozen_grid.png"):
        (out / name).write_text("{}", encoding="utf-8")
    (out / "adapter").mkdir()
    for name in ("data.json", "prompt_cache.pt"):
        (state / name).write_text("{}", encoding="utf-8")
    return out, state


def test_the_refusal_cell_is_section_4s_byod_cell() -> None:
    assert _byod_cell_record()["id"] == REFUSAL_CELL_ID


def test_a_folder_without_captions_csv_is_one_refused_line_and_clears_earlier_results(tmp_path: Path) -> None:
    """The hosted refusal run's input (``USE_BYOD=True;BYOD_PATH=/etc``): a folder that exists in every fresh runtime
    but holds no ``captions.csv``."""
    ns = _kernel_with_cells(tmp_path)
    out, state = _with_earlier_results(ns)
    folder = tmp_path / "etc"
    folder.mkdir()
    (folder / "hostname").write_text("colab-runtime\n", encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(str(folder)), "print('AFTER THE REFUSAL')"])
    line = _assert_refused(raised, shown, ran, "etc")
    assert "captions.csv" in line and "run the Section 4 cell again" in line
    assert "AFTER THE REFUSAL" not in shown
    assert sorted(p.name for p in out.iterdir()) == ["weights.json"]
    assert sorted(p.name for p in state.iterdir()) == ["prepare.error.json"]
    error = json.loads((state / "prepare.error.json").read_text(encoding="utf-8"))
    assert error["refused"] is True and error["file"] == "etc"


def test_a_file_that_is_not_a_zip_is_one_refused_line(tmp_path: Path) -> None:
    ns = _kernel_with_cells(tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-runtime\n", encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(str(hostname))])
    assert "must be a directory or a .zip" in _assert_refused(raised, shown, ran, "hostname")


def test_a_corrupt_zip_is_one_refused_line(tmp_path: Path) -> None:
    ns = _kernel_with_cells(tmp_path)
    fake = tmp_path / "photos.zip"
    fake.write_text("not an archive\n", encoding="utf-8")
    raised, shown, ran = _run_all(ns, [_byod_cell(str(fake))])
    _assert_refused(raised, shown, ran, "photos.zip")


def test_a_missing_path_is_one_refused_line(tmp_path: Path) -> None:
    ns = _kernel_with_cells(tmp_path)
    missing = str(tmp_path / "absent.zip")
    raised, shown, ran = _run_all(ns, [_byod_cell(missing)])
    assert "BYOD_PATH is not a zip or folder" in _assert_refused(raised, shown, ran, missing)


@pytest.mark.parametrize(
    ("uploads", "file"),
    [
        ({}, "(no file)"),
        ({"a.zip": b"PK", "b.zip": b"PK"}, "a.zip, b.zip"),
        ({"photos.tar": b"tar"}, "photos.tar"),
    ],
)
def test_upload_dialog_refusals_are_one_refused_line(tmp_path: Path, uploads: dict, file: str) -> None:
    ns = _kernel_with_cells(tmp_path)
    with _colab(uploads):
        raised, shown, ran = _run_all(ns, [_byod_cell("")])
    _assert_refused(raised, shown, ran, file)


def test_an_empty_path_outside_colab_is_one_refused_line(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # importing it raises ImportError
    ns = _kernel_with_cells(tmp_path)
    raised, shown, ran = _run_all(ns, [_byod_cell("")])
    _assert_refused(raised, shown, ran, "BYOD_PATH")


def test_a_genuine_failure_is_still_an_error_with_its_traceback(tmp_path: Path) -> None:
    """A stage run out of order is a notebook failure, not a refusal: RuntimeError, and the stage's traceback shows."""
    ns = _kernel_with_cells(tmp_path)
    raised, shown, ran = _run_all(ns, ["run_stage('evaluate')"])
    assert type(raised) is RuntimeError and "is missing" in str(raised)
    assert "Traceback" in shown and "Refused:" not in shown
