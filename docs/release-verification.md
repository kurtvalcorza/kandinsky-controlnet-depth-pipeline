# Release verification

`tutorials/kandinsky_controlnet_depth_colab.ipynb` (`E2E`, `GUIDED`, **standalone** carrier) stays a release
candidate for any revision until that exact notebook revision has executed top-to-bottom in a clean supported runtime **and** the pinned decoder
conversion has been checked against the upstream `main` weights. Unit tests, JSON validation, code-cell compilation,
the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime
evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release record.

## Automatic coverage (static, every pull request)

CI runs `ruff check src tests tools`, the offline unit suite, `tools/validate_release_assets.py` and
`tools/build_notebook.py --check`. The validator checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, registered in `tutorials/README.md` with its `E2E` profile, spec `2.2`, the
  standalone carrier and the isolated environment; `metadata.dimer` declares the profile, spec `2.2`, `GUIDED` mode,
  `standalone: true`, `requires_dimer_worker: false` and `generated_from` (repository, revision, package-module
  SHA-256, per-file hashes, generator);
- the carrier (ST1–ST8, SRC4, PAR1–PAR4): exactly one carrier cell whose `CARRIED_FILES` equal the repository files
  named by the template (the four package modules, `tools/tutorial_stages.py`, `tutorials/requirements-colab.lock.txt`,
  the four snapshot manifests, `LICENSE`) plus the generated `source.json`; every `CARRIED_HASHES` entry is the SHA-256
  of its text and is recorded in the cell and notebook metadata; the cell writes each file and raises on a hash
  mismatch; the notebook is byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision;
- the lock (ENV1, ENV2): it pins every `pyproject.toml` runtime pin at the same version, every entry carries
  `--hash=sha256:`, and its header records `--generate-hashes --only-binary :all:` and the manylinux x86_64 target;
- the isolated install (RUN10, ENV6, §25.13): no kernel cell runs `pip` except `uv pip install --python <isolated
  env> --require-hashes`; no `sys.executable`, `importlib`, `pip install` or `-m pip`; kernel imports limited to the
  standard library, `IPython.display` and `google.colab`; downloads (`urllib`) only in the install cell; `UV_URL`,
  `UV_BYTES` and a 64-hex `UV_SHA256` equal to the template's pinned `uv` wheel; `--managed-python` CPython 3.12.12,
  `--only-binary :all:`, the Hugging Face token and `PYTHONPATH` removal, `MPLBACKEND='Agg'`, the CUDA check in the
  isolated environment, and a `run_stage` that re-raises the stage's own error type and message;
- the learner path: the stages called in order (`weights`, `prepare`, `encode`, `frozen`, `activity`, `adapt`,
  `evaluate`, `reload`); both BYOD form fields exactly as `USE_BYOD = False  # @param {type:"boolean"}` and
  `BYOD_PATH = ''  # @param {type:"string"}` in the cell that runs `prepare` (EXE1/EXE2); `USE_BYOD` and
  `ACTIVITY_HINT` each assigned once; no learner prose that asks for a runtime restart;
- the carried stage runner's required calls (`KandinskyDepthPipeline.from_pretrained(..., depth_dir=..., use_lora=True)`,
  staging and verification of the four snapshots, `fetch_sample_dataset`, `load_byod_dataset`, `dataset_manifest`,
  `write_dataset_csv`, `validate_dataset` with the refusal probes, the dataset-digest re-check, `compute_hints`,
  `encode_prompts`, `release_prior` and the hint and prompt caches, the frozen evaluation, generation, CLIP scoring with
  `real_photo_baseline` and depth fidelity with `real_photo_depth_ceiling`, the flat-hint activity scored with
  `depth_fidelity_from_maps`, `pipe.adapt` with its explicit hyperparameters, the in-memory reference values,
  `save_artifact`, the fresh-process `from_artifact` evaluation with the guaranteed checks and the depth comparison
  rows, the second fresh-process reload with the parity check, the new-prompt generation, the provenance fields
  `safetensors_only: True`, `remote_code_executed: False` and the data base URL, the error record), and the expected
  exports;
- `MODEL_ID`/`MODEL_REVISION` never rebound in a kernel cell and the revision absent from every kernel cell (it lives in
  the carried package and manifests); forbidden patterns in the kernel and in every carried file: credential-in-URL,
  `git clone` / `github.com`, an editable install, a mutable `revision='main'`, `trust_remote_code=True`, unsafe
  deserialization, `extractall`, magics, `--no-binary` / `--no-build-isolation` / `--trusted-host` /
  `--extra-index-url`; and in the kernel only, a repository-package import;
- the §3.5 guided layer: audience, how-to-use (including where the code runs), roadmap, input → model → output,
  glossary, predictions, what-to-notice notes, check-your-reasoning answers, the change-one-thing activity, section
  types, troubleshooting, the conclusion scaffold, the four infrastructure cells titled **Infrastructure** and
  collapsed (`cellView: form`, `source_hidden`), and the learner cells not collapsed;
- the identity string and revision in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md`, and every SHA-256 digest and
  byte count quoted there traced to a manifest or a labelled allowlist entry;
- `MODEL_CARD.md` front matter, single H1, the 19 required headings in order, and the immutable provenance section.

In CI, which has no `torch`, `tests/test_tutorial_stages.py` runs its kernel-side tests (the generated carrier writes
and verifies every file, and `run_stage` re-raises an invalid BYOD zip's refusal text) and skips the CPU pre-flight.

## CPU pre-flight of the stage runner (not runtime evidence)

On 2026-09-30, on Windows (CPython 3.12, `torch 2.14.0+cpu`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`,
in a scratch virtual environment), `tests/test_tutorial_stages.py` passed 9 of 9:

- the generated carrier cell wrote the 11 carried files and verified each against `CARRIED_HASHES`;
- the generated `run_stage`, driving the carried `tutorial_stages.py` in a subprocess, raised
  `RuntimeError: Stage 'prepare' failed (exit 2): ValueError: captions.csv is missing columns ['caption']` for a zip
  without a `caption` column, and `… ValueError: records[0]: image sides must be within 256..4096 px, got (200, 200)`
  for 200 × 200 images;
- every stage (`weights`, `prepare`, `encode`, `frozen`, `activity`, `adapt`, `evaluate`, `reload`) ran in order on CPU
  against a stub UNet/MoVQ (from `tests/test_adaptation.py`), a stub prior, a stub depth estimator, a stub CLIP scorer,
  a stub generator and 12 synthetic records, each stage building its own pipeline so that everything crossed over
  through the run directory; all expected exports were written, reload parity held within tolerance, the activity
  accepted `flat`, `mismatched` and `own` and refused any other hint with the notebook's message, and a stage run
  before `prepare` was refused with `data.json is missing`.

This proves the stage plumbing and the file hand-offs only. The real snapshot staging, the real prior, UNet, MoVQ, DPT
estimator, diffusers generation and CLIP scorer, the `uv` bootstrap and the locked install were not exercised: they
need a Linux x86_64 GPU runtime.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better, ≥ 15 GB) | The runtime the tutorial is written for; a clean top-to-bottom run is promotion evidence |
| Kaggle kernel or equivalent fresh container | Fresh Linux x86_64 GPU container; the committed notebook executed verbatim with no repository checkout | Clean-room executor of the same class; promotion evidence |
| Kaggle CPU kernel | Fresh CPU container with `torch`, `safetensors`, `huggingface-hub` | Runs `tools/verify_conversion.py`; conversion evidence only |

## Conversion check procedure

1. In a fresh CPU kernel with about 11 GB of free disk, install the pinned `torch`, `safetensors` and `huggingface-hub`.
2. Run `python tools/verify_conversion.py` from a checkout of the revision under review.
3. Record the printed JSON summary (tensor counts, element counts, mismatch counts, runtime) in the table below.
4. Exit code 0 is required. Any mismatch blocks promotion and must be reported against the pinned revision.

## Supported notebook verification procedure

1. Resolve the exact commit under review and confirm static CI is green.
2. Open that notebook revision in a new Linux x86_64 GPU runtime with **no repository checkout**, an empty Hugging
   Face cache and no pre-staged files under `weights/`; the runtime needs about 31 GB of free disk and a GPU of at
   least 15 GB.
3. Run the notebook top-to-bottom with `Run all` and **no runtime restart**, with every form field at its default
   (`USE_BYOD = False`, `BYOD_PATH = ''`, `NEW_PROMPT` unchanged, `STEPS = 20`, `GUIDANCE_SCALE = 4.0`,
   `ACTIVITY_HINT = 'flat'`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`).
4. Verify that the Section 2 carrier reports the revision recorded in `metadata.dimer.generated_from`, that the
   isolated environment reports CPython 3.12.12 and the locked versions (`torch 2.14.0`, `diffusers 0.40.0`,
   `transformers 5.17.0`, `peft 0.21.0`) with `'cuda': True`, and that the kernel's own packages were not changed.
5. Verify every stage completes, including the two guaranteed checks of the `evaluate` stage (Section 9) and the
   reload-parity check of the `reload` stage (Section 10).
6. Record the notebook Git blob id, commit, runtime, wall time and the observed metrics below. Record no secrets.
7. For BYOD evidence, rerun Section 4 onward with `USE_BYOD = True` and `BYOD_PATH` set to a valid zip, and once with an
   invalid zip to record the refusal.

## Recorded executions

No run of the isolated-environment notebook (generator `build_notebook.py/3.0`) is recorded yet. Both runs below are of
the previous notebook, which pip-installed its pins into the kernel (**pre-fix**); they are evidence for the stage
logic it shared, not for this revision's environment bootstrap.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-29 (pre-fix) | `f4fe86ae19d99ad039554f1bcc075ccb47903492` / blob `671cf4c8f722e776e1854b4c98a20eaa0a26cbc1` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | Default `Run all` path, form fields at their defaults (`USE_BYOD = False`, `BYOD_PATH = ''`) | 831.5 s | **PASSED** — 12/12 code cells, 0 errors; the install cell's restart guard fired once (preloaded `numpy`, `protobuf`, `cuda-bindings`) and the kernel was restarted and re-run from the top. Held-out test `denoising_mse` 0.072207 (frozen) → 0.071615 (adapted); `depth_correlation` 0.864233 → 0.858579 (mismatched-hint baseline 0.25601); `depth_aligned_mae` 0.102967 → 0.108653 (baseline 0.241902); `label_accuracy` 0.75 → 0.8333 (real photographs 0.9167). Reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0. One run on 12 held-out photographs; sample-sanity evidence, not a benchmark |
| 2026-09-29 (pre-fix) | `f4fe86ae19d99ad039554f1bcc075ccb47903492` / blob `671cf4c8f722e776e1854b4c98a20eaa0a26cbc1` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | REL12 BYOD journey: in the executed copy only (not committed), `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay, 12 observers, not in the sample corpus), each checked against a pinned SHA-256; the committed cell source was checked by SHA-256 before the edit. After `Run all`, one appended harness cell re-ran the committed Section 4 source against two incompatible zips | 697.7 s | **PASSED** — 13/13 code cells, 0 errors. Positive: 12 records split 8 / 2 / 2 by caption and carried through depth hints, adaptation, evaluation, generation, export and reload (`denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0); `depth_correlation` 0.818885 → 0.841563 (baseline 0.423756). Negative: a `captions.csv` without `caption` was refused with `captions.csv is missing columns ['caption']`, and a 200 × 200 image with `records[0]: image sides must be within 256..4096 px, got (200, 200)`, both before any model ran on them. With one test photograph per caption these numbers show that the path runs, not how well it performs |

## Recorded conversion checks

The conversion check runs `tools/verify_conversion.py`, which does not depend on the notebook; the record below stays
valid for the isolated-environment notebook.

| Date (UTC) | Commit | Executor | Components | Outcome |
|---|---|---|---|---|
| 2026-09-29 | `f4fe86ae19d99ad039554f1bcc075ccb47903492` (script blob `c8640b27f0b3b2c3d87cf8a6745325b568f0989d`) | Kaggle CPU kernel, Python 3.12.13, `torch 2.10.0+cpu`; script fetched at the commit and blob-verified | UNet: 740/740 tensors, 1,253,429,212 elements; MoVQ: 431/431 tensors, 67,832,495 elements | **PASSED** — identical key sets, shapes, dtypes and `torch.equal` values for both components; every download matched its recorded SHA-256 (UNet `.bin` `3418cd4f…` vs safetensors `6549f8c8…`; MoVQ `.bin` `772e0973…` vs safetensors `43a5860f…`) |

## Current status

**Candidate** — the fix for the hosted-runtime restart is implemented but has not yet run on hardware. The previous notebook pip-installed its pins into the kernel, which on Google Colab and Kaggle replaced preloaded `numpy`, `protobuf` and `cuda-bindings` and stopped `Run all` with a manual-restart request (Notebook Specification 2.2 RUN1/RUN10). The notebook now follows the version 2.2 reference pattern (§25.13): nothing is installed into the kernel; a pinned `uv` builds an isolated environment from the committed hash lock (`tutorials/requirements-colab.lock.txt`) and every stage runs there in its own process. Promotion needs a clean Colab or Kaggle `Run all` of this notebook revision and the REL12 BYOD journey, recorded in `docs/release-verification.md`; the earlier Kaggle notebook runs there are of the previous in-kernel-install notebook and are not evidence for this one. The conversion check recorded there does not depend on the notebook and stays valid.
