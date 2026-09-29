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
- exactly one tutorial notebook, registered in `tutorials/README.md` with its `E2E` profile, spec `2.2` and the
  standalone carrier; `metadata.dimer` declares the profile, spec `2.2`, `GUIDED` mode, `standalone: true` and
  `generated_from`;
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import; one cell per carried
  module equal to its source after the documented rewrite; the inline `MANIFEST`, `PRIOR_MANIFEST`, `DEPTH_MANIFEST`
  and `SCORER_MANIFEST` equal to the committed manifests; inline `PINS` equal to `pyproject.toml`; the notebook
  byte-identical (on LF) to the generator output;
- the §3.5 guided layer: audience, how-to-use, roadmap, input → model → output, glossary, predictions, what-to-notice
  notes, check-your-reasoning answers, the change-one-thing activity, section types, troubleshooting, the conclusion
  scaffold, and the install, carried-module and model cells titled **Infrastructure** and collapsed (`cellView: form`);
- the public-API calls of every stage, the expected `outputs/` paths, the provenance fields, the gated-off BYOD default
  with its location field, and the forbidden patterns (mutable revisions, direct `huggingface_hub` / `safetensors` /
  `diffusers` / `transformers` / `peft` use outside the carried modules, `torch.load(` without `weights_only=True`,
  `trust_remote_code=True`);
- the identity string and revision in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md`, and every SHA-256 digest and
  byte count quoted there traced to a manifest or a labelled allowlist entry;
- `MODEL_CARD.md` front matter, single H1, the 19 required headings in order, and the immutable provenance section.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better, ≥ 15 GB) | The runtime the tutorial is written for; a clean top-to-bottom run is promotion evidence |
| Kaggle kernel or equivalent fresh container | Fresh GPU container, Python 3.12; the committed notebook executed verbatim with no repository checkout | Clean-room executor of the same class; promotion evidence |
| Kaggle CPU kernel | Fresh CPU container with `torch`, `safetensors`, `huggingface-hub` | Runs `tools/verify_conversion.py`; conversion evidence only |

## Conversion check procedure

1. In a fresh CPU kernel with about 11 GB of free disk, install the pinned `torch`, `safetensors` and `huggingface-hub`.
2. Run `python tools/verify_conversion.py` from a checkout of the revision under review.
3. Record the printed JSON summary (tensor counts, element counts, mismatch counts, runtime) in the table below.
4. Exit code 0 is required. Any mismatch blocks promotion and must be reported against the pinned revision.

## Supported notebook verification procedure

1. Resolve the exact commit under review and confirm static CI is green.
2. Open that notebook revision in a new GPU runtime with **no repository checkout**, an empty Hugging Face cache and no
   pre-staged files under `weights/`; the runtime needs about 25 GB of free disk and a GPU of at least 15 GB.
3. Run the notebook top-to-bottom with every form field at its default (`USE_BYOD = False`, `BYOD_PATH = ''`,
   `STEPS = 20`, `GUIDANCE_SCALE = 4.0`, `ACTIVITY_HINT = 'flat'`, `EPOCHS = 4`, `LEARNING_RATE = 1e-4`,
   `BATCH_SIZE = 1`).
4. Verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`
   and that the installed versions equal the inline `PINS`.
5. Verify every stage completes, including both assertions in Section 9 and the reload-parity assertion in Section 10.
6. Record the notebook Git blob id, commit, runtime, wall time and the observed metrics below. Record no secrets.
7. For BYOD evidence, rerun Section 4 onward with `USE_BYOD = True` and `BYOD_PATH` set to a valid zip, and once with an
   invalid zip to record the refusal.

## Recorded executions

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-29 | `f4fe86ae19d99ad039554f1bcc075ccb47903492` / blob `671cf4c8f722e776e1854b4c98a20eaa0a26cbc1` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | Default `Run all` path, form fields at their defaults (`USE_BYOD = False`, `BYOD_PATH = ''`) | 831.5 s | **PASSED** — 12/12 code cells, 0 errors; the install cell's restart guard fired once (preloaded `numpy`, `protobuf`, `cuda-bindings`) and the kernel was restarted and re-run from the top. Held-out test `denoising_mse` 0.072207 (frozen) → 0.071615 (adapted); `depth_correlation` 0.864233 → 0.858579 (mismatched-hint baseline 0.25601); `depth_aligned_mae` 0.102967 → 0.108653 (baseline 0.241902); `label_accuracy` 0.75 → 0.8333 (real photographs 0.9167). Reload parity: `denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0. One run on 12 held-out photographs; sample-sanity evidence, not a benchmark |
| 2026-09-29 | `f4fe86ae19d99ad039554f1bcc075ccb47903492` / blob `671cf4c8f722e776e1854b4c98a20eaa0a26cbc1` | Kaggle batch kernel, Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`; notebook fetched at the commit and blob-verified, run in a fresh interpreter with `nbclient`, Hugging Face cache empty at start | REL12 BYOD journey: in the executed copy only (not committed), `USE_BYOD = True` and `BYOD_PATH` = a zip built in the kernel from 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay, 12 observers, not in the sample corpus), each checked against a pinned SHA-256; the committed cell source was checked by SHA-256 before the edit. After `Run all`, one appended harness cell re-ran the committed Section 4 source against two incompatible zips | 697.7 s | **PASSED** — 13/13 code cells, 0 errors. Positive: 12 records split 8 / 2 / 2 by caption and carried through depth hints, adaptation, evaluation, generation, export and reload (`denoising_mse_diff` 0.0, `mean_abs_pixel_diff` 0.0); `depth_correlation` 0.818885 → 0.841563 (baseline 0.423756). Negative: a `captions.csv` without `caption` was refused with `captions.csv is missing columns ['caption']`, and a 200 × 200 image with `records[0]: image sides must be within 256..4096 px, got (200, 200)`, both before any model ran on them. With one test photograph per caption these numbers show that the path runs, not how well it performs |

## Recorded conversion checks

| Date (UTC) | Commit | Executor | Components | Outcome |
|---|---|---|---|---|
| 2026-09-29 | `f4fe86ae19d99ad039554f1bcc075ccb47903492` (script blob `c8640b27f0b3b2c3d87cf8a6745325b568f0989d`) | Kaggle CPU kernel, Python 3.12.13, `torch 2.10.0+cpu`; script fetched at the commit and blob-verified | UNet: 740/740 tensors, 1,253,429,212 elements; MoVQ: 431/431 tensors, 67,832,495 elements | **PASSED** — identical key sets, shapes, dtypes and `torch.equal` values for both components; every download matched its recorded SHA-256 (UNet `.bin` `3418cd4f…` vs safetensors `6549f8c8…`; MoVQ `.bin` `772e0973…` vs safetensors `43a5860f…`) |

## Current status

**Candidate.** The notebook stops at its install cell on hosted runtimes that preload `numpy`, `protobuf` and `cuda-bindings` (Google Colab and Kaggle), because the pinned install replaces those loaded packages and the cell then asks for a manual runtime restart. Notebook Specification 2.2 RUN1 and RUN10 forbid a manual restart on the `Run all` path, so the tutorial is not release-ready. The Kaggle runs recorded in `docs/release-verification.md` completed only because the executor restarted the kernel automatically; they remain valid evidence for everything after the install cell. Found in a Colab `Run all` on 2026-09-29; the fix (an isolated, hash-locked environment for the tutorial stages) is in progress.
