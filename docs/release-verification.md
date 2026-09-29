# Release verification

`tutorials/kandinsky_controlnet_depth_colab.ipynb` (`E2E`, `GUIDED`, **standalone** carrier) is a **release candidate**
until the exact notebook revision has executed top-to-bottom in a clean supported runtime **and** the pinned decoder
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
| — | — | — | No execution with the pinned weights has been recorded yet | — | — |

## Recorded conversion checks

| Date (UTC) | Commit | Executor | Components | Outcome |
|---|---|---|---|---|
| — | — | — | `tools/verify_conversion.py` has not been run yet | pending |

## Current status

**Candidate.** Static validation, parity and the offline unit suite pass. Both the conversion check and a clean-runtime
execution are outstanding.
