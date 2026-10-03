# Kandinsky 2.2 ControlNet-depth guided notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/kandinsky-controlnet-depth-pipeline`  
**Notebook:** `tutorials/kandinsky_controlnet_depth_colab.ipynb`  
**Reviewed commit:** `8fe8c4f2a84146eaa1485724f3199a95e8d371a2` (`main`)  
**Notebook Git blob:** `10145c0374408a493a284b985aeacc609bd57de0`  
**Finding prefix:** `KCD`  
**Framework:** Notebook Review Framework v1; requirements baseline DIMER Notebook Specification **2.2** (2026-09-26, `ml-worker` `origin/main`)

## Executive assessment

This is a strong notebook. It runs every stage in an isolated hash-locked `uv` environment, installs nothing into the
kernel, carries its code byte for byte with hash checks, pins four snapshots by commit and digest, and scores a generator
honestly: a held-out denoising loss on identical noise, CLIP scores beside real photographs, depth fidelity against a
mismatched-hint baseline, a validation-selected epoch, and a fresh-process reload with parity. The default `Run all` path
and the REL12 BYOD journey both have documented hosted evidence **at this exact notebook blob**, with no restart.

One explanation teaches the opposite of what the learner sees. Section 6 tells the learner to expect the per-timestep
denoising loss to **rise** with the noise level and explains it as "highest at large timesteps". The recorded runs at this
blob show it **falling** monotonically by a factor of about 460, from 0.273 at t = 100 to 0.0006 at t = 900, which is the
expected behaviour of an ε-prediction model. That is the one Major finding (KCD-M1).

Six Minor findings follow. Two of them leave an applicable spec **MUST** unmet: the random split's independence
assumption is not stated (SPL3), and the notebook does not say that the public sample photographs may overlap the
pretraining data of the generator and the CLIP scorer (DAT9). The others concern a "ceiling" that the generated images
exceed, a stale "duration not yet recorded" line, an activity rerun that leaves an older activity block in the
evaluation report, and a BYOD minimum-size statement that the code does not honour.

None of these findings says the default run fails. The hosted evidence shows it passes.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Revision | `8fe8c4f` (merge of PR #2). The notebook blob equals the blob of the recorded hosted runs at `c86e3fe`; `c86e3fe..8fe8c4f` changes only `MODEL_CARD.md`, `README.md`, `STATUS.md`, `docs/release-verification.md` and `tutorials/README.md` |
| Profile / mode | `E2E` / `GUIDED`, declared in metadata and the opening |
| Spec declared / applied | 2.2 / 2.2 |
| Audience | Can run hosted-notebook cells and read basic Python; new to conditioned diffusion models |
| Prerequisites stated | Linux x86_64 GPU ≥ 15 GB (T4), about 31 GB disk, about an hour |
| Supported runtime | Google Colab T4 (primary), Kaggle T4, Linux Jupyter with CUDA; CPU-only explicitly unsupported |
| Promised outcomes | Depth hints from DPT; prompt embeddings from the prior; frozen held-out denoising loss, CLIP and depth-fidelity scores against references; a change-one-thing hint activity; bounded LoRA (rank 8, 1,646,592 parameters); paired before/after comparison from the exported adapter; fresh-process reload with parity; a new prompt; BYOD through the same stages |
| Generator | `tools/build_notebook.py` from `tools/notebook_template.py`; stage logic in `tools/tutorial_stages.py` (carried) |

### Evidence actually obtained

**Source inspection.** All 34 cells (11 code, 23 markdown), the carried stage runner, `samples.py` split logic,
`pipeline.evaluate`, the decoder scheduler config (`prediction_type: epsilon`), the template, `README.md`,
`tutorials/README.md`, `docs/release-verification.md`, `STATUS.md` and `MODEL_CARD.md`.

**Documented execution evidence (this exact blob).** `docs/release-verification.md` records three runs of blob
`10145c03…`: Google Colab T4 (2026-09-30, 11/11 code cells, no restart), a strict single-pass Kaggle T4 run
(2026-09-29, 1,156.2 s) and the REL12 BYOD journey (positive zip plus two refusals). The Kaggle executor outputs for the
strict default and BYOD runs are archived under `.agent/backups/kandinsky-kaggle-2026-09-29/out-strict/`;
`run_summary.json` there names commit `c86e3fe` and blob `10145c03…` with `runner.ok: true`. This review read the
`evaluation_report.json` files from those runs (sha256 in `source_manifest.json`) for the per-timestep losses and CLIP
values used below. The executed Colab notebook itself was not re-read.

**Direct execution (CPU, Windows, CPython 3.12.10, numpy 2.5.3, Pillow 11.3.0; no torch, no model weights).**
13 probes in `run_probes.py`: notebook parse/compile/hygiene, blob identity, carrier parity against the repository
(11 files, 0 mismatches), the isolated-install pattern, the repository's offline suite (56 passed, 5 skipped),
`tools/validate_release_assets.py` (PASS) and `tools/build_notebook.py --check` (up to date), and the carried
BYOD loader and splitter on synthetic zips. These static and validator-only runs are **not** execution evidence under
REL8.

**Not verified.** Any GPU stage in this review; the `mismatched` and `own` activity values on real weights (the release
record's CPU stub pre-flight is the only evidence); learner understanding.

### Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Well scaffolded: audience, how-to-use, roadmap, I/O contract, glossary, predictions, What to notice, Check your reasoning, troubleshooting, conclusion template. Section 6 misleads on the loss curve (KCD-M1) and the CLIP "ceiling" (KCD-m1) |
| Clean default | Documented execution evidence at this blob | Passed on Colab T4 and strict Kaggle T4, one pass, no restart, reload parity 0.0 / 0.0. Not re-run here |
| Active learning | Documented evidence for the default `flat` hint (own 0.9204 → flat 0.4705, baseline 0.2560); source inspection for reruns | `mismatched` / `own` on real weights not verified. A rerun leaves an older activity block in the evaluation report (KCD-m5) |
| Reuse and recovery | Documented REL12 evidence at this blob; direct CPU execution of the validator | Positive BYOD and two refusals documented; the two refusals reproduced here. The stated minimum dataset size is wrong (KCD-m6). Reload from files with parity is documented |

## 2. Separate judgments

| Dimension | Judgment |
|---|---|
| Technical correctness | Sound. Isolated environment, carrier parity, digest-pinned snapshots, safetensors only, one process per stage, paired seeded evaluation, validation-selected epoch, fresh-process reload with explicit tolerance. One rerun-state issue (KCD-m5) |
| Promise fulfilment | Every promised stage runs and is evidenced at this blob, including BYOD. The bit-identical conversion claim is correctly scoped out of the notebook and recorded separately |
| Scientific validity | Comparisons are paired and honest, and negative results are kept. Gaps: the split's independence assumption (KCD-m2) and pretraining overlap of the sample (KCD-m3) are unstated, and "ceiling" is the wrong word for one reference (KCD-m1) |
| Learner experience | Above fleet average. One explanation states the opposite of the observed loss curve (KCD-M1); one sample answer does not match the recorded numbers (KCD-m1) |
| Spec conformance | RUN1–RUN14, ST1–ST8, ENV1–ENV9, MOD1–MOD9, VER1–VER5, REL1–REL12 and GDL1–GDL15 are met on the evidence above. **Unmet MUSTs:** SPL3 (KCD-m2), DAT9 (KCD-m3), DAT12 (KCD-m6, the stated limit is inaccurate) |

## 3. Findings

### KCD-M1 — Major: Section 6 says the denoising loss rises with the noise level; it falls by about 460×

**Location:** Section 6 *What to notice* (cell 19): "the per-timestep loss rising with the noise level". Section 6
*Check your reasoning* (cell 21): "The denoising loss is highest at large timesteps, where almost all of the latent is
noise." Generator: `tools/notebook_template.py` lines 376 and 394.

**Observed issue:** the decoder's scheduler is `DDPMScheduler` with `prediction_type: "epsilon"`
(`weights/kandinsky-2-2-controlnet-depth/scheduler/scheduler_config.json`), and `pipeline.evaluate` scores the MSE of
the predicted noise. At a large timestep the noisy latent is almost pure noise, so predicting that noise is easy and the
error is small. At a small timestep the noise is a small perturbation of the image and is hard to separate, so the error
is large. The recorded run at this blob shows exactly that:

| t | 100 | 300 | 500 | 700 | 900 |
|---|---|---|---|---|---|
| frozen test MSE | 0.273275 | 0.065541 | 0.017646 | 0.003976 | 0.000596 |
| adapted test MSE | 0.272064 | 0.064637 | 0.017117 | 0.003731 | 0.000557 |

The BYOD run shows the same monotone fall, from 0.353 to 0.00086.

**Consequence:** the learner is told to expect the opposite of what the cell prints, and the sample answer supplies a
wrong mechanism ("almost all of the latent is noise" means the loss is high) for the notebook's principal metric. A
learner who trusts the answer learns the wrong thing about ε-prediction. One who trusts the output is left with
an unexplained contradiction at the first evaluation checkpoint. The frozen test mean is also about 76 % driven by
t = 100 (0.2733 / 5 of 0.0722), which the learner needs to know to read the before/after change in Section 9.

**Evidence:** documented execution evidence (strict Kaggle default and BYOD `evaluation_report.json`, blob `10145c03…`),
plus source inspection. Probe `P05`.

**Recommended correction:** in the template, change the *What to notice* to "the per-timestep loss **falling** as the
noise level rises", and rewrite the sample answer. For an ε-prediction model, predicting the noise is easy when the
latent is almost all noise (large t) and hard when only a little noise has been added (small t), so the mean is
dominated by the smallest timestep. Optionally, print each timestep's share of the mean in the `frozen` stage.

**Acceptance check:** the regenerated cells 19 and 21 state that loss decreases with timestep and give the ε-prediction
reason. No learner-facing text says the loss is highest at large timesteps. A hosted run's `by_timestep` output matches the
direction the prose states. `tools/build_notebook.py --check` passes.

### KCD-m1 — Minor: the CLIP "ceiling" is exceeded, and a sample answer contradicts the recorded label accuracy

**Location:** Section 6 (cell 19, "`real_photo_baseline` scores the real photographs the same way: the ceiling"),
Section 9 *What to notice* (cell 27, "the real-photo ceiling for CLIP"), Section 6 *Check your reasoning* (cell 21,
"often it cannot"). Generator: `tools/notebook_template.py` lines 367, 393 and 472.

**Observed issue:** in the recorded default run the frozen model's CLIP prompt similarity is **32.305** against the real
photographs' **30.25**. In the BYOD run it is 32.577 against 26.485. Generated images routinely beat this "ceiling",
because a generator renders the prompt more literally than a real photograph does. The Section 6 answer says the
pretrained model "often cannot" render the six birds distinctly. The recorded frozen label accuracy is 0.75 (9 of 12),
against chance 0.167 and real photographs 0.9167. Adaptation left it at 0.75 in both hosted runs.

**Consequence:** a learner told to read changes "beside the ceiling" sees a value above it with no explanation, and the
worked answer predicts a weakness the run does not show. This misleads on a secondary metric; the core workflow is
unaffected.

**Evidence:** documented execution evidence and source inspection. Probe `P06`.

**Recommended correction:** call `real_photo_baseline` the **real-photo reference** and say per metric what it bounds:
it is a meaningful reference for label accuracy and reference similarity, and generated images can exceed it on prompt
similarity. Rewrite the answer to "the frozen model already separates most of the six species; adaptation has
little room on 12 images, and one image is 0.083".

**Acceptance check:** no learner-facing text calls the real-photo scores a ceiling for prompt similarity. The Section 6
answer is consistent with a frozen label accuracy of about 0.75 on the default sample.

### KCD-m2 — Minor (spec MUST SPL3): the random split's independence assumption is unstated, and "hold out by caption" describes the opposite split

**Location:** Section 4 (cell 14). *Interpretation and limits* "Three things to carry to real data" (cell 33):
"**Hold out by caption, not by image:** the split keeps every caption's images across sets". BYOD splitter
`samples.split_dataset` (docstring "grouped by caption"). Generator: `tools/notebook_template.py` line 567;
`src/kandinsky_controlnet_depth_pipeline/samples.py` line 795.

**Observed issue:** both the sample and the BYOD split are seeded shuffles **stratified within caption**, so every
caption appears in train, validation and test (probe `P07`: 3 captions → all 3 in every split). The headline "hold out by
caption" names a group split that holds whole captions out, which is the opposite mechanism; only the explanatory clause
after it is right. The word "independent" appears nowhere in the notebook, so SPL3 ("Random splitting MUST be identified
as assuming sufficiently independent rows") is unmet. The sample corpus mitigates the risk by design ("one per observer
per species"), but BYOD de-duplicates only byte-identical images. A one-pixel variant of a photo was accepted next to
its original (probe `P07`), so burst shots or crops of one scene can land on both sides of the split.

**Consequence:** a learner applying the transfer advice may build the wrong split, and a BYOD user is not warned that
near-duplicate photos inflate the held-out numbers.

**Evidence:** source inspection plus direct CPU execution of `split_dataset` on synthetic images.

**Recommended correction:** in Section 4 and the BYOD text, state that the split assumes the photographs are
independent and that near-duplicates (bursts, crops, the same scene) should be removed or kept together. Rename the
transfer point to "**Stratify by caption**: every caption has held-out images, so the test measures new photographs of
seen captions, not unseen captions". Fix the `split_dataset` docstring.

**Acceptance check:** the notebook states the independence assumption before the split runs. No text says "hold out by
caption" for a within-caption stratified split. The BYOD contract mentions near-duplicates.

### KCD-m3 — Minor (spec MUST DAT9): possible pretraining overlap of the public sample is not stated

**Location:** Prerequisites (cell 1), Section 4 (cell 14), *Interpretation and limits* (cell 33).

**Observed issue:** the default sample is 60 public iNaturalist photographs. The CLIP scorer is
`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, trained on LAION-2B web images, and the Kandinsky 2.2 models were trained on
web-scale image–text data (`MODEL_CARD.md` line 155). Overlap with these public photographs cannot be ruled out, but
neither the notebook nor the model card says so for the sample (probe `P08`: no overlap statement in any notebook cell).

**Consequence:** the real-photo reference, the label accuracy and the reference similarity may be inflated by
memorisation, and the learner is not told. DAT9 is an applicable MUST.

**Evidence:** source inspection.

**Recommended correction:** add one sentence to *Interpretation and limits* (and the Section 4 data note): the sample
photographs are public web images that the generator's and the scorer's training data may include, so the CLIP numbers
on them may be optimistic. BYOD on private photographs avoids this.

**Acceptance check:** a learner-facing cell states the pretraining-overlap limitation for the default sample.

### KCD-m4 — Minor: the opening says the run time has not been recorded; the release record has it

**Location:** opening *Run all* paragraph (cell 0): "its duration on a T4 has not yet been recorded". `tutorials/README.md`
registry row, *Run-all* column: "not yet recorded for this revision (the previous in-kernel-install revision took
831.5 s …)". Generator: `tools/notebook_template.py` line 127.

**Observed issue:** `docs/release-verification.md` records 1,156.2 s for this blob on a strict Kaggle T4 (probe `P09`).

**Consequence:** a learner planning a session is told no measurement exists and sees an older, shorter one in the
registry. This is localised friction (UX12 asks for measured claims with their environment).

**Evidence:** source inspection.

**Recommended correction:** state "about 19 minutes on a Kaggle T4 (1,156.2 s, measured 2026-09-29), plus downloads".
Update the registry row and regenerate.

**Acceptance check:** neither the notebook nor `tutorials/README.md` says the duration is unrecorded while the release
record holds a measurement for the same blob.

### KCD-m5 — Minor: rerunning the activity leaves an older activity block in the evaluation report

**Location:** Section 7 (cells 22–23: "Try `mismatched` next"; "The activity does not feed into training or the final
comparison"); `tools/tutorial_stages.py` `stage_evaluate` (reads `state/activity.json` and writes it as `"activity"` into
`kandinsky_controlnet_depth_evaluation_report.json`) and `stage_activity` (overwrites the state and
`outputs/activity.json`).

**Observed issue:** after `Run all`, a learner following the invitation reruns cell 23 with `mismatched`.
`outputs/activity.json` then holds the `mismatched` result, while the evaluation report still embeds the `flat` summary
that existed when `evaluate` ran. Nothing tells the learner which file reflects which run.

**Consequence:** two exports in one run directory disagree about the activity. This is minor because the comparison
block is unaffected.

**Evidence:** source inspection (probe `P10`). Not executed on GPU.

**Recommended correction:** drop the activity block from the evaluation report, or write each activity result to a
per-hint file (`activity_<hint>.json`) and have the report list every hint run with its timestamp. Say so in Section 7.

**Acceptance check:** after `Run all` followed by a `mismatched` rerun of cell 23, no export presents the `flat` summary
as the current activity without labelling it.

### KCD-m6 — Minor (spec MUST DAT12): the stated BYOD minimum is accepted by the text but refused by the code

**Location:** opening *Bring Your Own Data* (cell 0): "at least four images, and at least one caption with three or more
images". Troubleshooting (cell 32): "one caption with three or more images". Code: `samples.split_dataset` with
`MIN_TRAIN_RECORDS = 4` and 20 % / 20 % per-caption validation and test shares. Generator: `tools/notebook_template.py`
line 132.

**Observed issue:** zips that satisfy the stated contract are refused (probe `P13`, direct execution of the carried
loader and splitter on synthetic 320 px PNGs):

| Layout | Meets stated contract | Result |
|---|---|---|
| 4 images, 1 caption | yes | refused: `split leaves 2 training records; at least 4 are required` |
| 5 images, 1 caption | yes | refused: `split leaves 3 training records; …` |
| 6 images, 2 captions × 3 | yes | refused: `split leaves 2 training records; …` |
| 6 images, 1 caption | yes | accepted (4 / 1 / 1) |
| 8 images, 2 captions × 4 | yes | accepted (4 / 2 / 2) |

**Consequence:** a user who prepares the documented minimum gets a refusal. The refusal message is actionable (it names
the rule), so this is friction rather than a dead end, but DAT12 requires the limits to be stated correctly before
upload.

**Evidence:** direct execution (CPU, validator and splitter only; no model) and source inspection.

**Recommended correction:** state the rule the code enforces, for example "at least 4 training images after a per-caption
split that keeps about 20 % for validation and 20 % for test, so at least 6 images; each caption with three or more
images contributes a held-out image". Add the `split leaves N training records` message to the troubleshooting row.

**Acceptance check:** every layout that meets the stated minimum is accepted by `split_dataset(load_byod_dataset(...))`.
The troubleshooting table names the training-record refusal.

### KCD-S1 — Suggestion: let the activity take a seed

Section 7's answer says "repeat with other seeds before drawing a firm conclusion", but no form field changes the seed.
An optional `ACTIVITY_SEED` (default = the Section 6 seed) would make that advice actionable without touching the
canonical path.

### KCD-S2 — Suggestion: show the per-timestep paired change in Section 9

The comparison prints the per-timestep table, but the prose reads only the mean, which is mostly the t = 100 term. One
sentence pointing at `denoising_mse_test_by_timestep` would let the learner see where adaptation moved the loss: in the
recorded run about −0.0012 at t = 100, −0.0002 at t = 700 and −0.00004 at t = 900.

## 4. Readiness

**Needs revision.** There is no Blocker, and the default and BYOD journeys have hosted evidence at this exact blob with
no restart. One Major remains (KCD-M1), and three applicable MUSTs are unmet: SPL3 (KCD-m2), DAT9 (KCD-m3) and DAT12
(KCD-m6). All seven findings are prose or small generator changes. The stage logic needs no change except optionally
KCD-m5. Because the notebook bytes would change, a regenerated notebook needs a new hosted `Run all` record before it
returns to Release-grade.

## 5. Verified versus inferred

- **Verified:** the notebook blob equals the hosted-evidence blob; carrier parity (11 files); no kernel install;
  static gates pass; the per-timestep losses and CLIP values come from the executor's own reports for this blob; the BYOD
  refusals and the minimum-size boundary were directly executed on CPU.
- **Inferred:** the ε-prediction explanation for KCD-M1 is standard diffusion behaviour that matches the scheduler config
  and the numbers, but no ablation was run. KCD-m5's stale block is read from source, not executed.
- **Only Kurt can confirm:** whether a 20 % / 20 % per-caption split is the intended BYOD contract (fix the text) or the
  text is the intended contract (fix the code).
- **Most likely to be wrong:** KCD-m3's severity. A reader may judge the DAT9 statement in the model card's general data
  section sufficient, or the overlap moot for a sanity-level tutorial.
