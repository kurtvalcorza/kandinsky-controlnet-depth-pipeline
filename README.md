# Kandinsky 2.2 ControlNet-depth Pipeline

DIMER-oriented pipeline for **Kandinsky 2.2 ControlNet-depth** (`kandinsky-community/kandinsky-2-2-controlnet-depth`,
the 1.25 B-parameter hint-conditioned UNet diffusion model): a photograph becomes a depth hint, and a prompt plus that
hint becomes a 512 × 512 image whose content follows the words and whose layout follows the photograph. The decoder is
pinned to an immutable Hugging Face commit together with the shared `kandinsky-community/kandinsky-2-2-prior`, the
depth estimator `Intel/dpt-large` and a CLIP scorer for evaluation. The repository exposes depth-conditioned
generation, a held-out denoising-loss, CLIP and depth-fidelity evaluation, a captioned-image contract with explicit
ceilings, a bounded LoRA fine-tuning contract with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model
Card Specification 1.2, and a standalone, guided `E2E` tutorial at DIMER Notebook Specification 2.2.

## Upstream alignment

- Model: `kandinsky-community/kandinsky-2-2-controlnet-depth`
- Revision: `08632524a2b7e3bed39a7901d92cdd07e37544d0` — the head of the upstream repository's **open, unmerged
  pull request #5** (safetensors conversion by patrickvonplaten). The upstream `main` branch
  (`4ecd717e8c9086cf4a16ca28b64894f70a42cd08`) ships only pickled `.bin` weights, which this package never loads.
  The conversion is bit-identical to `main` (checked by `tools/verify_conversion.py`; see `docs/WEIGHTS.md`).
- Prior: `kandinsky-community/kandinsky-2-2-prior` at `9fc51ad5732afc5d031724219d22e6c42179c5a8`
- Depth estimator: `Intel/dpt-large` at `bc15f29aa3a80d532f2ed650b5e16ac48d8958f9`, the `transformers` 5.17.0
  `depth-estimation` default, loaded explicitly by id and revision
- Scorer: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at `1a25a446712ba5ee05982a381eed697ef9b435cf` (evaluation only)
- Upstream weight licences: **Apache-2.0** for the decoder, prior and depth estimator; MIT for the scorer
- Runtime: `diffusers==0.40.0` + `transformers==5.17.0` + `peft==0.21.0` + `torch==2.14.0` — every loaded weight file
  is safetensors, **nothing is unpickled and no Hub-hosted code is executed**
- Repository adaptation: **E2E** (bounded LoRA fine-tuning of the UNet attention projections on captioned images whose
  depth hints come from the pinned estimator, with a portable safetensors adapter)

## Two things to know before you start

**The hint decides the layout, the prompt decides the content.** `compute_hints()` runs DPT on each 512² centre crop
and min-max normalises the relative depth to 0..255; `generate(prompts, hints)` takes one hint per prompt. The tutorial's
Section 7 changes only the hint (flat, mismatched or original) to show the split.

**Generation has no ground truth, and the numbers say what they are.** `evaluate()` reports the held-out *denoising
loss* with each record's own hint, at fixed timesteps with seeded noise, so frozen and adapted models see identical
inputs. `metrics.score_generations()` reports CLIP prompt similarity, label accuracy and similarity to real
photographs, and `real_photo_baseline()` the same on the real photographs — the ceiling. `metrics.score_depth_fidelity()`
reports the Pearson correlation and the scale/offset-aligned error between each generated image's estimated depth and
its hint, next to a mismatched-hint baseline. None of these is a human judgement of image quality.

## Quick start

```python
from kandinsky_controlnet_depth_pipeline import KandinskyDepthPipeline, fetch_sample_dataset, sample_prompts
from kandinsky_controlnet_depth_pipeline.metrics import ClipScorer, score_depth_fidelity, score_generations

pipe = KandinskyDepthPipeline.from_pretrained(use_lora=True, allow_download=True)  # verifies 3 snapshots, loads UNet + LoRA, MoVQ
splits = fetch_sample_dataset()                                    # 36 / 12 / 12 pinned CC0 iNaturalist bird photographs
records = splits["train"] + splits["validation"] + splits["test"]
pipe.compute_hints(records)                                        # DPT depth hints, cached by pixel digest
pipe.encode_prompts(sample_prompts(records)); pipe.release_prior()
print(pipe.evaluate(splits["test"])["denoising_mse"])              # frozen model, held-out, hint-conditioned
test = splits["test"]
images = pipe.generate([r["caption"] for r in test], [pipe.depth_hint(r) for r in test], seed=1000)["images"]
print(score_generations(ClipScorer(allow_download=True), images, references=test), score_depth_fidelity(pipe, images))
pipe.adapt(splits["train"], splits["validation"])                 # bounded LoRA fine-tuning, epoch kept by validation loss
pipe.save_artifact("outputs/adapter")
```

## Weights layout

```
weights/kandinsky-2-2-controlnet-depth/  README.md  model_index.json  scheduler/  movq/  unet/  dimer-base-manifest.json
                                         unet/diffusion_pytorch_model.safetensors  (git-ignored, 5.01 GB)
                                         movq/diffusion_pytorch_model.safetensors  (git-ignored, 271 MB)
weights/kandinsky-2-2-prior/             configs, tokenizer, dimer-base-manifest.json  (safetensors git-ignored, 10.57 GB)
weights/dpt-large/                       README.md  config.json  preprocessor_config.json  dimer-base-manifest.json
                                         model.safetensors  (git-ignored, 1.37 GB)
weights/clip-vit-b-32-laion2b/           config, tokenizer and preprocessor files  dimer-base-manifest.json  (605 MB model git-ignored)
weights/inat-birds/                      the 60 pinned photographs, cached on first fetch (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()`, `stage_missing_prior_files()` and `stage_missing_depth_files()`
(fetch only absent manifest entries, only at the pinned 40-character commits, with `allow_download=True`), then
`verify_snapshot()`, `verify_prior_snapshot()` and `verify_depth_snapshot()` (byte size + SHA-256 of every entry).
`docs/WEIGHTS.md` records the provenance of all four snapshots, including the conversion pull request.

## Sample data

`fetch_sample_dataset()` fetches 60 research-grade iNaturalist photographs of six North American birds (10 per species,
CC0 1.0) from the public open-data bucket, each pinned by byte size and SHA-256 in `SAMPLE_RECORDS`. Captions come from
one template per species. `build_sample_dataset` draws a seeded 6 / 2 / 2 stratified split per species (36 / 12 / 12)
and `check_split_disjoint` asserts no image appears twice. `write_dataset_csv` / `load_byod_dataset` support BYOD.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the 176 LoRA tensors — rank 8 on `to_q`, `to_k`, `to_v`, `to_out.0`,
1,646,592 parameters) and `manifest.json` with the decoder, prior and depth-estimator identities, the LoRA
configuration, the training configuration and history, and the file digest.
`KandinskyDepthPipeline.from_artifact(dir)` re-verifies the snapshots and checks the format, base model, estimator,
tensor scope and digest before deserialising.

## Conversion check

`python tools/verify_conversion.py` downloads `main`'s `.bin` files and the pinned safetensors at their exact commits,
verifies their SHA-256, and compares every tensor (keys, shapes, dtypes, `torch.equal`). It needs `torch`,
`safetensors` and `huggingface-hub`, about 11 GB of disk and no GPU. Its 2026-09-29 run on a Kaggle CPU kernel
found both components bit-identical; the record is in `docs/release-verification.md`.

## Tests

```
pip install -e . --no-deps
pytest
```

The core suite runs offline with only `numpy` and `pillow`: temporary manifests, synthetic images and a stub UNet.
`tests/test_adaptation.py` and `tests/test_diffusers_path.py` also exercise `torch`, `diffusers` and `peft` on tiny
random models when those libraries are installed, and are skipped otherwise.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/kandinsky-controlnet-depth-pipeline/blob/main/tutorials/kandinsky_controlnet_depth_colab.ipynb)

`tutorials/kandinsky_controlnet_depth_colab.ipynb` is declared `E2E` in `GUIDED` mode and is **standalone** (DIMER
Notebook Specification 2.2 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds
the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim, the four pinned identities and manifests,
and the runtime pins. It carries the §3.5 guided layer (how-to-use, roadmap, glossary, predictions, check-your-reasoning
answers, a change-one-thing activity, troubleshooting and a conclusion scaffold).

## Release status

**Candidate** — the notebook stops at its install cell on hosted runtimes that preload `numpy`, `protobuf` and `cuda-bindings` (Google Colab and Kaggle), because the pinned install replaces those loaded packages and the cell then asks for a manual runtime restart. Notebook Specification 2.2 RUN1 and RUN10 forbid a manual restart on the `Run all` path, so the tutorial is not release-ready. The Kaggle runs recorded in `docs/release-verification.md` completed only because the executor restarted the kernel automatically; they remain valid evidence for everything after the install cell. Found in a Colab `Run all` on 2026-09-29; the fix (an isolated, hash-locked environment for the tutorial stages) is in progress. See `STATUS.md` and `docs/release-verification.md`.

## Licensing

- Upstream weights: Apache-2.0 (`kandinsky-community/kandinsky-2-2-controlnet-depth`, `kandinsky-community/kandinsky-2-2-prior`, `Intel/dpt-large`); the scorer is MIT.
- Tutorial data: iNaturalist research-grade photographs, each CC0 1.0 (observers credited in `samples.py`).
- This repository's code and documentation: Apache-2.0 (`LICENSE`).

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
