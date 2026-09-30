---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: text-to-image
task: "Depth-conditioned text-to-image generation (Kandinsky 2.2 ControlNet-depth UNet + MoVQ, DPT depth hints)"
base_model: kandinsky-community/kandinsky-2-2-controlnet-depth
date_published: "2023-07-13"
date_published_source: "Hugging Face Hub commit `741f080424a3450bc27ea13c591a8773ad09daec` (2023-07-13) uploaded the current UNet checkpoint, `unet/diffusion_pytorch_model.bin` with SHA-256 `3418cd4f977d51ec902d095ff67482232c6097edfd91b474443148113b8f8a36`. The pinned safetensors files are a later conversion of that checkpoint, published on 2023-07-25 in the repository's open pull request #5."
---

# Kandinsky 2.2 ControlNet-depth — Depth-Conditioned Generation and Bounded LoRA Fine-Tuning (conversion commit `08632524`)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kandinsky--2--2--controlnet--depth-ffcc4d?style=flat)](https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-ai--forever%2FKandinsky--2-181717?style=flat&logo=github&logoColor=white)](https://github.com/ai-forever/Kandinsky-2)
[![Depth estimator](https://img.shields.io/badge/%F0%9F%A4%97%20Depth-Intel%2Fdpt--large-ffcc4d?style=flat)](https://huggingface.co/Intel/dpt-large)
[![arXiv ControlNet](https://img.shields.io/badge/arXiv-2302.05543-b31b1b.svg)](https://arxiv.org/abs/2302.05543)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth/blob/main/README.md)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are fetched unmodified from the Hugging Face Hub under their upstream Apache-2.0 licence; the accompanying code and notebook are Apache-2.0. Nothing here is validated for production, and no benchmark result is claimed.

> [!IMPORTANT]
> **The decoder weights come from an open, unmerged conversion pull request.** The upstream `main` branch (commit `4ecd717e8c9086cf4a16ca28b64894f70a42cd08`) ships only pickled `.bin` files. This pipeline pins the safetensors files of pull request #5, opened by patrickvonplaten, by its commit `08632524a2b7e3bed39a7901d92cdd07e37544d0`. `tools/verify_conversion.py` checked that conversion against the `main` weights: a Kaggle CPU run of `tools/verify_conversion.py` on 2026-09-29 found identical key sets, shapes, dtypes and values for all 740 UNet tensors (1,253,429,212 elements) and all 431 MoVQ tensors (67,832,495 elements).

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run, self-contained Google Colab notebook. It carries the repository's code in its own cells and runs end to end without cloning the repository. It installs nothing into the notebook kernel: every stage runs in an isolated environment built from a committed hash lock, so `Run all` needs no runtime restart:

- **Guided End-to-End Depth-Conditioned Fine-Tuning Notebook**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/kandinsky-controlnet-depth-pipeline/blob/main/tutorials/kandinsky_controlnet_depth_colab.ipynb) [`kandinsky_controlnet_depth_colab.ipynb`](https://github.com/kurtvalcorza/kandinsky-controlnet-depth-pipeline/blob/main/tutorials/kandinsky_controlnet_depth_colab.ipynb)  
  *The notebook verifies four pinned snapshots and fetches 60 digest-pinned CC0 iNaturalist bird photographs. It computes depth hints with the pinned DPT estimator and scores the frozen model by held-out denoising loss, CLIP and depth fidelity. It then changes only the hint in a guided activity, runs a bounded LoRA fine-tuning and compares both models on identical held-out inputs. Finally it exports the adapter as safetensors and verifies reload parity.*

---

#### Description

`kandinsky-community/kandinsky-2-2-controlnet-depth` is the depth-conditioned decoder of Kandinsky 2.2, a latent diffusion text-to-image model from ai-forever (Shakhmatov et al., 2023). This pipeline pins it at commit `08632524a2b7e3bed39a7901d92cdd07e37544d0`, the safetensors conversion described above. Generation runs in three steps:

1. The separately pinned prior `kandinsky-community/kandinsky-2-2-prior` maps a text prompt to a CLIP ViT-G/14 image embedding.
2. A depth hint is computed from a photograph by the pinned estimator `Intel/dpt-large` (DPT; Ranftl et al., 2021).
3. The conditional UNet (`UNet2DConditionModel`, 1,253,429,212 parameters, `addition_embed_type="image_hint"`) removes noise from a 64 × 64 × 4 latent, conditioned on the embedding and the hint. A MoVQ autoencoder (67,832,495 parameters) decodes the latent to a 512 × 512 RGB image.

The idea is ControlNet-style spatial conditioning (Zhang et al., 2023). The prompt decides what appears, and the hint decides where objects are and how near they seem. The UNet's `input_hint_block` encodes the 3 × 512 × 512 hint to a 4-channel map, which is concatenated with the latent, so the UNet has 8 input channels.

This repository adds the `KandinskyDepthPipeline` class in `src/kandinsky_controlnet_depth_pipeline/pipeline.py`, the captioned-image contract in `samples.py`, and CLIP and depth-fidelity scoring in `metrics.py`. The code verifies four snapshots before any model library is imported. It then computes hints, encodes prompts and releases the prior, and generates seeded images. It evaluates held-out denoising MSE and trains a rank-8 LoRA adapter on the UNet attention projections. It writes a safetensors adapter whose digest is checked before it is loaded again.

#### Intended Use and Limitations

The pipeline is a teaching and research reference for depth-conditioned generation and for adapting such a generator to a small captioned photo collection. It is small enough to read in full. It is bounded by explicit input limits, and it states what its scores do and do not measure. The subsections below give its tasks, its users and the uses it does not support.

###### Primary Intended Uses

The pipeline exposes two tasks:

- **Depth-conditioned generation.** The input is a prompt (1..1,000 characters) and a hint, which is a 512 × 512 `uint8` relative depth map. A hint usually comes from `estimate_depth` or `compute_hints` on a photograph. Further inputs are a seed, a step count (1..100, default `20`) and a guidance scale (1..20, default `4.0`). The output is one 512 × 512 RGB image per prompt, hint and seed.
- **Adaptation to captioned images.** The input is a list of `{id, image, caption}` records (4..2,000 records), whose hints are computed by the pinned estimator. The output is a LoRA adapter trained on the hint-conditioned noise-prediction objective, written as `adapter.safetensors` with a `manifest.json`.

The envisioned uses are teaching how spatial conditioning and parameter-efficient fine-tuning interact, and research on adapting a depth-conditioned generator to a narrow subject. A third use is producing illustrative synthetic images whose layout follows a reference photograph. The design role is a reference implementation that a reader's own code can embed or copy. It is not a production image service.

###### Primary Intended Users

The intended users are machine-learning engineers, researchers and students who work with diffusion models and parameter-efficient fine-tuning. The envisioned settings are research, teaching and self-hosted experimentation on a GPU the user controls.

The pipeline assumes these user competencies:

- how diffusion sampling and classifier-free guidance affect an image;
- that a relative depth map from an estimator is not measured depth;
- that a lower denoising loss does not imply better-looking images;
- that CLIP similarity and depth correlation are automated proxies, not human judgements;
- how to check the licence and consent status of the images used for hints or training.

The pipeline enforces only structural limits on records, prompts and hints. It cannot judge whether an image, caption or hint is appropriate, so it is not robust to careless or adversarial inputs.

###### Out-of-scope use cases

1. **Capability boundary:** the pipeline generates 512 × 512 images from a prompt and a depth hint, and fine-tunes a LoRA on the UNet. It does not perform text-only generation, inpainting, image-to-image or other control types such as edges or pose. It does not generate at other resolutions or produce metric depth.
2. **Input boundary:** training records must be images whose sides are within 256..4,096 px, with captions of 1..1,000 characters. A dataset must have 4..2,000 records, and record ids must be unique strings of at most 64 characters. Each image is resized so its shorter side is 512 px and centre-cropped to 512 × 512, and its hint is computed from that crop. Hints must be 512 × 512 `uint8` arrays. Prompts are refused outside 1..1,000 characters, steps outside 1..100 and guidance outside 1..20.
3. **Data-size boundary:** the adaptation is designed for tens to hundreds of images and at most 50 epochs. It is not a full fine-tuning recipe, and it is not tested on thousands of images.
4. **Decision boundary:** the pipeline is not for images presented as photographs of real events or identifiable people. It is not for any decision that treats a generated image as factual evidence, such as news, legal, insurance or scientific evidence.
5. **Provenance boundary:** the pipeline refuses weights that do not match the committed manifests. It does not load the upstream `.bin` files.

---

#### Factors

The pipeline's behaviour varies with the subject named in the prompt, the scene layout encoded in the hint, and the photographs used for adaptation. It also varies with the depth estimator's accuracy on those photographs and with the hardware it runs on. The subsections below describe the groups, instruments and environments that matter.

###### Groups

The pipeline is not human-centric by design. The bundled sample contains photographs of six North American bird species, and the evaluation measures no demographic attribute. The model can still depict people when a prompt asks for them. A hint taken from a photograph of a person also carries that person's pose and body outline into the generated image. The upstream training corpus is web image–text data that the upstream authors did not document by demographic group, so its representation of age, gender, skin tone or culture is unknown.

This repository therefore makes no group-level fairness claim. An operator who generates or adapts images depicting people takes on that audit. They should compare depictions across the relevant groups with fixed prompts and hints that vary only the described group.

###### Instrumentation

The upstream models were trained on image–text pairs collected from the web, captured by unknown cameras. The upstream authors do not document the capture instruments. The tutorial sample consists of 60 research-grade iNaturalist photographs, taken by volunteer observers with consumer cameras and phones. They vary in resolution, focus and lighting.

The depth estimator is a second instrument. DPT predicts relative depth, and each hint is min-max normalised per image, so 255 means the nearest thing in that photograph rather than a distance. Estimator errors, such as a blurred background read as near or a reflection read as far, become part of the condition the generator follows. The pipeline cannot detect such errors or camera defects. Captions are a third instrument: the sample captions come from one template per species and describe no pose, background or lighting.

###### Environment

**Operating environment.** Python 3.12 with the pinned packages `torch==2.14.0`, `torchvision==0.29.0`, `diffusers==0.40.0`, `transformers==5.17.0`, `peft==0.21.0`, `accelerate==1.15.0`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`, `numpy==2.5.3` and `pillow==11.3.0`. The UNet, prior, MoVQ and depth estimator run in float16 on CUDA. The LoRA parameters are kept in float32, and training uses float16 autocast. The tutorial notebook does not install these into its kernel: it builds a separate CPython 3.12.12 environment with a pinned `uv` from `tutorials/requirements-colab.lock.txt`, which locks those pins and all their dependencies to exact versions and SHA-256 digests for Linux x86_64 (the Linux `torch` 2.14.0 wheel is the CUDA 13.0 build), and runs each stage in its own process there. The tutorial requires a Linux x86_64 runtime with a CUDA GPU of at least 15 GB of memory, such as a 16 GB T4, and about 31 GB of free disk: 17.8 GB of pinned weights and about 12 GB for the isolated environment. The previous in-kernel-install notebook ran on a Kaggle T4 (see *Verification records*); the isolated-environment notebook ran on a Colab T4 and a Kaggle T4 on 2026-09-29 and 2026-09-30. On CPU the code runs in float32 but is impractically slow.

**Data environment.** Adaptation assumes that the training photographs resemble the images the user later wants to generate, in subject, framing and photographic style. Generation assumes a hint from the same estimator and the same crop as in training. Prompts and layouts far from the adapted data revert towards the base model's behaviour. Held-out denoising loss is meaningful only when the validation photographs come from the same distribution as the training photographs.

---

#### Metrics

The metrics are chosen because generation has no ground truth. Each metric measures one property of the model, and none of them measures image quality as a person would judge it. Together they cover the training objective, agreement with the prompt and with real photographs, and agreement with the depth hint. The subsections below define them and their limits.

###### Performance Measures

The code reports these measures, named as the code reports them:

1. **`denoising_mse`** from `evaluate`: the mean squared error between the UNet's noise prediction and the true Gaussian noise added to MoVQ latents of held-out photographs. The UNet receives each record's caption embedding and its own hint. Timesteps are `100`, `300`, `500`, `700` and `900`, with a per-timestep breakdown. It is the training objective on photographs the adapter never trained on, and it compares frozen and adapted models on identical inputs.
2. **`clip_prompt_similarity`**, **`label_accuracy`** and **`reference_similarity`** from `score_generations`: the cosine similarity × 100 between each generated image and its prompt; the fraction of images whose nearest caption is their own; and the similarity to held-out real photographs with the same caption. The scorer is the pinned CLIP ViT-B/32.
3. **`real_photo_baseline`**: the same three CLIP measures on the held-out real photographs, which is the ceiling a generator imitating them can approach.
4. **`depth_correlation`** and **`depth_aligned_mae`** from `score_depth_fidelity`: the pinned estimator measures the depth of each generated image, and both maps are divided by 255. The correlation is the Pearson correlation between that map and the hint. The aligned error is the mean absolute difference after a least-squares scale and offset alignment, in hint units from 0 to 1.
5. **`mismatched_depth_correlation`** and **`mismatched_depth_aligned_mae`**: the same numbers against another image's hint. They are the chance baseline for an image that ignores its hint.

The measures are complementary. The loss is sensitive to adaptation but shows nothing about appearance. CLIP measures describe content but ignore layout. Depth fidelity describes layout but ignores content, and it shares its estimator with the hints, so it rewards agreement with the estimator's own errors. No FID or human preference score is computed, because 12 held-out images are far too few for FID.

###### Decision thresholds

The pipeline applies two implicit decision rules:

- `label_accuracy` uses an `argmax` over the CLIP cosine similarities to the distinct captions, with no minimum similarity.
- Adaptation keeps the epoch with the lowest validation `denoising_mse`, including epoch 0, the frozen model. The kept adapter therefore never has a higher validation loss than the frozen model.

No acceptance threshold on any measure was set during development, and no quality or depth-fidelity threshold is shipped. A score from this pipeline cannot, on its own, decide whether an image is fit for use. The operator who deploys generated images owns any acceptance rule. A false accept, where a misleading image is published, usually costs more than a false reject, where a usable image is discarded. The operator should combine automated scores with human review and set any threshold on their own validation images.

###### Approaches to uncertainty and variability

Every number the tutorial reports comes from a single run on one seeded split of the sample: 36 training, 12 validation and 12 test photographs, with seed `42`. No repeated runs, cross-validation or bootstrap are performed, and no standard deviation or confidence interval is reported. A frozen-versus-adapted difference is one observation on 12 photographs and 12 generated images, not a population estimate.

Seeds control the split, the evaluation noise and latents, the training noise and ordering, and each generated image (`seed + index`). The remaining sources of run-to-run variability are these:

- **Non-deterministic GPU kernels and float16 autocast** can change low-order digits between runs, and more between GPU models.
- **Prompt encoding** seeds the prior's sampler with `prompt_seed(prompt)`, a 31-bit integer taken from the prompt's SHA-256. The same prompt gets the same prior seed in every process, but its embedding is still subject to GPU nondeterminism.
- **Depth hints** are deterministic for a given image up to the same GPU effects.

CLIP similarities and depth correlations are not probabilities, and `label_accuracy` is not calibrated. A caller who needs calibrated measures must label their own images and calibrate against them.

---

#### Ethical considerations and biases

No external ethics board or group review has assessed this pipeline. The considerations below are the developers' own, written from the documentation of the pinned upstream models and from what this repository's code does. They cover the data, uses that affect human life, the mitigations the code implements, the remaining risks and the uses that are unacceptable.

###### Data

The upstream Kandinsky 2.2 models were trained on large web-scale image–text datasets, which the upstream authors describe only at a high level. The DPT estimator was trained on a mix of depth datasets described in its paper. It is not ruled out that the training data includes personal images, faces, copyrighted works or other sensitive material. Whether it does is unknown.

This repository distributes code, snapshot manifests, configuration files and documentation. It does not distribute model weights in Git; they are fetched from the Hugging Face Hub at pinned commits. It does not distribute training images either. The tutorial downloads 60 CC0 1.0 photographs from the iNaturalist open-data bucket at run time, and the adapter it exports is trained only on those.

The operator is responsible for the images, captions and hints they supply. A depth hint can reveal the layout of a private room or a person's silhouette even when the photograph is not shared. The pipeline does not check inputs for faces, personal data, copyrighted content or confidential material. An adapter can memorise its training images, so an adapter trained on restricted images must be treated as restricted too.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life. It produces synthetic images, and no generated image should be treated as a record of a real person, place or event. Nobody has validated it for any such domain. Its only checks are unit tests, static validation and a tutorial designed around bird photographs.

Foreseeable misuse includes images that follow the exact layout of a real scene and could be mistaken for medical, forensic or news imagery. Such use would require, at minimum, human review of every image and disclosure that it is synthetic. It would also require validation by the responsible domain authority. This repository provides none of these.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_REVISION`, `PRIOR_REVISION`, `DEPTH_REVISION` and `SCORER_REVISION` are immutable 40-character commit hashes. Every file's byte count and SHA-256 must match its committed `dimer-base-manifest.json` before loading. Staging refuses a manifest that names a different model or revision, and refuses any revision that is not a 40-character commit.
2. **No executable serialization:** every weight file is safetensors, and the snapshot check rejects `.bin` and other code-bearing types. No pickle is deserialized, and no Hub-hosted code runs. The depth estimator is loaded by pinned id and revision, never as a task default.
3. **Conversion check:** `tools/verify_conversion.py` compares the pinned safetensors with the `main` `.bin` files tensor by tensor. It loads the `.bin` files with `torch.load(weights_only=True)`. On 2026-09-29 it found every UNet and MoVQ tensor identical (see *Verification records*).
4. **Adapter integrity:** `load_adapter` refuses an artifact whose `format` is not `org.valcorza.kandinsky-controlnet-depth.adapter.v1`. It also refuses a different base model or depth estimator, a tensor set outside the LoRA scope, or an `adapter.safetensors` digest that differs from its manifest.
5. **Input integrity:** `validate_dataset`, `validate_prompts` and `validate_hint` reject malformed records, prompts and hints before any model runs.
6. **Bounded adaptation:** `adapt` refuses more than 50 epochs or a learning rate above `1e-2`, and keeps the epoch with the lowest validation loss.
7. **Reproducibility:** splits, noise and generated images are seeded, and prompt encoding is seeded from SHA-256 rather than Python's salted `hash()`. Runtime packages are pinned exactly in `pyproject.toml`; the notebook installs them, with every transitive dependency, from a hash lock into an isolated environment and never into the hosted runtime's own interpreter.

The pipeline has no content filter or safety checker on prompts or generated images. It adds no watermark or provenance metadata to generated images.

###### Risks and harms

1. **Misleading synthetic imagery.** A depth hint lets the model reproduce the exact layout of a real scene, which makes a fake more convincing. Third parties who see an image without disclosure bear the harm. It is likely under normal use, because no watermark is added, and its magnitude ranges from confusion to serious harm when an image is used as evidence.
2. **Harmful or non-consensual content.** No content filter runs, so a prompt with a hint from a real person's photograph can produce violent, sexual or defamatory depictions in that person's pose. The people depicted bear the harm, which can be severe.
3. **Privacy leakage through hints.** A hint or an adapter can carry the layout of a private space or a person's silhouette. The people and places photographed bear the harm. It is likely when hints are computed from personal photographs and shared.
4. **Bias amplification.** Web-trained generators reproduce stereotypes, and an adapter trained on a skewed set of images narrows the output further. The groups depicted and the viewers bear the harm.
5. **Training-data leakage.** A LoRA trained on a few images can reproduce them closely, and the data subjects and rights holders bear the harm.
6. **Automation bias.** Users may read a rising CLIP score, a high depth correlation or a falling loss as proof of better images. The downstream audience then bears the cost of worse outputs.
7. **Conversion from an unmerged pull request.** The pinned safetensors are bit-identical to the upstream `.bin` weights at `4ecd717e`, so outputs match the upstream checkpoint. The files still come from a pull request the upstream maintainers have not merged; if upstream later publishes a different checkpoint, results obtained with this pin will differ from it. Users comparing against upstream releases bear that harm.

###### Use cases

The following uses are unacceptable even where the pipeline would work:

1. creating sexual or intimate imagery of real people, or any sexual imagery of minors, including from a hint of their photograph;
2. creating images of real people, places or events presented as authentic, including disinformation, fabricated evidence and impersonation;
3. generating harassment, hate imagery or material intended to intimidate or demean a person or group;
4. surveillance, biometric identification or profiling, and computing hints or training adapters from photographs of people or private spaces without consent;
5. producing images used to discriminate in employment, housing, credit, insurance, education or healthcare access;
6. deceptive, manipulative or fraudulent applications, such as fake product photographs or fake identity documents;
7. any use that violates the upstream Apache-2.0 licences, the rights attached to the input images, or applicable law.

## Immutable provenance

- Model: `kandinsky-community/kandinsky-2-2-controlnet-depth`
- Revision: `08632524a2b7e3bed39a7901d92cdd07e37544d0`, the head of the repository's open, unmerged pull request #5 ("Adding `safetensors` variant of this model", patrickvonplaten, 2023-07-25)
- Upstream `main` at `4ecd717e8c9086cf4a16ca28b64894f70a42cd08` ships only `.bin` weights: `unet/diffusion_pytorch_model.bin` (5,013,996,233 bytes, SHA-256 `3418cd4f977d51ec902d095ff67482232c6097edfd91b474443148113b8f8a36`) and `movq/diffusion_pytorch_model.bin` (271,492,131 bytes, SHA-256 `772e09739d742ddee6807add2d3c2fd2a32db53896b5d07a92c729d8c879ce59`). These files are never staged or loaded.
- Manifest: `weights/kandinsky-2-2-controlnet-depth/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 7 files, `totalBytes` 5285192144
- UNet `unet/diffusion_pytorch_model.safetensors` (5,013,798,992 bytes) SHA-256: `6549f8c8471357ed8ed6b700a80ffdd8fe45bd5b54f4c486d7f81ac9fe5f343b`; 1,253,429,212 parameters in 740 tensors
- MoVQ `movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes) SHA-256: `43a5860fea195a7116f2471396c5cc9535fade9b63c4857d8a192ffd924b7002`; this digest is identical to the MoVQ in the pinned `kandinsky-community/kandinsky-2-2-decoder` snapshot
- Conversion equivalence: the pinned safetensors are bit-identical to the `main` `.bin` weights: a Kaggle CPU run of `tools/verify_conversion.py` on 2026-09-29 found identical key sets, shapes, dtypes and values for all 740 UNet tensors (1,253,429,212 elements) and all 431 MoVQ tensors (67,832,495 elements)
- Prior: `kandinsky-community/kandinsky-2-2-prior` at `9fc51ad5732afc5d031724219d22e6c42179c5a8`; manifest `weights/kandinsky-2-2-prior/dimer-base-manifest.json`, 14 files, `totalBytes` 10574964619
- Depth estimator: `Intel/dpt-large` at `bc15f29aa3a80d532f2ed650b5e16ac48d8958f9` (Apache-2.0), the default model of the `transformers` 5.17.0 `depth-estimation` task, loaded explicitly; manifest `weights/dpt-large/dimer-base-manifest.json`, 4 files, `totalBytes` 1367465032
- Scorer (evaluation only): `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at `1a25a446712ba5ee05982a381eed697ef9b435cf` (MIT); manifest `weights/clip-vit-b-32-laion2b/dimer-base-manifest.json`, 9 files, `totalBytes` 608782299

## Input/output contract

- `KandinskyDepthPipeline.from_pretrained(device=None, weights_dir=None, prior_dir=None, depth_dir=None, allow_download=False, use_lora=False)`: stages and verifies the decoder, prior and depth snapshots; loads the UNet, MoVQ and scheduler config.
- `compute_hints(records) -> dict`, `estimate_depth(images) -> list[np.ndarray]`, `depth_hint(record) -> np.ndarray`: 512 × 512 `uint8` relative depth maps from the pinned estimator.
- `encode_prompts(prompts) -> dict`, `release_prior() -> bool`: prompt embeddings from the prior, cached; the prior is then dropped.
- `generate(prompts, hints, *, seed=0, steps=20, guidance_scale=4.0) -> dict`: one image per prompt and hint.
- `evaluate(records, *, seed=0) -> dict`: held-out hint-conditioned denoising MSE.
- `adapt(train, val=None, *, epochs=4, lr=1e-4, batch_size=1, seed=0) -> dict`: bounded AdamW LoRA fine-tuning.
- `save_artifact(path, metadata=None) -> dict` and `from_artifact(path, ...)`: `adapter.safetensors` (176 LoRA tensors, 1,646,592 parameters) and `manifest.json`.
- `metrics.score_generations`, `metrics.score_depth_fidelity`, `metrics.real_photo_baseline`, `metrics.real_photo_depth_ceiling`.

## Deployment notes

| Field | Status |
|---|---|
| Licence | Apache-2.0 for the decoder, prior and depth estimator; MIT for the scorer; code Apache-2.0 |
| Weights | About 17.2 GB served (5.29 GB decoder, 10.57 GB prior, 1.37 GB depth estimator), plus the 0.6 GB evaluation scorer |
| Decoder source | safetensors from open pull request #5, pinned by commit; bit-identical to `main`'s `.bin` weights (checked 2026-09-29) |
| Remote code | Not required: standard `diffusers`, `transformers` and `peft` classes |
| Executable serialization | None loaded: safetensors only |
| Runtime | PyTorch 2.14, float16 on CUDA |

## Verification records

`docs/release-verification.md` holds the procedure and every record. The current notebook, which runs every stage in an isolated hash-locked environment, was run three times at commit `c86e3fe`:

- **Date:** 2026-09-30
- **Subject:** `tutorials/kandinsky_controlnet_depth_colab.ipynb` at commit `c86e3fe`, blob `10145c037440`
- **Runtime:** Google Colab, Tesla T4 (15,360 MiB); the notebook kernel ran Python 3.13.15, and the isolated environment ran Python 3.12.12 with `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0` and `peft 0.21.0`
- **Procedure:** `Run all` from a fresh runtime with the form fields at their defaults
- **Observed result:** 11 of 11 code cells ran in one pass without error or restart. Held-out test `denoising_mse` was 0.072207 for the frozen model and 0.071639 after adaptation. `depth_correlation` was 0.864 frozen and 0.859 adapted, against 0.256 for a mismatched hint. A reload in a fresh process reproduced the adapted model's results exactly.
- **Caveats:** one run on one seeded split. This is sample-sanity evidence, not a benchmark.

The same commit also passed a Kaggle T4 run in strict single-pass mode (a restart request fails the run), with the same results, and the BYOD journey: 12 representative photographs were carried through every stage, and a `captions.csv` without its `caption` column and a 200 × 200 image were each refused with the validator's message.

Three earlier executions are also recorded, all on 2026-09-29 at commit `f4fe86a`. The conversion check does not depend on the notebook and stays valid. The two notebook runs are of the **previous** notebook revision, which installed its pins into the kernel and needed a manual restart on hosted runtimes.

The conversion check:

- **Date:** 2026-09-29
- **Subject:** `tools/verify_conversion.py` at commit `f4fe86a`, blob `c8640b27f0b3` (full identifiers in `docs/release-verification.md`)
- **Runtime:** Kaggle CPU kernel, Python 3.12.13, `torch 2.10.0+cpu`
- **Procedure:** the script was fetched at that commit and blob-verified, then run. It downloaded `main`'s `.bin` files and the pinned safetensors at their exact commits and verified every download against its recorded SHA-256.
- **Observed result:** all 740 UNet tensors (1,253,429,212 elements) and all 431 MoVQ tensors (67,832,495 elements) have identical key sets, shapes, dtypes and values.
- **Caveats:** this proves the files are the same checkpoint. It says nothing about the checkpoint's quality.

The clean-runtime run of the previous tutorial notebook:

- **Date:** 2026-09-29
- **Subject:** `tutorials/kandinsky_controlnet_depth_colab.ipynb` at commit `f4fe86a`, blob `671cf4c8f722`
- **Runtime:** Kaggle batch kernel on a Tesla T4 (15,360 MiB), Python 3.12.13, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`
- **Procedure:** the notebook was fetched at that commit and run with `Run all` in a fresh interpreter, with an empty Hugging Face cache, no repository checkout and the form fields at their defaults. The install cell's restart guard fired once because the kernel had preloaded older `numpy` and `protobuf`, and the kernel was restarted and run again from the top.
- **Observed result:** 12 of 12 code cells ran without error in 831.5 s. Held-out test `denoising_mse` was 0.072207 for the frozen model and 0.071615 after adaptation. `depth_correlation` was 0.864 frozen and 0.859 adapted, against 0.256 for a mismatched hint. `depth_aligned_mae` was 0.103 frozen and 0.109 adapted, against 0.242 for a mismatched hint. `label_accuracy` rose from 0.75 to 0.8333, against 0.9167 for the real photographs. The reloaded adapter reproduced the in-memory results exactly.
- **Caveats:** one run on one seeded split with 12 held-out photographs. This is sample-sanity evidence, not a benchmark.

The BYOD branch at the same commit:

- **Date:** 2026-09-29
- **Subject:** the same notebook and commit, with `USE_BYOD = True` and `BYOD_PATH` set in the executed copy only
- **Runtime:** as above
- **Procedure:** a zip of 12 CC0 research-grade iNaturalist photographs (6 Northern Cardinal, 6 Blue Jay) with a `captions.csv` was built inside the kernel, each photograph checked against a pinned SHA-256. After `Run all`, the committed Section 4 source was re-run against two incompatible zips.
- **Observed result:** 13 of 13 code cells ran without error in 697.7 s. The 12 records were split 8 / 2 / 2 by caption and passed through depth-hint extraction, fine-tuning, evaluation, export and an exact reload. A `captions.csv` without its `caption` column and a 200 × 200 image were each refused with a message naming the failed rule, before any model ran on them.
- **Caveats:** with one test photograph per caption, these numbers show that the BYOD path runs, not how well the model adapts to such data.

The offline unit tests, the static validator `tools/validate_release_assets.py`, and a CPU test that runs the `diffusers` and `peft` code paths on a tiny randomly initialised model are checks of the code, not executions of the pipeline. The tutorial follows DIMER Notebook Specification 2.2 as a standalone notebook.

## References

- Shakhmatov, A., et al. (2023). Kandinsky 2.2. ai-forever. https://github.com/ai-forever/Kandinsky-2
- Zhang, L., Rao, A., & Agrawala, M. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. ICCV. https://arxiv.org/abs/2302.05543
- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers for Dense Prediction. ICCV. https://arxiv.org/abs/2103.13413
- Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Pinned repositories: https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth · https://huggingface.co/kandinsky-community/kandinsky-2-2-prior · https://huggingface.co/Intel/dpt-large · https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
