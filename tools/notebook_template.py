"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier, §3.5 guided layer).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline modules
(pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package; `infrastructure: True` labels and collapses them.

This template configures an E2E depth-conditioned text-to-image fine-tuning workflow: the pinned Kandinsky 2.2
ControlNet-depth decoder, the shared diffusion prior, the DPT depth estimator and the CLIP scorer are staged and
digest-verified; 60 pinned CC0 iNaturalist bird photographs are fetched, validated and split; the estimator turns
every photograph into a depth hint and the prior turns every caption into an image embedding (the prior is then
released); the frozen model is scored (held-out denoising loss, CLIP-scored and depth-scored generations) against
the real-photo reference and a mismatched-hint baseline; one variable (the hint) is changed in a guided activity;
a bounded LoRA fine-tuning runs in the kernel; the held-out scores are read again in a paired comparison; a new
prompt is rendered on an existing depth layout; and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "kandinsky-controlnet-depth-pipeline"
NOTEBOOK = "kandinsky_controlnet_depth_colab.ipynb"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/{NOTEBOOK}",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kandinsky--2--2--controlnet--depth-ffcc4d?style=flat",
        "https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-ai--forever%2FKandinsky--2-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/ai-forever/Kandinsky-2",
    ),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", "https://www.apache.org/licenses/LICENSE-2.0"),
]

INTRO = """Kandinsky 2.2 ControlNet-depth is a version of the Kandinsky 2.2 latent diffusion decoder that was retrained to follow a **depth map** as well as a text prompt. A frozen diffusion prior turns the prompt into a CLIP image embedding; a 1.25 B-parameter UNet removes noise from a small latent image while reading that embedding and a 512 × 512 depth hint; a MoVQ decoder turns the latent into pixels. The hint is produced by a separate model, the DPT depth estimator `Intel/dpt-large`, from an ordinary photograph. The result is an image whose *content* follows the words and whose *layout* follows the photograph's depth.

### Who this notebook is for

This notebook is for a learner who can open a hosted notebook, run cells and read basic Python, and who is new to conditioned diffusion models. No prior experience with diffusion training is assumed; the terms are explained where they first matter and collected in the glossary below. You need a GPU runtime (a free Colab T4 is enough) and about fifty minutes, most of it downloads.

### How to use this notebook

1. In Colab choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Nothing needs to be edited for the default path.
2. Cells titled **Infrastructure** (Sections 1–3) install packages, carry the repository's code and download the pinned models. They are collapsed; run them without studying them.
3. Sections 4–10 are the lesson. Each principal step starts with a **question** and, where it makes sense, asks you to **predict** the result first. After the cell, read **What to notice**, then open **Check your reasoning** to compare your answer.
4. Values you may change are **form fields** (the `# @param` lines). The canonical run uses their defaults; Section 7 is the place to change one of them deliberately.
5. Every output is written under `outputs/` in the working directory.

### Roadmap

| Section | Kind | What happens |
|---|---|---|
| 1–3 | *Engineering* | install pinned packages, carry the code, verify four model snapshots |
| 4 | *Core concept* | real captioned photographs, validation, a seeded split |
| 5 | *Core concept* | photograph → depth hint; caption → image embedding |
| 6 | *Evaluation practice* | the frozen model: loss, generations, three kinds of score |
| 7 | *Core concept* | **change one thing**: give the model a different hint |
| 8 | *Core concept* | bounded LoRA fine-tuning |
| 9 | *Evaluation practice* | the paired before/after comparison |
| 10 | *Engineering* | a new prompt, adapter export and fresh reload |
| 11 | *Engineering* | troubleshooting |

**Fast path:** if time is short, run everything and read only Sections 5, 6, 7 and 9.

### Input → Model/System → Output

`photograph` → **DPT depth estimator** → `depth hint (512 × 512)`; `caption` → **diffusion prior** → `image embedding`; `(hint, embedding, noise)` → **UNet + MoVQ** → `512 × 512 generated image`. For adaptation: `{id, image, caption}` records → hints + embeddings + latents → **LoRA training** → `adapter.safetensors` + `manifest.json`.

<details>
<summary><b>Glossary</b> (open when a term is unfamiliar)</summary>

- **Latent diffusion:** the model works on a compressed 64 × 64 × 4 *latent* instead of pixels; generation starts from noise and removes a little of it at each *step*.
- **Depth hint:** a greyscale map where brighter means nearer. Here it is estimated by DPT, so it is *relative* depth, not metres.
- **ControlNet-style conditioning:** an extra input (the hint) that constrains *where* things are, while the prompt says *what* they are.
- **Prior / image embedding:** the prior turns text into the CLIP image embedding the decoder expects.
- **Classifier-free guidance:** each step is computed with and without the prompt and pushed towards the prompt by `guidance_scale`.
- **Denoising loss (MSE):** how far the model's noise prediction is from the true noise added to a real photograph's latent — the training objective.
- **LoRA:** small low-rank matrices added beside frozen weights; only they are trained.
- **CLIP similarity:** cosine similarity between a CLIP embedding of an image and of a text (or another image), × 100.
- **Depth fidelity:** how closely the estimated depth of a generated image follows the hint it was given (correlation, and error after aligning scale and offset).
- **Held-out:** records never used for training or for choosing the kept epoch.
</details>

Two properties are handled in the open. **The prior pipeline can be released before training:** Section 5 encodes every prompt the notebook uses and then drops the prior, so the UNet, the depth estimator, the scorer and a training graph fit comfortably on a 16 GB GPU. **Generation has no ground truth**, so the notebook reads three kinds of number: the held-out *denoising loss*, CLIP scores against the real-photo ceiling, and *depth fidelity* against a mismatched-hint baseline. None of these is a human judgement of image quality.

**About the decoder weights.** The upstream repository's `main` branch ships only pickled `.bin` files. This notebook downloads the safetensors files of the repository's open, unmerged conversion pull request, addressed by its immutable commit and verified by SHA-256. Whether that conversion is bit-identical to `main` is checked separately by `tools/verify_conversion.py` in the repository; that check is recorded as pending in the model card."""

TEMPLATE = {
    "package": "kandinsky_controlnet_depth_pipeline",
    "repo_name": REPO,
    "stem": "kandinsky_controlnet_depth",
    "notebook_name": NOTEBOOK,
    "profile": "E2E",
    "mode": "GUIDED",
    "infrastructure": True,
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) installs the pinned "
        "dependencies, stages and digest-verifies four pinned snapshots from the Hub — the 5.29 GB Kandinsky 2.2 ControlNet-depth "
        "decoder (UNet + MoVQ), the 10.57 GB Kandinsky 2.2 diffusion prior, the 1.37 GB DPT depth estimator and a 0.6 GB CLIP "
        "scorer — loads the UNet in float16 with an untrained LoRA adapter attached, fetches 60 CC0 iNaturalist bird photographs "
        "as digest-verified JPEGs (6 MB, no credential), validates them and splits them 36 / 12 / 12 by seed, turns every photograph "
        "into a depth hint and every caption into an image embedding (then releases the prior), scores the frozen model — the "
        "held-out denoising loss, and twelve depth-conditioned generations scored for prompt alignment, similarity to the held-out "
        "photographs and depth fidelity against a mismatched-hint baseline — runs a guided change-one-thing activity on the hint, "
        "runs a bounded LoRA fine-tuning (4 epochs over 36 images), scores the adapted model on identical inputs, renders a new "
        "prompt on an existing depth layout, exports the adapter as safetensors with a manifest, and reloads that artifact into a "
        "fresh pipeline to verify parity. The default path needs no repository clone, no DIMER worker or service, no credential, "
        "no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). Model time on a T4 is an estimate of about twenty "
        "minutes after roughly 18 GB of downloads; it has not yet been measured."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "captioned photographs as a zip holding `captions.csv` (columns `id`, `file`, `caption`) beside the image files (JPEG or "
        "PNG, shorter side 256..4096 px; at least four images, and at least one caption with three or more images so a held-out "
        "record exists). Set `BYOD_PATH` to a zip already in the runtime to skip the upload dialog. Your photographs get depth hints "
        "from the same estimator, are split by caption, and flow through the same contract — validation, hints and prompt "
        "encoding, frozen baseline, adaptation, held-out evaluation, generation, artifact export and reload parity. The expected "
        "schema, the ceilings and the privacy guidance are stated in the Prerequisites and in Section 4, and uploaded files stay "
        "inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "KandinskyDepthPipeline",
    "model_load": "KandinskyDepthPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, prior_dir=PRIOR_WEIGHTS_DIR, depth_dir=DEPTH_WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), use_lora=True)",
    "weights_key": "kandinsky-2-2-controlnet-depth",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "rewrites": [
        [
            r"^_WEIGHTS_ROOT = Path\(__file__\)[^\n]*$",
            '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ]
    ],
    "extra_weights": [
        {
            "key": "kandinsky-2-2-prior",
            "var": "PRIOR_MANIFEST",
            "dir": "PRIOR_WEIGHTS_DIR",
            "identity": ["PRIOR_ID", "PRIOR_REVISION"],
            "stage": "stage_missing_prior_files",
            "verify": "verify_prior_snapshot",
        },
        {
            "key": "dpt-large",
            "var": "DEPTH_MANIFEST",
            "dir": "DEPTH_WEIGHTS_DIR",
            "identity": ["DEPTH_ID", "DEPTH_REVISION"],
            "stage": "stage_missing_depth_files",
            "verify": "verify_depth_snapshot",
        },
        {
            "key": "clip-vit-b-32-laion2b",
            "var": "SCORER_MANIFEST",
            "dir": "SCORER_WEIGHTS_DIR",
            "identity": ["SCORER_ID", "SCORER_REVISION"],
            "stage": "stage_missing_scorer_files",
            "verify": "verify_scorer_snapshot",
        },
    ],
    "runtime_imports": ["torch", "diffusers", "transformers", "peft"],
    "title": "Kandinsky 2.2 ControlNet-depth — DIMER guided notebook: depth-conditioned generation and LoRA fine-tuning (standalone)",
    "badges": BADGES,
    "capability": "depth-conditioned text-to-image generation (photograph → depth hint → prompt + hint → 512 × 512 image), held-out denoising-loss, CLIP and depth-fidelity evaluation, and bounded LoRA fine-tuning to a set of captioned photographs",
    "intro": INTRO,
    "learning_objectives": (
        "after this notebook you should be able to **explain** how a depth hint and a text prompt divide the work of "
        "conditioning a diffusion model; **produce** a depth hint from a photograph with a pinned estimator and **inspect** it; "
        "**predict** and then **observe** what happens to a generation when only the hint changes; **interpret** a held-out "
        "denoising loss, CLIP scores against a real-photo ceiling and a depth-fidelity score against a mismatched-hint baseline; "
        "**run** a bounded LoRA fine-tuning with explicit hyperparameters and **compare** the adapted and frozen models on "
        "identical held-out inputs; and **verify** that an exported safetensors adapter reloads against the pinned base with "
        "the same outputs."
    ),
    "exclusions": (
        "other ControlNet conditions (edges, pose, segmentation), image-to-image or inpainting, resolutions other than 512 × 512, "
        "metric (absolute) depth, full fine-tuning, safety filtering of prompts or images, human preference benchmarks, prompt "
        "engineering, and any claim that a CLIP score, a depth-fidelity score or a denoising loss measures image quality. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Audience:** comfortable running notebook cells and reading short Python; no diffusion-training experience needed.",
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU of at least 15 GB and Python 3.12). The prior runs in float16 while it encodes prompts and is then released; the UNet runs in float16 (2.5 GB) with the LoRA parameters in float32; the MoVQ decoder and the DPT estimator run in float16. CPU-only runtimes are not supported for this notebook. About 25 GB of disk is needed for the snapshots.",
        "- **Knowledge:** what a diffusion model does at inference (noise → latent → pixels over several steps), what classifier-free guidance is, what a LoRA adapter changes and what it does not, and why a training loss is not a quality score. The glossary above covers each term.",
        "- **Weights:** the decoder, the prior, the depth estimator and the CLIP scorer are all loaded from safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers` and `peft` on PyPI. The decoder safetensors come from an open, unmerged conversion pull request of the upstream repository, pinned by commit. Upstream weights are Apache-2.0; the scorer is MIT.",
        "- **Data contract:** a record is `{id, image, caption}` — an RGB image with shorter side 256..4096 px (resized so the shorter side is 512 px and centre-cropped to 512 × 512; the crop is reported) and a caption of 1..1000 characters. Its depth hint is computed from the cropped image. Validation is structural: nothing checks that a caption describes its image, that a depth map is plausible, or that the model can render the prompt.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. A depth map can still reveal the layout of a private space. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, validation and splits\n\n"
                "**Section type:** Core concept (data).  \n"
                "**Question:** what does the model learn from, and how do we keep some of it aside for an honest test?\n\n"
                "The default dataset is 60 research-grade iNaturalist photographs of six common North American birds — 10 per "
                "species, one per observer per species, every one CC0 — fetched by photo id from the open-data bucket and "
                "refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). Each photo's caption is generated from its "
                "species by one template. `build_sample_dataset` draws a seeded stratified split — 6 / 2 / 2 per species for "
                "training, validation and test — and `dataset_manifest` validates every split, checks that no image appears "
                "twice and records a digest. **Validation** is the check that runs *before* any model: it refuses a malformed "
                "record with a message that names the broken rule.\n\n"
                "**What to notice:** 36 / 12 / 12 records, six distinct captions, a shorter side around 300..500 px (every photo "
                "is centre-cropped to 512²), a written `outputs/{stem}_sample_captions.csv` in the shape BYOD expects, and three "
                "refusal probes — a missing caption, a 200 px image, a duplicate id — each rejected before the model runs."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from IPython.display import display\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH:\n"
                "        byod_path = Path(BYOD_PATH)\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/inat-birds')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'captions': dataset_report['splits']['train']['n_captions'], 'disjoint': dataset_report['disjoint']}})\n"
                "print({{'shorter_side': dataset_report['splits']['train']['shorter_side'], 'centre_cropped': dataset_report['splits']['train']['centre_cropped'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "print({{'first_test_record': validate_inputs(test_records[0]), 'caption': test_records[0]['caption']}})\n"
                "prompts = sample_prompts(train_records)\n"
                "print({{'prompts': prompts}})\n"
                "sample_csv = write_dataset_csv(test_records, 'outputs/{stem}_sample_captions.csv')\n"
                "print({{'sample_csv': str(sample_csv)}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'missing caption': [{{'id': r['id'], 'image': r['image']}} for r in train_records[:4]],\n"
                "    'image too small': [{{**train_records[0], 'image': Image.new('RGB', (200, 200))}}, *train_records[1:4]],\n"
                "    'duplicate id': [train_records[0], *train_records[:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Depth hints and prompt embeddings\n\n"
                "**Section type:** Core concept.  \n"
                "**Question:** what exactly does the model receive besides the noise?\n\n"
                "Two frozen helper models prepare the two conditions. `pipe.compute_hints` runs the pinned DPT estimator on every "
                "photograph after the same 512² centre crop the training uses; the predicted relative depth is resized back to "
                "512² and min-max normalised to 0..255 (brighter = nearer). The UNet reads it as three identical channels divided "
                "by 255. `pipe.encode_prompts` loads the diffusion prior and turns each distinct caption — plus one new prompt for "
                "Section 10 and the empty negative prompt that guidance needs — into an image embedding, seeded from a SHA-256 of "
                "the prompt so every run gets the same embedding. `release_prior` then drops the 1.03 B-parameter prior.\n\n"
                "**Predict first:** in the hint of a bird photographed on a branch against the sky, which part will be brightest?\n\n"
                "**What to notice:** 60 hints computed; a grid pairing each cropped photograph with its hint "
                "(`outputs/{stem}_hints.jpg`); each hint spanning 0..255 because of the min-max normalisation; seven prompts "
                "encoded; and GPU memory falling after the prior is released."
            ),
            "code": (
                "import time\n\n"
                "NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'  # @param {{type:\"string\"}}\n\n"
                "def gpu_memory_gb():\n"
                "    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None\n\n"
                "def grid(images, path, columns=6):\n"
                "    tiles = [im.resize((256, 256)) for im in images]\n"
                "    rows = (len(tiles) + columns - 1) // columns\n"
                "    sheet = Image.new('RGB', (256 * columns, 256 * rows), 'white')\n"
                "    for i, tile in enumerate(tiles):\n"
                "        sheet.paste(tile, (256 * (i % columns), 256 * (i // columns)))\n"
                "    sheet.save(path)\n"
                "    display(sheet)\n"
                "    return path\n\n"
                "def depth_image(depth):\n"
                "    return Image.fromarray(depth).convert('RGB')\n\n"
                "all_records = train_records + val_records + test_records\n"
                "hint_report = pipe.compute_hints(all_records)\n"
                "print({{**hint_report, 'gpu_memory_gb': gpu_memory_gb()}})\n"
                "first_hint = pipe.depth_hint(test_records[0])\n"
                "print({{'hint_shape': first_hint.shape, 'dtype': str(first_hint.dtype), 'min': int(first_hint.min()), 'max': int(first_hint.max()), 'mean': round(float(first_hint.mean()), 1)}})\n"
                "pairs = []\n"
                "for record in test_records[:6]:\n"
                "    pairs += [preprocess_image(record['image']), depth_image(pipe.depth_hint(record))]\n"
                "print({{'hint_grid': str(grid(pairs, 'outputs/{stem}_hints.jpg', columns=4))}})\n\n"
                "all_prompts = sample_prompts(all_records) + [NEW_PROMPT]\n"
                "encode_report = pipe.encode_prompts(all_prompts)\n"
                "print({{**encode_report, 'gpu_memory_gb_with_prior': gpu_memory_gb()}})\n"
                "released = pipe.release_prior()\n"
                "print({{'prior_released': released, 'gpu_memory_gb_after_release': gpu_memory_gb(), 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', '')}})"
            ),
        },
        {
            "md": (
                "<details>\n<summary><b>Check your reasoning</b> — Section 5</summary>\n\n"
                "Usually the bird and the branch it stands on are brightest, because they are nearest the camera, and open sky "
                "is darkest. Because each hint is min-max normalised, 255 means *the nearest thing in this photo*, not a distance "
                "in metres: two hints cannot be compared in absolute terms. Errors in the estimator — a blurred background read as "
                "near, a reflection read as far — become part of the condition the generator is asked to follow.\n</details>"
            ),
        },
        {
            "md": (
                "## 6. The frozen model: held-out denoising loss and depth-conditioned generations\n\n"
                "**Section type:** Evaluation practice.  \n"
                "**Question:** before any training, how well does the pretrained model reproduce these photographs' depth "
                "layouts and subjects?\n\n"
                "The pipeline was built with `use_lora=True`, but the adapter's B matrices start at zero, so until Section 8 this "
                "is exactly the pretrained model. Three kinds of number are read and kept for the comparison.\n\n"
                "1. **Held-out denoising loss** (`pipe.evaluate`): each validation and test photograph is MoVQ-encoded, noised at "
                "five fixed timesteps (100, 300, 500, 700, 900) with a seeded noise tensor, and the UNet — given the caption's "
                "embedding and the photograph's own hint — predicts that noise. The score is the mean squared error. The same "
                "seed gives the same latents, noise and timesteps later, so the adapted number is a *paired* comparison.\n"
                "2. **CLIP scores** (`score_generations`): each of the 12 test photographs contributes its caption and its hint, "
                "and one image is generated from them at a fixed seed (20 steps, guidance 4.0). CLIP ViT-B/32 scores prompt "
                "alignment, which caption each image is nearest to, and similarity to the held-out photographs. "
                "`real_photo_baseline` scores the real photographs the same way: the ceiling.\n"
                "3. **Depth fidelity** (`score_depth_fidelity`): the estimator measures the depth of each generated image, and "
                "the notebook reports its Pearson correlation with the hint and the mean error after aligning scale and offset. "
                "The same numbers against *another* image's hint are the mismatched-hint baseline — what an image that ignored its "
                "hint would score. `real_photo_depth_ceiling` re-measures the real photographs as a check of the measurement itself.\n\n"
                "**Predict first:** will the depth correlation of the frozen model's images be closer to 1.0 or to the mismatched "
                "baseline? Will CLIP label accuracy be closer to the real-photo ceiling or to chance (1 in 6)?\n\n"
                "**What to notice:** the per-timestep loss rising with the noise level; generated birds whose layout follows the "
                "hint; depth correlation well above the mismatched baseline; and a real-photo depth ceiling near 1.0, which shows "
                "the measurement is consistent rather than that 1.0 is reachable."
            ),
            "code": (
                "STEPS = 20  # @param {{type:\"integer\"}}\n"
                "GUIDANCE_SCALE = 4.0  # @param {{type:\"number\"}}\n"
                "EVAL_SEED = 0\n"
                "GEN_SEED = 1000\n\n"
                "scorer = ClipScorer(weights_dir=SCORER_WEIGHTS_DIR, device=pipe.device)\n"
                "t0 = time.perf_counter()\n"
                "frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "print({{'frozen_denoising_mse': {{'validation': frozen_val['denoising_mse'], 'test': frozen_test['denoising_mse']}}, 'by_timestep_test': frozen_test['by_timestep'], 'seconds': round(time.perf_counter() - t0, 1)}})\n\n"
                "generation_prompts = [r['caption'] for r in test_records]\n"
                "generation_hints = [pipe.depth_hint(r) for r in test_records]\n"
                "frozen_generation = pipe.generate(generation_prompts, generation_hints, seed=GEN_SEED, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "print({{'generated': len(frozen_generation['images']), 'steps': frozen_generation['steps'], 'guidance_scale': frozen_generation['guidance_scale'], 'seconds': frozen_generation['seconds'], 'adapted': frozen_generation['model']['adapted']}})\n"
                "frozen_scores = score_generations(scorer, frozen_generation['images'], references=test_records)\n"
                "frozen_depth = score_depth_fidelity(pipe, frozen_generation['images'])\n"
                "real_ceiling = real_photo_baseline(scorer, test_records)\n"
                "depth_floor = real_photo_depth_ceiling(pipe, test_records)\n"
                "CLIP_KEYS = ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')\n"
                "DEPTH_KEYS = ('depth_correlation', 'depth_aligned_mae', 'mismatched_depth_correlation', 'mismatched_depth_aligned_mae')\n"
                "print({{'frozen_clip': {{k: frozen_scores[k] for k in CLIP_KEYS}}, 'real_photo_ceiling': {{k: real_ceiling[k] for k in CLIP_KEYS}}}})\n"
                "print({{'frozen_depth_fidelity': {{k: frozen_depth[k] for k in DEPTH_KEYS}}, 'real_photo_depth_check': {{k: depth_floor[k] for k in DEPTH_KEYS[:2]}}}})\n"
                "tiles = []\n"
                "for entry in frozen_generation['images'][:6]:\n"
                "    tiles += [depth_image(entry['hint']), entry['image']]\n"
                "print({{'grid': str(grid(tiles, 'outputs/{stem}_frozen_grid.jpg', columns=4))}})"
            ),
        },
        {
            "md": (
                "<details>\n<summary><b>Check your reasoning</b> — Section 6</summary>\n\n"
                "A ControlNet-style model is trained to follow its hint, so the frozen model's depth correlation is expected to sit "
                "well above the mismatched baseline even before any adaptation; how far below 1.0 it stays reflects both the "
                "generator and the estimator's own noise. CLIP label accuracy depends on whether the pretrained model can render "
                "six similar small birds distinctly from their names — often it cannot, which is what adaptation targets. The "
                "denoising loss is highest at large timesteps, where almost all of the latent is noise. None of these numbers says "
                "whether an image *looks good*; that needs a person.\n</details>"
            ),
        },
        {
            "md": (
                "## 7. Change one thing: the depth hint\n\n"
                "**Section type:** Core concept (guided activity: Predict → Change one thing → Run → Observe → Explain).  \n"
                "**Question:** how much of the image's layout comes from the hint rather than from the prompt?\n\n"
                "The next cell regenerates the first four test prompts with **the same prompts, seeds, steps and guidance** as "
                "Section 6 — only the hint changes, selected by `ACTIVITY_HINT`:\n\n"
                "- `flat` (default): a constant depth map, i.e. *no* spatial information;\n"
                "- `mismatched`: the hint of a different test photograph;\n"
                "- `own`: the original hint, which should reproduce Section 6 (a control).\n\n"
                "Depth fidelity is measured against each record's **own** hint, the layout Section 6 asked for, and compared with "
                "the Section 6 numbers for the same four images. The activity does not feed into training or the final comparison.\n\n"
                "**Predict first:** with a flat hint, will the images still show a bird? Will their depth still correlate with the "
                "original photographs' layouts? Write down a number for the correlation before you run the cell.\n\n"
                "**What to notice:** the prompt still decides *what* appears; the hint decides *where* and *how near*. With `flat` "
                "the correlation with the original layout should drop towards the mismatched baseline of Section 6; with `own` "
                "it should match Section 6 up to GPU nondeterminism. Try `mismatched` next and explain the difference."
            ),
            "code": (
                "ACTIVITY_HINT = 'flat'  # @param [\"flat\", \"mismatched\", \"own\"]\n"
                "ACTIVITY_IMAGES = 4\n\n"
                "activity_records = test_records[:ACTIVITY_IMAGES]\n"
                "own_hints = [pipe.depth_hint(r) for r in activity_records]\n"
                "if ACTIVITY_HINT == 'flat':\n"
                "    changed_hints = [flat_hint() for _ in activity_records]\n"
                "elif ACTIVITY_HINT == 'mismatched':\n"
                "    changed_hints = [pipe.depth_hint(test_records[(i + ACTIVITY_IMAGES) % len(test_records)]) for i in range(len(activity_records))]\n"
                "elif ACTIVITY_HINT == 'own':\n"
                "    changed_hints = own_hints\n"
                "else:\n"
                "    raise ValueError(\"ACTIVITY_HINT must be 'flat', 'mismatched' or 'own'\")\n"
                "activity_generation = pipe.generate([r['caption'] for r in activity_records], changed_hints, seed=GEN_SEED, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "activity_depth = depth_fidelity_from_maps(pipe.estimate_depth([g['image'] for g in activity_generation['images']]), own_hints)\n"
                "activity_clip = score_generations(scorer, activity_generation['images'])\n"
                "canonical = frozen_depth['per_image'][:ACTIVITY_IMAGES]\n"
                "activity_summary = {{\n"
                "    'hint_given': ACTIVITY_HINT,\n"
                "    'depth_correlation_with_original_layout': {{'section_6_own_hint': round(sum(e['depth_correlation'] for e in canonical) / len(canonical), 4), 'this_activity': activity_depth['depth_correlation']}},\n"
                "    'section_6_mismatched_baseline': frozen_depth['mismatched_depth_correlation'],\n"
                "    'clip_prompt_similarity': activity_clip['clip_prompt_similarity'],\n"
                "}}\n"
                "print(activity_summary)\n"
                "tiles = []\n"
                "for before, after, hint in zip(frozen_generation['images'][:ACTIVITY_IMAGES], activity_generation['images'], changed_hints):\n"
                "    tiles += [before['image'], depth_image(hint), after['image']]\n"
                "print({{'grid': str(grid(tiles, 'outputs/{stem}_activity_grid.jpg', columns=3))}})"
            ),
        },
        {
            "md": (
                "<details>\n<summary><b>Check your reasoning</b> — Section 7</summary>\n\n"
                "With a flat hint the model still paints a bird, because the prompt embedding carries the subject, but the scene's "
                "arrangement is no longer tied to the original photograph, so its depth correlation with that layout falls towards "
                "the level of an unrelated hint. With a mismatched hint the bird is placed where the *other* photograph's bird was. "
                "That separation — words for content, hint for layout — is what the depth conditioning buys. Four images at one "
                "seed are an illustration, not a measurement: repeat with other seeds before drawing a firm conclusion.\n</details>"
            ),
        },
        {
            "md": (
                "## 8. Bounded LoRA fine-tuning\n\n"
                "**Section type:** Core concept (adaptation).  \n"
                "**Question:** can a small adapter teach the model these six species without disturbing its depth conditioning?\n\n"
                "`pipe.adapt` trains the 176 LoRA tensors (rank 8, 1,646,592 parameters — about 0.13 % of the UNet) that `peft` "
                "attached to the attention projections `to_q`, `to_k`, `to_v` and `to_out.0`, and nothing else: the UNet base "
                "weights, its hint encoder, the MoVQ, the prior and the depth estimator are frozen. This is **parameter-efficient "
                "gradient training**, not full fine-tuning. Each step takes one training photograph's latent (MoVQ-encoded once), "
                "its caption embedding and its own depth hint, draws a timestep and a noise tensor (both seeded), and minimises the "
                "MSE between the predicted and the true noise. The optimiser is AdamW at a fixed learning rate with gradient-norm "
                "clipping at 1.0 and float16 autocast with loss scaling on CUDA. Epoch 0 records the frozen model's validation "
                "loss, and the epoch with the lowest validation loss is kept.\n\n"
                "**What to notice:** the training loss is noisy, because every step draws a random timestep; the validation loss "
                "is the number that decides which epoch is kept. A falling training loss alone is not evidence of better images."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 1  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_denoising_mse': entry['val_loss']}}\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=EVAL_SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['adapter']['n_trainable'], 'total_parameters': adapt_result['adapter']['n_total'], 'steps': adapt_result['steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['adapter']['precision'], 'seconds': adapt_seconds, 'gpu_memory_gb': gpu_memory_gb()}})"
            ),
        },
        {
            "md": (
                "## 9. Held-out evaluation: the paired comparison\n\n"
                "**Section type:** Evaluation practice.  \n"
                "**Question:** on photographs the adapter never saw, did adaptation help, and did it cost depth fidelity?\n\n"
                "The test photographs were never used for training or for choosing the epoch. The adapted model is scored exactly "
                "as the frozen model was in Section 6 — the same seed, so the same latents, noise and timesteps, and the same 12 "
                "prompt/hint/seed triples for generation — and the table puts frozen, adapted and reference numbers side by side. "
                "The cell asserts only what the procedure guarantees: the kept epoch's validation loss is no higher than the frozen "
                "model's, and re-scoring the validation set reproduces the recorded value. The test numbers are printed, not "
                "asserted: a worse held-out result is a valid result and is kept.\n\n"
                "**Predict first:** which number will move most — the denoising loss, CLIP label accuracy or depth correlation?\n\n"
                "**What to notice:** read each change beside its reference — the real-photo ceiling for CLIP and the mismatched "
                "baseline for depth. With 12 images, a difference of one image in label accuracy is 0.083."
            ),
            "code": (
                "adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "adapted_generation = pipe.generate(generation_prompts, generation_hints, seed=GEN_SEED, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "adapted_scores = score_generations(scorer, adapted_generation['images'], references=test_records)\n"
                "adapted_depth = score_depth_fidelity(pipe, adapted_generation['images'])\n"
                "comparison = {{\n"
                "    'denoising_mse_validation': {{'frozen': frozen_val['denoising_mse'], 'adapted': adapted_val['denoising_mse']}},\n"
                "    'denoising_mse_test': {{'frozen': frozen_test['denoising_mse'], 'adapted': adapted_test['denoising_mse']}},\n"
                "    'denoising_mse_test_by_timestep': {{t: {{'frozen': frozen_test['by_timestep'][t], 'adapted': adapted_test['by_timestep'][t]}} for t in adapted_test['by_timestep']}},\n"
                "    'clip_prompt_similarity': {{'frozen': frozen_scores['clip_prompt_similarity'], 'adapted': adapted_scores['clip_prompt_similarity'], 'real_photos': real_ceiling['clip_prompt_similarity']}},\n"
                "    'label_accuracy': {{'frozen': frozen_scores['label_accuracy'], 'adapted': adapted_scores['label_accuracy'], 'real_photos': real_ceiling['label_accuracy']}},\n"
                "    'reference_similarity': {{'frozen': frozen_scores['reference_similarity'], 'adapted': adapted_scores['reference_similarity'], 'real_photos': real_ceiling['reference_similarity']}},\n"
                "    'depth_correlation': {{'frozen': frozen_depth['depth_correlation'], 'adapted': adapted_depth['depth_correlation'], 'mismatched_baseline': frozen_depth['mismatched_depth_correlation']}},\n"
                "    'depth_aligned_mae': {{'frozen': frozen_depth['depth_aligned_mae'], 'adapted': adapted_depth['depth_aligned_mae'], 'mismatched_baseline': frozen_depth['mismatched_depth_aligned_mae']}},\n"
                "}}\n"
                "for name, row in comparison.items():\n"
                "    print({{name: row}})\n"
                "tiles = []\n"
                "for before, after in zip(frozen_generation['images'][:6], adapted_generation['images'][:6]):\n"
                "    tiles += [depth_image(before['hint']), before['image'], after['image']]\n"
                "print({{'grid': str(grid(tiles, 'outputs/{stem}_adapted_grid.jpg', columns=3))}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'components': {{'prior': {{'id': PRIOR_ID, 'revision': PRIOR_REVISION}}, 'depth_estimator': {{'id': DEPTH_ID, 'revision': DEPTH_REVISION}}}},\n"
                "    'scorer': frozen_scores['scorer'],\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'generation': {{'steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'seed': GEN_SEED, 'images': len(generation_prompts)}},\n"
                "    'frozen': {{'validation': frozen_val, 'test': frozen_test, 'clip': frozen_scores, 'depth_fidelity': frozen_depth}},\n"
                "    'adapted': {{'validation': adapted_val, 'test': adapted_test, 'clip': adapted_scores, 'depth_fidelity': adapted_depth}},\n"
                "    'real_photo_ceiling': real_ceiling,\n"
                "    'real_photo_depth_check': depth_floor,\n"
                "    'activity': activity_summary,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result['adapter'].items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "best = adapt_result['history'][adapt_result['best_epoch']]\n"
                "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['denoising_mse'] - best['val_loss']) < 1e-4\n"
                "print({{'test_denoising_mse_change': round(adapted_test['denoising_mse'] - frozen_test['denoising_mse'], 6), 'note': 'held-out observation, not asserted'}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "<details>\n<summary><b>Check your reasoning</b> — Section 9</summary>\n\n"
                "The denoising loss usually moves first and most reliably, because it is the quantity being optimised and it is "
                "measured with identical noise before and after. CLIP label accuracy and reference similarity can improve when "
                "the adapter learns what the six species look like, but with 12 images one image changes accuracy by 0.083, so "
                "small changes are within noise. Depth correlation should stay close to its frozen value: the hint encoder is "
                "frozen and the adapter only changes attention. A clear drop would mean the adaptation traded layout control for "
                "subject fidelity — a finding to report, not to hide.\n</details>"
            ),
        },
        {
            "md": (
                "## 10. A new prompt, artifact export and fresh reload\n\n"
                "**Section type:** Engineering (artifact and reproducibility).  \n"
                "**Question:** does the adapter work as a file, apart from this Python session?\n\n"
                "The adapted model renders `NEW_PROMPT` — a composition that appears in no training caption — on the depth layouts "
                "of two test photographs; the CLIP and depth numbers are printed as a sanity check, not an evaluation.\n\n"
                "`pipe.save_artifact` writes the 176 trained tensors as `adapter.safetensors` with a `manifest.json` recording the "
                "artifact format, the decoder's id and revision, the prior and depth-estimator identities, the LoRA configuration, "
                "the tensor names, the file size and SHA-256, the training configuration and the epoch history. "
                "`KandinskyDepthPipeline.from_artifact` re-verifies the snapshots and checks the manifest, the estimator identity, "
                "the LoRA scope and the digest **before** deserialising; it loads a fresh UNet with the adapter attached and "
                "overlays the tensors — a new object from files, not the in-memory model. The fresh pipeline adopts the prompt "
                "embeddings and hints already computed, and the cell asserts that it reproduces the same held-out denoising loss "
                "and the same image for the same prompt, hint and seed. Loading is not the same as reproducing: only the second "
                "assertion shows the artifact is equivalent.\n\n"
                "**What to notice:** an artifact of a few MB, a parity difference of 0 (or below the stated tolerance), and the "
                "list of files under `outputs/`."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "new_hints = [generation_hints[0], generation_hints[1]]\n"
                "new_generation = pipe.generate([NEW_PROMPT, NEW_PROMPT], new_hints, seed=2000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)\n"
                "new_scores = score_generations(scorer, new_generation['images'])\n"
                "new_depth = score_depth_fidelity(pipe, new_generation['images'])\n"
                "print({{'new_prompt': NEW_PROMPT, 'clip_prompt_similarity': new_scores['clip_prompt_similarity'], 'depth_correlation': new_depth['depth_correlation'], 'seconds': new_generation['seconds'], 'note': 'sanity check, not an evaluation'}})\n"
                "for i, entry in enumerate(new_generation['images']):\n"
                "    entry['image'].save(f'outputs/{stem}_new_prompt_{{i}}.png')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': artifact_manifest['weights']['n_tensors'], 'bytes': artifact_manifest['weights']['bytes'], 'sha256': artifact_manifest['weights']['sha256'][:16] + '...'}})\n\n"
                "reloaded = KandinskyDepthPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, prior_dir=PRIOR_WEIGHTS_DIR, depth_dir=DEPTH_WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded.import_prompt_cache(pipe.export_prompt_cache())\n"
                "reloaded.import_hint_cache(pipe.export_hint_cache())\n"
                "reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)\n"
                "before = pipe.generate([generation_prompts[0]], [generation_hints[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "after = reloaded.generate([generation_prompts[0]], [generation_hints[0]], seed=3000, steps=STEPS, guidance_scale=GUIDANCE_SCALE)['images'][0]['image']\n"
                "parity = {{'denoising_mse_diff': round(abs(reloaded_test['denoising_mse'] - adapted_test['denoising_mse']), 8), 'mean_abs_pixel_diff': round(float(np.abs(np.asarray(before, dtype=np.float32) - np.asarray(after, dtype=np.float32)).mean()), 4)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch'], 'tolerance': {{'denoising_mse_diff': 1e-6, 'mean_abs_pixel_diff': 1.0}}}})\n"
                "assert parity['denoising_mse_diff'] < 1e-6 and parity['mean_abs_pixel_diff'] < 1.0\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'precision': str(pipe.dtype).replace('torch.', ''), 'source': pipe.source}},\n"
                "    'components': {{'prior': {{'id': PRIOR_ID, 'revision': PRIOR_REVISION, 'license': PRIOR_LICENSE}}, 'depth_estimator': {{'id': DEPTH_ID, 'revision': DEPTH_REVISION, 'license': DEPTH_LICENSE}}}},\n"
                "    'scorer': {{**evaluation_report['scorer'], 'license': SCORER_LICENSE}},\n"
                "    'provenance': {{\n"
                "        'snapshots': {{'decoder': len(MANIFEST['files']), 'prior': len(PRIOR_MANIFEST['files']), 'depth_estimator': len(DEPTH_MANIFEST['files']), 'scorer': len(SCORER_MANIFEST['files'])}},\n"
                "        'safetensors_only': True,\n"
                "        'remote_code_executed': False,\n"
                "        'decoder_weights': 'safetensors from an open, unmerged upstream conversion pull request, pinned by commit',\n"
                "        'prior_released_before_training': released,\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['weights']['sha256'], 'bytes': artifact_manifest['weights']['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
        {
            "md": (
                "## 11. Troubleshooting\n\n"
                "**Section type:** Engineering.\n\n"
                "| Observation | What it means | What to do |\n"
                "|---|---|---|\n"
                "| `device='cuda' requested but CUDA is not available`, or `cuda: False` in Section 1 | the runtime has no GPU | Runtime → Change runtime type → T4 GPU, then Run all |\n"
                "| `Core dependencies changed while older modules were loaded` | a pinned install replaced an imported package | Runtime → Restart, then run from the top |\n"
                "| `sha256 ... != manifest` or `size ... != manifest` | a download is incomplete or altered | delete the named file under `weights/` and re-run Section 3; never edit the manifest |\n"
                "| `unet has N parameters ... expected` | the snapshot or `diffusers` version differs from the pins | re-run Section 1 in a fresh runtime so the exact pins install |\n"
                "| `CUDA out of memory` | another notebook or a previous run holds GPU memory | restart the runtime; keep `BATCH_SIZE = 1`; do not report a skipped stage as a result |\n"
                "| A photo fetch fails in Section 4 | the open-data bucket is unreachable or a file changed | re-run the cell; a digest mismatch is refused on purpose |\n"
                "| BYOD: `captions.csv is missing columns` / `image sides must be within` / `split leaves no test record` | the upload does not meet the data contract | fix the zip as the message says: columns `id,file,caption`, sides 256..4096 px, one caption with three or more images |\n"
                "| `adapter depth estimator ... does not match` | an adapter from another estimator or revision | reload only adapters exported by this pipeline revision |\n\n"
                "A negative or surprising held-out result is not an error. Keep it, and read Section 9's *Check your reasoning*."
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A LoRA of 1.6 million parameters trained for a few minutes on 36 photographs and their depth hints changes the held-out "
        "denoising loss on twelve photographs the model never saw, and may move its depth-conditioned generations towards the "
        "held-out photographs of the same species. That is the claim the notebook can support: the adaptation contract teaches the "
        "UNet a narrow visual domain from a handful of captioned images while the depth conditioning stays in place, the held-out "
        "objective is measured on identical inputs before and after, and the artifact reloads with the same outputs.\n\n"
        "The numbers are sample-sanity evidence. A denoising loss is the training objective, not a quality score. CLIP similarity "
        "and CLIP's nearest-caption vote are a frozen model's opinion, not a human judgement. Depth fidelity is measured by the same "
        "estimator that made the hints, so it rewards agreement with DPT, including DPT's mistakes. Twelve images from one seeded "
        "run give no dispersion estimate, and nothing here measures aesthetics, diversity or artefacts. Fine-tuning on a narrow "
        "domain can also erode the model elsewhere — the new prompt in Section 10 is a sanity check on one composition.\n\n"
        "## Conclude with evidence\n\n"
        "Complete this in your own words, using only numbers this notebook printed:\n\n"
        "> On [n] held-out photographs of [subject], the frozen model reached a denoising loss of [x] and a depth correlation of "
        "[r] against a mismatched-hint baseline of [b]; CLIP label accuracy was [a] against a real-photo ceiling of [c]. After "
        "[epochs] epochs of rank-8 LoRA training, these became [x'], [r'] and [a']. Changing only the hint to [flat / mismatched] "
        "changed the depth correlation from [r] to [r'']. Because [one seed, twelve images, one scorer, one estimator], these "
        "results do not show [generality / image quality / ...]. Next I would [collect ... / repeat with seeds ... / ask people to rate ...].\n\n"
        "**Transfer:** try BYOD with photographs of a subject whose depth layout matters — rooms, streets, tabletop objects — "
        "and check whether the adapted model keeps following the hint.\n\n"
        "Three things to carry to real data. **Captions and hints are both the contract:** the adapter learns caption text, "
        "image and depth together, so a caption that does not describe its image, or a hint estimated from a distorted crop, "
        "teaches noise. **Hold out by caption, not by image:** the split keeps every caption's images across sets so the "
        "held-out loss measures generalisation within the domain. **Licences travel with the outputs:** the Kandinsky and DPT "
        "weights are Apache-2.0 and the training photographs here are CC0; with your own data, the rights are yours to establish.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can stage and digest-verify four pinned safetensors snapshots, fetch and validate digest-pinned real "
        "photographs, compute depth hints, encode prompts and release the prior, execute bounded LoRA fine-tuning, evaluate the "
        "frozen and the adapted model on identical held-out inputs with a real-photo ceiling and a mismatched-hint baseline, and "
        "emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, production fitness, image quality beyond the checks shown, or that the pinned safetensors conversion is "
        "bit-identical to the upstream `.bin` weights.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/kandinsky-controlnet-depth-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/kandinsky-controlnet-depth-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/kandinsky-controlnet-depth-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face decoder repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth (revision `{MODEL_REVISION}`, conversion pull request #5)\n"
        "- Hugging Face prior repository: https://huggingface.co/kandinsky-community/kandinsky-2-2-prior\n"
        "- Hugging Face depth estimator: https://huggingface.co/Intel/dpt-large\n"
        "- Shakhmatov, A., et al. (2023). Kandinsky 2.2: https://github.com/ai-forever/Kandinsky-2\n"
        "- Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. ICCV: https://arxiv.org/abs/2302.05543\n"
        "- Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. ICCV: https://arxiv.org/abs/2103.13413\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n"
    ),
}
