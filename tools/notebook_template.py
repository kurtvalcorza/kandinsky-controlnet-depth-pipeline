"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, snapshot
staging) from repository files; this template holds the learner-facing prose, the list of carried files and the
learner cells. Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the
repository) runs one stage per process in an isolated, hash-locked environment, so nothing is installed into the
notebook kernel.

This template configures an E2E depth-conditioned text-to-image fine-tuning workflow: the pinned Kandinsky 2.2
ControlNet-depth decoder, the shared diffusion prior, the DPT depth estimator and the CLIP scorer are staged and
digest-verified; 60 pinned CC0 iNaturalist bird photographs are fetched, validated and split; the estimator turns
every photograph into a depth hint and the prior turns every caption into an image embedding (the prior is then
released); the frozen model is scored (held-out denoising loss, CLIP-scored and depth-scored generations) against
the real-photo reference and a mismatched-hint baseline; one variable (the hint) is changed in a guided activity;
a bounded LoRA fine-tuning runs and exports the adapter; a fresh process scores the exported adapter on identical
held-out inputs; and a second fresh process reloads it, checks parity with the trained in-memory model and renders a
new prompt on an existing depth layout.
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

This notebook is for a learner who can open a hosted notebook, run cells and read basic Python, and who is new to conditioned diffusion models. No prior experience with diffusion training is assumed; the terms are explained where they first matter and collected in the glossary below. You need a Linux GPU runtime (a free Colab T4 is enough), about 31 GB of free disk and roughly an hour, much of it downloads.

### How to use this notebook

1. In Colab choose **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. Nothing needs to be edited for the default path, and no runtime restart is needed.
2. Cells titled **Infrastructure** (Sections 1–3) check the runtime, carry the repository's code, build an isolated Python environment from hash-locked packages and download the pinned models. They are collapsed; run them without studying them.
3. **Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it fails. Stages hand results to each other only through files in the run directory — the verified snapshots, the depth hints, the prompt embeddings, the frozen model's images, the adapter artifact and JSON records — and GPU memory is released when each stage ends.
4. Sections 4–10 are the lesson. Each principal step starts with a **question** and, where it makes sense, asks you to **predict** the result first. After the cell, read **What to notice**, then open **Check your reasoning** to compare your answer.
5. Values you may change are **form fields** (the `# @param` lines). The canonical run uses their defaults; Section 7 is the place to change one of them deliberately.
6. Every output is written under `outputs/` in the run directory that Section 1 creates.

### Roadmap

| Section | Kind | What happens |
|---|---|---|
| 1–3 | *Engineering* | check the runtime, carry the code, build the isolated environment, verify four model snapshots |
| 4 | *Core concept* | real captioned photographs, validation, a seeded split |
| 5 | *Core concept* | photograph → depth hint; caption → image embedding |
| 6 | *Evaluation practice* | the frozen model: loss, generations, three kinds of score |
| 7 | *Core concept* | **change one thing**: give the model a different hint |
| 8 | *Core concept* | bounded LoRA fine-tuning and adapter export |
| 9 | *Evaluation practice* | the paired before/after comparison, from the exported adapter |
| 10 | *Engineering* | a fresh reload with parity, and a new prompt |
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
- **Hash-locked environment:** a separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests; the installer refuses anything else.
- **Stage:** one step of the workflow run as its own process by `run_stage`; it reads the files earlier stages wrote and writes its own.
</details>

Two properties are handled in the open. **The prior pipeline can be released before training:** Section 5 encodes every prompt the notebook uses and then drops the prior, so the UNet, the depth estimator, the scorer and a training graph fit comfortably on a 16 GB GPU. **Generation has no ground truth**, so the notebook reads three kinds of number: the held-out *denoising loss*, CLIP scores against the real-photo ceiling, and *depth fidelity* against a mismatched-hint baseline. None of these is a human judgement of image quality.

**About the decoder weights.** The upstream repository's `main` branch ships only pickled `.bin` files. This notebook downloads the safetensors files of the repository's open, unmerged conversion pull request, addressed by its immutable commit and verified by SHA-256. Whether that conversion is bit-identical to `main` is checked separately by `tools/verify_conversion.py` in the repository; that check is recorded in the model card."""

TEMPLATE = {
    "package": "kandinsky_controlnet_depth_pipeline",
    "repo_name": REPO,
    "stem": "kandinsky_controlnet_depth",
    "notebook_name": NOTEBOOK,
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh Linux **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) builds an isolated "
        "Python environment from the carried hash-locked requirements (torch, diffusers, transformers, peft, accelerate, "
        "sentencepiece, safetensors, huggingface-hub, numpy, pillow and their dependencies) without touching the notebook kernel's "
        "own packages, then runs each stage below in its own process: it stages and digest-verifies four pinned snapshots from the "
        "Hub — the 5.29 GB Kandinsky 2.2 ControlNet-depth decoder (UNet + MoVQ), the 10.57 GB Kandinsky 2.2 diffusion prior, the "
        "1.37 GB DPT depth estimator and a 0.6 GB CLIP scorer — fetches 60 CC0 iNaturalist bird photographs as digest-verified JPEGs "
        "(6 MB, no credential), validates them and splits them 36 / 12 / 12 by seed, turns every photograph into a depth hint and "
        "every caption into an image embedding (then releases the prior), scores the frozen model — the held-out denoising loss, "
        "and twelve depth-conditioned generations scored for prompt alignment, similarity to the held-out photographs and depth "
        "fidelity against a mismatched-hint baseline — runs a guided change-one-thing activity on the hint, runs a bounded LoRA "
        "fine-tuning (4 epochs over 36 images) and exports the adapter as safetensors with a manifest, scores the exported adapter "
        "in a fresh process on identical inputs, and reloads it in a second fresh process to verify parity with the trained "
        "in-memory model before rendering a new prompt on an existing depth layout. The default path needs no repository clone, "
        "no DIMER worker or service, no credential, no upload dialog, no configuration edit and no runtime restart (NOTEBOOK_SPEC "
        "2.2 §5). This version adds the environment build and a snapshot re-verification in every stage process; its duration on "
        "a T4 has not yet been recorded."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "captioned photographs as a zip holding `captions.csv` (columns `id`, `file`, `caption`) beside the image files (JPEG or "
        "PNG, shorter side 256..4096 px; at least four images, and at least one caption with three or more images so a held-out "
        "record exists). Set `BYOD_PATH` to a zip already in the runtime to skip the upload dialog. Your photographs get depth hints "
        "from the same estimator, are split by caption, and flow through the same contract — validation, hints and prompt "
        "encoding, frozen baseline, adaptation, held-out evaluation, generation, artifact export and reload parity. The expected "
        "schema, the ceilings and the privacy guidance are stated in the Prerequisites and in Section 4, and uploaded files stay "
        "inside this runtime. An invalid zip stops Section 4 with the validator's own message. BYOD is optional and never part of "
        "the default path."
    ),
    "weights_key": "kandinsky-2-2-controlnet-depth",
    # Carried byte for byte (UTF-8 text, LF newlines) into the run directory and verified against CARRIED_HASHES.
    "carried": {
        "src/kandinsky_controlnet_depth_pipeline/__init__.py": "src/kandinsky_controlnet_depth_pipeline/__init__.py",
        "src/kandinsky_controlnet_depth_pipeline/pipeline.py": "src/kandinsky_controlnet_depth_pipeline/pipeline.py",
        "src/kandinsky_controlnet_depth_pipeline/samples.py": "src/kandinsky_controlnet_depth_pipeline/samples.py",
        "src/kandinsky_controlnet_depth_pipeline/metrics.py": "src/kandinsky_controlnet_depth_pipeline/metrics.py",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/kandinsky-2-2-controlnet-depth/dimer-base-manifest.json": "weights/kandinsky-2-2-controlnet-depth/dimer-base-manifest.json",
        "weights/kandinsky-2-2-prior/dimer-base-manifest.json": "weights/kandinsky-2-2-prior/dimer-base-manifest.json",
        "weights/dpt-large/dimer-base-manifest.json": "weights/dpt-large/dimer-base-manifest.json",
        "weights/clip-vit-b-32-laion2b/dimer-base-manifest.json": "weights/clip-vit-b-32-laion2b/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    # The four snapshots total 16.6 GiB (17.84 GB); the isolated environment needs about 12 GiB.
    "disk_gib": {"weights": 19, "environment": 12},
    "runtime_modules": ["torch", "diffusers", "transformers", "peft"],
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
        "identical held-out inputs; and **verify** that an exported safetensors adapter, reloaded in a fresh process, reproduces "
        "the trained model's outputs."
    ),
    "exclusions": (
        "other ControlNet conditions (edges, pose, segmentation), image-to-image or inpainting, resolutions other than 512 × 512, "
        "metric (absolute) depth, full fine-tuning, safety filtering of prompts or images, human preference benchmarks, prompt "
        "engineering, and any claim that a CLIP score, a depth-fidelity score or a denoising loss measures image quality. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Audience:** comfortable running notebook cells and reading short Python; no diffusion-training experience needed.",
        "- **Runtime:** a fresh supported **Linux x86_64 GPU** runtime (Google Colab T4 or better, Kaggle T4, or a Linux Jupyter kernel with a CUDA GPU of at least 15 GB). The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (torch 2.14.0, whose Linux wheel is the CUDA 13.0 build). The prior runs in float16 while it encodes prompts and is then released; the UNet runs in float16 (2.5 GB) with the LoRA parameters in float32; the MoVQ decoder and the DPT estimator run in float16. CPU-only runtimes are not supported for this notebook. About 18 GB of disk is needed for the snapshots (the Section 1 check asks for 19 GiB, with headroom) and about 12 GB for the isolated environment.",
        "- **Knowledge:** what a diffusion model does at inference (noise → latent → pixels over several steps), what classifier-free guidance is, what a LoRA adapter changes and what it does not, and why a training loss is not a quality score. The glossary above covers each term.",
        "- **Weights:** the decoder, the prior, the depth estimator and the CLIP scorer are all loaded from safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers` and `peft` on PyPI. The decoder safetensors come from an open, unmerged conversion pull request of the upstream repository, pinned by commit. Upstream weights are Apache-2.0; the scorer is MIT.",
        "- **Data contract:** a record is `{{id, image, caption}}` — an RGB image with shorter side 256..4096 px (resized so the shorter side is 512 px and centre-cropped to 512 × 512; the crop is reported) and a caption of 1..1000 characters. Its depth hint is computed from the cropped image. Validation is structural: nothing checks that a caption describes its image, that a depth map is plausible, or that the model can render the prompt.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. A depth map can still reveal the layout of a private space. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime\n\n"
                "**Section type:** Engineering.\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with a CUDA GPU and enough free "
                "disk, and creates a new run directory. **Output:** the GPU name and the directories this run will use. Each run "
                "writes to a new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current result. "
                "The verified snapshots are kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**What to notice:** one dictionary naming the GPU (for example `Tesla T4, 15360 MiB`), the kernel's Python "
                "version, the run directory, the weights directory, the isolated environment's directory and the free disk. "
                "If the cell stops with a GPU or disk message, see Section 11."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime\n\n"
                "**Section type:** Engineering.\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so "
                "that this notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under "
                "`src/kandinsky_controlnet_depth_pipeline/` (identity constants, snapshot verification and staging, validation, the "
                "pipeline class with its depth hints, the sample corpus, and the CLIP and depth-fidelity metrics), the stage runner "
                "`tutorial_stages.py`, the hash-locked `requirements.txt` ({n_locked} packages), the four snapshot manifests and the "
                "licence. It writes each file into the run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on any "
                "mismatch. The text is the repository's files byte for byte; the repository's parity test "
                "(`tests/test_notebook_parity.py`) fails whenever the two diverge, so what runs here is what the repository tests. "
                "Nothing in this cell runs a model."
            ),
            "after": (
                "**What to notice:** `carried_files`, `verified: True`, and the repository revision the notebook was generated "
                "from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer "
                "wheel, refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython "
                "3.12.12, and installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must "
                "be the locked version, a prebuilt wheel, and match a locked digest. The hosted runtime's own packages are never "
                "replaced, which is why no restart is needed. The cell also defines `run_stage`, `load_record` and `show_image`, "
                "the three helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few "
                "minutes. A failed download or a hash mismatch stops the cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**What to notice:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), "
                "`torch`, `diffusers`, `transformers` and `peft` versions, `'cuda': True`, the number of locked packages and the "
                "setup time. If `'cuda'` is `False` the cell stops: this notebook is not supported on a CPU-only runtime; see "
                "Section 11."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the model\n\n"
                "**Section type:** Engineering.\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads about 17.8 GB of pinned weights and checks "
                "every file's size and SHA-256; you may run it without studying its implementation.\n\n"
                "The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the carried `pipeline.py` and the manifest "
                "of each snapshot (paths, byte sizes, SHA-256) — and the `weights` stage first checks that they agree. It installs "
                "each carried manifest into `weights/`, fetches exactly the files that are absent from the Hugging Face Hub **at "
                "the pinned revisions** (never `main`), and re-hashes every file, raising on the first size or digest mismatch. "
                "Four snapshots are staged: the decoder (`{MODEL_ID}` at `{MODEL_REVISION}`), the shared diffusion prior, the DPT "
                "depth estimator `Intel/dpt-large` (loaded explicitly by id and revision, never through an auto-download) and the "
                "CLIP scorer. There is no fallback to a different download, and no remote model code is executed. Every later "
                "stage verifies the snapshots it loads again before loading them."
            ),
            "after": (
                "**What to notice:** for each of the four snapshots, its id, revision, licence and file count; a `fetched` list of "
                "the files downloaded on this run (empty on a rerun, because staging only fetches files that are absent); and a "
                "count of verified files. A size or SHA-256 mismatch stops the cell with a `ValueError` naming the file — see "
                "Section 11, and never edit a manifest to get past one."
            ),
        },
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, validation and splits\n\n"
                "**Section type:** Core concept (data).  \n"
                "**Question:** what does the model learn from, and how do we keep some of it aside for an honest test?\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`; its printed dictionaries "
                "appear under the cell. This cell runs the `prepare` stage. The default dataset is 60 research-grade iNaturalist "
                "photographs of six common North American birds — 10 per species, one per observer per species, every one CC0 — "
                "fetched by photo id from the open-data bucket and refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). "
                "Each photo's caption is generated from its species by one template. `build_sample_dataset` draws a seeded "
                "stratified split — 6 / 2 / 2 per species for training, validation and test — and `dataset_manifest` validates "
                "every split, checks that no image appears twice and records a digest. **Validation** is the check that runs "
                "*before* any model: it refuses a malformed record with a message that names the broken rule. The stage records "
                "the split so that every later stage rebuilds exactly these records and refuses to run if they changed.\n\n"
                "**What to notice:** 36 / 12 / 12 records, six distinct captions, a shorter side around 300..500 px (every photo "
                "is centre-cropped to 512²), a written `outputs/{stem}_sample_captions.csv` in the run directory in the shape BYOD "
                "expects, and three refusal probes — a missing caption, a 200 px image, a duplicate id — each rejected before the "
                "model runs. With `USE_BYOD = True`, an invalid zip stops this cell with a `RuntimeError` that repeats the "
                "validator's message."
            ),
            "code": (
                'from pathlib import Path\n\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = \'\'  # @param {{type:"string"}}\n\n'
                'prepare_options = []\n'
                'if USE_BYOD:\n'
                '    if BYOD_PATH:\n'
                '        byod_path = Path(BYOD_PATH)\n'
                '    else:\n'
                '        from google.colab import files\n'
                '        uploaded = files.upload()\n'
                '        file_name, payload = next(iter(uploaded.items()))\n'
                "        byod_path = ROOT / 'byod' / Path(file_name).name\n"
                '        byod_path.parent.mkdir(parents=True, exist_ok=True)\n'
                '        byod_path.write_bytes(payload)\n'
                "    prepare_options = ['--byod', byod_path.resolve()]\n"
                "run_stage('prepare', *prepare_options)"
            ),
        },
        {
            "md": (
                "## 5. Depth hints and prompt embeddings\n\n"
                "**Section type:** Core concept.  \n"
                "**Question:** what exactly does the model receive besides the noise?\n\n"
                "Two frozen helper models prepare the two conditions, in the `encode` stage. `pipe.compute_hints` runs the pinned DPT "
                "estimator on every photograph after the same 512² centre crop the training uses; the predicted relative depth is "
                "resized back to 512² and min-max normalised to 0..255 (brighter = nearer). The UNet reads it as three identical "
                "channels divided by 255. `pipe.encode_prompts` loads the diffusion prior and turns each distinct caption — plus "
                "`NEW_PROMPT` for Section 10 and the empty negative prompt that guidance needs — into an image embedding, seeded from "
                "a SHA-256 of the prompt so every run gets the same embedding. `release_prior` then drops the 1.03 B-parameter prior. "
                "The hints and the embeddings are written to two safetensors caches in the run directory, and every later stage reads "
                "them from there instead of loading the prior again.\n\n"
                "**Predict first:** in the hint of a bird photographed on a branch against the sky, which part will be brightest?\n\n"
                "**What to notice:** 60 hints computed; a grid pairing each cropped photograph with its hint "
                "(`outputs/{stem}_hints.jpg`); each hint spanning 0..255 because of the min-max normalisation; seven prompts "
                "encoded; GPU memory falling after the prior is released; and the paths of the two caches."
            ),
            "code": (
                "NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'  # @param {{type:\"string\"}}\n\n"
                "run_stage('encode', '--new-prompt', NEW_PROMPT)\n"
                "show_image('{stem}_hints.jpg', 'Cropped test photographs beside their depth hints')"
            ),
        },
        {
            "md": (
                "<details>\n<summary><b>Check your reasoning</b> — Section 5</summary>\n\n"
                "Usually the bird and the branch it stands on are brightest, because they are nearest the camera, and open sky "
                "is darkest. Because each hint is min-max normalised, 255 means *the nearest thing in this photo*, not a distance "
                "in metres: two hints cannot be compared in absolute terms. Errors in the estimator — a blurred background read as "
                "near, a reflection read as far — become part of the condition the generator is asked to follow. Both conditions "
                "are fixed once computed, which is why a file of hints and a file of embeddings can serve every later process.\n</details>"
            ),
        },
        {
            "md": (
                "## 6. The frozen model: held-out denoising loss and depth-conditioned generations\n\n"
                "**Section type:** Evaluation practice.  \n"
                "**Question:** before any training, how well does the pretrained model reproduce these photographs' depth "
                "layouts and subjects?\n\n"
                "The `frozen` stage builds the pipeline with `use_lora=True`, but the adapter's B matrices start at zero, so until "
                "Section 8 this is exactly the pretrained model. Three kinds of number are read and kept for the comparison.\n\n"
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
                "The stage also keeps the twelve generated images in the run directory, so Sections 7 and 9 can show them beside "
                "their own results.\n\n"
                "**Predict first:** will the depth correlation of the frozen model's images be closer to 1.0 or to the mismatched "
                "baseline? Will CLIP label accuracy be closer to the real-photo ceiling or to chance (1 in 6)?\n\n"
                "**What to notice:** the per-timestep loss rising with the noise level; generated birds whose layout follows the "
                "hint; depth correlation well above the mismatched baseline; and a real-photo depth ceiling near 1.0, which shows "
                "the measurement is consistent rather than that 1.0 is reachable."
            ),
            "code": (
                "STEPS = 20  # @param {{type:\"integer\"}}\n"
                "GUIDANCE_SCALE = 4.0  # @param {{type:\"number\"}}\n\n"
                "run_stage('frozen', '--steps', STEPS, '--guidance', GUIDANCE_SCALE)\n"
                "show_image('{stem}_frozen_grid.jpg', 'Frozen model: each hint beside the image generated from it')"
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
                "The `activity` stage regenerates the first four test prompts with **the same prompts, seeds, steps and guidance** "
                "as Section 6 — only the hint changes, selected by `ACTIVITY_HINT`:\n\n"
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
                "ACTIVITY_HINT = 'flat'  # @param [\"flat\", \"mismatched\", \"own\"]\n\n"
                "run_stage('activity', '--hint', ACTIVITY_HINT)\n"
                "show_image('{stem}_activity_grid.jpg', 'Section 6 image, the changed hint, and the image generated from it')"
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
                "The `adapt` stage calls `pipe.adapt`, which trains the 176 LoRA tensors (rank 8, 1,646,592 parameters — about 0.13 % "
                "of the UNet) that `peft` attached to the attention projections `to_q`, `to_k`, `to_v` and `to_out.0`, and nothing "
                "else: the UNet base weights, its hint encoder, the MoVQ, the prior and the depth estimator are frozen. This is "
                "**parameter-efficient gradient training**, not full fine-tuning. Each step takes one training photograph's latent "
                "(MoVQ-encoded once), its caption embedding and its own depth hint, draws a timestep and a noise tensor (both seeded), "
                "and minimises the MSE between the predicted and the true noise. The optimiser is AdamW at a fixed learning rate with "
                "gradient-norm clipping at 1.0 and float16 autocast with loss scaling on CUDA. Epoch 0 records the frozen model's "
                "validation loss, and the epoch with the lowest validation loss is kept. Before its process ends, the stage records "
                "two reference values of the trained model still in memory — its test denoising loss and one image at a fixed "
                "prompt, hint and seed — and exports the adapter as `outputs/{stem}_adapter/` (`adapter.safetensors` + "
                "`manifest.json`).\n\n"
                "**What to notice:** the training loss is noisy, because every step draws a random timestep; the validation loss "
                "is the number that decides which epoch is kept. A falling training loss alone is not evidence of better images. "
                "The exported artifact holds 176 tensors of a few MB, with its SHA-256."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 1  # @param {{type:\"integer\"}}\n\n"
                "run_stage('adapt', '--epochs', EPOCHS, '--lr', LEARNING_RATE, '--batch-size', BATCH_SIZE)"
            ),
        },
        {
            "md": (
                "## 9. Held-out evaluation: the paired comparison\n\n"
                "**Section type:** Evaluation practice.  \n"
                "**Question:** on photographs the adapter never saw, did adaptation help, and did it cost depth fidelity?\n\n"
                "The test photographs were never used for training or for choosing the epoch. The `evaluate` stage is a fresh "
                "process: it loads the adapter from the exported files — the artifact you would ship, not the object that was "
                "trained — and scores it exactly as the frozen model was scored in Section 6 — the same seed, so the same latents, "
                "noise and timesteps, and the same 12 prompt/hint/seed triples for generation — and the table puts frozen, adapted "
                "and reference numbers side by side. The stage asserts only what the procedure guarantees: the kept epoch's "
                "validation loss is no higher than the frozen model's, and the exported adapter reproduces the recorded value. The "
                "test numbers are printed, not asserted: a worse held-out result is a valid result and is kept.\n\n"
                "**Predict first:** which number will move most — the denoising loss, CLIP label accuracy or depth correlation?\n\n"
                "**What to notice:** read each change beside its reference — the real-photo ceiling for CLIP and the mismatched "
                "baseline for depth. With 12 images, a difference of one image in label accuracy is 0.083. The grid shows each "
                "hint, the frozen image and the adapted image; the full record is `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "run_stage('evaluate')\n"
                "show_image('{stem}_adapted_grid.jpg', 'Hint, frozen model, adapted model: same prompts, same seeds')"
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
                "## 10. Fresh reload and a new prompt\n\n"
                "**Section type:** Engineering (artifact and reproducibility).  \n"
                "**Question:** does the adapter work as a file, apart from the process that trained it?\n\n"
                "The `reload` stage is a second fresh process. `KandinskyDepthPipeline.from_artifact` re-verifies the snapshots and "
                "checks the manifest — the artifact format, the decoder's id and revision, the prior and depth-estimator identities, "
                "the LoRA scope and the weights' SHA-256 — **before** deserialising; it loads a new UNet with the adapter attached "
                "and overlays the tensors. Nothing of the trained model survives in memory between processes, so this is a reload "
                "from files in the strictest sense. The stage then checks parity against the two reference values the `adapt` "
                "process recorded from the trained model while it was still in memory: the same held-out denoising loss and the "
                "same image for the same prompt, hint and seed. Loading is not the same as reproducing: only the parity check shows "
                "the artifact is equivalent. Only after parity holds does it render `NEW_PROMPT` — a composition that appears in no "
                "training caption — on the depth layouts of two test photographs; the CLIP and depth numbers are printed as a sanity "
                "check, not an evaluation. Finally it writes `outputs/{stem}_result.json` with the provenance, the runtime versions "
                "and the comparison, and lists every file in the run's `outputs/`.\n\n"
                "**What to notice:** an artifact of a few MB, a parity difference of 0 (or below the stated tolerance), the "
                "new-prompt numbers labelled as a sanity check, and the list of files under `outputs/`."
            ),
            "code": (
                "run_stage('reload')\n"
                "show_image('{stem}_new_prompt_0.png', 'New prompt on the first test layout (reloaded adapter)')\n"
                "show_image('{stem}_new_prompt_1.png', 'New prompt on the second test layout (reloaded adapter)')"
            ),
        },
        {
            "md": (
                "## 11. Troubleshooting\n\n"
                "**Section type:** Engineering.\n\n"
                "| Observation | What it means | What to do |\n"
                "|---|---|---|\n"
                "| Section 1 stops with `No GPU driver was found` or `CUDA was not detected`, or Section 2 stops with `The isolated environment cannot see a CUDA GPU` | the runtime has no GPU | Runtime → Change runtime type → T4 GPU, then Run all. This notebook is not supported on a CPU-only runtime; if Colab offers no GPU, your quota may be exhausted |\n"
                "| Section 1 stops with `This notebook needs a Linux x86_64 GPU runtime` | a local Windows or macOS kernel, or an ARM machine | use Google Colab, Kaggle, or a Linux x86_64 machine with a CUDA GPU: the locked environment is built for manylinux x86_64 wheels |\n"
                "| Section 1 stops with `Not enough free disk` | the four snapshots need about 17.8 GB and the isolated environment about 12 GB | start a fresh runtime with about 31 GB free; a `weights/` directory from an earlier run is reused and counted |\n"
                "| `Carried file integrity failure` in Section 2 | a carried file was edited in the notebook | do not edit the infrastructure cells; open a fresh copy of the notebook from the repository |\n"
                "| `uv 0.12.15 wheel size/hash mismatch`, or a `URLError` / timeout while downloading it | a network failure or an unexpected response from PyPI | re-run the Section 2 install cell; never replace the pinned URL or digest |\n"
                "| `CalledProcessError` from `uv venv` or `uv pip install` (a hash mismatch, `Failed to download`, HTTP 5xx) | a transient PyPI or network failure, or a package that no longer matches the lock | re-run the Section 2 install cell (`uv` reuses what it already downloaded); if a hash mismatch repeats, stop and report it — never remove `--require-hashes`, a pin or a hash |\n"
                "| `RuntimeError: Stage '…' failed (exit 2): …` | the stage raised an error; the text after the colon is the stage's own message, and its full log (with the traceback) is printed above and kept in the run directory's `logs/` | find the message in the rows below; a stage reads only files, so after fixing the cause re-run that cell and the cells after it |\n"
                "| `… is missing: run the stage that writes it before …` | a learner cell was run before an earlier stage | run the notebook from the top, or re-run the earlier cells in order |\n"
                "| `the dataset changed since 'prepare'` | the BYOD zip or the photo cache changed after Section 4 | re-run from Section 4 |\n"
                "| `sha256 ... != manifest` or `size ... != manifest` | a download is incomplete or altered | delete the named file under `weights/` and re-run Section 3; never edit the manifest |\n"
                "| `No space left on device` during a stage | the disk filled after the Section 1 check | start a fresh runtime with about 31 GB free |\n"
                "| `CUDA out of memory` | a form field was raised, or another program holds GPU memory | keep `BATCH_SIZE = 1` and re-run that cell; each stage is its own process, so a failed stage's memory was released when it stopped; do not report a skipped stage as a result |\n"
                "| A photo fetch fails in Section 4 | the open-data bucket is unreachable or a file changed | re-run the cell; a digest mismatch is refused on purpose, and verified photographs in `weights/inat-birds/` are reused |\n"
                "| `ModuleNotFoundError: No module named 'google.colab'` with `USE_BYOD = True` | the upload dialog needs Google Colab | set `BYOD_PATH` to a zip already in the runtime, or keep `USE_BYOD = False` |\n"
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
        "objective is measured on identical inputs before and after, and the exported artifact reloads in a fresh process with the "
        "same outputs.\n\n"
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
        "Successful execution proves that the recorded repository revision's pipeline modules and stage runner, carried in this "
        "standalone notebook and run in an isolated hash-locked environment, can stage and digest-verify four pinned safetensors "
        "snapshots, fetch and validate digest-pinned real photographs, compute depth hints, encode prompts and release the prior, "
        "execute bounded LoRA fine-tuning, evaluate the frozen and the adapted model on identical held-out inputs with a real-photo "
        "ceiling and a mismatched-hint baseline, reload the exported adapter in a fresh process with parity to the trained model, "
        "and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, production fitness, image quality beyond the checks shown, or that the pinned safetensors "
        "conversion is bit-identical to the upstream `.bin` weights.\n\n"
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
        "- uv (the installer that builds the isolated environment): https://docs.astral.sh/uv/\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n"
    ),
}
