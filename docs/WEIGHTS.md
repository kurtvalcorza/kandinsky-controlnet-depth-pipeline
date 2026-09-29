# Weight provenance: the four pinned snapshots

This repository pins **four** Hugging Face snapshots, each with its own `dimer-base-manifest.json` (format
`dimer_hf_snapshot` v1: the immutable revision and the byte size and SHA-256 of every file). Each is staged and
verified separately by `src/kandinsky_controlnet_depth_pipeline/pipeline.py`. Every file that is loaded is
safetensors or JSON/text; nothing is unpickled, and the model classes come from `diffusers`, `transformers` and
`peft` on PyPI — no Hub-hosted code is executed. Staging refuses any revision that is not a 40-character commit.

## 1. The decoder — Kandinsky 2.2 ControlNet-depth

- Upstream: `kandinsky-community/kandinsky-2-2-controlnet-depth`
- Pinned revision: `08632524a2b7e3bed39a7901d92cdd07e37544d0`
- Upstream weight licence: **Apache-2.0** (`license: apache-2.0` in the pinned README front matter).

### Why the pin is a pull-request commit

The upstream `main` branch, at commit `4ecd717e8c9086cf4a16ca28b64894f70a42cd08` (2023-10-09), ships **only pickled
`.bin` weights**:

| File on `main` | Bytes | SHA-256 |
|---|---|---|
| `unet/diffusion_pytorch_model.bin` | 5,013,996,233 bytes | `3418cd4f977d51ec902d095ff67482232c6097edfd91b474443148113b8f8a36` |
| `movq/diffusion_pytorch_model.bin` | 271,492,131 bytes | `772e09739d742ddee6807add2d3c2fd2a32db53896b5d07a92c729d8c879ce59` |

This package never stages or loads a `.bin` file. The safetensors files it pins come from the repository's **open,
unmerged pull request #5**, "Adding `safetensors` variant of this model", opened by patrickvonplaten on 2023-07-25.
Its head commit `08632524a2b7e3bed39a7901d92cdd07e37544d0` is addressed directly. Downloads pass that full commit SHA
as `revision`, never `main` and never the moving `refs/pr/5` name. A later open pull request, #7, by the automated
SFconvertbot, carries safetensors files with the same two SHA-256 digests.

Both pull requests are parented on commits after 2023-07-13, when the current UNet `.bin` (digest `3418cd4f…`) was
uploaded in commit `741f080424a3450bc27ea13c591a8773ad09daec`. The earlier 2023-06-28 upload had a different digest.
The safetensors are therefore conversions of the current checkpoint by provenance. **Byte-level equivalence of the
converted tensors to the `main` `.bin` files is not yet verified.** `tools/verify_conversion.py` is the pending
check: it downloads both at their exact commits, verifies the digests above, loads the `.bin` files with
`torch.load(weights_only=True)` and asserts identical key sets, shapes, dtypes and `torch.equal` values for the UNet
and the MoVQ. It needs about 11 GB of disk and is meant for a disposable CPU kernel.

The MoVQ safetensors digest `43a5860fea195a7116f2471396c5cc9535fade9b63c4857d8a192ffd924b7002` (271,380,364 bytes)
is identical to the MoVQ in the pinned `kandinsky-community/kandinsky-2-2-decoder` snapshot used by the text-to-image
sibling pipeline, so the MoVQ is the same autoencoder.

### Local layout

`weights/kandinsky-2-2-controlnet-depth/` holds the 7 manifest entries (5,285,192,144 bytes in total): upstream
`README.md`, `model_index.json`, `scheduler/scheduler_config.json`, `movq/config.json`,
`movq/diffusion_pytorch_model.safetensors` (271,380,364 bytes), `unet/config.json`, and
`unet/diffusion_pytorch_model.safetensors` (5,013,798,992 bytes, SHA-256
`6549f8c8471357ed8ed6b700a80ffdd8fe45bd5b54f4c486d7f81ac9fe5f343b`). Only the small files are committed.

### Architecture facts (read from the safetensors headers and confirmed by instantiating the configs)

- `UNet2DConditionModel` with `addition_embed_type="image_hint"`, `in_channels=8`, `out_channels=8`:
  1,253,429,212 parameters in 740 float32 tensors. Compared with the text-to-image decoder UNet it adds the 16
  `add_embedding.input_hint_block.*` tensors and a wider `conv_in` (8 input channels).
- LoRA scope (`to_q`, `to_k`, `to_v`, `to_out.0`, rank 8): 176 tensors, 1,646,592 parameters.
- MoVQ `VQModel`: 67,832,495 parameters in 431 tensors.

## 2. The prior — Kandinsky 2.2 Prior (shared)

- Upstream: `kandinsky-community/kandinsky-2-2-prior`, revision `9fc51ad5732afc5d031724219d22e6c42179c5a8`,
  Apache-2.0. The manifest is shared byte-for-byte with the text-to-image sibling pipeline.
- `weights/kandinsky-2-2-prior/` holds the 14 manifest entries (10,574,964,619 bytes): the `PriorTransformer`
  (4,104,940,968 bytes), the CLIP ViT-G/14 image encoder (3,689,912,664 bytes), the CLIP text encoder
  (2,778,702,976 bytes), tokenizer, processor and scheduler files.
- The prior is loaded by `encode_prompts` and released by `release_prior`, so it never shares GPU memory with training.

## 3. The depth estimator — DPT-Large

- Upstream: `Intel/dpt-large`, revision `bc15f29aa3a80d532f2ed650b5e16ac48d8958f9`, **Apache-2.0**.
- Why this model: the upstream ControlNet-depth model card builds its hint with `transformers`'
  `pipeline("depth-estimation")` and no model id. In `transformers` 5.17.0 (the pinned version) that task's default is
  `("Intel/dpt-large", "bc15f29")`, recorded in `src/transformers/pipelines/__init__.py` (`SUPPORTED_TASKS`). This
  package pins the same model and loads it explicitly by id and revision through `AutoModelForDepthEstimation`; it
  never relies on the task default at run time. `Intel/dpt-hybrid-midas` was not used because it ships only `.bin`.
- `weights/dpt-large/` holds 4 manifest entries (1,367,465,032 bytes): `README.md`, `config.json`,
  `preprocessor_config.json` and `model.safetensors` (1,367,456,044 bytes).
- The checkpoint stores 341,850,305 parameters in 458 tensors. `DPTForDepthEstimation` builds 343,030,465 parameters
  in 462 tensors: the extra 4 are the residual branch of the first fusion layer, which `transformers` never calls
  because that layer receives no residual input. `build_depth_estimator` asserts the 462-tensor count.
- The hint recipe follows the upstream model card: the predicted depth is resized to 512 × 512 (bicubic), min-max
  normalised to 0..255, repeated over three channels and divided by 255.

## 4. The scorer — CLIP ViT-B/32 (evaluation only)

- Upstream: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, revision `1a25a446712ba5ee05982a381eed697ef9b435cf`, MIT.
- `weights/clip-vit-b-32-laion2b/`: 9 manifest entries (608,782,299 bytes), including `model.safetensors`
  (605,157,884 bytes). Used by `metrics.ClipScorer` only; never part of generation and never trained.

## Totals and publication

- Served weights: about 17.2 GB (5.29 GB decoder, 10.57 GB prior, 1.37 GB depth estimator), plus the 0.6 GB scorer.
- Licences: Apache-2.0 for the decoder, prior and depth estimator; MIT for the scorer. All four permit
  redistribution with attribution.
- `date_published` in `MODEL_CARD.md` is 2023-07-13, the upload of the current UNet checkpoint; the safetensors
  conversion followed on 2023-07-25.
