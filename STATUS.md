# Release status

Current status: **Release-grade** — at `c86e3fe` the guided `E2E` tutorial notebook `tutorials/kandinsky_controlnet_depth_colab.ipynb` runs its stages in an isolated hash-locked environment and installs nothing into the kernel. It passed `Run all` in one pass on Google Colab (T4, 2026-09-30) and on a clean Kaggle T4 in strict single-pass mode, and the REL12 BYOD journey on Kaggle (2026-09-29). The decoder safetensors from the upstream repository's open, unmerged pull request #5 are proven bit-identical to `main`'s `.bin` weights by `tools/verify_conversion.py`. All runs are recorded in `docs/release-verification.md`.

The served weights total about 17.2 GB (decoder 5.29 GB, shared prior 10.57 GB, depth estimator 1.37 GB), plus the 0.6 GB evaluation scorer; provenance is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
