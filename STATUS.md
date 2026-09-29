# Release status

Current status: **Release-grade** — at `f4fe86a` the guided `E2E` tutorial notebook `tutorials/kandinsky_controlnet_depth_colab.ipynb` passed a clean-runtime `Run all` of the default path on a Kaggle T4 and the REL12 BYOD journey (representative photographs accepted through `BYOD_PATH` and carried through depth-hint extraction, adaptation, evaluation, export and reload; two incompatible inputs refused). The decoder safetensors from the upstream repository's open, unmerged pull request #5 were proven bit-identical to `main`'s `.bin` weights by `tools/verify_conversion.py`. All three runs are dated 2026-09-29 and recorded in `docs/release-verification.md`.

The served weights total about 17.2 GB (decoder 5.29 GB, shared prior 10.57 GB, depth estimator 1.37 GB), plus the 0.6 GB evaluation scorer; provenance is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
