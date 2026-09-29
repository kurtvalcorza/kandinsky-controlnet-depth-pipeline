# Release status

Current status: **Candidate** — the notebook stops at its install cell on hosted runtimes that preload `numpy`, `protobuf` and `cuda-bindings` (Google Colab and Kaggle), because the pinned install replaces those loaded packages and the cell then asks for a manual runtime restart. Notebook Specification 2.2 RUN1 and RUN10 forbid a manual restart on the `Run all` path, so the tutorial is not release-ready. The Kaggle runs recorded in `docs/release-verification.md` completed only because the executor restarted the kernel automatically; they remain valid evidence for everything after the install cell. Found in a Colab `Run all` on 2026-09-29; the fix (an isolated, hash-locked environment for the tutorial stages) is in progress.

The served weights total about 17.2 GB (decoder 5.29 GB, shared prior 10.57 GB, depth estimator 1.37 GB), plus the 0.6 GB evaluation scorer; provenance is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
