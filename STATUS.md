# Release status

Current status: **Candidate** — the guided `E2E` tutorial notebook `tutorials/kandinsky_controlnet_depth_colab.ipynb` is generated from the package and passes the static validator, the generator parity checks and the offline unit suite. No execution with the pinned weights has been recorded: neither a clean-runtime run of the notebook nor a GPU pre-flight.

Two items are open before promotion to **Release-grade**:

1. **Conversion equivalence.** The decoder safetensors come from the upstream repository's open, unmerged pull request #5 (commit `08632524…`); `main` ships only `.bin` weights. `tools/verify_conversion.py` must be run (CPU, about 11 GB of disk) and its result recorded in `docs/release-verification.md`.
2. **Clean-runtime execution.** A top-to-bottom run of the notebook in a fresh GPU runtime with no pre-staged weights must be recorded in `docs/release-verification.md`.

The served weights total about 17.2 GB (decoder 5.29 GB, shared prior 10.57 GB, depth estimator 1.37 GB), plus the 0.6 GB evaluation scorer; provenance is recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.
