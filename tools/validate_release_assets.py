"""Static release-asset validation for the Kandinsky 2.2 ControlNet-depth DIMER pipeline.

Checks the STANDALONE tutorial notebook (DIMER Notebook Specification 2.2 §4, isolated hash-locked environment of
§25.13, including the §3.5 guided layer), the carried stage runner, the tutorial registry, model card, README, STATUS.md and weight documentation for
source conformance and cross-document identity consistency, and runs the generator parity checks (PAR1-PAR3).

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.
"""
# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "kandinsky_controlnet_depth_pipeline"
REPO_NAME = "kandinsky-controlnet-depth-pipeline"
NOTEBOOK_NAME = "kandinsky_controlnet_depth_colab.ipynb"
EXPECTED_PROFILE = "E2E"
EXPECTED_MODEL_ID = "kandinsky-community/kandinsky-2-2-controlnet-depth"
STAGE_RUNNER = ROOT / "tools" / "tutorial_stages.py"
# Other full commits the documents may cite: the shared prior, the depth estimator, the CLIP scorer, the upstream
# `main` commit that ships only .bin weights, and the commit that published the current UNet checkpoint.
KNOWN_SHAS: frozenset[str] = frozenset(
    {
        "9fc51ad5732afc5d031724219d22e6c42179c5a8",
        "bc15f29aa3a80d532f2ed650b5e16ac48d8958f9",
        "1a25a446712ba5ee05982a381eed697ef9b435cf",
        "4ecd717e8c9086cf4a16ca28b64894f70a42cd08",
        "741f080424a3450bc27ea13c591a8773ad09daec",
    }
)
BYOD_GATES = ("USE_BYOD", "ACTIVITY_HINT")
# EXE1/EXE2 form fields, exactly as the Kaggle REL12 harness edits them in a run copy.
BYOD_FIELD_LINES = ('USE_BYOD = False  # @param {type:"boolean"}', "BYOD_PATH = ''  # @param {type:\"string\"}")
# NOTEBOOK_SPEC 2.2 §3.5 guided layer (GDL1-GDL15): the learner-facing scaffolding the template carries.
GUIDED_MARKERS = (
    "### Who this notebook is for",  # GDL1
    "### How to use this notebook",  # GDL2
    "**Where the code runs.**",  # GDL2 (isolated environment)
    "### Roadmap",  # GDL3
    "### Input → Model/System → Output",  # GDL4
    "you should be able to **explain**",  # GDL5
    "<summary><b>Glossary</b>",  # GDL6
    "**Predict first:**",  # GDL7
    "**What to notice:**",  # GDL8
    "<summary><b>Check your reasoning</b>",  # GDL9
    "Predict → Change one thing → Run → Observe → Explain",  # GDL10
    "> **Infrastructure.**",  # GDL11
    "**Section type:** Core concept",  # GDL12
    "**Section type:** Evaluation practice",  # GDL12
    "**Section type:** Engineering",  # GDL12
    "## 11. Troubleshooting",  # GDL13
    "## Conclude with evidence",  # GDL14
)
GUIDED_MIN_COUNTS = {
    "**Predict first:**": 4,
    "**What to notice:**": 10,
    "<summary><b>Check your reasoning</b>": 4,
}
INFRASTRUCTURE_TITLES = {
    "check": "# @title Infrastructure: check the runtime, GPU and disk; create a fresh run directory",
    "carrier": "# @title Infrastructure: write and verify the carried package, stage runner, lock and manifests",
    "install": "# @title Infrastructure: install the locked runtime into an isolated environment and define the stage runner",
    "weights": "# @title Infrastructure: stage and digest-verify the four pinned snapshots",
}
# The learner cells run the stages in this order (RUN1: the default path is one pass from the top).
STAGE_CALLS = (
    "run_stage('weights')",
    "run_stage('prepare', *prepare_options)",
    "run_stage('encode', '--new-prompt', NEW_PROMPT)",
    "run_stage('frozen', '--steps', STEPS, '--guidance', GUIDANCE_SCALE)",
    "run_stage('activity', '--hint', ACTIVITY_HINT)",
    "run_stage('adapt', '--epochs', EPOCHS, '--lr', LEARNING_RATE, '--batch-size', BATCH_SIZE)",
    "run_stage('evaluate')",
    "run_stage('reload')",
)
# Kernel cells may import only the standard library, IPython's display helpers and (for the BYOD upload) google.colab.
ALLOWED_KERNEL_IMPORTS = frozenset(
    {"hashlib", "io", "json", "os", "pathlib", "platform", "shutil", "subprocess", "tempfile", "time", "urllib.error", "urllib.request", "uuid", "zipfile", "IPython.display", "google.colab"}
)
KERNEL_CODE_MARKERS = (
    "CARRIED_FILES = {",
    "CARRIED_HASHES = {",
    "for name, text in CARRIED_FILES.items():",
    "if hashlib.sha256(path.read_bytes()).hexdigest() != CARRIED_HASHES[name]:",
    "NOTEBOOK_SOURCE = json.loads(",
    "if len(wheel) != UV_BYTES or hashlib.sha256(wheel).hexdigest() != UV_SHA256:",
    "'venv', '--managed-python', '--python', '3.12.12'",
    "'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:'",
    "MPLBACKEND='Agg'",
    "for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH', 'PYTHONHOME'):",
    "def run_stage(stage, *options):",
    "detail = error['type'] + ': ' + error['message']",
    "raise RuntimeError(f'Stage {stage!r} failed (exit {process.returncode}): {detail}')",
    "if not RUNTIME['cuda']:",
    "from google.colab import files",
    "files.upload()",
    "byod_path = Path(BYOD_PATH)",
    "prepare_options = ['--byod', byod_path.resolve()]",
    "ACTIVITY_HINT = 'flat'",
)
# What the carried stage runner must do (checked on tools/tutorial_stages.py, which the carrier holds byte for byte).
RUNNER_MARKERS = (
    "KandinskyDepthPipeline.from_pretrained(",
    "use_lora=True",
    "depth_dir=run.snapshot(SNAPSHOT_KEYS[2])",
    "KandinskyDepthPipeline.from_artifact(",
    "fetched = stage(target, allow_download=True)",
    "verified = verify(target)",
    "stage_missing_depth_files, verify_depth_snapshot",
    "splits = load_sample_splits(run)",
    "fetch_sample_dataset(cache_dir=run.weights / SAMPLE_CACHE)",
    "splits = split_dataset(load_byod_dataset(byod_path), seed=0)",
    "dataset_report = dataset_manifest(",
    "write_dataset_csv(test_records,",
    "validate_dataset(records)",
    'if digest != data["dataset_digest"]:',
    "hint_report = pipe.compute_hints(all_records)",
    "encode_report = pipe.encode_prompts(all_prompts)",
    "released = pipe.release_prior()",
    "save_prompt_cache(run, cache, new_prompt)",
    "save_hint_cache(run, hints)",
    "load_caches(run, pipe,",
    "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)",
    "frozen_generation = pipe.generate(generation_prompts, generation_hints, seed=GEN_SEED, steps=opts.steps, guidance_scale=opts.guidance)",
    'frozen_scores = score_generations(scorer, frozen_generation["images"], references=test_records)',
    'frozen_depth = score_depth_fidelity(pipe, frozen_generation["images"])',
    "real_ceiling = real_photo_baseline(scorer, test_records)",
    "depth_floor = real_photo_depth_ceiling(pipe, test_records)",
    "changed_hints = [flat_hint() for _ in activity_records]",
    'activity_depth = depth_fidelity_from_maps(pipe.estimate_depth([g["image"] for g in activity_generation["images"]]), own_hints)',
    "adapt_result = pipe.adapt(",
    "lr=opts.lr",
    "batch_size=opts.batch_size",
    "in_memory_test = pipe.evaluate(",
    "manifest = pipe.save_artifact(artifact_dir, metadata=",
    'pipe = load_from_artifact(run, Path(adapted_state["artifact"]["dir"]))',
    "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)",
    'adapted_scores = score_generations(scorer, adapted_generation["images"], references=test_records)',
    'adapted_depth = score_depth_fidelity(pipe, adapted_generation["images"])',
    '"depth_correlation": {"frozen": frozen_depth["depth_correlation"]',
    '"depth_aligned_mae": {"frozen": frozen_depth["depth_aligned_mae"]',
    'if not best["val_loss"] <= history[0]["val_loss"]:',
    'if not abs(adapted_val["denoising_mse"] - best["val_loss"]) < 1e-4:',
    "reloaded = load_from_artifact(run, artifact_dir)",
    'if not (parity["denoising_mse_diff"] < 1e-6 and parity["mean_abs_pixel_diff"] < 1.0):',
    "new_generation = reloaded.generate([new_prompt, new_prompt], new_hints, seed=NEW_PROMPT_SEED",
    '"safetensors_only": True',
    '"remote_code_executed": False',
    '"data_base_url": CORPUS_BASE_URL',
    "error_file.write_text(json.dumps(",
    'print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)',
)
EXPECTED_OUTPUTS = (
    "_sample_captions.csv",
    "_hints.jpg",
    "_frozen_grid.jpg",
    "_adapted_grid.jpg",
    "_evaluation_report.json",
    "_adapter",
    "_result.json",
    "_new_prompt_",
    "_activity_grid.jpg",
)
MARKDOWN_MARKERS = (
    "**Capability:** depth-conditioned text-to-image generation (photograph → depth hint → prompt + hint → 512 × 512 image), held-out denoising-loss, CLIP and depth-fidelity evaluation, and bounded LoRA fine-tuning to a set of captioned photographs",
    "prior pipeline can be released",
    "Generation has no ground truth",
    "denoising loss",
    "real-photo ceiling",
    "mismatched-hint baseline",
    "depth fidelity",
    "not a human judgement",
    "open, unmerged conversion pull request",
    "Apache-2.0",
    "sample-sanity",
    "CC0",
    "Nothing is installed into the notebook kernel",
    "--require-hashes",
)
# Direct model-library use that must stay inside the carried files (G2): the kernel imports no model library at all.
FORBIDDEN_IN_KERNEL = (
    "huggingface_hub",
    "hf_hub_download(",
    "safetensors",
    "pickle",
    "sys.executable",
    "importlib",
    "pip install",
    "'-m', 'pip'",
)

NOTEBOOK_SPEC = "2.2"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "- **External access:** the Hugging Face Hub",
    "## 1. Check the runtime",
    "## 2. Carry the code and install the locked runtime",
    "## 3. Pin, stage and verify the model",
    "## Interpretation and limits",
    "Successful execution proves that the recorded repository revision",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("repository package import in the kernel (ST1)", re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(
            r"\bpickle\.load"
            r"|\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)"
            r"|weights_only\s*=\s*False"
            r"|getattr\(\s*torch\s*,\s*['\"]load['\"]"
        ),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
    ("unhashed or source install", re.compile(r"--no-binary|--no-build-isolation|--trusted-host|--extra-index-url")),
)
# Checked on the carried files (package, runner, lock): everything above except the kernel-only package-import rule.
CARRIED_FORBIDDEN = tuple(item for item in FORBIDDEN_PATTERNS if not item[0].startswith("repository package import"))


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_identity() -> tuple[str, str]:
    text = _read(ROOT / "src" / PACKAGE / "pipeline.py")
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        "pipeline.py must define MODEL_ID and MODEL_REVISION",
    )
    _check(SHA40.match(revision.group(1)) is not None, "MODEL_REVISION must be a 40-hex immutable commit")
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check('model_card_spec: "1.2"' in front or 'model_card_spec: "1.1"' in front, "MODEL_CARD.md model_card_spec must be 1.2 or 1.1")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def validate_identity_consistency() -> None:
    model_id, revision = _package_identity()
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the immutable revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


# --- weight-facts check (fleet rollout 2026-09-24) ---
# Every SHA-256 digest and byte count quoted in the weight prose must come from a committed
# weights/*/dimer-base-manifest.json, or be declared below with a label saying what it describes
# (dataset files, upstream files that are not staged, origin checkpoints, totals). Declared entries
# that no document cites any more are rejected, so the allowlist cannot go stale.
WEIGHT_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
EXTERNAL_WEIGHT_BYTES: dict[int, str] = {
    5_013_996_233: "upstream main 4ecd717e unet/diffusion_pytorch_model.bin (never staged; the conversion reference)",
    271_492_131: "upstream main 4ecd717e movq/diffusion_pytorch_model.bin (never staged; the conversion reference)",
}
EXTERNAL_WEIGHT_DIGESTS: dict[str, str] = {
    "3418cd4f977d51ec902d095ff67482232c6097edfd91b474443148113b8f8a36": "upstream main unet .bin, uploaded in commit 741f0804 (date_published source)",
    "772e09739d742ddee6807add2d3c2fd2a32db53896b5d07a92c729d8c879ce59": "upstream main movq .bin (never staged; the conversion reference)",
}
_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_GROUPED = r"(\d{1,3}(?:[,\u202f\u00a0 ]\d{3})+|\d+)"
_BYTE_COUNT = re.compile(r"(?<![\d,\-])" + _GROUPED + r"\s*bytes\b|totalBytes`?\s*" + _GROUPED)


def _manifest_facts(root: Path = ROOT) -> tuple[set[str], set[int]]:
    digests: set[str] = set()
    sizes: set[int] = set()
    for path in sorted(root.glob("weights/*/dimer-base-manifest.json")):
        manifest = json.loads(_read(path))
        sizes.add(manifest["totalBytes"])
        for entry in manifest["files"]:
            digests.add(entry["sha256"])
            sizes.add(entry["bytes"])
    return digests, sizes


def validate_weight_facts(root: Path = ROOT) -> None:
    """Every SHA-256 and byte count quoted in the weight prose must come from a manifest or a labelled allowlist entry."""
    digests, sizes = _manifest_facts(root)
    _check(bool(digests), "no weights/*/dimer-base-manifest.json found to check weight facts against")
    cited_digests: set[str] = set()
    cited_sizes: set[int] = set()
    for name in WEIGHT_DOCS:
        path = root / name
        if not path.exists():
            continue
        text = _read(path)
        found_digests = set(_DIGEST.findall(text))
        found_sizes = {int(re.sub(r"[,\u202f\u00a0 ]", "", m.group(1) or m.group(2))) for m in _BYTE_COUNT.finditer(text)}
        cited_digests |= found_digests
        cited_sizes |= found_sizes
        bad_digests = sorted(found_digests - digests - set(EXTERNAL_WEIGHT_DIGESTS))
        _check(not bad_digests, f"{name} cites SHA-256 digests absent from every manifest and from EXTERNAL_WEIGHT_DIGESTS: {bad_digests}")
        bad_sizes = sorted(found_sizes - sizes - set(EXTERNAL_WEIGHT_BYTES))
        _check(not bad_sizes, f"{name} cites byte counts absent from every manifest and from EXTERNAL_WEIGHT_BYTES: {bad_sizes}")
    stale = sorted(set(EXTERNAL_WEIGHT_BYTES) - cited_sizes) + sorted(set(EXTERNAL_WEIGHT_DIGESTS) - cited_digests)
    _check(not stale, f"EXTERNAL_WEIGHT_* entries no weight document cites any more: {stale}")


# --- end weight-facts check ---

def validate_release_status() -> None:
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )




def _validate_notebook_structure(path: Path, notebook: dict) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == EXPECTED_PROFILE, f"{path.name}: profile {profile!r} != declared {EXPECTED_PROFILE!r}")
    spec = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(spec == NOTEBOOK_SPEC, f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'")
    _check(dimer.get("notebook_mode") in ("REFERENCE", "GUIDED", "WORKSHOP"), f"{path.name}: metadata.dimer.notebook_mode must declare a §3.3 pedagogical mode")
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    _check(dimer.get("requires_dimer_worker") is False, f"{path.name}: metadata.dimer.requires_dimer_worker must be false")
    generated = dimer.get("generated_from")
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(generated.get("repository") == REPO_NAME, f"{path.name}: generated_from.repository must be {REPO_NAME}")
    _check(generated.get("module") == f"src/{PACKAGE}/pipeline.py", f"{path.name}: generated_from.module must name the package entry module")
    _check(bool(generated.get("generator")), f"{path.name}: generated_from.generator is required")
    cells = notebook.get("cells", [])
    _check(bool(cells) and cells[0].get("cell_type") == "markdown", f"{path.name}: first cell must be markdown")
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(index > 0 and cells[index - 1].get("cell_type") == "markdown", f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell")
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    kernel = "\n".join(source for index, source, _ in code_cells if not _is_carrier(cells[index]))
    _check(not PLACEHOLDER.search(kernel + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _is_carrier(cell: dict) -> bool:
    return bool(cell.get("metadata", {}).get("dimer", {}).get("embedded_sources"))


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValidationError(f"{name} is not assigned as a literal at the top level of its cell")


def _validate_carrier(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], build) -> tuple[int, dict[str, str]]:
    """PAR1/SRC4: one carrier cell; every carried file is the repository file and its CARRIED_HASHES entry is correct;
    the cell verifies each written file against CARRIED_HASHES and stops on a mismatch."""
    carriers = [(index, tree) for index, _source, tree in code_cells if _is_carrier(notebook["cells"][index])]
    _check(len(carriers) == 1, f"{path.name}: exactly one carrier cell (metadata.dimer.embedded_sources) is expected, found {len(carriers)}")
    index, tree = carriers[0]
    files, hashes = _literal(tree, "CARRIED_FILES"), _literal(tree, "CARRIED_HASHES")
    template = _load_tool("notebook_template").TEMPLATE
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    ctx = build.load_context(ROOT, template, recorded)
    _check(files == ctx["files"], f"{path.name}: carried files differ from the repository sources (PAR1); regenerate the notebook")
    for name, text in files.items():
        _check(hashes.get(name) == hashlib.sha256(text.encode("utf-8")).hexdigest(), f"{path.name}: CARRIED_HASHES[{name!r}] is wrong (SRC4)")
    _check(set(hashes) == set(files), f"{path.name}: CARRIED_HASHES and CARRIED_FILES name different files")
    _check(notebook["cells"][index]["metadata"]["dimer"].get("files") == hashes, f"{path.name}: carrier metadata must record the carried hashes")
    _check(notebook["metadata"]["dimer"]["generated_from"].get("files") == hashes, f"{path.name}: generated_from.files must record the carried hashes")
    _check(notebook["metadata"]["dimer"]["generated_from"].get("module_sha256") == ctx["module_sha256"], f"{path.name}: generated_from.module_sha256 does not match src/ (PAR4)")
    verifies = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and "CARRIED_FILES.items()" in ast.unparse(node.iter):
            body = ast.unparse(node)
            verifies = "CARRIED_HASHES[name]" in body and "raise RuntimeError(" in body and "hashlib.sha256(path.read_bytes())" in body
    _check(verifies, f"{path.name}: the carrier must verify every written file against CARRIED_HASHES and raise on a mismatch (SRC4)")
    runner = template["stage_runner"]
    _check(files.get(runner) == _read(STAGE_RUNNER), f"{path.name}: the carried stage runner is not tools/tutorial_stages.py (PAR1)")
    lock_source = ROOT / template["carried"][template["lock"]]
    _check(files.get(template["lock"]) == _read(lock_source), f"{path.name}: the carried lock is not {lock_source.relative_to(ROOT).as_posix()} (PAR2)")
    return index, files


def _validate_lock(path: Path, files: dict[str, str], build) -> None:
    """ENV1/ENV2: the carried lock pins every pyproject pin, hashes every entry, and was compiled wheel-only."""
    template = _load_tool("notebook_template").TEMPLATE
    lock = files[template["lock"]]
    try:
        build.check_lock(build._pins(ROOT), lock)
    except SystemExit as exc:
        raise ValidationError(f"{path.name}: {exc}") from exc
    header = "\n".join(lock.splitlines()[:3])
    _check("--generate-hashes" in header and "--only-binary :all:" in header, f"{path.name}: the lock must be compiled with --generate-hashes --only-binary :all:")
    _check("--python-platform x86_64-manylinux" in header, f"{path.name}: the lock must target manylinux x86_64 (Colab/Kaggle)")


def _validate_isolated_install(path: Path, code_cells: list[tuple[int, str, ast.Module]], carrier: int) -> None:
    """RUN10/ENV6 (§25.13): nothing is pip-installed into the kernel; the only installer is the pinned uv, which installs
    the lock with --require-hashes into a separate environment; run_stage re-raises a stage's own error message."""
    template = _load_tool("notebook_template").TEMPLATE
    uv = template["uv"]
    install = [(index, source, tree) for index, source, tree in code_cells if source.startswith(INFRASTRUCTURE_TITLES["install"])]
    _check(len(install) == 1, f"{path.name}: exactly one isolated-install cell is expected")
    install_index, source, tree = install[0]
    _check(_literal(tree, "UV_URL") == uv["url"] and uv["url"].startswith("https://files.pythonhosted.org/"), f"{path.name}: UV_URL must be the pinned PyPI wheel")
    _check(_literal(tree, "UV_BYTES") == uv["bytes"], f"{path.name}: UV_BYTES must be the pinned wheel size")
    _check(SHA64.match(str(_literal(tree, "UV_SHA256"))) is not None and _literal(tree, "UV_SHA256") == uv["sha256"], f"{path.name}: UV_SHA256 must pin the uv wheel (64-hex)")
    run_stage = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_stage"]
    _check(len(run_stage) == 1, f"{path.name}: the install cell must define run_stage")
    _check("raise RuntimeError(" in ast.unparse(run_stage[0]) and "error_file" in ast.unparse(run_stage[0]), f"{path.name}: run_stage must re-raise the stage's error message")
    for index, cell_source, cell_tree in code_cells:
        if index == carrier:
            continue
        stripped = _strip_comments(cell_source)
        leaked = [marker for marker in FORBIDDEN_IN_KERNEL if marker in stripped]
        _check(not leaked, f"{path.name}: cell {index} does kernel-side work that belongs in the isolated environment: {leaked}")
        for node in ast.walk(cell_tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                bad = [n for n in names if n not in ALLOWED_KERNEL_IMPORTS]
                _check(not bad, f"{path.name}: cell {index} imports {bad} into the kernel; only the standard library, IPython.display and google.colab are allowed")
            if isinstance(node, ast.List) and any(isinstance(e, ast.Constant) and e.value == "pip" for e in node.elts):
                call = ast.unparse(node)
                _check(
                    call.startswith("[str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes'"),
                    f"{path.name}: cell {index} runs pip other than `uv pip install --python <isolated env> --require-hashes`: {call[:80]}",
                )
            if isinstance(node, ast.Name) and node.id == "urllib" and index != install_index:
                raise ValidationError(f"{path.name}: cell {index} downloads outside the isolated-install cell")


def _validate_identity(path: Path, code_cells: list[tuple[int, str, ast.Module]], carrier: int, revision: str) -> None:
    for index, source, tree in code_cells:
        if index == carrier:
            continue
        for node in ast.walk(tree):
            rebound = [name for name in _assignment_targets(node) if name in IDENTITY_NAMES]
            _check(not rebound, f"{path.name}: {rebound} must not be rebound outside the carried package (cell {index})")
        _check(revision not in source, f"{path.name}: the model revision may appear only in the carried package and manifests (cell {index})")


def _validate_parity(path: Path, notebook: dict, build) -> None:
    template = _load_tool("notebook_template").TEMPLATE
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")
    _check(current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate")


def _validate_notebook_content(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], markdown: str, carrier: int, files: dict[str, str]) -> None:
    model_id, _revision = _package_identity()
    kernel_cells = [(index, source) for index, source, _ in code_cells if index != carrier]
    kernel = "\n".join(_strip_comments(source) for _, source in kernel_cells)
    carrier_source = next(source for index, source, _ in code_cells if index == carrier)
    missing = [marker for marker in KERNEL_CODE_MARKERS if marker not in kernel + "\n" + carrier_source]
    _check(not missing, f"{path.name}: missing required kernel-code markers: {missing}")
    for title in INFRASTRUCTURE_TITLES.values():
        _check(sum(source.startswith(title) for _, source, _ in code_cells) == 1, f"{path.name}: expected exactly one cell titled {title!r}")
    positions = []
    for call in STAGE_CALLS:
        hits = [index for index, source in kernel_cells if call in source]
        _check(len(hits) == 1, f"{path.name}: expected exactly one `{call}`, found {len(hits)}")
        positions.append(hits[0])
    _check(positions == sorted(positions), f"{path.name}: the stages must run in order {[c.split(chr(39))[1] for c in STAGE_CALLS]} (RUN1)")
    byod_cells = [source for _, source in kernel_cells if all(line in source.splitlines() for line in BYOD_FIELD_LINES)]
    _check(len(byod_cells) == 1, f"{path.name}: one learner cell must hold both BYOD form fields exactly: {BYOD_FIELD_LINES} (EXE1/EXE2)")
    _check("run_stage('prepare', *prepare_options)" in byod_cells[0], f"{path.name}: the BYOD fields must drive the prepare stage in the same cell")
    for label, pattern in FORBIDDEN_PATTERNS:
        _check(not pattern.search(kernel), f"{path.name}: forbidden/insecure kernel source: {label}")
    for name, text in files.items():
        present = [label for label, pattern in CARRIED_FORBIDDEN if pattern.search(text)]
        _check(not present, f"{path.name}: forbidden/insecure source in carried {name}: {present}")
    runner = files[_load_tool("notebook_template").TEMPLATE["stage_runner"]]
    missing = [marker for marker in RUNNER_MARKERS if marker not in runner]
    _check(not missing, f"{path.name}: the carried stage runner is missing required markers: {missing}")
    missing = [name for name in EXPECTED_OUTPUTS if name not in runner]
    _check(not missing, f"{path.name}: the stage runner must export {missing}")
    _validate_gates(path, code_cells)
    missing_md = [marker for marker in COMMON_MARKDOWN_MARKERS + MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check("restart" not in markdown.lower().replace("no runtime restart", "").replace("no restart", ""), f"{path.name}: learner prose must not instruct a runtime restart (RUN10)")
    _check(f"**Profile:** `{EXPECTED_PROFILE}`" in markdown, f"{path.name}: markdown must state the profile")
    _check(f"https://huggingface.co/{model_id}" in markdown, f"{path.name}: references must link {model_id}")


def _validate_gates(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    for gate in BYOD_GATES:
        assignments = []
        for index, _source, tree in code_cells:
            for node in ast.walk(tree):
                if gate in _assignment_targets(node):
                    assignments.append((index, node))
        _check(len(assignments) == 1, f"{path.name}: {gate} must be assigned exactly once, found {len(assignments)}")


def _validate_guided_layer(path: Path, notebook: dict, markdown: str) -> None:
    """GDL1–GDL15 (§3.5): orientation, predictions, checkpoints, the change-one-thing activity, collapsed infrastructure."""
    missing = [marker for marker in GUIDED_MARKERS if marker not in markdown]
    _check(not missing, f"{path.name}: guided layer (§3.5) is missing: {missing}")
    for marker, minimum in GUIDED_MIN_COUNTS.items():
        found = markdown.count(marker)
        _check(found >= minimum, f"{path.name}: guided layer needs at least {minimum} × {marker!r}, found {found}")
    _check(markdown.count("<details>") == markdown.count("</details>"), f"{path.name}: unbalanced <details> blocks")
    _check(not re.search(r"\bworkshop\b", markdown, re.I), f"{path.name}: learner prose must say 'notebook', not 'workshop' (GDL15)")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        meta = cell.get("metadata", {})
        infrastructure = source.startswith("# @title Infrastructure: ")
        collapsed = meta.get("cellView") == "form" and meta.get("jupyter", {}).get("source_hidden") is True
        if infrastructure or _is_carrier(cell):
            _check(collapsed and infrastructure, f"{path.name}: infrastructure cell {index} must be titled '# @title Infrastructure: …' and collapsed (GDL11)")
        else:
            _check(not collapsed, f"{path.name}: learner cell {index} must not be collapsed")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    notebooks = sorted(tutorials.glob("*.ipynb"))
    _check(len(notebooks) == 1, f"exactly one tutorial notebook is expected, found {len(notebooks)}")
    path = notebooks[0]
    _check(path.name == NOTEBOOK_NAME, f"tutorial notebook must be named {NOTEBOOK_NAME}, found {path.name}")
    build = _load_tool("build_notebook")
    notebook = json.loads(_read(path))
    code_cells, markdown = _validate_notebook_structure(path, notebook)
    carrier, files = _validate_carrier(path, notebook, code_cells, build)
    _validate_lock(path, files, build)
    _validate_isolated_install(path, code_cells, carrier)
    _model_id, revision = _package_identity()
    _validate_identity(path, code_cells, carrier, revision)
    _validate_parity(path, notebook, build)
    _validate_notebook_content(path, notebook, code_cells, markdown, carrier, files)
    _validate_guided_layer(path, notebook, markdown)
    registry = _read(tutorials / "README.md")
    _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
    _check(f"`{EXPECTED_PROFILE}`" in registry, f"tutorials/README.md must record `{EXPECTED_PROFILE}`")
    _check(f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry, "tutorials/README.md must name the notebook spec version")
    _check("standalone" in registry.lower(), "tutorials/README.md must record that the notebook is standalone")
    _check("isolated" in registry.lower(), "tutorials/README.md must describe the isolated environment")


def validate_all() -> list[str]:
    validate_model_card()
    validate_identity_consistency()
    validate_weight_facts()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "identity-consistency", "weight-facts", "release-status", "notebook+carrier+lock+parity"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
