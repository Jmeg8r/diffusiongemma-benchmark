"""Central configuration for the DiffusionGemma vs autoregressive-Gemma benchmark.

WHY: one place for every knob so runs are reproducible and there are no magic
numbers scattered across scripts. Model weights default to the external
"AI Education" drive (the internal disk is nearly full) but HF_HOME is honored
if already set in the environment.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = PROJECT_ROOT / "charts"
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "prompts.yaml"

# WHAT: HuggingFace cache lives on the big external drive.
# WHY: the 26B 8-bit weights (~36 GB total) won't fit the ~54 GB-free internal disk.
DEFAULT_HF_HOME = "/Volumes/AI Education/hf-cache"


def ensure_hf_home() -> str:
    """Set HF_HOME (env override wins) and return the resolved path."""
    hf_home = os.environ.get("HF_HOME") or DEFAULT_HF_HOME
    os.environ["HF_HOME"] = hf_home
    return hf_home


# --- Models under test -------------------------------------------------------
# WHY 8-bit: near-lossless, and the quant tier MUST match across both models for
# a fair speed/quality comparison. Both are the SAME architecture (Gemma 4 26B
# A4B MoE) — only diffusion-vs-autoregressive differs. That is the whole point.
QUANT = "8bit"
MODELS = {
    "diffusion": {
        "id": "mlx-community/diffusiongemma-26B-A4B-it-8bit",
        "kind": "diffusion",
        "label": "DiffusionGemma 26B (8-bit)",
    },
    "autoregressive": {
        "id": "mlx-community/gemma-4-26b-a4b-it-8bit",
        "kind": "autoregressive",
        "label": "Gemma 4 26B A4B (8-bit, AR)",
    },
}

# --- Generation settings (held constant across every timed run) --------------
MAX_TOKENS = 512
TEMPERATURE = 0.0          # deterministic for AR; diffusion still uses its EB schedule
SEED = 42
DIFFUSION_SAMPLER = "entropy-bound"   # mlx-vlm default; matches Google's model-card eval
DENOISING_STEPS_DEFAULT = 48          # model-card evaluation setting
STEP_SWEEP = [8, 16, 24, 48]          # the speed<->quality knob for run_step_sweep.py

# --- Trial structure ---------------------------------------------------------
WARMUP_RUNS = 2            # discarded (warms Metal kernels / caches)
TIMED_TRIALS = 5          # measured; report mean +/- std

# --- Reference numbers for the "myth" chart (CITED, not measured here) -------
# Source: Google DiffusionGemma blog / model card (NVIDIA H100 FP8, RTX 5090).
REPORTED_TPS = {
    "H100 (reported)": 1008,
    "RTX 5090 (reported)": 700,
}
