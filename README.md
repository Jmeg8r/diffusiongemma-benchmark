# DiffusionGemma Speed Benchmark vs. Autoregressive Gemma 4 (Apple Silicon)

A reproducible, hands-on benchmark testing the claim that **DiffusionGemma**
(Google DeepMind's open-weights diffusion LLM) generates "1,000+ tokens/sec" by
denoising 256 tokens in parallel — and whether that speed survives on **Apple
Silicon**, where inference is memory-bandwidth-bound rather than compute-bound.

It pits DiffusionGemma against the **autoregressive Gemma 4 26B A4B** — the *same
architecture and weights*, only diffusion-vs-autoregressive differs — at matched
8-bit quantization, on the same Mac, through the same runner (`mlx-vlm`). That
isolates the one variable that matters.

## What it measures

- **Throughput** — output tokens ÷ end-to-end wall time (the fair single number).
- **Time-to-first-token (TTFT)** — where diffusion structurally loses (it must
  denoise a whole block before emitting anything).
- **Actual denoising steps used** — the entropy-bound sampler early-stops.
- **Quality** — objective where possible (code via execution, math via numeric
  match, instruction-following via deterministic checks), blind LLM-as-judge for
  long-form writing.
- **Speed↔quality tradeoff** — a denoising-step sweep (8/16/24/48).

## Requirements

- Apple Silicon Mac (built/validated on a Mac Studio, 256 GB RAM).
- Python 3.13 (pyenv). `mlx-vlm >= 0.6.3` (DiffusionGemma needs `mlx-vlm`, **not**
  `mlx-lm`).
- ~40 GB free for the two 8-bit models — defaults to an external drive via `HF_HOME`.
- For judging: the local `claude` CLI logged in (no API key needed).

## Setup

```bash
pyenv exec python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # adjust HF_HOME if your model drive differs
```

The models download automatically on first run, or pre-fetch them:

```bash
HF_HOME="/Volumes/AI Education/hf-cache" .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/diffusiongemma-26B-A4B-it-8bit')
snapshot_download('mlx-community/gemma-4-26b-a4b-it-8bit')"
```

## Run

```bash
# Fast smoke validation (1 prompt/category, 1 trial, no judge)
.venv/bin/python run_benchmark.py --quick

# Full benchmark (5 trials, all 30 prompts, blind judge for writing)
.venv/bin/python run_benchmark.py

# Denoising-step sweep (speed<->quality knob)
.venv/bin/python run_step_sweep.py

# Render the article-ready PNG charts from the latest run + sweep
.venv/bin/python make_charts.py
```

## Outputs

- `results/run-N/` — `raw.jsonl` (every generation), `results.json` (aggregates +
  quality), `summary.md` (human-readable).
- `results/sweep-N/` — the step-sweep data.
- `charts/*.png` — article-ready figures.
- `findings.md` — the writeup of what was actually observed.

## Methodology notes

- Each model loads once; 2 warmup runs (discarded) then ≥5 timed trials; results
  are mean ± std.
- Everything held constant: prompt, `max_tokens` (512), 8-bit quant, machine,
  runner, denoising steps (48, matching Google's model-card eval).
- **Determinism asymmetry (disclosed):** the AR baseline runs at `temperature=0`;
  DiffusionGemma still uses its entropy-bound schedule, so its outputs vary
  slightly run-to-run. We report variance rather than hiding it.

## Layout

```
bench/        config, runner (mlx-vlm wrapper), metrics, prompts, judge, charts
prompts/      prompts.yaml — 30 prompts across 4 categories
run_benchmark.py / run_step_sweep.py / make_charts.py
```
