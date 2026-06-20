# CLAUDE.md — diffusiongemma-benchmark

Repo-specific notes for Claude Code. General workflow/conventions live in the global
`~/.claude/CLAUDE.md`; this file only captures what's specific to this project.

## What this is
A hands-on benchmark comparing **DiffusionGemma vs autoregressive Gemma 4 on Apple Silicon**
(via `mlx-vlm`): speed (throughput/TTFT), a denoising-step sweep, and blind LLM-judged writing
quality. Public, MIT. Findings in `findings.md`; the article draft in `article/` is gitignored.

## Where things are
- `bench/` — `runner.py` (mlx-vlm wrapper), `worker.py` (per-model subprocess), `metrics.py`,
  `judge.py`, `prompts.py`, `charts.py`, `config.py`.
- Top-level: `run_benchmark.py` (`--quick` for smoke), `run_step_sweep.py`, `make_charts.py`.
- `prompts/prompts.yaml` (30 prompts × 4 categories); `results/run-N/` + `results/sweep-N/`.

## Run / verify
```bash
.venv/bin/python run_benchmark.py --quick    # fast smoke (1 prompt/cat, no judge)
.venv/bin/python run_benchmark.py            # full run (5 trials, blind judge)
.venv/bin/python make_charts.py              # render charts/*.png
```
Apple-Silicon only (MLX). The writing judge needs `ANTHROPIC_API_KEY` (see `.env.example`).

## Gotchas (load-bearing — don't regress)
- **Subprocess isolation is required.** MLX diffusion throughput collapses (TTFT degrades
  severely) when many generations run in one process; `mx.clear_cache`/`mx.synchronize` do **not**
  fix it. `worker.py` runs one model per subprocess on purpose — keep generations isolated.
- **Empty `ANTHROPIC_API_KEY`:** Claude Code injects `ANTHROPIC_API_KEY=""` into subprocesses;
  `judge.py` strips an empty value before invoking the judge. Don't remove that guard.
- **Hold methodology constant** when changing the harness: `max_tokens=512`, 8-bit quant,
  48 denoising steps (matches Google's model-card eval), 1 discarded warmup per worker, mean ± std.
- **Determinism asymmetry is disclosed, not a bug:** AR baseline runs at `temperature=0`;
  DiffusionGemma varies run-to-run (entropy-bound schedule). Report variance, don't hide it.
- Public repo — keep results reproducible and claims grounded in `results/`.
