#!/usr/bin/env python
"""Diffusion-only denoising-step sweep: the speed<->quality knob.

Runs DiffusionGemma at several `max_denoising_steps` values over the objective
prompts (code/math/instruction) and records, per step setting, mean throughput
and objective accuracy. Generation is delegated to subprocess workers (see
bench/worker.py) to avoid the Metal driver-state runaway.

  .venv/bin/python run_step_sweep.py                 # full sweep (3 trials)
  .venv/bin/python run_step_sweep.py --quick         # 1 prompt/category, 1 trial
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import config, judge
from bench.prompts import load_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark.sweep")

OBJECTIVE = {"code", "numeric", "programmatic"}
GEN_BUDGET_PER_WORKER = 10


def next_dir() -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("-")[1]) for p in config.RESULTS_DIR.glob("sweep-*")
                if p.name.split("-")[1].isdigit()]
    n = (max(existing) + 1) if existing else 1
    d = config.RESULTS_DIR / f"sweep-{n}"
    d.mkdir()
    return d


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    ap.add_argument("--steps", type=int, nargs="+", default=config.STEP_SWEEP)
    ap.add_argument("--limit-per-category", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.trials, args.max_tokens, args.limit_per_category = 1, 128, 1

    config.ensure_hf_home()
    prompts = [p for p in load_prompts() if p.scoring in OBJECTIVE]
    if args.limit_per_category:
        seen: dict[str, int] = defaultdict(int)
        sel = []
        for p in prompts:
            if seen[p.category] < args.limit_per_category:
                sel.append(p); seen[p.category] += 1
        prompts = sel
    pmap = {p.id: p for p in prompts}

    out_dir = next_dir()
    raw = out_dir / "raw.jsonl"
    raw.touch()
    ppw = max(1, GEN_BUDGET_PER_WORKER // max(1, args.trials))
    log.info("sweep dir: %s  steps: %s  prompts: %d  prompts/worker: %d",
             out_dir, args.steps, len(prompts), ppw)

    env = dict(os.environ)
    failures: list[tuple] = []
    for steps in args.steps:
        for chunk in chunks(prompts, ppw):
            ids = ",".join(p.id for p in chunk)
            cmd = [sys.executable, "-m", "bench.worker", "--model-key", "diffusion",
                   "--prompt-ids", ids, "--trials", str(args.trials),
                   "--max-tokens", str(args.max_tokens), "--denoising-steps", str(steps),
                   "--out", str(raw)]
            log.info("worker: steps=%d [%s]", steps, ids)
            proc = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT), env=env)
            if proc.returncode != 0:
                failures.append((steps, ids, proc.returncode))
                log.error("sweep worker failed (rc=%d) steps=%d [%s]", proc.returncode, steps, ids)
    if failures:
        raise RuntimeError(f"{len(failures)} worker(s) failed; aborting sweep to avoid partial metrics.")

    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    by_step: dict[int, dict] = {}
    for steps in args.steps:
        srows = [r for r in rows if r["phase"] == "timed" and r["requested_denoising_steps"] == steps]
        tput = [r["throughput_tps"] for r in srows if r["throughput_tps"] is not None]
        ttft = [r["ttft_s"] for r in srows if r["ttft_s"] is not None]
        used = [r["denoising_steps"] for r in srows if r["denoising_steps"] is not None]
        rep = {r["prompt_id"]: r["output_text"] for r in srows if r["trial"] == 0}
        scores = [judge.score_objective(pmap[pid].category, pmap[pid].scoring,
                                        out, pmap[pid].gold, pmap[pid].checks)["score"]
                  for pid, out in rep.items()]
        by_step[steps] = {
            "throughput": sum(tput) / len(tput) if tput else 0,
            "ttft": sum(ttft) / len(ttft) if ttft else None,
            "accuracy": sum(scores) / len(scores) if scores else 0,
            "denoising_steps_used": sum(used) / len(used) if used else None,
            "n_runs": len(tput),
        }
        log.info("steps=%d -> %.1f tok/s, acc=%.2f", steps,
                 by_step[steps]["throughput"], by_step[steps]["accuracy"])

    payload = {"config": {"trials": args.trials, "max_tokens": args.max_tokens,
                          "steps": args.steps, "model": config.MODELS["diffusion"]["id"],
                          "n_prompts": len(prompts)},
               "by_step": by_step}
    (out_dir / "sweep.json").write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", out_dir / "sweep.json")
    for s, d in by_step.items():
        print(f"steps={s:>3}  {d['throughput']:7.1f} tok/s   acc={d['accuracy']*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
