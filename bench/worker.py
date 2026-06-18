#!/usr/bin/env python
"""Single-subprocess generation worker: one model, a few prompts, then exit.

WHY this exists: DiffusionGemma accumulates Metal driver-level state across
generations. After ~36 cumulative 256-token canvas blocks in one process, TTFT
runs away (measured 1s -> 124s) and never recovers. No in-process reset
(mx.clear_cache, mx.synchronize) prevents it. Isolating a small handful of
generations per process keeps every measured run inside the flat zone; process
exit fully reclaims the driver state. Model load is ~5s (weights stay in the OS
page cache on a 256 GB machine), so the isolation overhead is small.

Invoked by run_benchmark.py / run_step_sweep.py via `python -m bench.worker`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import config
from bench.prompts import load_prompts
from bench.runner import load_model, result_to_dict, run_once


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(config.MODELS))
    ap.add_argument("--prompt-ids", required=True, help="comma-separated prompt ids")
    ap.add_argument("--trials", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    ap.add_argument("--denoising-steps", type=int, default=config.DENOISING_STEPS_DEFAULT)
    ap.add_argument("--out", required=True, help="jsonl path to append results to")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()

    config.ensure_hf_home()
    spec = config.MODELS[a.model_key]
    pmap = {p.id: p for p in load_prompts()}
    sel = [pmap[i] for i in a.prompt_ids.split(",")]

    model, processor, mcfg = load_model(spec["id"])

    common = dict(chat_config=mcfg, model_id=spec["id"], model_label=spec["label"],
                  kind=spec["kind"], max_tokens=a.max_tokens, denoising_steps=a.denoising_steps)

    # One warmup per process to absorb kernel compilation before timed trials.
    if not a.no_warmup:
        run_once(model, processor, prompt_text=sel[0].text, prompt_id="__warmup__",
                 category="warmup", trial=0, phase="warmup", **common)

    with open(a.out, "a") as f:
        for p in sel:
            for t in range(a.trials):
                r = run_once(model, processor, prompt_text=p.text, prompt_id=p.id,
                             category=p.category, trial=t, phase="timed", **common)
                f.write(json.dumps(result_to_dict(r)) + "\n")
                f.flush()
                ttft = r.ttft_s if r.ttft_s is not None else -1
                print(f"[{a.model_key}] {p.id} t{t}: {r.throughput_tps:6.1f} tok/s  "
                      f"ttft={ttft:.2f}s  steps={r.denoising_steps}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
