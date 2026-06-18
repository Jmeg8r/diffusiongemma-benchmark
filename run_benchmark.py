#!/usr/bin/env python
"""Run the DiffusionGemma vs autoregressive-Gemma benchmark.

Generation is delegated to short-lived subprocess workers (bench/worker.py) —
a few prompts each, then exit. This is REQUIRED, not an optimization:
DiffusionGemma accumulates Metal driver state across generations and TTFT runs
away after ~36 cumulative canvas blocks in a single process; process isolation
bounds that. After all workers finish, this script scores quality (objective
categories deterministically; writing via the blind Claude-CLI judge) and writes
results/<run-N>/{raw.jsonl,results.json,summary.md}.

  .venv/bin/python run_benchmark.py --quick            # fast smoke validation
  .venv/bin/python run_benchmark.py                    # full run (5 trials, all prompts)
  .venv/bin/python run_benchmark.py --no-judge         # skip LLM judging
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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `bench` importable

from bench import config, judge, metrics
from bench.prompts import load_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark.run")

# Keep timed generations per worker well inside the flat zone (knee ~36 blocks).
GEN_BUDGET_PER_WORKER = 10


def next_run_dir(prefix: str = "run") -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("-")[1]) for p in config.RESULTS_DIR.glob(f"{prefix}-*")
                if p.name.split("-")[1].isdigit()]
    n = (max(existing) + 1) if existing else 1
    d = config.RESULTS_DIR / f"{prefix}-{n}"
    d.mkdir()
    return d


def select_prompts(prompts, limit_per_category: int | None):
    if not limit_per_category:
        return prompts
    seen: dict[str, int] = defaultdict(int)
    out = []
    for p in prompts:
        if seen[p.category] < limit_per_category:
            out.append(p)
            seen[p.category] += 1
    return out


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def run_workers(run_dir: Path, raw_path: Path, prompts, trials: int, max_tokens: int) -> list[tuple]:
    ppw = max(1, GEN_BUDGET_PER_WORKER // max(1, trials))
    env = dict(os.environ)
    failures: list[tuple] = []
    for key in config.MODELS:                      # all of one model before the next
        for chunk in chunks(prompts, ppw):
            ids = ",".join(p.id for p in chunk)
            cmd = [sys.executable, "-m", "bench.worker",
                   "--model-key", key, "--prompt-ids", ids,
                   "--trials", str(trials), "--max-tokens", str(max_tokens),
                   "--out", str(raw_path)]
            log.info("worker: %s [%s]", key, ids)
            proc = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT), env=env)
            if proc.returncode != 0:
                failures.append((key, ids, proc.returncode))
                log.error("worker failed (rc=%d) for %s [%s]", proc.returncode, key, ids)
    return failures


def load_rows(raw_path: Path) -> list[dict]:
    return [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]


def build_rep(rows: list[dict]) -> dict:
    """rep[kind][prompt_id] = representative (trial-0 timed) output text."""
    rep: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        if r["phase"] == "timed" and r["trial"] == 0:
            rep[r["kind"]][r["prompt_id"]] = r["output_text"]
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=config.TIMED_TRIALS)
    ap.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS)
    ap.add_argument("--limit-per-category", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="trials=1, 1 prompt/category, max_tokens=128, no judge")
    args = ap.parse_args()
    if args.quick:
        args.trials, args.max_tokens = 1, 128
        args.limit_per_category, args.no_judge = 1, True

    config.ensure_hf_home()
    run_dir = next_run_dir()
    raw_path = run_dir / "raw.jsonl"
    raw_path.touch()
    log.info("run dir: %s", run_dir)

    prompts = select_prompts(load_prompts(), args.limit_per_category)
    log.info("prompts: %d  trials: %d  max_tokens: %d", len(prompts), args.trials, args.max_tokens)

    failures = run_workers(run_dir, raw_path, prompts, args.trials, args.max_tokens)
    if failures:
        # WHY: aggregating a partial raw.jsonl would silently bias every reported
        # metric. Fail loudly instead of publishing skewed numbers.
        raise RuntimeError(
            f"{len(failures)} worker(s) failed; aborting to avoid partial benchmark metrics.")

    rows = load_rows(raw_path)
    rep = build_rep(rows)
    quality = score_quality(prompts, rep, no_judge=args.no_judge)

    agg = metrics.aggregate_by_model(rows)
    by_cat = metrics.aggregate_by_category(rows)
    results = {
        "config": {
            "trials": args.trials, "max_tokens": args.max_tokens, "quant": config.QUANT,
            "denoising_steps": config.DENOISING_STEPS_DEFAULT, "sampler": config.DIFFUSION_SAMPLER,
            "temperature": config.TEMPERATURE,
            "models": {k: v["id"] for k, v in config.MODELS.items()},
            "n_prompts": len(prompts), "judged": not args.no_judge,
            "isolation": f"subprocess worker, {max(1, GEN_BUDGET_PER_WORKER // max(1, args.trials))} prompt(s)/worker",
        },
        "speed_by_model": agg,
        "speed_by_category": by_cat,
        "quality": quality,
        "reported_tps": config.REPORTED_TPS,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "summary.md").write_text(render_summary(results))
    log.info("wrote %s", run_dir / "results.json")
    print("\n" + render_summary(results))
    return 0


def score_quality(prompts, rep, no_judge: bool) -> dict:
    """Per-model accuracy per category. Objective = pass rate; writing = judge 0-1."""
    per = defaultdict(lambda: defaultdict(list))   # kind -> category -> [scores]
    details = []
    diff_key, ar_key = "diffusion", "autoregressive"
    for p in prompts:
        if p.scoring == "judge":
            if no_judge or p.id not in rep.get(diff_key, {}) or p.id not in rep.get(ar_key, {}):
                continue
            res = judge.score_writing(p.text, rep[diff_key][p.id], rep[ar_key][p.id], p.rubric or "")
            if res.get("ok"):
                per[diff_key][p.category].append(res["diffusion"])
                per[ar_key][p.category].append(res["autoregressive"])
                details.append({"prompt": p.id, "type": "judge", **res})
        else:
            for key in (diff_key, ar_key):
                if p.id not in rep.get(key, {}):
                    continue
                s = judge.score_objective(p.category, p.scoring, rep[key][p.id], p.gold, p.checks)
                per[key][p.category].append(s["score"])
                details.append({"prompt": p.id, "model": key, "type": p.scoring, **s})

    label = {k: config.MODELS[k]["label"] for k in config.MODELS}
    by_category = {label[k]: {c: (sum(v) / len(v) if v else None) for c, v in cats.items()}
                   for k, cats in per.items()}
    overall = {}
    for k, cats in per.items():
        flat = [x for v in cats.values() for x in v]
        overall[label[k]] = (sum(flat) / len(flat)) if flat else None
    return {"by_category": by_category, "overall": overall, "details": details}


def _fmt(ms: dict, unit: str = "", fmt: str = "{:.1f}") -> str:
    if not ms or ms.get("mean") is None:
        return "n/a"
    return f"{fmt.format(ms['mean'])}{unit} ± {fmt.format(ms['std'])}"


def render_summary(results: dict) -> str:
    c = results["config"]
    lines = ["# Benchmark Summary", "",
             f"- Models: {', '.join(c['models'].values())}",
             f"- Quant: {c['quant']}  |  trials: {c['trials']}  |  max_tokens: {c['max_tokens']}"
             f"  |  denoising steps: {c['denoising_steps']} ({c['sampler']})",
             f"- Prompts: {c['n_prompts']}  |  judged: {c['judged']}  |  {c.get('isolation','')}", "",
             "## Speed (timed trials, mean ± std)", "",
             "| Model | Throughput tok/s | TTFT (s) | Denoising steps used | Peak mem |",
             "|---|---|---|---|---|"]
    for model, m in results["speed_by_model"].items():
        lines.append(f"| {model} | {_fmt(m['throughput_tps'])} | {_fmt(m['ttft_s'], 's', '{:.2f}')} "
                     f"| {_fmt(m.get('denoising_steps'), '', '{:.0f}')} "
                     f"| {_fmt(m.get('peak_memory'), ' GB', '{:.1f}')} |")
    q = results.get("quality", {})
    if q.get("overall"):
        cats = sorted({c for v in q["by_category"].values() for c in v})
        lines += ["", "## Quality (accuracy / judge score, 0-1)", "",
                  "| Model | Overall | " + " | ".join(cats) + " |",
                  "|---|---|" + "---|" * len(cats)]
        for model, ov in q["overall"].items():
            row = [model, f"{ov:.2f}" if ov is not None else "n/a"]
            for cat in cats:
                val = q["by_category"].get(model, {}).get(cat)
                row.append(f"{val:.2f}" if val is not None else "n/a")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
