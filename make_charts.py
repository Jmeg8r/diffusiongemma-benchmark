#!/usr/bin/env python
"""Render the article-ready PNG charts from the latest results.

  .venv/bin/python make_charts.py                  # use latest run-* and sweep-*
  .venv/bin/python make_charts.py --run results/run-3 --sweep results/sweep-1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import charts, config


def latest(prefix: str) -> Path | None:
    dirs = [p for p in config.RESULTS_DIR.glob(f"{prefix}-*")
            if p.name.split("-")[1].isdigit()]
    return max(dirs, key=lambda p: int(p.name.split("-")[1])) if dirs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None)
    ap.add_argument("--sweep", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run or latest("run")
    if not run_dir:
        print("No run-* results found. Run run_benchmark.py first.")
        return 1
    results = json.loads((run_dir / "results.json").read_text())
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    agg = results["speed_by_model"]
    diff_label = config.MODELS["diffusion"]["label"]
    measured = agg.get(diff_label, {}).get("throughput_tps", {}).get("mean") or 0

    made = []
    made.append(charts.chart_throughput(
        agg, config.CHARTS_DIR / "01_throughput.png",
        "Throughput on Apple Silicon (8-bit, 512 tok)"))
    made.append(charts.chart_reported_vs_measured(
        measured, results.get("reported_tps", config.REPORTED_TPS),
        config.CHARTS_DIR / "02_reported_vs_measured.png",
        "DiffusionGemma: reported (NVIDIA) vs measured (Mac)"))
    made.append(charts.chart_ttft(
        agg, config.CHARTS_DIR / "03_ttft.png",
        "Time to first token (diffusion's latency cost)"))

    by_cat = results.get("quality", {}).get("by_category") or {}
    if by_cat:
        made.append(charts.chart_quality_by_category(
            by_cat, config.CHARTS_DIR / "05_quality_by_category.png",
            "Quality by task category (8-bit, equal settings)"))

    sweep_dir = args.sweep or latest("sweep")
    if sweep_dir and (sweep_dir / "sweep.json").exists():
        sweep = json.loads((sweep_dir / "sweep.json").read_text())
        by_step = {int(k): v for k, v in sweep["by_step"].items()}
        made.append(charts.chart_step_tradeoff(
            by_step, config.CHARTS_DIR / "04_step_tradeoff.png",
            "DiffusionGemma: denoising steps vs speed & quality"))

    for p in made:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
