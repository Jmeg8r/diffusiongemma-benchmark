"""Aggregate per-trial CallResults into mean/std summaries.

WHY: keep statistics out of the runner and orchestrator so each stays simple and
testable. Everything here is pure-stdlib and operates on dicts so it works on
results loaded from raw.jsonl just as well as on live CallResult objects.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Iterable

# Numeric fields we summarize across timed trials.
NUMERIC_FIELDS = (
    "throughput_tps",
    "ttft_s",
    "wall_time_s",
    "generation_tokens",
    "denoising_steps",
    "peak_memory",
    "prompt_tps",
    "lib_generation_tps",
)


def _as_dict(r) -> dict:
    if is_dataclass(r):
        return asdict(r)
    return dict(r)


def _mean_std(values: Iterable) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def aggregate_by_model(results) -> dict:
    """{model_label: {field: {mean,std,min,max,n}}} over timed trials only."""
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for raw in results:
        r = _as_dict(raw)
        if r.get("phase") != "timed":
            continue
        for f in NUMERIC_FIELDS:
            buckets[r["model_label"]][f].append(r.get(f))
    return {m: {f: _mean_std(v) for f, v in fields.items()} for m, fields in buckets.items()}


def aggregate_by_category(results) -> dict:
    """{model_label: {category: {throughput_mean, ttft_mean, n}}} over timed trials."""
    buckets: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for raw in results:
        r = _as_dict(raw)
        if r.get("phase") != "timed":
            continue
        key = (r["model_label"], r["category"])
        buckets[key]["throughput_tps"].append(r.get("throughput_tps"))
        buckets[key]["ttft_s"].append(r.get("ttft_s"))
    out: dict[str, dict] = defaultdict(dict)
    for (model, cat), fields in buckets.items():
        out[model][cat] = {
            "throughput_tps": _mean_std(fields["throughput_tps"]),
            "ttft_s": _mean_std(fields["ttft_s"]),
        }
    return out
