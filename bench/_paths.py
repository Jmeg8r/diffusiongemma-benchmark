"""Shared filesystem + prompt-selection helpers for the top-level scripts.

WHY a module: run_benchmark.py, run_step_sweep.py, and make_charts.py each carried
their own copy of the run-dir numbering, prompt chunking, and per-category prompt
limiting. One copy keeps the directory naming (run-N, sweep-N), ordering, and
selection semantics identical across all three.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import config


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def next_run_dir(prefix: str = "run") -> Path:
    """Create and return results/<prefix>-N, with N one past the current max."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name.split("-")[1])
        for p in config.RESULTS_DIR.glob(f"{prefix}-*")
        if p.name.split("-")[1].isdigit()
    ]
    n = (max(existing) + 1) if existing else 1
    d = config.RESULTS_DIR / f"{prefix}-{n}"
    d.mkdir()
    return d


def latest_run_dir(prefix: str) -> Path | None:
    """Return the highest-numbered results/<prefix>-N, or None if none exist."""
    dirs = [
        p
        for p in config.RESULTS_DIR.glob(f"{prefix}-*")
        if p.name.split("-")[1].isdigit()
    ]
    return max(dirs, key=lambda p: int(p.name.split("-")[1])) if dirs else None


def select_prompts(prompts, limit_per_category: int | None):
    """First `limit_per_category` prompts of each category (all if falsy)."""
    if not limit_per_category:
        return prompts
    seen: dict[str, int] = defaultdict(int)
    out = []
    for p in prompts:
        if seen[p.category] < limit_per_category:
            out.append(p)
            seen[p.category] += 1
    return out
