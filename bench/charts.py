"""Render article-ready PNG charts from aggregated benchmark data.

PNG specifically (not markdown tables) so they drop straight into an ASTGL
article. Pure functions that take plain dicts so they work off results.json.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

DIFFUSION_COLOR = "#e8743b"   # warm orange
AR_COLOR = "#1f6f8b"          # cool teal
REPORTED_COLOR = "#9aa0a6"    # grey for "cited, not measured"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def _model_color(label: str) -> str:
    return DIFFUSION_COLOR if "diffusion" in label.lower() else AR_COLOR


def _annotate_bars(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 2),
                    textcoords="offset points")


def chart_throughput(agg: dict, out: Path, title: str) -> Path:
    labels = list(agg.keys())
    means = [agg[l]["throughput_tps"]["mean"] or 0 for l in labels]
    errs = [agg[l]["throughput_tps"]["std"] or 0 for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=errs, capsize=5,
                  color=[_model_color(l) for l in labels])
    _annotate_bars(ax, bars)
    ax.set_ylabel("Throughput (tokens / sec, end-to-end)")
    ax.set_title(title)
    plt.xticks(rotation=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_reported_vs_measured(measured_diffusion_tps: float, reported: dict,
                               out: Path, title: str) -> Path:
    labels = list(reported.keys()) + ["Your Mac Studio (measured)"]
    values = list(reported.values()) + [measured_diffusion_tps]
    colors = [REPORTED_COLOR] * len(reported) + [DIFFUSION_COLOR]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, values, color=colors)
    _annotate_bars(ax, bars)
    ax.set_ylabel("DiffusionGemma throughput (tokens / sec)")
    ax.set_title(title)
    plt.xticks(rotation=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_ttft(agg: dict, out: Path, title: str) -> Path:
    labels = list(agg.keys())
    means = [(agg[l]["ttft_s"]["mean"] or 0) for l in labels]
    errs = [(agg[l]["ttft_s"]["std"] or 0) for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=errs, capsize=5,
                  color=[_model_color(l) for l in labels])
    _annotate_bars(ax, bars, fmt="{:.2f}s")
    ax.set_ylabel("Time to first token (seconds)")
    ax.set_title(title)
    plt.xticks(rotation=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_step_tradeoff(sweep: dict, out: Path, title: str) -> Path:
    """sweep: {steps:int -> {"throughput": float, "accuracy": float}}."""
    steps = sorted(sweep.keys())
    tput = [sweep[s]["throughput"] for s in steps]
    acc = [sweep[s]["accuracy"] * 100 for s in steps]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    l1, = ax1.plot(steps, tput, "o-", color=DIFFUSION_COLOR, label="Throughput (tok/s)")
    ax1.set_xlabel("Max denoising steps")
    ax1.set_ylabel("Throughput (tokens / sec)", color=DIFFUSION_COLOR)
    ax1.tick_params(axis="y", labelcolor=DIFFUSION_COLOR)
    ax2 = ax1.twinx()
    ax2.grid(False)
    l2, = ax2.plot(steps, acc, "s--", color=AR_COLOR, label="Objective accuracy (%)")
    ax2.set_ylabel("Objective accuracy (%)", color=AR_COLOR)
    ax2.tick_params(axis="y", labelcolor=AR_COLOR)
    ax2.set_ylim(0, 100)
    ax1.set_title(title)
    ax1.legend(handles=[l1, l2], loc="center right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_quality_by_category(by_category: dict, out: Path, title: str) -> Path:
    """by_category: {model_label: {category: accuracy_0_1}}."""
    models = list(by_category.keys())
    cats = sorted({c for m in by_category.values() for c in m})
    x = range(len(cats))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(models):
        vals = [by_category[m].get(c, 0) * 100 for c in cats]
        offs = [xi + i * width for xi in x]
        bars = ax.bar(offs, vals, width, label=m, color=_model_color(m))
        _annotate_bars(ax, bars, fmt="{:.0f}")
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(cats)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
