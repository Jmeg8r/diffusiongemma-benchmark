"""mlx-vlm wrapper: run one timed generation and capture a CallResult.

WHY a custom wrapper: we need an apples-to-apples timing definition across two
fundamentally different architectures. The single fair number is
    throughput = output tokens / end-to-end wall time
measured the SAME way for both. We also record time-to-first-token (TTFT) — the
metric where diffusion structurally loses, because it must denoise a whole block
before emitting anything — plus the library's own counters for cross-checking.

We use stream_generate so we can stamp TTFT at the first visible token, and read
the final GenerationResult chunk for token counts, peak memory, and (for
diffusion) the actual denoising steps used by the entropy-bound sampler.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import mlx.core as mx
from mlx_vlm import apply_chat_template, load, stream_generate
from mlx_vlm.utils import load_config

from . import config

logger = logging.getLogger("benchmark.runner")


@dataclass
class CallResult:
    """One generation. Serialized verbatim to results/<run>/raw.jsonl."""
    model_id: str
    model_label: str
    kind: str                      # "diffusion" | "autoregressive"
    category: str
    prompt_id: str
    trial: int
    phase: str                     # "warmup" | "timed"
    # request knobs
    requested_max_tokens: int
    requested_denoising_steps: Optional[int]
    temperature: float
    sampler: Optional[str]
    # output + token counts
    output_text: str
    prompt_tokens: int
    generation_tokens: int
    total_tokens: int
    # timing (our measurement)
    wall_time_s: float
    ttft_s: Optional[float]
    throughput_tps: float          # generation_tokens / wall_time_s  <-- headline metric
    # library-reported counters (cross-check)
    prompt_tps: Optional[float] = None
    lib_generation_tps: Optional[float] = None
    diffusion_work_tps: Optional[float] = None
    diffusion_canvas_tps: Optional[float] = None
    denoising_steps: Optional[int] = None   # ACTUAL steps used (diffusion only)
    peak_memory: Optional[float] = None
    finish_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def load_model(model_id: str):
    """Load model + processor + config once; caller reuses across all trials.

    The config is needed to apply the chat template — instruction-tuned ("-it")
    models degenerate badly if fed raw text without their turn markers.
    """
    config.ensure_hf_home()
    logger.info("loading %s", model_id)
    t0 = time.perf_counter()
    model, processor = load(model_id)
    model_config = load_config(model_id)
    logger.info("loaded %s in %.1fs", model_id, time.perf_counter() - t0)
    return model, processor, model_config


def _reset_peak_memory() -> None:
    # WHY: peak_memory is cumulative; reset per call so the number is per-generation.
    for fn in ("reset_peak_memory",):
        if hasattr(mx, fn):
            getattr(mx, fn)()
            return
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "reset_peak_memory"):
        metal.reset_peak_memory()


def _build_kwargs(kind: str, max_tokens: int, denoising_steps: Optional[int]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": config.TEMPERATURE}
    # seed where supported -> reproducible sampling/diffusion
    kwargs["seed"] = config.SEED
    if kind == "diffusion":
        kwargs["diffusion_sampler"] = config.DIFFUSION_SAMPLER
        if denoising_steps is not None:
            kwargs["max_denoising_steps"] = denoising_steps
    return kwargs


def run_once(
    model,
    processor,
    *,
    chat_config,
    model_id: str,
    model_label: str,
    kind: str,
    prompt_text: str,
    prompt_id: str,
    category: str,
    trial: int,
    phase: str,
    max_tokens: int = config.MAX_TOKENS,
    denoising_steps: Optional[int] = config.DENOISING_STEPS_DEFAULT,
) -> CallResult:
    """Generate once, measuring end-to-end wall time and TTFT."""
    requested_steps = denoising_steps if kind == "diffusion" else None
    kwargs = _build_kwargs(kind, max_tokens, requested_steps)

    # Apply the model's chat template; -it models need their turn markers or
    # they echo/degenerate. add_generation_prompt defaults True.
    prompt = apply_chat_template(processor, chat_config, prompt_text, num_images=0)

    _reset_peak_memory()
    parts: list[str] = []
    last: Any = None
    ttft: Optional[float] = None

    t0 = time.perf_counter()
    try:
        for chunk in stream_generate(model, processor, prompt, **kwargs):
            delta = getattr(chunk, "text", chunk)
            if not isinstance(chunk, str):
                last = chunk
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(delta)
    except TypeError:
        # WHY: if seed/sampler kwargs aren't accepted for this build, retry minimal.
        kwargs.pop("seed", None)
        t0 = time.perf_counter()
        parts, last, ttft = [], None, None
        for chunk in stream_generate(model, processor, prompt, **kwargs):
            delta = getattr(chunk, "text", chunk)
            if not isinstance(chunk, str):
                last = chunk
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(delta)
    wall = time.perf_counter() - t0
    # WHY: mx.synchronize() drains the Metal command queue and mx.clear_cache()
    # frees buffers between generations. Without the synchronize, undrained GPU
    # command buffers pile up and DiffusionGemma's TTFT runs away after ~18
    # generations (observed climbing 1s -> 225s); clear_cache alone does NOT
    # prevent it. Both run after timing so neither counts toward wall time.
    mx.synchronize()
    mx.clear_cache()

    text = "".join(parts)
    g = lambda name, default=None: getattr(last, name, default) if last is not None else default

    gen_tokens = int(g("generation_tokens", 0) or 0)
    if gen_tokens == 0 and text:
        # Fallback: count tokens ourselves if the stream didn't surface counts.
        try:
            gen_tokens = len(processor.tokenizer.encode(text))
        except Exception:
            gen_tokens = 0

    throughput = (gen_tokens / wall) if wall > 0 and gen_tokens else 0.0

    return CallResult(
        model_id=model_id,
        model_label=model_label,
        kind=kind,
        category=category,
        prompt_id=prompt_id,
        trial=trial,
        phase=phase,
        requested_max_tokens=max_tokens,
        requested_denoising_steps=requested_steps,
        temperature=config.TEMPERATURE,
        sampler=config.DIFFUSION_SAMPLER if kind == "diffusion" else None,
        output_text=text,
        prompt_tokens=int(g("prompt_tokens", 0) or 0),
        generation_tokens=gen_tokens,
        total_tokens=int(g("total_tokens", 0) or 0),
        wall_time_s=wall,
        ttft_s=ttft,
        throughput_tps=throughput,
        prompt_tps=g("prompt_tps"),
        lib_generation_tps=g("generation_tps"),
        diffusion_work_tps=g("diffusion_work_tps"),
        diffusion_canvas_tps=g("diffusion_canvas_tps"),
        denoising_steps=g("diffusion_denoising_steps") if kind == "diffusion" else None,
        peak_memory=g("peak_memory"),
        finish_reason=g("finish_reason"),
    )


def result_to_dict(r: CallResult) -> dict:
    return asdict(r)
