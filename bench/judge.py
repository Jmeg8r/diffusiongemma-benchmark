"""Quality scoring for benchmark outputs.

Four scoring strategies, chosen per prompt so the objective categories never
depend on a subjective judge:
  - code:        run the generated function against asserts in an isolated
                 subprocess (python -I, stripped env; best-effort, NOT a true
                 OS sandbox). Fully objective.
  - numeric:     extract the answer and compare to gold (pass/fail). Objective.
  - programmatic: IFEval-style deterministic checks (length/format/keywords).
  - judge:       blind pairwise LLM-as-judge via the local `claude` CLI, 1-5.

Only the `writing` category uses the LLM judge; everything else is deterministic.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from .config import SEED

logger = logging.getLogger("benchmark.judge")

CODE_TIMEOUT_S = 12
JUDGE_TIMEOUT_S = 120
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_GSM_RE = re.compile(r"####\s*(.+)")
_BOLD_NUM_RE = re.compile(r"\*\*\s*(-?\d[\d,]*\.?\d*)")   # models bold the final answer
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


# --- extraction --------------------------------------------------------------
def extract_code(text: str) -> str:
    """Pull the first fenced code block, else return the raw text."""
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def extract_numeric_answer(text: str) -> Optional[str]:
    """Find a final answer: \\boxed{}, a GSM-style '#### x', else the last number."""
    for rx in (_BOXED_RE, _GSM_RE):
        m = list(rx.finditer(text))
        if m:
            return m[-1].group(1).strip().rstrip(".")
    bold = _BOLD_NUM_RE.findall(text)
    if bold:
        return bold[-1].replace(",", "")
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _num_equal(a: str, b: Any) -> bool:
    try:
        return abs(float(str(a).replace(",", "")) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


# --- objective scorers -------------------------------------------------------
def _sandbox_env() -> dict:
    """Minimal env for running generated code so it can't read inherited
    secrets (API keys, tokens). Best-effort, not a full OS sandbox."""
    return {k: os.environ[k] for k in ("PATH", "TMPDIR") if k in os.environ}


def score_code(output_text: str, gold: dict) -> dict:
    """Execute generated code + asserts in a subprocess; pass iff exit code 0."""
    code = extract_code(output_text)
    tests = gold.get("tests", "")
    program = f"{code}\n\n# --- tests ---\n{tests}\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "candidate.py"
        f.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(f)],   # -I: isolated mode, ignores env/user-site
                capture_output=True, text=True, timeout=CODE_TIMEOUT_S, cwd=d,
                env=_sandbox_env(),
            )
            passed = proc.returncode == 0
            return {"passed": passed, "score": 1.0 if passed else 0.0,
                    "detail": (proc.stderr or "").strip()[-300:]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "score": 0.0, "detail": "timeout"}


def score_numeric(output_text: str, gold: Any) -> dict:
    ans = extract_numeric_answer(output_text)
    passed = ans is not None and _num_equal(ans, gold)
    return {"passed": passed, "score": 1.0 if passed else 0.0,
            "extracted": ans, "gold": gold}


def score_programmatic(output_text: str, checks: dict) -> dict:
    """Run deterministic IFEval-style checks; score = fraction passed."""
    text = output_text.strip()
    words = text.split()
    results: list[tuple[str, bool]] = []

    def add(name, ok):
        results.append((name, bool(ok)))

    if "min_words" in checks:
        add(f"min_words>={checks['min_words']}", len(words) >= checks["min_words"])
    if "max_words" in checks:
        add(f"max_words<={checks['max_words']}", len(words) <= checks["max_words"])
    if "max_lines" in checks:
        add(f"max_lines<={checks['max_lines']}", len(text.splitlines()) <= checks["max_lines"])
    for sub in checks.get("must_include", []):
        add(f"includes:{sub}", sub.lower() in text.lower())
    for sub in checks.get("must_not_include", []):
        add(f"excludes:{sub}", sub.lower() not in text.lower())
    if "starts_with" in checks:
        add("starts_with", text.lower().startswith(checks["starts_with"].lower()))
    if "ends_with" in checks:
        add("ends_with", text.rstrip().lower().endswith(checks["ends_with"].lower()))
    if "regex" in checks:
        add("regex", re.search(checks["regex"], text, re.DOTALL) is not None)
    if checks.get("json_valid"):
        try:
            json.loads(_strip_fence(text))
            add("json_valid", True)
        except Exception:
            add("json_valid", False)

    passed = sum(1 for _, ok in results if ok)
    total = len(results) or 1
    failed = [n for n, ok in results if not ok]
    return {"passed": passed == total, "score": passed / total, "failed_checks": failed}


def _strip_fence(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


# --- blind LLM-as-judge (writing only) ---------------------------------------
def _judge_env() -> dict:
    # WHY: Claude Code injects ANTHROPIC_API_KEY="" into subprocesses; an empty
    # key can break the CLI's session auth. Strip it so the CLI uses the login.
    env = dict(os.environ)
    if env.get("ANTHROPIC_API_KEY", None) == "":
        env.pop("ANTHROPIC_API_KEY", None)
    return env


def judge_pairwise(prompt_text: str, out_a: str, out_b: str, rubric: str = "",
                   judge_cli: str = "claude") -> Optional[dict]:
    """Ask the local Claude CLI to rate two responses 1-5 each. Returns {A,B}."""
    instructions = (
        "You are a strict, fair evaluator. A task was given to two assistants. "
        "Rate EACH response from 1 (poor) to 5 (excellent) on accuracy, coherence, "
        "and completeness combined. " + (rubric + " " if rubric else "") +
        "Respond with ONLY compact JSON, no prose, exactly like: {\"A\": 4, \"B\": 3}\n\n"
        f"=== TASK ===\n{prompt_text}\n\n"
        f"=== RESPONSE A ===\n{out_a}\n\n"
        f"=== RESPONSE B ===\n{out_b}\n"
    )
    try:
        proc = subprocess.run(
            [judge_cli, "-p", instructions],
            capture_output=True, text=True, timeout=JUDGE_TIMEOUT_S, env=_judge_env(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("judge CLI failed: %s", e)
        return None
    m = re.search(r"\{[^{}]*\}", proc.stdout)
    if not m:
        logger.warning("judge returned no JSON: %s", proc.stdout[:200])
        return None
    try:
        data = json.loads(m.group(0))
        return {"A": int(data["A"]), "B": int(data["B"])}
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def score_writing(prompt_text: str, out_diffusion: str, out_ar: str, rubric: str = "",
                  judge_cli: str = "claude") -> dict:
    """Blind, order-randomized pairwise judging. Maps A/B scores back to models."""
    # Stable hash (Python's built-in hash() is salted per process -> not reproducible).
    stable = int(hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(SEED + stable % 10_000)
    diffusion_is_a = rng.random() < 0.5
    out_a, out_b = (out_diffusion, out_ar) if diffusion_is_a else (out_ar, out_diffusion)
    verdict = judge_pairwise(prompt_text, out_a, out_b, rubric, judge_cli)
    if verdict is None:
        return {"diffusion": None, "autoregressive": None, "ok": False}
    if diffusion_is_a:
        diff_score, ar_score = verdict["A"], verdict["B"]
    else:
        diff_score, ar_score = verdict["B"], verdict["A"]
    return {"diffusion": diff_score / 5.0, "autoregressive": ar_score / 5.0,
            "raw": verdict, "diffusion_was_A": diffusion_is_a, "ok": True}


# --- unified entry for objective categories ----------------------------------
def score_objective(category: str, scoring: str, output_text: str,
                    gold: Any, checks: Optional[dict]) -> dict:
    if scoring == "code":
        return score_code(output_text, gold or {})
    if scoring == "numeric":
        return score_numeric(output_text, gold)
    if scoring == "programmatic":
        return score_programmatic(output_text, checks or {})
    raise ValueError(f"score_objective called with non-objective scoring {scoring!r}")
