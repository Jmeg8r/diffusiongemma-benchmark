"""Load and represent the curated benchmark prompt set.

Each prompt declares how it is scored:
  - code:        execution-based pass/fail (sandboxed subprocess)
  - numeric:     extract a number/string and compare to gold
  - programmatic: deterministic IFEval-style checks (length/format/keywords)
  - judge:       blind LLM-as-judge (local claude CLI), 1-5 rubric
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import PROMPTS_FILE

VALID_CATEGORIES = {"code", "math", "instruction", "writing"}
VALID_SCORING = {"code", "numeric", "programmatic", "judge"}


@dataclass
class Prompt:
    id: str
    category: str
    text: str
    scoring: str
    gold: Optional[Any] = None        # numeric answer, or {"entrypoint","tests"} for code
    checks: Optional[dict] = None     # programmatic checks for instruction-following
    rubric: Optional[str] = None      # extra judge guidance for writing

    def __post_init__(self):
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"{self.id}: bad category {self.category!r}")
        if self.scoring not in VALID_SCORING:
            raise ValueError(f"{self.id}: bad scoring {self.scoring!r}")


def load_prompts(path: Path = PROMPTS_FILE) -> list[Prompt]:
    data = yaml.safe_load(Path(path).read_text())
    prompts = [Prompt(**p) for p in data["prompts"]]
    ids = [p.id for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt ids in prompts.yaml")
    return prompts
