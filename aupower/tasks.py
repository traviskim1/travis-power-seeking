"""Odysseys tasks, rubrics, difficulty pools and seeded auxiliary-goal sampling (SPEC.md sections 2 and 3)."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

LEVELS = ("easy", "medium", "hard")


@dataclass(frozen=True)
class RubricItem:
    id: str
    requirement: str
    verification: str
    weight: float | None = None   # recorded when the data carries one; the released data has none

    def to_dict(self) -> dict:
        return {"id": self.id, "requirement": self.requirement, "verification": self.verification,
                "weight": self.weight}


@dataclass(frozen=True)
class Task:
    task_id: str
    text: str
    website: str
    level: str
    reference_length: int
    rubrics: tuple[RubricItem, ...]
    categories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def m(self) -> int:
        """Number of rubric items (m in equation 7)."""
        return len(self.rubrics)

    @property
    def weights(self) -> list[float | None]:
        return [r.weight for r in self.rubrics]

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "text": self.text, "website": self.website, "level": self.level,
                "reference_length": self.reference_length, "rubrics": [r.to_dict() for r in self.rubrics],
                "categories": list(self.categories)}


def _rubric_items(raw: dict) -> tuple[RubricItem, ...]:
    items = []
    for rid, value in raw.items():
        weight = value.get("weight")
        items.append(RubricItem(
            id=str(rid),
            requirement=str(value.get("requirement", "")),
            verification=str(value.get("verification", "")),
            weight=float(weight) if weight is not None else None,
        ))
    return tuple(items)


def load_tasks(path: Path | str) -> list[Task]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    tasks = []
    for item in raw:
        tasks.append(Task(
            task_id=item["task_id"],
            text=item["confirmed_task"],
            website=item["website"],
            level=item["level"],
            reference_length=int(item.get("reference_length", 0)),
            rubrics=_rubric_items(item["rubrics"]),
            categories=tuple(item.get("categories", [])),
        ))
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task_id in task file")
    return tasks


class TaskIndex:
    def __init__(self, tasks: Iterable[Task]):
        self.tasks = list(tasks)
        self.by_id = {t.task_id: t for t in self.tasks}

    def __getitem__(self, task_id: str) -> Task:
        return self.by_id[task_id]

    def __len__(self) -> int:
        return len(self.tasks)

    def pool(self, level: str, exclude: Iterable[str] = ()) -> list[Task]:
        """All tasks at ``level`` (the goal distribution D for that level), minus ``exclude``."""
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        ex = set(exclude)
        return [t for t in self.tasks if t.level == level and t.task_id not in ex]


def derive_seed(base: int, *parts: object) -> int:
    """A reproducible 62-bit seed from the base seed and a sequence of labels."""
    h = hashlib.sha256(("|".join([str(base)] + [str(p) for p in parts])).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") >> 2


def sample_trajectory_tasks(pool: list[Task], n: int, seed: int) -> list[str]:
    """Pick ``n`` trajectory tasks g from the level's pool with a fixed seed."""
    ids = sorted(t.task_id for t in pool)
    if n > len(ids):
        raise ValueError(f"asked for {n} trajectory tasks but the pool has {len(ids)}")
    return random.Random(seed).sample(ids, n)


def sample_aux_goals(pool: list[Task], trajectory_task_id: str, n: int, seed: int) -> list[str]:
    """Draw the auxiliary goals for a trajectory (section 3).

    Uniformly at random, without replacement, from the other tasks at g's difficulty; the trajectory
    task itself is excluded. The same draw is reused at every sampled state on that trajectory, at
    the baseline, and for every model, so the seed must be fixed per trajectory.
    """
    candidates = sorted(t.task_id for t in pool if t.task_id != trajectory_task_id)
    if n > len(candidates):
        raise ValueError(f"asked for {n} auxiliary goals but only {len(candidates)} candidates at this level")
    return random.Random(seed).sample(candidates, n)
