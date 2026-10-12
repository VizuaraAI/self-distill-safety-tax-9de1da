"""Config loading. Thin on purpose: a dict with attribute access and a few
invariants checked up front, so a typo fails at second zero rather than after
a forty minute rollout job.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, optionally applying dotted-key overrides.

    Overrides use dotted keys so they can come straight from a CLI:
        {"distill.lora_rank": 64, "seed": 3}
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = copy.deepcopy(raw)
    for dotted, value in (overrides or {}).items():
        _set_dotted(cfg, dotted, value)

    _validate(cfg)
    return Config(cfg)


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override into non-mapping key: {dotted}")
    node[keys[-1]] = value


def _validate(cfg: dict[str, Any]) -> None:
    for section in ("models", "distill", "eval"):
        if section not in cfg:
            raise ValueError(f"config is missing the '{section}' section")

    distill = cfg["distill"]

    fraction = float(distill.get("on_policy_fraction", 1.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"on_policy_fraction must be in [0, 1], got {fraction}")

    direction = distill.get("kl_direction", "forward")
    if direction not in {"forward", "reverse", "jsd"}:
        raise ValueError(f"unknown kl_direction: {direction}")

    if int(distill.get("rollouts_per_prompt", 1)) < 1:
        raise ValueError("rollouts_per_prompt must be at least 1")

    if not cfg["distill"].get("prompt_sources"):
        raise ValueError("distill.prompt_sources is empty; nothing to sample on")


def resolve_teacher_id(cfg: Config) -> str:
    """`teacher_id: base` means true self-distillation (teacher = theta_base).

    Any other value is an explicit checkpoint, which is how the
    teacher-student capacity-gap sweep is run.
    """
    teacher = cfg.models.teacher_id
    return cfg.models.base_id if teacher == "base" else teacher
