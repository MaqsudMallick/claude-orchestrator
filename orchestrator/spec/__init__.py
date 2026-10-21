"""The spec: what a run is asked to do, and the difficulty -> model mapping."""

from .models import DIFFICULTIES, MODEL_BY_DIFFICULTY, model_for
from .schema import Agent, Defaults, ReviewConfig, ReviewOverride, Spec, Task, load

__all__ = [
    "Agent",
    "Defaults",
    "ReviewConfig",
    "ReviewOverride",
    "Spec",
    "Task",
    "load",
    "DIFFICULTIES",
    "MODEL_BY_DIFFICULTY",
    "model_for",
]
