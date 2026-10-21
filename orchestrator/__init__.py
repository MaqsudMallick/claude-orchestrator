"""Run a DAG of Claude tasks in parallel, without giving up clarifying questions."""

from .spec import MODEL_BY_DIFFICULTY, Spec, Task, load, model_for

__all__ = ["MODEL_BY_DIFFICULTY", "Spec", "Task", "load", "model_for"]
__version__ = "0.1.0"
