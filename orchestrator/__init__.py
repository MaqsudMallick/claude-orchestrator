"""Run a DAG of Claude tasks in parallel, without giving up clarifying questions."""

from .models import MODEL_BY_DIFFICULTY, model_for
from .spec import Spec, Task, load

__all__ = ["MODEL_BY_DIFFICULTY", "Spec", "Task", "load", "model_for"]
__version__ = "0.1.0"
