"""The gate on what an agent may do: the deny policy, the permission callback, questions."""

from .callback import PermissionResult, make_callback
from .policy import DANGEROUS, PATH_TOOLS, PUSH, RM_CALL, SAFE_REDIRECT_TARGETS, check_denied
from .questions import SEP, ask_human

__all__ = [
    "DANGEROUS",
    "PATH_TOOLS",
    "PUSH",
    "RM_CALL",
    "SAFE_REDIRECT_TARGETS",
    "SEP",
    "PermissionResult",
    "ask_human",
    "check_denied",
    "make_callback",
]
