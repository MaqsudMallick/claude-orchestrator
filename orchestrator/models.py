"""Difficulty aliases -> concrete model ids.

The spec never names a model id directly: tasks declare how hard they are and
the mapping lives here, so retuning cost/quality is a one-file change.
"""

MODEL_BY_DIFFICULTY: dict[str, str] = {
    "easy": "claude-haiku-5-5",
    "medium": "claude-opus-5-5",
    "hard": "claude-fable-5-1",
}

DIFFICULTIES = tuple(MODEL_BY_DIFFICULTY)


def model_for(difficulty: str) -> str:
    try:
        return MODEL_BY_DIFFICULTY[difficulty]
    except KeyError:
        raise ValueError(
            f"unknown difficulty {difficulty!r}; expected one of {', '.join(DIFFICULTIES)}"
        ) from None
