"""Pattern feature utilities and optional ML scoring for FastScalp."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

try:  # Optional dependency
    import joblib  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    joblib = None
    np = None

from . import config


class _PatternModel:
    def __init__(self, model) -> None:  # noqa: ANN001 - sklearn object
        self._model = model

    def score(self, features: Sequence[float], direction: str) -> Optional[float]:
        if np is None:  # numpy not available
            return None
        vec = np.asarray(list(features) + [_direction_sign(direction)], dtype=float)
        try:
            proba = self._model.predict_proba(vec.reshape(1, -1))
        except AttributeError:
            pred = self._model.predict(vec.reshape(1, -1))
            return float(pred[0]) if pred is not None else None
        except Exception as exc:  # pragma: no cover - runtime safety
            logging.debug("[SCALP-PATTERN] model scoring failed: %s", exc)
            return None
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0]) if proba.size else None


def _direction_sign(direction: str) -> int:
    return 1 if direction == "long" else -1


def _load_model(path: str) -> Optional[_PatternModel]:
    if not path or joblib is None or np is None:
        return None
    model_path = Path(path)
    if not model_path.exists():
        logging.warning("[SCALP-PATTERN] model path not found: %s", model_path)
        return None
    try:
        model = joblib.load(model_path)
    except Exception as exc:  # pragma: no cover - runtime I/O
        logging.error("[SCALP-PATTERN] failed to load model: %s", exc)
        return None
    return _PatternModel(model)


PATTERN_MODEL = _load_model(config.PATTERN_MODEL_PATH)


def pattern_score(features: Optional[Sequence[float]], direction: str) -> Optional[float]:
    """Return probability of success for the given feature vector."""
    if PATTERN_MODEL is None or not features:
        return None
    return PATTERN_MODEL.score(features, direction)

