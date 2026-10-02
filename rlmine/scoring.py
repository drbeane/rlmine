"""Scoring functions.

A score function turns the stats dict returned by ``rltools.utils.evaluate``
into a single number that drives roulette selection and drift comparison.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mean_minus_std",
    "mean_return",
    "success_rate",
    "resolve_score_fn",
    "score_fn_name",
    "score_digits",
    "round_to_digits",
    "with_score_digits",
]


def mean_minus_std(stats):
    """Mean return penalised by one standard deviation.

    The default: it prefers agents that are reliably good over agents that are
    occasionally brilliant.
    """
    return float(stats["mean_return"]) - float(stats["stdev_return"])


def mean_return(stats):
    return float(stats["mean_return"])


def success_rate(stats):
    """Fraction of episodes the environment reported as successful.

    Requires the trial to evaluate with ``check_success=True``.
    """
    if "sr" not in stats:
        raise KeyError(
            "stats has no 'sr' key; evaluate with check_success=True to score "
            "on success rate"
        )
    return float(stats["sr"])


_NAMED = {
    "mean_minus_std": mean_minus_std,
    "mean": mean_return,
    "mean_return": mean_return,
    "success_rate": success_rate,
    "sr": success_rate,
}


def resolve_score_fn(spec):
    if spec is None:
        return mean_minus_std
    if callable(spec):
        return spec
    if isinstance(spec, str) and spec in _NAMED:
        return _NAMED[spec]
    raise ValueError(
        f"Unknown score function {spec!r}. Pass a callable or one of {sorted(_NAMED)}."
    )


def score_fn_name(fn):
    """Stable name stored on the evaluation bucket.

    A different formula is a different config. An unnamed callable is stored
    under its ``__name__``.
    """
    if fn is mean_minus_std:
        return "mean_minus_std"
    if fn is mean_return:
        return "mean_return"
    if fn is success_rate:
        return "success_rate"
    return getattr(fn, "__name__", "custom")


def score_digits(evaluation):
    """Decimal places declared on an evaluation block, or ``None``.

    ``digits`` is part of the config. A missing key leaves values unrounded.
    """
    if not isinstance(evaluation, dict) or "digits" not in evaluation:
        return None
    return coerce_digits(evaluation["digits"])


def coerce_digits(value):
    """A non-negative integer number of decimal places."""
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise ValueError(
            f"evaluation digits must be a non-negative integer, got {value!r}."
        )
    if isinstance(value, (int, np.integer)):
        digits = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        digits = int(value)
    else:
        raise ValueError(
            f"evaluation digits must be a non-negative integer, got {value!r}."
        )
    if digits < 0:
        raise ValueError(
            f"evaluation digits must be a non-negative integer, got {value!r}."
        )
    return digits


def with_score_digits(evaluation, digits):
    """Copy an evaluation block, adding ``digits`` when it is set."""
    evaluation = dict(evaluation)
    if digits is not None:
        evaluation["digits"] = coerce_digits(digits)
    return evaluation


def round_to_digits(value, digits):
    """Round ``value`` to ``digits`` decimal places.

    Dicts, lists, tuples, and arrays are rounded element by element, so every
    piece of a score component is stored at the declared precision. ``None``
    digits leaves the value unchanged.
    """
    if digits is None:
        return value
    return _round_value(value, int(digits))


def _round_value(value, digits):
    if isinstance(value, dict):
        return {key: _round_value(item, digits) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        rounded = [_round_value(item, digits) for item in value.reshape(-1).tolist()]
        return np.array(rounded, dtype=float).reshape(value.shape)
    if isinstance(value, (list, tuple)):
        rounded = [_round_value(item, digits) for item in value]
        return tuple(rounded) if isinstance(value, tuple) else rounded
    if isinstance(value, (bool, np.bool_)) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(number):
        return value
    return float(f"{number:.{digits}f}")
