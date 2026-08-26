"""Scoring functions.

A score function turns the stats dict returned by ``rltools.utils.evaluate``
into a single number that drives roulette selection and drift comparison.
"""

from __future__ import annotations

__all__ = ["mean_minus_std", "mean_return", "success_rate", "resolve_score_fn"]


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
