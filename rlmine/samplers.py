"""Distributions that draw a value when a trial runs.

Constructing one does not sample. ``Study.run`` draws a fresh value for each
trial, including each of ``n`` repeats, and stores that number.

    from rlmine import samplers as s

    study.run(
        n=10,
        agent={"initial_lr": s.loguniform(1e-2, 1e-4, sig=2)},
    )

``sig`` rounds to significant digits (learning rates). ``digits`` rounds to
decimal places (gamma). Either bound order is accepted.

Available samplers:

    uniform(low, high)           flat on an interval (gamma)
    loguniform(low, high)        flat in log space (learning rates)
    normal(mean, std)            bell curve, optionally clipped with low/high
    randint(low, high)           whole number, both ends included (n_steps)
    choice([a, b, c])            one option; any value, including lists
                                 (net_arch), optionally weighted

To change a value that is already in the config instead of drawing a new one,
see :mod:`rlmine.mutations`.
"""

from __future__ import annotations

import copy

import numpy as np

from .mutations import Mutation
from .utils import round_sig

__all__ = [
    "Sampler",
    "loguniform",
    "uniform",
    "normal",
    "randint",
    "choice",
    "resolve_samplers",
]


class Sampler:
    """A distribution. Call ``draw`` to sample; constructing it does not."""

    def __init__(self, *, sig=None, digits=None):
        self.sig = _count(sig, "sig", minimum=1)
        self.digits = _count(digits, "digits", minimum=0)

    def draw(self, rng=None):
        """One value. ``rng`` is a ``numpy`` Generator; a fresh one is used if omitted."""
        if rng is None:
            rng = np.random.default_rng()
        value = self._round(self._sample(rng))
        return self._clip(value)

    def _sample(self, rng):
        raise NotImplementedError

    def _clip(self, value):
        return value

    def _round(self, value):
        value = float(value)
        if self.sig is not None:
            value = float(round_sig(value, self.sig))
        if self.digits is not None:
            value = round(value, self.digits)
        return value


class Uniform(Sampler):
    """Uniform on an interval. NumPy's half-open ``[low, high)``."""

    def __init__(self, low, high, *, sig=None, digits=None):
        super().__init__(sig=sig, digits=digits)
        self.low, self.high = _interval(low, high)

    def _sample(self, rng):
        if self.low == self.high:
            return self.low
        return float(rng.uniform(self.low, self.high))

    def _clip(self, value):
        return _clip(value, self.low, self.high)

    def __repr__(self):
        return _repr("uniform", self.low, self.high, sig=self.sig, digits=self.digits)


class LogUniform(Sampler):
    """Log-uniform on a positive interval. Either bound order is accepted."""

    def __init__(self, low, high, *, sig=None, digits=None):
        super().__init__(sig=sig, digits=digits)
        self.low, self.high = _interval(low, high)
        if self.low <= 0 or self.high <= 0:
            raise ValueError("loguniform bounds must be positive")

    def _sample(self, rng):
        if self.low == self.high:
            return self.low
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))

    def _clip(self, value):
        return _clip(value, self.low, self.high)

    def __repr__(self):
        return _repr("loguniform", self.low, self.high, sig=self.sig, digits=self.digits)


class Normal(Sampler):
    """Normal with mean and standard deviation.

    ``low`` and ``high``, when given, clip the rounded value. Mass that falls
    outside the interval piles up on the boundary.
    """

    def __init__(self, mean, std, *, low=None, high=None, sig=None, digits=None):
        super().__init__(sig=sig, digits=digits)
        self.mean = float(mean)
        self.std = float(std)
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std < 0:
            raise ValueError("normal needs a finite mean and a non-negative std")
        self.low = None if low is None else float(low)
        self.high = None if high is None else float(high)
        if (
            self.low is not None
            and self.high is not None
            and self.low > self.high
        ):
            raise ValueError("low cannot exceed high")

    def _sample(self, rng):
        if self.std == 0:
            return self.mean
        return float(rng.normal(self.mean, self.std))

    def _clip(self, value):
        return _clip(value, self.low, self.high)

    def __repr__(self):
        return _repr(
            "normal",
            self.mean,
            self.std,
            low=self.low,
            high=self.high,
            sig=self.sig,
            digits=self.digits,
        )


class RandInt(Sampler):
    """A whole number from ``low`` to ``high``, both included."""

    def __init__(self, low, high):
        super().__init__()
        for name, bound in (("low", low), ("high", high)):
            if isinstance(bound, bool) or not isinstance(bound, (int, np.integer)):
                raise ValueError(f"randint {name} must be an integer")
        low, high = int(low), int(high)
        self.low, self.high = (low, high) if low <= high else (high, low)

    def draw(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.integers(self.low, self.high + 1))

    def __repr__(self):
        return _repr("randint", self.low, self.high)


class Choice(Sampler):
    """One option from a list. Options may be any value, such as a ``net_arch`` list.

    ``weights`` makes some options more likely. They need not sum to one.
    """

    def __init__(self, options, *, weights=None):
        super().__init__()
        if isinstance(options, (str, bytes)) or not isinstance(options, (list, tuple)):
            raise TypeError(
                "choice takes a list of options, for example choice([5, 8, 16])"
            )
        if len(options) == 0:
            raise ValueError("choice needs at least one option")
        self.options = list(options)
        self.weights = None
        if weights is not None:
            weights = [float(w) for w in weights]
            if len(weights) != len(self.options):
                raise ValueError("choice needs one weight per option")
            if any(not np.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError("choice weights must be non-negative and not all zero")
            self.weights = weights

    def draw(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.weights is None:
            index = int(rng.integers(len(self.options)))
        else:
            probabilities = np.asarray(self.weights) / sum(self.weights)
            index = int(rng.choice(len(self.options), p=probabilities))
        return copy.deepcopy(self.options[index])

    def __repr__(self):
        return _repr("choice", self.options, weights=self.weights)


def uniform(low, high, *, sig=None, digits=None):
    """Uniform sampler. Does not draw a value."""
    return Uniform(low, high, sig=sig, digits=digits)


def loguniform(low, high, *, sig=None, digits=None):
    """Log-uniform sampler. Does not draw a value."""
    return LogUniform(low, high, sig=sig, digits=digits)


def normal(mean, std, *, low=None, high=None, sig=None, digits=None):
    """Normal sampler. Does not draw a value."""
    return Normal(mean, std, low=low, high=high, sig=sig, digits=digits)


def randint(low, high):
    """Whole-number sampler, both ends included. Does not draw a value."""
    return RandInt(low, high)


def choice(options, *, weights=None):
    """Sampler that picks one of ``options``. Does not draw a value."""
    return Choice(options, weights=weights)


_MISSING = object()


def resolve_samplers(value, rng, *, drawn=None, base=_MISSING, _path=()):
    """Replace every sampler and mutation with one result. Other values pass through.

    A sampler draws a new value. A mutation (:mod:`rlmine.mutations`) changes
    the value found at the same path in ``base``, the config the trial is
    built from before any overrides. A mutation with no number or option to
    start from there is an error.

    Dicts, lists, and tuples are walked in their existing order, so a seeded
    generator is reproducible for a given config. When ``drawn`` is a list,
    each result is appended as ``(path, value)``. ``path`` is the key tuple
    from the root of ``value``.
    """
    if isinstance(value, Sampler):
        sample = value.draw(rng)
        if drawn is not None:
            drawn.append((_path, sample))
        return sample
    if isinstance(value, Mutation):
        changed = _apply_mutation(value, base, rng, _path)
        if drawn is not None:
            drawn.append((_path, changed))
        return changed
    if _is_param_object(value):
        raise TypeError(
            f"{value!r} belongs to a parameter space. "
            "Inside a config, use rlmine.samplers to draw a value, for example "
            "s.loguniform(1e-4, 1e-2, sig=2), or rlmine.mutations to change "
            "one, for example m.proportional(0.2)."
        )
    if isinstance(value, dict):
        return {
            key: resolve_samplers(
                item, rng, drawn=drawn, base=_child(base, key), _path=_path + (key,)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_samplers(
                item, rng, drawn=drawn, base=_child(base, i), _path=_path + (i,)
            )
            for i, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            resolve_samplers(
                item, rng, drawn=drawn, base=_child(base, i), _path=_path + (i,)
            )
            for i, item in enumerate(value)
        )
    return value


def _child(base, key):
    """The value at ``key`` inside ``base``, or ``_MISSING``."""
    if isinstance(base, dict):
        return base.get(key, _MISSING)
    if isinstance(base, (list, tuple)) and isinstance(key, int) and 0 <= key < len(base):
        return base[key]
    return _MISSING


def _apply_mutation(mutation, base, rng, path):
    name = ".".join(str(part) for part in path)
    if base is _MISSING:
        raise ValueError(
            f"{name}: {mutation!r} changes an existing value, but the config "
            "being run has no value there."
        )
    if isinstance(base, (Sampler, Mutation)):
        raise ValueError(
            f"{name}: {mutation!r} needs a fixed value to start from, "
            f"but the config holds {base!r} there."
        )
    try:
        return mutation.apply(base, rng)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: {mutation!r} could not change {base!r}: {exc}") from exc


_PARAM_MUTATIONS = {"Fixed", "Scale", "Times", "Shift", "Pick", "Resample", "Flip"}


def _is_param_object(value):
    """True for ``P.uniform`` / ``P.loguniform`` / ``P.choice`` / ``P.scale`` and friends.

    Those belong to a :class:`Space` and ``run`` cannot use them in a config.
    """
    if isinstance(value, (Sampler, Mutation)):
        return False
    cls = type(value)
    if cls.__name__ not in {"Uniform", "ChoiceSampler"} | _PARAM_MUTATIONS:
        return False
    module = cls.__module__ or ""
    # The Colab bundle defines the parameter-space classes in the ``rlmine``
    # module itself. Config samplers live in ``rlmine.samplers``.
    return module.endswith(".params") or module == "rlmine"


def _interval(low, high):
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("bounds must be finite")
    if low == high:
        return low, high
    return (low, high) if low < high else (high, low)


def _clip(value, low, high):
    if low is not None and value < low:
        value = low
    if high is not None and value > high:
        value = high
    return value


def _count(value, name, minimum):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _repr(kind, *positional, **keywords):
    bits = [repr(part) for part in positional]
    for key, value in keywords.items():
        if value is not None:
            bits.append(f"{key}={value!r}")
    return f"{kind}({', '.join(bits)})"
