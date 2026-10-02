"""Declarative parameter definitions.

Each parameter declares a default (used when not sampling or mutating) and
hard bounds (always applied). Sample range and mutation live next to those
so a run can pick a behaviour per variable: a given value, the default,
a draw from the sample range, or a mutation of a parent.

    from rlmine import params as P

    space = dict(
        timesteps     = P.Int(1_000_000, bounds=(1, None)),
        learning_rate = P.Float(7e-4, sig=2, bounds=(1e-6, 1),
                                sample=P.loguniform(1e-5, 1e-2),
                                mutate=P.scale(0.2)),
        gamma         = P.Float(0.99, digits=3, bounds=(0, 1),
                                sample=(0.9, 0.999),
                                mutate=P.scale(0.02)),
        net_arch      = P.Choice([[128, 128], [256, 256]], default=[256, 256]),
        use_sde       = P.Bool(False),
    )
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from .utils import round_sig

__all__ = [
    "Param",
    "Int",
    "Float",
    "Bool",
    "Choice",
    "Action",
    "DEFAULT",
    "SAMPLE",
    "MUTATE",
    "PARENT",
    "scale",
    "times",
    "shift",
    "pick",
    "resample",
    "flip",
    "fixed",
    "uniform",
    "loguniform",
    "choice",
]


# ---------------------------------------------------------------------------
# Per-run actions: how this call should obtain a value for one parameter
# ---------------------------------------------------------------------------


class Action:
    """Instruction used as a parameter value in ``study.run``.

    Bare sentinels use the sample range or mutation declared on the Param::

        study.run(n=10, learning_rate=P.SAMPLE, gamma=P.DEFAULT)

    Calling a sentinel states a range or transformation for this run only::

        study.run(n=10, learning_rate=P.SAMPLE(P.loguniform(1e-5, 1e-2)))
        study.run(n=10, parent=184203, gamma=P.MUTATE(P.scale(0.05)))
    """

    def __init__(self, kind, spec=None):
        self.kind = kind
        self.spec = spec

    def __call__(self, spec):
        return Action(self.kind, spec)

    def __repr__(self):
        label = self.kind.upper()
        return label if self.spec is None else f"{label}({self.spec!r})"

    def __eq__(self, other):
        return (
            isinstance(other, Action)
            and self.kind == other.kind
            and self.spec == other.spec
        )


DEFAULT = Action("default")
SAMPLE = Action("sample")
MUTATE = Action("mutate")
PARENT = Action("parent")


def _is_action(value):
    """True for an Action, including one from a previous import of rlmine."""
    if isinstance(value, Action):
        return True
    return type(value).__name__ == "Action" and getattr(value, "kind", None) in (
        "default",
        "sample",
        "mutate",
        "parent",
    )


# ---------------------------------------------------------------------------
# Mutation strategies: how an inherited value is perturbed
# ---------------------------------------------------------------------------


class Mutation:
    def __call__(self, value, param, rng):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}()"


class Fixed(Mutation):
    """Inherit the parent value untouched, even when selected for mutation."""

    def __call__(self, value, param, rng):
        return value


class Scale(Mutation):
    """Multiply by a uniform factor in [1 - p, 1 + p]."""

    def __init__(self, max_prop=0.2):
        if not 0 < max_prop < 1:
            raise ValueError("max_prop must be between 0 and 1")
        self.max_prop = max_prop

    def __call__(self, value, param, rng):
        return value * rng.uniform(1 - self.max_prop, 1 + self.max_prop)

    def __repr__(self):
        return f"scale({self.max_prop})"


class Times(Mutation):
    """Multiply by a factor drawn from an explicit list, e.g. halve or double."""

    def __init__(self, factors=(0.5, 1, 2)):
        self.factors = list(factors)

    def __call__(self, value, param, rng):
        return value * self.factors[rng.integers(len(self.factors))]

    def __repr__(self):
        return f"times({self.factors})"


class Shift(Mutation):
    """Add a uniform offset in [-amount, +amount]."""

    def __init__(self, amount=1):
        self.amount = amount

    def __call__(self, value, param, rng):
        return value + rng.uniform(-self.amount, self.amount)

    def __repr__(self):
        return f"shift({self.amount})"


class Pick(Mutation):
    """Choose from an explicit list of options, ignoring the parent value."""

    def __init__(self, options=None):
        self.options = list(options) if options is not None else None

    def __call__(self, value, param, rng):
        options = self.options
        if options is None:
            options = getattr(param, "options", None)
        if not options:
            raise ValueError(
                "pick() needs options, either passed directly or from a Choice parameter"
            )
        return options[rng.integers(len(options))]

    def __repr__(self):
        return f"pick({self.options})"


class Resample(Mutation):
    """Ignore the parent and draw a fresh value from the parameter's sampler."""

    def __call__(self, value, param, rng):
        return param.draw_raw(rng)


class Flip(Mutation):
    """Flip a boolean with the given probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, value, param, rng):
        return (not value) if rng.uniform() < self.p else value

    def __repr__(self):
        return f"flip({self.p})"


class _CallableMutation(Mutation):
    """Wraps a user function of (value, rng) or (value) for full control."""

    def __init__(self, fn):
        self.fn = fn
        try:
            self.arity = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            self.arity = 2

    def __call__(self, value, param, rng):
        if self.arity >= 2:
            return self.fn(value, rng)
        return self.fn(value)

    def __repr__(self):
        return f"callable({getattr(self.fn, '__name__', 'fn')})"


def scale(max_prop=0.2):
    return Scale(max_prop)


def times(factors=(0.5, 1, 2)):
    return Times(factors)


def shift(amount=1):
    return Shift(amount)


def pick(options=None):
    return Pick(options)


def resample():
    return Resample()


def flip(p=0.5):
    return Flip(p)


fixed = Fixed()


def _as_mutation(spec):
    if spec is None:
        return None
    if isinstance(spec, Mutation):
        return spec
    if isinstance(spec, str):
        key = spec.lower()
        if key in ("fixed", "repeat", "none"):
            return Fixed()
        if key == "resample":
            return Resample()
        raise ValueError(f"Unknown mutation {spec!r}")
    if callable(spec):
        return _CallableMutation(spec)
    raise TypeError(f"Cannot interpret {spec!r} as a mutation")


# ---------------------------------------------------------------------------
# Samplers: how a fresh value is produced when there is no parent
# ---------------------------------------------------------------------------


class Sampler:
    def __call__(self, param, rng):
        raise NotImplementedError


class Uniform(Sampler):
    def __init__(self, a, b, log=False):
        if log and (a <= 0 or b <= 0):
            raise ValueError("loguniform bounds must be positive")
        self.a, self.b, self.log = a, b, log

    def __call__(self, param, rng):
        if self.log:
            return float(np.exp(rng.uniform(np.log(self.a), np.log(self.b))))
        return float(rng.uniform(self.a, self.b))

    def __repr__(self):
        kind = "loguniform" if self.log else "uniform"
        return f"{kind}({self.a}, {self.b})"


class ChoiceSampler(Sampler):
    def __init__(self, options):
        self.options = list(options)

    def __call__(self, param, rng):
        return self.options[rng.integers(len(self.options))]

    def __repr__(self):
        return f"choice({self.options})"


def uniform(a, b):
    return Uniform(a, b, log=False)


def loguniform(a, b):
    return Uniform(a, b, log=True)


def choice(options):
    return ChoiceSampler(options)


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _as_sampler(spec, numeric=False):
    if spec is None or isinstance(spec, Sampler):
        return spec
    if (
        numeric
        and isinstance(spec, (list, tuple))
        and len(spec) == 2
        and _is_number(spec[0])
        and _is_number(spec[1])
    ):
        return Uniform(spec[0], spec[1], log=False)
    if isinstance(spec, (list, tuple)):
        return ChoiceSampler(spec)
    if callable(spec):
        return lambda param, rng: spec(rng)
    raise TypeError(f"Cannot interpret {spec!r} as a sampler")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class Param:
    """Base class. Subclasses supply coercion and a default mutation."""

    def __init__(self, default, *, bounds=None, mutate=None, sample=None, doc=None):
        self.default = default
        self.bounds = bounds
        self.mutation = _as_mutation(mutate)
        self.sampler = _as_sampler(sample, numeric=isinstance(self, (Int, Float)))
        self.doc = doc

    # -- subclass hooks ----------------------------------------------------

    def coerce(self, value):
        return value

    def round_value(self, value):
        return value

    def default_mutation(self):
        return Fixed()

    def parse(self, raw):
        """Convert a value read back from a spreadsheet or CSV cell."""
        return self.coerce(raw)

    # -- shared behaviour --------------------------------------------------

    def clip(self, value):
        if not self.bounds:
            return value
        low, high = self.bounds
        if low is not None:
            value = max(value, low)
        if high is not None:
            value = min(value, high)
        return value

    def clean(self, value):
        """Clip to bounds, round, then coerce to the declared type."""
        value = self.clip(value)
        value = self.round_value(value)
        return self.coerce(value)

    def draw_raw(self, rng):
        if self.sampler is None:
            return self.default
        return self.sampler(self, rng)

    def draw(self, rng):
        """A fresh value from the sampler, or the default if none is declared."""
        return self.clean(self.draw_raw(rng))

    def sample_value(self, rng, sampler=None, name=None):
        """Draw from a stated range. Errors if this parameter has none."""
        numeric = isinstance(self, (Int, Float))
        use = self.sampler if sampler is None else _as_sampler(sampler, numeric=numeric)
        if use is None:
            label = repr(name) if name is not None else type(self).__name__
            raise ValueError(
                f"Cannot sample {label}: no sample range declared. "
                "Add sample=(low, high) or sample=P.loguniform(...) to the Param."
            )
        return self.clean(use(self, rng))

    def perturb(self, value, rng, mutation=None):
        """A mutated value derived from ``value``."""
        if mutation is None:
            mutation = self.mutation or self.default_mutation()
        else:
            mutation = _as_mutation(mutation)
        return self.clean(mutation(value, self, rng))

    def __repr__(self):
        bits = [repr(self.default)]
        if self.bounds:
            bits.append(f"bounds={self.bounds}")
        if self.mutation:
            bits.append(f"mutate={self.mutation!r}")
        if self.sampler:
            bits.append(f"sample={self.sampler!r}")
        return f"{type(self).__name__}({', '.join(bits)})"


class Int(Param):
    """An integer parameter. Defaults to halving or doubling when mutated."""

    def coerce(self, value):
        if isinstance(value, bool):
            return int(value)
        return int(round(float(value)))

    def default_mutation(self):
        return Times((0.5, 1, 2))

    def parse(self, raw):
        if isinstance(raw, str):
            raw = raw.replace(",", "").replace("_", "").strip()
            if raw == "":
                return self.default
        return self.coerce(float(raw))


class Float(Param):
    """A float parameter.

    ``sig`` rounds to significant digits (right for learning rates), ``digits``
    rounds to decimal places (right for gamma). Defaults to a 10% jitter when
    mutated.
    """

    def __init__(self, default, *, sig=None, digits=None, **kwargs):
        super().__init__(default, **kwargs)
        self.sig = sig
        self.digits = digits

    def coerce(self, value):
        return float(value)

    def round_value(self, value):
        value = float(value)
        if self.sig is not None:
            value = round_sig(value, self.sig)
        if self.digits is not None:
            value = round(value, self.digits)
        return value

    def default_mutation(self):
        return Scale(0.1)

    def parse(self, raw):
        if isinstance(raw, str):
            raw = raw.replace(",", "").replace("_", "").strip()
            if raw == "":
                return self.default
        return float(raw)


class Bool(Param):
    """A boolean parameter. Defaults to a coin flip when mutated."""

    TRUTHY = {"true", "t", "yes", "y", "1"}

    def __init__(self, default, **kwargs):
        kwargs.setdefault("sample", [False, True])
        super().__init__(default, **kwargs)

    def coerce(self, value):
        return bool(value)

    def default_mutation(self):
        return Flip(0.5)

    def parse(self, raw):
        if isinstance(raw, str):
            return raw.strip().lower() in self.TRUTHY
        return bool(raw)


class Choice(Param):
    """A categorical parameter drawn from an explicit list of options.

    Options may be arbitrary objects, including lists such as network
    architectures. Defaults to picking a different option when mutated.
    """

    def __init__(self, options, default=None, **kwargs):
        options = list(options)
        if not options:
            raise ValueError("Choice needs at least one option")
        if default is None:
            default = options[0]
        kwargs.setdefault("sample", ChoiceSampler(options))
        super().__init__(default, **kwargs)
        self.options = options

    def clean(self, value):
        return value

    def default_mutation(self):
        return Pick()

    def parse(self, raw):
        if isinstance(raw, str):
            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return raw
        return raw
