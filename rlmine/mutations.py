"""Changes to a value that is already in the config.

Constructing one does not change anything. ``Study.run`` applies it when the
trial is built, once per trial, and stores the result. The value it changes
is the one in the config being run: the study default, or the stored config
named by ``config_id``.

    from rlmine import mutations as m

    study.run(
        config_id=927170,
        agent={"initial_lr": m.proportional(0.2, sig=2)},
    )

``sig`` rounds to significant digits (learning rates). ``digits`` rounds to
decimal places (gamma). ``low`` and ``high`` clip the result. An integer
value stays an integer unless ``sig`` or ``digits`` is set.
"""

from __future__ import annotations

import numpy as np

from .utils import round_sig

__all__ = [
    "Mutation",
    "proportional",
    "log_proportional",
    "shift",
    "normal",
    "factor",
    "choice",
    "flip",
]


class Mutation:
    """A change to an existing value. Call ``apply`` to use it."""

    def apply(self, value, rng):
        """The new value. ``rng`` is a ``numpy`` Generator."""
        raise NotImplementedError


class _NumericMutation(Mutation):
    def __init__(self, *, sig=None, digits=None, low=None, high=None):
        self.sig = _count(sig, "sig", minimum=1)
        self.digits = _count(digits, "digits", minimum=0)
        self.low, self.high = _bounds(low, high)

    def apply(self, value, rng):
        number = _require_number(value)
        updated = float(self._mutate(number, rng))
        updated = self._round(updated)
        window = self._window(number)
        if window is not None:
            updated = _clip(updated, window[0], window[1])
        updated = _clip(updated, self.low, self.high)
        if _keeps_int(value, self):
            return _as_int(updated, self.low, self.high)
        return updated

    def _mutate(self, number, rng):
        raise NotImplementedError

    def _window(self, number):
        """Inclusive range the rounded result is pulled back into."""
        return None

    def _round(self, value):
        value = float(value)
        if self.sig is not None:
            value = float(round_sig(value, self.sig))
        if self.digits is not None:
            value = round(value, self.digits)
        return value


class Proportional(_NumericMutation):
    """Multiply by a factor drawn uniformly from ``[1 - spread, 1 + spread]``.

    ``spread`` is between 0 and 1, so the factor stays positive. ``0.2`` moves
    a learning rate by up to 20 percent. For a change of "half to double",
    use :func:`log_proportional`.
    """

    def __init__(self, spread, *, sig=None, digits=None, low=None, high=None):
        super().__init__(sig=sig, digits=digits, low=low, high=high)
        spread = float(spread)
        if not np.isfinite(spread) or not 0 < spread < 1:
            raise ValueError("proportional spread must be between 0 and 1")
        self.spread = spread

    def _mutate(self, number, rng):
        low = 1 - self.spread
        high = 1 + self.spread
        return number * float(rng.uniform(low, high))

    def _window(self, number):
        return _span(number, 1 - self.spread, 1 + self.spread)

    def __repr__(self):
        return _repr("proportional", self.spread, sig=self.sig, digits=self.digits, low=self.low, high=self.high)


class LogProportional(_NumericMutation):
    """Multiply by a log-uniform factor in ``[1 / factor, factor]``.

    ``factor`` is greater than 1. ``2`` keeps the new value between half and
    double the current one, with every scale equally likely.
    """

    def __init__(self, factor, *, sig=None, digits=None, low=None, high=None):
        super().__init__(sig=sig, digits=digits, low=low, high=high)
        factor = float(factor)
        if not np.isfinite(factor) or factor <= 1:
            raise ValueError("log_proportional factor must be greater than 1")
        self.factor = factor

    def _mutate(self, number, rng):
        if number <= 0:
            raise ValueError("log_proportional needs a positive value")
        low = 1.0 / self.factor
        high = self.factor
        return number * float(np.exp(rng.uniform(np.log(low), np.log(high))))

    def _window(self, number):
        return _span(number, 1.0 / self.factor, self.factor)

    def __repr__(self):
        return _repr(
            "log_proportional",
            self.factor,
            sig=self.sig,
            digits=self.digits,
            low=self.low,
            high=self.high,
        )


class Shift(_NumericMutation):
    """Add an offset drawn uniformly from ``[-amount, amount]``."""

    def __init__(self, amount, *, sig=None, digits=None, low=None, high=None):
        super().__init__(sig=sig, digits=digits, low=low, high=high)
        amount = float(amount)
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("shift amount must be positive")
        self.amount = amount

    def _mutate(self, number, rng):
        return number + float(rng.uniform(-self.amount, self.amount))

    def _window(self, number):
        return (number - self.amount, number + self.amount)

    def __repr__(self):
        return _repr("shift", self.amount, sig=self.sig, digits=self.digits, low=self.low, high=self.high)


class Normal(_NumericMutation):
    """Add noise drawn from a normal distribution with the given standard deviation.

    ``low`` and ``high``, when given, clip the rounded value.
    """

    def __init__(self, std, *, sig=None, digits=None, low=None, high=None):
        super().__init__(sig=sig, digits=digits, low=low, high=high)
        std = float(std)
        if not np.isfinite(std) or std < 0:
            raise ValueError("normal needs a non-negative std")
        self.std = std

    def _mutate(self, number, rng):
        if self.std == 0:
            return number
        return number + float(rng.normal(0.0, self.std))

    def __repr__(self):
        return _repr("normal", self.std, sig=self.sig, digits=self.digits, low=self.low, high=self.high)


class Factor(_NumericMutation):
    """Multiply by one factor from a list, chosen uniformly.

    ``factor([0.5, 1, 2])`` halves, keeps, or doubles the current value.
    """

    def __init__(self, factors, *, sig=None, digits=None, low=None, high=None):
        super().__init__(sig=sig, digits=digits, low=low, high=high)
        if isinstance(factors, (str, bytes)) or not isinstance(factors, (list, tuple)):
            raise TypeError(
                "factor takes a list of multipliers, for example factor([0.5, 1, 2])"
            )
        if len(factors) == 0:
            raise ValueError("factor needs at least one multiplier")
        cleaned = []
        for factor in factors:
            if isinstance(factor, (bool, np.bool_)) or not isinstance(
                factor, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"factor multipliers must be numbers, got {factor!r}")
            factor = float(factor)
            if not np.isfinite(factor):
                raise ValueError("factor multipliers must be finite")
            cleaned.append(factor)
        self.factors = list(factors)
        self._factors = cleaned

    def _mutate(self, number, rng):
        return number * self._factors[int(rng.integers(len(self._factors)))]

    def _window(self, number):
        products = [number * factor for factor in self._factors]
        return min(products), max(products)

    def __repr__(self):
        return _repr("factor", self.factors, sig=self.sig, digits=self.digits, low=self.low, high=self.high)


class Choice(Mutation):
    """Replace the current value with one option from a list, chosen uniformly."""

    def __init__(self, options):
        if isinstance(options, (str, bytes)) or not isinstance(options, (list, tuple)):
            raise TypeError(
                "choice takes a list of options, for example choice([5, 8, 16])"
            )
        self.options = list(options)
        if not self.options:
            raise ValueError("choice needs at least one option")

    def apply(self, value, rng):
        return self.options[int(rng.integers(len(self.options)))]

    def __repr__(self):
        return f"choice({self.options!r})"


class Flip(Mutation):
    """Flip a boolean with probability ``p``. ``p=0`` keeps it, ``p=1`` always flips."""

    def __init__(self, p=0.5):
        p = float(p)
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("flip probability must be between 0 and 1")
        self.p = p

    def apply(self, value, rng):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"flip needs a boolean, got {value!r}")
        if float(rng.uniform()) < self.p:
            return not bool(value)
        return bool(value)

    def __repr__(self):
        return f"flip({self.p})"


def proportional(spread, *, sig=None, digits=None, low=None, high=None):
    """Multiply by a uniform factor in ``[1 - spread, 1 + spread]``."""
    return Proportional(spread, sig=sig, digits=digits, low=low, high=high)


def log_proportional(factor, *, sig=None, digits=None, low=None, high=None):
    """Multiply by a log-uniform factor in ``[1 / factor, factor]``."""
    return LogProportional(factor, sig=sig, digits=digits, low=low, high=high)


def shift(amount, *, sig=None, digits=None, low=None, high=None):
    """Add a uniform offset in ``[-amount, amount]``."""
    return Shift(amount, sig=sig, digits=digits, low=low, high=high)


def normal(std, *, sig=None, digits=None, low=None, high=None):
    """Add normal noise with the given standard deviation."""
    return Normal(std, sig=sig, digits=digits, low=low, high=high)


def factor(factors, *, sig=None, digits=None, low=None, high=None):
    """Multiply by a factor drawn from a list."""
    return Factor(factors, sig=sig, digits=digits, low=low, high=high)


def choice(options):
    """Replace the value with one of ``options``."""
    return Choice(options)


def flip(p=0.5):
    """Flip a boolean with probability ``p``."""
    return Flip(p)


def _require_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"expected a number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"expected a finite number, got {value!r}")
    return number


def _keeps_int(value, mutation):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and mutation.sig is None
        and mutation.digits is None
    )


def _as_int(value, low, high):
    value = int(np.rint(value))
    if low is not None:
        value = max(value, int(np.ceil(float(low) - 1e-9)))
    if high is not None:
        value = min(value, int(np.floor(float(high) + 1e-9)))
    return int(value)


def _span(number, low_factor, high_factor):
    low = number * low_factor
    high = number * high_factor
    if low <= high:
        return low, high
    return high, low


def _bounds(low, high):
    low = None if low is None else float(low)
    high = None if high is None else float(high)
    if low is not None and not np.isfinite(low):
        raise ValueError("low must be finite")
    if high is not None and not np.isfinite(high):
        raise ValueError("high must be finite")
    if low is not None and high is not None and low > high:
        raise ValueError("low cannot exceed high")
    return low, high


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
