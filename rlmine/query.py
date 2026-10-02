"""Conditions for ``Results.filter``.

``config`` names a setting by the same path the config is written with, and
``run`` names a field of the recorded run. Comparing either one gives a
condition:

    from rlmine import config, run

    results.filter(config.training.timesteps <= 50_000)
    results.filter(run.score > 200, config.runtime == "gpu")
    results.filter((config.agent.gamma >= 0.99) | (run.minutes < 5))
    results.filter(~config.agent.net_arch.isin([[64, 64]]))

Several conditions passed to ``filter`` must all hold. Combine them yourself
with ``&`` (and), ``|`` (or), and ``~`` (not); each comparison needs its own
parentheses. A run that has no value for a setting never matches a comparison
on it. Use ``.isnull()`` to find those runs.
"""

from __future__ import annotations

import math

__all__ = ["config", "run", "Expr"]

_MISSING = object()


def _blank(value):
    if value is _MISSING or value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


class Expr:
    """A condition on one run. Build it by comparing ``config`` or ``run`` fields."""

    __hash__ = None

    def __and__(self, other):
        return _Logic("&", self, _need_expr(other))

    def __rand__(self, other):
        return _Logic("&", _need_expr(other), self)

    def __or__(self, other):
        return _Logic("|", self, _need_expr(other))

    def __ror__(self, other):
        return _Logic("|", _need_expr(other), self)

    def __invert__(self):
        return _Not(self)

    def __bool__(self):
        raise TypeError(
            "A condition has no truth value. Join conditions with & | ~ and "
            "put each comparison in parentheses; chained comparisons "
            "(a < x < b) and and/or/not do not work. Use .between(a, b)."
        )

    def refs(self):
        """Every ``config`` or ``run`` field the condition reads."""
        return []

    def test(self, settings, record):
        """True when the run, with its config ``settings``, matches."""
        raise NotImplementedError


def _need_expr(value):
    if not isinstance(value, Expr):
        raise TypeError(f"Expected a condition, got {type(value).__name__}.")
    return value


class Ref(Expr):
    """A field of the config or of the run, reached by attribute or ``[key]``."""

    def __init__(self, root, path=()):
        self._root = root
        self._path = tuple(path)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return Ref(self._root, (*self._path, name))

    def __getitem__(self, key):
        return Ref(self._root, (*self._path, key))

    def __repr__(self):
        text = self._root
        for key in self._path:
            text += f"[{key!r}]" if not isinstance(key, str) or not key.isidentifier() else f".{key}"
        return text

    def refs(self):
        return [self]

    def value(self, settings, record):
        """The field's value, or ``_MISSING`` when this run has none."""
        node = settings if self._root == "config" else record
        for key in self._path:
            try:
                node = node[key]
            except (KeyError, IndexError, TypeError):
                return _MISSING
        return node

    def test(self, settings, record):
        raise TypeError(f"{self!r} is not a condition. Compare it to a value.")

    def __lt__(self, other):
        return _Compare("<", self, other)

    def __le__(self, other):
        return _Compare("<=", self, other)

    def __gt__(self, other):
        return _Compare(">", self, other)

    def __ge__(self, other):
        return _Compare(">=", self, other)

    def __eq__(self, other):
        return _Compare("==", self, other)

    def __ne__(self, other):
        return _Compare("!=", self, other)

    def between(self, low, high):
        """``low <= field <= high``."""
        return (self >= low) & (self <= high)

    def isin(self, values):
        """The field equals one of ``values``."""
        return _IsIn(self, list(values))

    def isnull(self):
        """The run has no value for the field."""
        return _Null(self, True)

    def notnull(self):
        """The run has a value for the field."""
        return _Null(self, False)


_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class _Compare(Expr):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right

    def __repr__(self):
        return f"{self.left!r} {self.op} {self.right!r}"

    def refs(self):
        return [side for side in (self.left, self.right) if isinstance(side, Ref)]

    def test(self, settings, record):
        a, b = (
            side.value(settings, record) if isinstance(side, Ref) else side
            for side in (self.left, self.right)
        )
        if _blank(a) or _blank(b):
            return False
        try:
            return bool(_OPS[self.op](a, b))
        except TypeError:  # a schedule string against a number, say
            return False


class _IsIn(Expr):
    def __init__(self, ref, values):
        self.ref, self.values = ref, values

    def __repr__(self):
        return f"{self.ref!r}.isin({self.values!r})"

    def refs(self):
        return [self.ref]

    def test(self, settings, record):
        value = self.ref.value(settings, record)
        return not _blank(value) and any(value == v for v in self.values)


class _Null(Expr):
    def __init__(self, ref, want_null):
        self.ref, self.want_null = ref, want_null

    def __repr__(self):
        return f"{self.ref!r}.{'isnull' if self.want_null else 'notnull'}()"

    def refs(self):
        return [self.ref]

    def test(self, settings, record):
        return _blank(self.ref.value(settings, record)) == self.want_null


class _Logic(Expr):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right

    def __repr__(self):
        return f"({self.left!r}) {self.op} ({self.right!r})"

    def refs(self):
        return self.left.refs() + self.right.refs()

    def test(self, settings, record):
        if self.op == "&":
            return self.left.test(settings, record) and self.right.test(settings, record)
        return self.left.test(settings, record) or self.right.test(settings, record)


class _Not(Expr):
    def __init__(self, inner):
        self.inner = inner

    def __repr__(self):
        return f"~({self.inner!r})"

    def refs(self):
        return self.inner.refs()

    def test(self, settings, record):
        return not self.inner.test(settings, record)


config = Ref("config")
run = Ref("run")
