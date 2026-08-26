"""A search space: named parameters plus optional cross-parameter constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Param

__all__ = ["Space"]


class Space:
    """An ordered collection of named :class:`~rlmine.params.Param` objects.

    ``constraints`` is an optional callable taking the parameter dict and
    returning a corrected one, for rules that span parameters. For example::

        def constraints(p):
            p['final_lr'] = min(p['initial_lr'], p['final_lr'])
            return p
    """

    def __init__(self, params, constraints=None):
        if isinstance(params, Space):
            constraints = constraints or params.constraints
            params = params.params

        bad = {k: v for k, v in params.items() if not isinstance(v, Param)}
        if bad:
            raise TypeError(
                "Space values must be Param objects (P.Int, P.Float, P.Bool, "
                f"P.Choice). Got plain values for: {sorted(bad)}. "
                "Use P.Int(4) rather than 4."
            )

        self.params = dict(params)
        self.constraints = constraints

    # -- introspection -----------------------------------------------------

    @property
    def names(self):
        return list(self.params)

    def __contains__(self, name):
        return name in self.params

    def __getitem__(self, name):
        return self.params[name]

    def __iter__(self):
        return iter(self.params.items())

    def __len__(self):
        return len(self.params)

    def describe(self):
        """A readable table of the space, handy for documenting a study."""
        rows = []
        for name, param in self.params.items():
            rows.append(
                {
                    "parameter": name,
                    "type": type(param).__name__,
                    "default": param.default,
                    "bounds": param.bounds,
                    "mutate": repr(param.mutation or param.default_mutation()),
                    "sample": repr(param.sampler) if param.sampler else "",
                }
            )
        return pd.DataFrame(rows)

    # -- value generation --------------------------------------------------

    def _check_names(self, names, label):
        unknown = [n for n in names if n not in self.params]
        if unknown:
            raise KeyError(
                f"Unknown {label}: {unknown}. Space defines: {self.names}"
            )

    def _resolve_mutate(self, mutate):
        if mutate is None:
            return []
        if isinstance(mutate, str):
            if mutate == "all":
                return self.names
            mutate = [mutate]
        mutate = list(mutate)
        self._check_names(mutate, "parameter(s) in mutate")
        return mutate

    def defaults(self):
        return {name: p.clean(p.default) for name, p in self.params.items()}

    def draw(self, rng=None):
        """A fresh value for every parameter, from its sampler or default."""
        rng = _as_rng(rng)
        return self.apply_constraints({n: p.draw(rng) for n, p in self.params.items()})

    def derive(self, parent=None, mutate=(), overrides=None, rng=None, fresh=False):
        """Build a parameter set.

        Precedence, highest first: explicit ``overrides``, then a mutation of
        the ``parent`` value for parameters named in ``mutate``, then the
        inherited ``parent`` value, then a fresh draw (if ``fresh``), then the
        declared default.
        """
        rng = _as_rng(rng)
        overrides = dict(overrides or {})
        self._check_names(overrides, "parameter override(s)")
        mutate = self._resolve_mutate(mutate)

        values = {}
        for name, param in self.params.items():
            if name in overrides:
                values[name] = param.clean(overrides[name])
            elif name in mutate:
                if parent is not None and name in parent and _present(parent[name]):
                    values[name] = param.perturb(parent[name], rng)
                else:
                    values[name] = param.draw(rng)
            elif parent is not None and name in parent and _present(parent[name]):
                values[name] = param.clean(parent[name])
            elif fresh:
                values[name] = param.draw(rng)
            else:
                values[name] = param.clean(param.default)

        return self.apply_constraints(values)

    def apply_constraints(self, values):
        if self.constraints is None:
            return values
        corrected = self.constraints(dict(values))
        if corrected is None:
            raise ValueError("constraints callable must return the parameter dict")
        return {n: self.params[n].clean(v) for n, v in corrected.items()}

    # -- reading values back in -------------------------------------------

    def parse_row(self, row):
        """Extract and type-correct this space's parameters from a stored row.

        Accepts a dict or a pandas Series. Values that arrived as strings
        (which is everything, if they came from a spreadsheet) are converted
        back to their declared types, so ``net_arch`` becomes a list again and
        ``'TRUE'`` becomes ``True``.
        """
        if isinstance(row, pd.Series):
            row = row.to_dict()

        values = {}
        for name, param in self.params.items():
            if name in row and _present(row[name]):
                try:
                    values[name] = param.clean(param.parse(row[name]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Could not read parameter {name!r} from stored value "
                        f"{row[name]!r}: {exc}"
                    ) from None
            else:
                values[name] = param.clean(param.default)
        return values


def _present(value):
    """True unless the value is missing (None, NaN, or an empty string)."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def _as_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
