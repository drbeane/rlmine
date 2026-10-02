"""A search space: named parameters plus optional cross-parameter constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Param, _is_action

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
        if _is_space(params):
            constraints = constraints or params.constraints
            params = params.params

        bad = {k: v for k, v in params.items() if not _is_param(v)}
        if bad:
            raise TypeError(
                "Space values must be Param objects (P.Int, P.Float, P.Bool, "
                f"P.Choice). Got plain values for: {sorted(bad)}. "
                "Use P.Int(4) rather than 4. If you just reloaded rlmine, "
                "re-run the cell that defines space so it builds fresh Params."
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
            hint = ""
            reserved = [n for n in unknown if n in ("sample", "mutate", "parent")]
            if reserved:
                hint = (
                    f" {reserved} belong on Study.run / Study.mine "
                    "(sample=['gamma'], not as a space parameter)."
                )
            raise KeyError(
                f"Unknown {label}: {unknown}. Space defines: {self.names}.{hint}"
            )

    def _resolve_mutate(self, mutate):
        return list(self._as_spec_map(mutate, "parameter(s) in mutate"))

    def _as_spec_map(self, spec, label):
        """Normalise ``sample=`` / ``mutate=`` to ``{name: extra or None}``."""
        if spec is None or spec is False:
            return {}
        if isinstance(spec, str):
            if spec == "all":
                return {name: None for name in self.names}
            spec = [spec]
        if isinstance(spec, dict):
            self._check_names(spec, label)
            return {
                name: (None if extra is True else extra) for name, extra in spec.items()
            }
        spec = list(spec)
        self._check_names(spec, label)
        return {name: None for name in spec}

    def defaults(self):
        return {name: p.clean(p.default) for name, p in self.params.items()}

    def draw(self, rng=None):
        """A fresh value for every parameter, from its sampler or default."""
        rng = _as_rng(rng)
        return self.apply_constraints({n: p.draw(rng) for n, p in self.params.items()})

    def derive(self, parent=None, mutate=(), sample=(), overrides=None, rng=None, fresh=False):
        """Build a parameter set from per-parameter behaviours.

        For each parameter the first matching rule wins:

        1. A given value in ``overrides``
        2. ``P.SAMPLE`` / a name in ``sample`` — draw from the stated range
        3. ``P.MUTATE`` / a name in ``mutate`` — perturb the parent value
        4. ``P.PARENT`` — inherit the parent value
        5. ``P.DEFAULT`` — the declared default
        6. The parent value, if a parent was supplied
        7. The declared default

        ``fresh`` is accepted for older callers; it draws every parameter
        that does not already have a more specific instruction.
        """
        rng = _as_rng(rng)
        overrides = dict(overrides or {})
        sample, mutate, overrides = _take_run_controls(overrides, sample, mutate)
        self._check_names(overrides, "parameter override(s)")
        sample_map = self._as_spec_map(sample, "parameter(s) in sample")
        mutate_map = self._as_spec_map(mutate, "parameter(s) in mutate")

        values = {}
        for name, param in self.params.items():
            values[name] = self._value_for(
                name,
                param,
                parent=parent,
                overrides=overrides,
                sample_map=sample_map,
                mutate_map=mutate_map,
                fresh=fresh,
                rng=rng,
            )

        return self.apply_constraints(values)

    def _value_for(self, name, param, parent, overrides, sample_map, mutate_map, fresh, rng):
        kind, spec = _behaviour(name, overrides, sample_map, mutate_map, parent, fresh)
        if kind == "value":
            return param.clean(spec)
        if kind == "sample":
            return param.sample_value(rng, sampler=spec, name=name)
        if kind == "draw":
            return param.draw(rng)
        if kind == "mutate":
            if parent is not None and name in parent and _present(parent[name]):
                return param.perturb(parent[name], rng, mutation=spec)
            if param.sampler is not None:
                return param.sample_value(rng, name=name)
            return param.clean(param.default)
        if kind == "parent":
            if parent is None or name not in parent or not _present(parent[name]):
                raise ValueError(
                    f"Cannot inherit {name!r} from a parent: no parent value is available."
                )
            return param.clean(parent[name])
        return param.clean(param.default)

    def apply_constraints(self, values):
        values = dict(values)
        initial, final = values.get("initial_lr"), values.get("final_lr")
        if _present(initial) and _present(final) and final > initial:
            values["final_lr"] = initial
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


def _is_param(value):
    """True for a Param, including one built by a previous import of rlmine.

    After ``importlib.reload`` (or deleting ``sys.modules['rlmine']``), a
    space dict still holds Int/Float/Bool/Choice instances from the old
    module. ``isinstance(value, Param)`` is then False even though the
    objects are usable.
    """
    if isinstance(value, Param):
        return True
    if type(value).__name__ not in ("Int", "Float", "Bool", "Choice"):
        return False
    return callable(getattr(value, "clean", None)) and hasattr(value, "default")


def _is_space(obj):
    """True for a Space, including one built by a previous import of rlmine."""
    if isinstance(obj, Space):
        return True
    return callable(getattr(obj, "derive", None)) and hasattr(obj, "params")


def _take_run_controls(overrides, sample, mutate):
    """Pull ``sample=`` / ``mutate=`` out of overrides if they leaked in.

    Older ``Study.run(n=1, **overrides)`` treated those names as parameters.
    They are run controls, not space parameters.
    """
    if "sample" in overrides:
        leaked = overrides.pop("sample")
        if sample is None or sample == ():
            sample = leaked
    if "mutate" in overrides:
        leaked = overrides.pop("mutate")
        if mutate is None or mutate == ():
            mutate = leaked
    if "parent" in overrides:
        raise KeyError(
            "parent= is a Study.run argument, not a space parameter. "
            "Pass parent=184203 (a config id) or a row to run/mine."
        )
    return sample, mutate, overrides


def _behaviour(name, overrides, sample_map, mutate_map, parent, fresh):
    if name in overrides:
        value = overrides[name]
        if _is_action(value):
            return value.kind, value.spec
        return "value", value
    if name in sample_map:
        return "sample", sample_map[name]
    if name in mutate_map:
        return "mutate", mutate_map[name]
    if fresh:
        return "draw", None
    if parent is not None and name in parent and _present(parent[name]):
        return "parent", None
    return "default", None


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
