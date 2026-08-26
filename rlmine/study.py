"""The Study object: one engine, three ways to choose parameters."""

from __future__ import annotations

import inspect
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from .scoring import resolve_score_fn
from .space import Space
from .stores import resolve_store
from .utils import (
    _quiet_third_party_warnings,
    display_header,
    display_obj,
    env_info,
    new_run_id,
    now_iso,
    today,
)

__all__ = ["Study"]

META_COLUMNS = [
    "run_id",
    "study",
    "origin",
    "parent_id",
    "status",
    "score",
    "mean",
    "std_dev",
    "minutes",
    "date",
]

METRIC_COLUMNS = ["score", "mean", "std_dev", "minutes", "mean_length", "success_rate"]


class Study:
    """Runs trials, scores them, and records every result.

    ``trial`` is a callable taking a parameter dict and returning an
    ``evaluate``-style stats dict with ``mean_return`` and ``stdev_return``. It
    may optionally take a second argument, a context dict carrying ``run_id``,
    ``study``, ``origin`` and ``parent_id``.

    Three ways to choose parameters, all recorded identically:

    ``run``       explicit values, layered over the space defaults
    ``mine``      a roulette-selected parent, with chosen parameters perturbed
    ``recheck``   a stored row replayed verbatim, to detect drift
    """

    def __init__(
        self,
        name,
        trial,
        space,
        store=None,
        mirror=None,
        score_fn=None,
        constraints=None,
        seed=None,
        verbose=True,
    ):
        self.name = name
        self.trial = trial
        self.space = space if isinstance(space, Space) else Space(space, constraints)
        if constraints is not None and isinstance(space, Space):
            self.space.constraints = constraints
        self.store = resolve_store(store, name)
        self.mirrors = _as_list(mirror)
        self.score_fn = resolve_score_fn(score_fn)
        self.verbose = verbose
        self._rng = np.random.default_rng(seed)
        self._trial_wants_context = _accepts_two_args(trial)
        self._noted_no_positive = False
        _quiet_third_party_warnings()

    def __repr__(self):
        return (
            f"Study(name={self.name!r}, params={len(self.space)}, "
            f"runs={len(self.history())}, store={self.store.location!r})"
        )

    # -----------------------------------------------------------------
    # The three entry points
    # -----------------------------------------------------------------

    def run(self, n=1, **overrides):
        """Run an explicit parameter set: space defaults plus any overrides.

            study.run(learning_rate=1e-3, gamma=0.995)
            study.run(n=3, seed=P_ANY)   # three repeats of the same settings
        """
        records = []
        for _ in range(int(n)):
            params = self.space.derive(overrides=overrides, rng=self._rng)
            records.append(self._execute(params, origin="manual"))
        return pd.DataFrame(records)

    def mine(self, n=1, mutate=(), parent=None, positive_only=True, **overrides):
        """Explore near known-good settings.

        A parent row is chosen by roulette-wheel selection weighted by score
        (or pinned with ``parent=``), the parameters named in ``mutate`` are
        perturbed, and anything in ``overrides`` is pinned to your value.

            study.mine(n=5, mutate=['learning_rate', 'gamma'])
            study.mine(n=3, mutate='all', parent=12, timesteps=200_000)
        """
        records = []
        for _ in range(int(n)):
            parent_row, parent_id = self._select_parent(parent, positive_only)
            parent_params = (
                self.space.parse_row(parent_row) if parent_row is not None else None
            )
            params = self.space.derive(
                parent=parent_params,
                mutate=mutate,
                overrides=overrides,
                rng=self._rng,
                fresh=parent_params is None,
            )
            records.append(
                self._execute(params, origin="mine", parent_id=parent_id)
            )
        return pd.DataFrame(records)

    def recheck(self, rows=None, top=None, latest=None):
        """Re-run stored parameter sets to see whether results have drifted.

        Specify exactly one of ``rows`` (indices or run ids), ``top`` (the best
        k by score), or ``latest`` (the most recent k).

            study.recheck(top=3)
            study.recheck(rows=[0, 4, 9])
        """
        history = self.history(ok_only=True)
        if "origin" in history.columns:
            history = history[history["origin"] != "recheck"]
        if history.empty:
            raise ValueError(
                f"No completed runs to recheck in {self.store.location}."
            )

        if rows is not None:
            selected = self._lookup_rows(history, rows)
        elif top is not None:
            selected = history.nlargest(int(top), "score")
        elif latest is not None:
            selected = history.tail(int(latest))
        else:
            raise ValueError(
                "Choose what to recheck: rows=[...], top=k, or latest=k."
            )

        records = []
        for _, row in selected.iterrows():
            params = self.space.parse_row(row)
            records.append(
                self._execute(
                    params,
                    origin="recheck",
                    parent_id=row.get("run_id"),
                    parent_score=row.get("score"),
                )
            )
        return pd.DataFrame(records)

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    def _execute(self, params, origin, parent_id=None, parent_score=None):
        _quiet_third_party_warnings()
        run_id = new_run_id()

        if self.verbose:
            display_header(f"{self.name} &mdash; {origin} &mdash; {run_id}", level=3)
            display_obj(pd.DataFrame([_cellify(params)]))

        context = {
            "run_id": run_id,
            "study": self.name,
            "origin": origin,
            "parent_id": parent_id,
        }

        started = time.time()
        status, error, stats = "ok", None, {}
        try:
            if self._trial_wants_context:
                stats = self.trial(dict(params), context) or {}
            else:
                stats = self.trial(dict(params)) or {}
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                traceback.print_exc()

        minutes = round((time.time() - started) / 60, 2)

        score = None
        if status == "ok":
            try:
                score = float(self.score_fn(stats))
            except Exception as exc:
                status = "failed"
                error = f"score_fn failed: {type(exc).__name__}: {exc}"

        record = {
            "run_id": run_id,
            "study": self.name,
            "origin": origin,
            "parent_id": parent_id,
            "status": status,
            "score": score,
            "mean": _maybe_float(stats.get("mean_return")),
            "std_dev": _maybe_float(stats.get("stdev_return")),
            "minutes": minutes,
            "date": today(),
            "timestamp": now_iso(),
        }
        for source, target in (("mean_length", "mean_length"), ("sr", "success_rate")):
            if source in stats:
                record[target] = _maybe_float(stats[source])

        record.update(_jsonable(params))
        record.update(env_info())
        record["error"] = error

        self._write(record)

        if self.verbose:
            self._report(record, parent_score)

        return record

    def _write(self, record, mirror=True):
        self.store.append(record)
        if not mirror:
            return
        for target in self.mirrors:
            try:
                target.append(record)
            except Exception as exc:
                warnings.warn(f"Mirror write failed ({exc}).", stacklevel=2)

    def _report(self, record, parent_score=None):
        if record["status"] != "ok":
            print(f"FAILED after {record['minutes']} min: {record['error']}")
            return

        summary = {
            "score": round(record["score"], 2),
            "mean": round(record["mean"], 2),
            "std_dev": round(record["std_dev"], 2),
            "minutes": record["minutes"],
        }
        if parent_score is not None and pd.notna(parent_score):
            previous = float(parent_score)
            summary["previous"] = round(previous, 2)
            summary["delta"] = round(record["score"] - previous, 2)
        display_obj(pd.DataFrame([summary]))

    # -----------------------------------------------------------------
    # Parent selection
    # -----------------------------------------------------------------

    def _select_parent(self, parent, positive_only):
        history = self.history(ok_only=True)
        if history.empty:
            if self.verbose:
                print("No prior results; drawing fresh parameter values.")
            return None, None

        if parent is None:
            row = self._roulette(history, positive_only)
        elif isinstance(parent, (dict, pd.Series)):
            return (parent, None)
        else:
            row = self._lookup_rows(history, [parent]).iloc[0]

        if self.verbose:
            print(f"Parent: {row.get('run_id')} (score {round(float(row['score']), 2)})")
        return row, row.get("run_id")

    def _roulette(self, history, positive_only):
        """Score-weighted selection, matching the notebooks' original scheme."""
        pool = history
        if positive_only:
            positive = history[history["score"] > 0]
            if positive.empty:
                if self.verbose and not self._noted_no_positive:
                    print("No runs with a positive score yet; selecting from all runs.")
                    self._noted_no_positive = True
            else:
                pool = positive

        weights = pool["score"].to_numpy(dtype=float)
        weights = weights - weights.min() + 1e-6
        probabilities = weights / weights.sum()
        position = self._rng.choice(len(pool), p=probabilities)
        return pool.iloc[position]

    @staticmethod
    def _lookup_rows(history, refs):
        frames = []
        for ref in _as_list(refs):
            if isinstance(ref, str):
                match = history[history["run_id"] == ref]
            else:
                match = history[history["idx"] == int(ref)]
            if match.empty:
                raise KeyError(f"No completed run matching {ref!r}.")
            frames.append(match.head(1))
        return pd.concat(frames)

    # -----------------------------------------------------------------
    # Reading results
    # -----------------------------------------------------------------

    def history(self, ok_only=False):
        """Every recorded row, with ``idx`` giving stable file position.

        Because the store is append-only, ``idx`` never changes, so it is safe
        to refer to a run as ``parent=6``.
        """
        frame = self.store.load()
        if frame.empty:
            return frame

        frame = frame.copy()
        frame.insert(0, "idx", range(len(frame)))
        if "score" in frame.columns:
            frame["score"] = pd.to_numeric(frame["score"], errors="coerce")

        if ok_only:
            if "status" in frame.columns:
                frame = frame[frame["status"] == "ok"]
            frame = frame[frame["score"].notna()]
        return frame

    def table(self, n=None, sort="score", ascending=False, params=True, extra=()):
        """A compact, readable view of results."""
        frame = self.history()
        if frame.empty:
            print(f"No results yet in {self.store.location}")
            return frame

        columns = [c for c in ["idx", "run_id", "origin", "score", "mean", "std_dev", "minutes", "date"] if c in frame.columns]
        if params:
            columns += [c for c in self.space.names if c in frame.columns]
        columns += [c for c in extra if c in frame.columns]

        view = frame[columns]
        if sort in view.columns:
            view = view.sort_values(sort, ascending=ascending, na_position="last")
        if n is not None:
            view = view.head(int(n))

        # Round the metrics only. Parameter values were already rounded to
        # their declared precision when generated, and blanket rounding would
        # flatten small ones such as a 1e-5 learning rate to zero.
        metrics = [c for c in METRIC_COLUMNS if c in view.columns]
        if metrics:
            view = view.copy()
            view[metrics] = view[metrics].apply(pd.to_numeric, errors="coerce").round(2)
        return view

    def best(self, k=1):
        history = self.history(ok_only=True)
        if history.empty:
            return history
        return history.nlargest(int(k), "score")

    def params_of(self, ref):
        """The parameter dict for a run, ready to pass back into ``run``."""
        history = self.history()
        row = self._lookup_rows(history, [ref]).iloc[0]
        return self.space.parse_row(row)

    def drift_report(self):
        """Compare every recheck against the run it replayed."""
        history = self.history()
        if history.empty or "origin" not in history.columns:
            return pd.DataFrame()

        rechecks = history[history["origin"] == "recheck"]
        if rechecks.empty:
            return pd.DataFrame()

        originals = history.set_index("run_id")
        rows = []
        for _, recheck in rechecks.iterrows():
            parent_id = recheck.get("parent_id")
            if parent_id not in originals.index:
                continue
            original = originals.loc[parent_id]
            if isinstance(original, pd.DataFrame):
                original = original.iloc[0]
            rows.append(
                {
                    "parent_id": parent_id,
                    "original_score": original.get("score"),
                    "recheck_score": recheck.get("score"),
                    "delta": _safe_delta(recheck.get("score"), original.get("score")),
                    "original_date": original.get("date"),
                    "recheck_date": recheck.get("date"),
                    "sb3_then": original.get("stable_baselines3"),
                    "sb3_now": recheck.get("stable_baselines3"),
                    "gym_then": original.get("gymnasium"),
                    "gym_now": recheck.get("gymnasium"),
                    "status": recheck.get("status"),
                }
            )
        return pd.DataFrame(rows).round(3)

    # -----------------------------------------------------------------
    # Migration
    # -----------------------------------------------------------------

    def import_rows(self, frame, origin="imported", score_column="score", mirror=False):
        """Bring historical results (e.g. an old Google Sheet) into this store.

        Values are parsed through the space, so string cells become properly
        typed parameters once, here, instead of on every read.

        Mirroring is off by default: importing a sheet into a study that
        mirrors to that same sheet would otherwise duplicate every row.
        """
        if isinstance(frame, pd.Series):
            frame = frame.to_frame().T

        imported = 0
        for _, row in frame.iterrows():
            params = self.space.parse_row(row)
            record = {
                "run_id": new_run_id(),
                "study": self.name,
                "origin": origin,
                "parent_id": None,
                "status": "ok",
                "score": _maybe_float(row.get(score_column)),
                "mean": _maybe_float(row.get("mean")),
                "std_dev": _maybe_float(row.get("std_dev")),
                "minutes": _maybe_float(row.get("time") or row.get("minutes")),
                "date": str(row.get("date", "")) or None,
                "timestamp": now_iso(),
            }
            record.update(_jsonable(params))
            for column in ("sb3_version", "gym_version", "runtime"):
                if column in row:
                    record[column.replace("sb3_version", "stable_baselines3").replace("gym_version", "gymnasium")] = row[column]
            record["error"] = None
            self._write(record, mirror=mirror)
            imported += 1

        if self.verbose:
            print(f"Imported {imported} rows into {self.store.location}")
        return imported


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _accepts_two_args(fn):
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _maybe_float(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else result


def _safe_delta(new, old):
    new, old = _maybe_float(new), _maybe_float(old)
    if new is None or old is None:
        return None
    return new - old


def _jsonable(params):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()}


def _cellify(params):
    return {k: (str(v) if isinstance(v, (list, tuple, dict)) else v) for k, v in params.items()}
