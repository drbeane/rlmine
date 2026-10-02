"""Read a study results file without running anything.

The file is the JSON document written by :class:`~rlmine.stores.JSONStore`:
one object keyed by config id, each value holding ``config`` and ``runs``.
Opening it does not rewrite the file.

    results = Results("results/bipedal-walker-a2c.json")
    results.summary()
    results.best(5)
    results.configs()
    results.by("initial_lr")
    results.runs(323629)

Pass ``config`` (a dict, or the path of a JSON file) to shade, in ``best`` and
``table``, every setting that differs from it:

    Results("results/bipedal-walker-a2c.json", config=config).best(5)
"""

from __future__ import annotations

import copy
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .query import _MISSING, Expr
from .stores import (
    RUN_FIELDS,
    JSONStore,
    _flat_records,
    _flatten_config,
    _is_study_document,
    _json_default,
    _migrate_document,
    _normalize_config_ids,
    _set_sort_key,
    format_config_id,
    group_flat_records,
)

__all__ = ["Results"]

_LEAD = ["config_id", "idx", "status", "score", "mean", "std_dev", "minutes", "date"]
_EXTRA = ["mean_length", "success_rate"]
# One config, so the settings stay off this view. What can differ between
# runs is the result, when it was recorded, and which packages recorded it.
_RUN_VIEW = [
    "config_id",
    "idx",
    "status",
    "date",
    "score",
    "mean",
    "std_dev",
    "minutes",
    "runtime",
    "timesteps",
    *_EXTRA,
    "python",
    "gymnasium",
    "stable_baselines3",
    "torch",
    "numpy",
]
_CONFIG_LEAD = ["config_id", "n", "score", "best", "mean", "worst", "minutes"]
# The recorded result. A recheck that matches these is the same result; when it
# was recorded, and which packages did the recording, are not part of it.
_OUTCOME = ["score", "mean", "std_dev", "mean_length", "success_rate"]
_NUMERIC = ["score", "mean", "std_dev", "minutes", "mean_length", "success_rate"]
_DRIFT_COLUMNS = [
    "config_id",
    "earlier_score",
    "later_score",
    "delta",
    "earlier_date",
    "later_date",
    "sb3_then",
    "sb3_now",
    "gym_then",
    "gym_now",
]
_PARAMS = ("varying", "all", "none")
_SHADE = "background-color: #ffe08a; color: #000000"


def _reloads(method):
    """Reread the file before a public method runs, so it never reports a stale view.

    A filtered ``Results`` holds a subset of the file and is left as it is. A
    method called from another method does not reread again.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._filtered or self._reading:
            return method(self, *args, **kwargs)
        self._reading = True
        try:
            self._reload()
            return method(self, *args, **kwargs)
        finally:
            self._reading = False

    return wrapper


class Results:
    """A results file opened for analysis.

    ``path`` is a ``.jsonl`` or ``.json`` file in the study document shape.
    Each config keeps every run of that exact config, so a recheck is another
    row under the same id rather than a new config.

    ``best`` and ``table`` are one row per run. ``best`` keeps the latest run
    of each config. the run table marks each run ``latest``, ``current``, or
    ``outdated``. Pass ``latest=True`` to keep one row per config, the run
    with status ``latest``.
    ``configs`` is one row per config. Fields that are the same on every
    config are left off those views; pass ``params="all"`` to put them back.

    ``config`` is a reference config: a dict in the shape stored with each
    result, or the path of a JSON file holding one. With it, ``table`` and
    ``best`` shade each setting that differs from the reference, and a
    setting that differs from it is shown even if it never varies.
    A setting the reference does not name is never shaded.
    """

    def __init__(self, path, config=None):
        self.path = Path(path).expanduser()
        self.config = _load_config(config)
        self._reference = None if self.config is None else _flatten_config(self.config)
        if self.path.is_dir():
            raise IsADirectoryError(
                f"{self.path} is a directory. Pass the results file itself."
            )
        if not self.path.is_file():
            raise FileNotFoundError(f"No results file at {self.path}.")
        self.document = _load_document(self.path)
        self._history = None
        self._configs = None
        self._keep_status = False
        self._filtered = False
        self._reading = False

    def _reload(self):
        self.document = _load_document(self.path)
        self._history = None
        self._configs = None

    @_reloads
    def __repr__(self):
        info = self.summary()
        best = "none" if info["best"] is None else info["best"]
        worst = "none" if info["worst"] is None else info["worst"]
        return (
            f"Results({str(self.path)!r}, configs={int(info['configs'])}, "
            f"runs={int(info['runs'])}, best={best}, worst={worst})"
        )

    @_reloads
    def __len__(self):
        return len(self._all_runs())

    @_reloads
    def summary(self):
        """How much is in the file, as one table.

        ``best`` and ``worst`` are the latest run of each config, so an older
        score is not the headline; ``best_id`` and ``worst_id`` name those
        configs. ``mean`` is the mean score of every run. ``minutes`` is the
        sum of recorded run times.
        """
        frame = self._all_runs()
        newest = self._all_runs(latest=True)
        scores = _column(frame, "score")
        newest_scores = _column(newest, "score")
        minutes = _column(frame, "minutes")
        stamps = frame["timestamp"].dropna() if "timestamp" in frame.columns else pd.Series(dtype=object)
        scored = scores.dropna()
        newest_scored = newest_scores.dropna()
        best_id = worst_id = None
        if not newest_scored.empty:
            best_id = format_config_id(newest.loc[newest_scores.idxmax(), "config_id"])
            worst_id = format_config_id(newest.loc[newest_scores.idxmin(), "config_id"])
        return pd.Series(
            {
                "best": None if newest_scored.empty else float(newest_scored.max()),
                "best_id": best_id,
                "worst": None if newest_scored.empty else float(newest_scored.min()),
                "worst_id": worst_id,
                "mean": None if scored.empty else float(scored.mean()),
                "configs": int(frame["config_id"].nunique()) if "config_id" in frame.columns else 0,
                "runs": int(len(frame)),
                "minutes": float(minutes.sum(skipna=True)) if len(minutes) else 0.0,
                "first": None if stamps.empty else str(stamps.min()),
                "last": None if stamps.empty else str(stamps.max()),
            },
            name=self.path.name,
        )

    @_reloads
    def delete_runs(self, *run_ids):
        """Delete runs by ``idx``: ``results.delete_runs(4, 7, 9)`` or a list.

        A config whose last run is deleted is removed too. This rewrites the
        file and prints what was deleted and any ids not found.
        """
        self._delete("delete_runs", _flatten_ids(run_ids))

    @_reloads
    def delete_configs(self, *config_ids):
        """Delete whole configs, with every run, by ``config_id``.
        """
        self._delete("delete_configs", _flatten_ids(config_ids))

    @_reloads
    def delete_duplicates(self, dry_run=False):
        """Delete runs that match a more recent run of the same config.

        Two runs are duplicates when everything but ``idx``, ``date``,
        ``timestamp`` and ``minutes`` matches: the score and its components,
        and the package versions. The most recent run of each group is kept.
        ``dry_run=True`` only reports the run ids that would be deleted.
        """
        self._delete("delete_duplicates", dry_run=dry_run, dry=dry_run)

    def _delete(self, method, *args, dry=False, **kwargs):
        if self._filtered:
            raise ValueError(
                "Delete from the Results you opened, not from a filtered one. "
                "Read the ids off the filtered view, then pass them to delete_runs."
            )
        store = JSONStore(self.path.parent)
        store.file = self.path
        result = getattr(store, method)(*args, **kwargs)
        if not dry:
            self.document = _load_document(self.path)
            self._history = None
            self._configs = None
        if dry:
            print(f"Would delete {len(result['runs'])} run(s): {result['runs']}")
        else:
            print(f"Deleted {len(result['runs'])} run(s) and {len(result['configs'])} config(s).")
        if result["missing"]:
            print(f"Not found: {result['missing']}")

    def _all_runs(self, latest=False):
        """Every recorded run, one row per run, in timestamp order.

        ``status`` is ``latest`` for the most recent run of a config. It is
        ``current`` for an earlier run whose result matches that one, and
        ``outdated`` when the result differs. The result is the score and its
        components, not the timestamp or the package versions.

        ``latest=True`` keeps one row per config: the run with status
        ``latest``.
        """
        if self._history is None:
            self._history = _annotate_status(_runs_frame(self.document), keep=self._keep_status)
        frame = _latest_runs(self._history) if latest else self._history
        return frame.copy()

    @_reloads
    def table(self, n=None, params="varying", latest=False):
        """Runs sorted by score, with the config fields that are worth showing.

        ``params`` is ``"varying"`` (the default), ``"all"``, or ``"none"``.
        Package versions are not part of this view; ``runs`` shows them for one config.
        ``latest=True`` keeps one row per config, the run with status ``latest``.
        ``runtime`` and ``timesteps`` always come first among the settings.
        """
        return self._table(n, params, latest, _LEAD)

    def _table(self, n, params, latest, lead):
        fields = _core_first(self._param_fields(params))
        frame = self._all_runs(latest=latest)
        if frame.empty:
            return frame
        columns = [c for c in lead + _EXTRA if c in frame.columns]
        columns += [c for c in fields if c in frame.columns and c not in columns]
        view = frame[columns]
        if "score" in view.columns:
            view = view.sort_values("score", ascending=False, na_position="last", kind="mergesort")
        if n is not None:
            view = view.head(int(n))
        view = view.reset_index(drop=True)
        if self._reference is not None:
            view = _ShadedFrame(view)
            view._reference = self._reference
        else:
            view = _TrimmedFrame(view)
        return view

    @_reloads
    def best(self, k=10, params="varying", latest=True):
        """The ``k`` highest-scoring runs.

        The default keeps the latest run of each config, so an older score
        cannot outrank the result that config has now, and a repeated identical
        result appears once. Pass ``latest=False`` to rank every recorded run.
        ``runtime`` and ``timesteps`` come first among the settings, and
        ``date`` comes just before ``score``.
        """
        lead = [c for c in _LEAD if c != "date"]
        lead.insert(lead.index("score"), "date")
        return self._table(k, params, latest, lead)

    @_reloads
    def configs(self, params="varying"):
        """One row per config, ranked by its best score.

        ``score`` is the latest run. ``best``, ``mean``, and ``worst`` summarise
        every run, so ``best`` can be an older score than ``score``. ``n`` is
        how many times the config was run. ``minutes`` is their total time.
        """
        self._check_params(params)
        frame = self._config_frame()
        if frame.empty:
            return frame.copy()
        fields = _core_first(self._param_fields(params))
        columns = [c for c in _CONFIG_LEAD if c in frame.columns]
        columns += [c for c in fields if c in frame.columns and c not in columns]
        return frame[columns].copy()

    @_reloads
    def runtimes(self, k=10):
        """The ``k`` best setups, with each runtime they were checked on.

        Configs that differ only in ``runtime`` are one setup. Setups are
        ranked by the best latest score among their runtimes. Each row shows
        ``score`` (the best latest score among the runtimes), then for each
        runtime the latest score, the date of that run, its ``minutes``, and
        the config id, then ``timesteps`` and the same settings ``best`` shows
        (shaded the same way when a reference ``config`` is given).
        A runtime that was never checked is blank.
        """
        groups = {}
        for pid, entry in sorted(self.document.items(), key=_set_sort_key):
            config = entry.get("config") or {}
            latest = _latest_run(list(entry.get("runs") or []))
            if latest is None:
                continue
            runtime = config.get("runtime")
            runtime = "unknown" if _blank_value(runtime) else str(runtime)
            rest = {key: value for key, value in config.items() if key != "runtime"}
            stamp = latest.get("timestamp") if isinstance(latest.get("timestamp"), str) else ""
            seen = groups.setdefault(_group_key(rest), {"config": rest, "runtimes": {}})
            held = seen["runtimes"].get(runtime)
            if held is None or stamp >= held["stamp"]:
                seen["runtimes"][runtime] = {
                    "id": format_config_id(pid),
                    "score": _number(latest.get("score")),
                    "date": latest.get("date"),
                    "minutes": _number(latest.get("minutes")),
                    "stamp": stamp,
                }
        if not groups:
            return pd.DataFrame()

        names = sorted({name for g in groups.values() for name in g["runtimes"]})
        fields = [c for c in self._param_fields("varying") if c not in ("runtime", "timesteps")]
        columns = ["score"] + [
            f"{name}_{part}" for name in names for part in ("score", "date", "minutes", "id")
        ] + ["timesteps"] + fields
        ranked = []
        for group in groups.values():
            scores = [r["score"] for r in group["runtimes"].values() if r["score"] is not None]
            best = max(scores) if scores else None
            flat = _flatten_config(group["config"])
            row = {"score": best, "timesteps": flat.get("timesteps")}
            for field in fields:
                row[field] = flat.get(field)
            for name in names:
                found = group["runtimes"].get(name) or {}
                row[f"{name}_id"] = found.get("id")
                row[f"{name}_score"] = found.get("score")
                row[f"{name}_date"] = found.get("date")
                row[f"{name}_minutes"] = found.get("minutes")
            ranked.append((float("-inf") if best is None else best, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        rows = [row for _, row in ranked[: int(k)]]
        frame = pd.DataFrame(rows, columns=columns)
        # An unchecked runtime is NaN in every column. A date column holding
        # None would show it as None, and an id column holding None is a float.
        date_columns = [c for c in columns if c.endswith("_date")]
        frame[date_columns] = frame[date_columns].astype(object).where(frame[date_columns].notna(), float("nan"))
        frame = _RuntimeFrame(frame)
        frame._reference = self._reference
        return frame

    @_reloads
    def by(self, field, latest=False):
        """Scores grouped by one column, usually a config setting.

        Each row is one value of ``field``: how many runs used it, the best
        score among them, and the mean score. ``latest=True`` uses one run
        per config, the one with status ``latest``.
        """
        frame = self._all_runs(latest=latest)
        if frame.empty or field not in frame.columns:
            known = ", ".join(frame.columns) if not frame.empty else "none"
            raise KeyError(f"{field!r} is not a column of this file. Columns: {known}.")
        work = frame[[field]].copy()
        work["score"] = _column(frame, "score")
        work["_key"] = work[field].map(_group_key)
        rows = []
        for _, group in work.groupby("_key", dropna=False, sort=False):
            scores = group["score"].dropna()
            rows.append(
                {
                    field: group[field].iloc[0],
                    "n": int(len(group)),
                    "best": None if scores.empty else float(scores.max()),
                    "mean": None if scores.empty else float(scores.mean()),
                }
            )
        out = pd.DataFrame(rows, columns=[field, "n", "best", "mean"])
        if out.empty:
            return out
        return out.sort_values(
            ["best", "mean"], ascending=False, na_position="last", kind="mergesort"
        ).reset_index(drop=True)

    @_reloads
    def drift(self):
        """Each later score of a config against the one recorded before it.

        A config that was only run once is omitted, and so is any pair whose
        scores match (a zero delta) or cannot be compared. Package versions are
        included so a move in the score can be lined up with a library change.
        """
        frame = self._all_runs()
        if frame.empty or "config_id" not in frame.columns:
            return pd.DataFrame(columns=_DRIFT_COLUMNS)
        rows = []
        for config_id, group in frame.groupby("config_id", sort=False):
            ordered = group.reset_index(drop=True)
            if len(ordered) < 2:
                continue
            for i in range(1, len(ordered)):
                earlier = ordered.iloc[i - 1]
                later = ordered.iloc[i]
                delta = _delta(later.get("score"), earlier.get("score"))
                if not delta:
                    continue
                rows.append(
                    {
                        "config_id": format_config_id(config_id),
                        "earlier_score": _number(earlier.get("score")),
                        "later_score": _number(later.get("score")),
                        "delta": delta,
                        "earlier_date": earlier.get("date"),
                        "later_date": later.get("date"),
                        "sb3_then": earlier.get("stable_baselines3"),
                        "sb3_now": later.get("stable_baselines3"),
                        "gym_then": earlier.get("gymnasium"),
                        "gym_now": later.get("gymnasium"),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=_DRIFT_COLUMNS)
        return pd.DataFrame(rows, columns=_DRIFT_COLUMNS)

    @_reloads
    def runs(self, config_id):
        """Every recorded run of one config, in the order it was recorded.

        ``config_id`` is the six-digit id, as an int or a string. The settings
        are the same on every row, so this table keeps the result, when it
        was recorded, the timesteps and runtime, and the package versions.
        ``score`` is shaded as in ``table``. ``get_config`` returns the config.
        """
        pid, _ = self._entry(config_id)
        frame = self._all_runs()
        rows = frame[frame["config_id"] == pid]
        columns = [name for name in _RUN_VIEW if name in rows.columns]
        return _TrimmedFrame(rows[columns].reset_index(drop=True))

    @_reloads
    def get_config(self, config_id, diff=False):
        """The config stored under ``config_id``, as a nested dict.

        ``config_id`` is the six-digit id, as an int or a string. Runs and
        scores are not here; ``runs`` shows them.

        ``diff=True`` cuts the config down to the settings that differ from
        the reference config given to ``Results``, keeping the same nesting.
        As with shading, a setting the reference does not name is not a
        difference. ``runtime`` and ``training.timesteps`` are always kept.
        """
        _, entry = self._entry(config_id)
        config = copy.deepcopy(entry.get("config") or {})
        if not diff:
            return config
        if self.config is None:
            raise ValueError("diff=True needs a reference: pass config= to Results.")
        shown = _config_diff(config, self.config)
        if "runtime" in config:
            shown["runtime"] = config["runtime"]
        training = config.get("training")
        if isinstance(training, dict) and "timesteps" in training:
            shown.setdefault("training", {})["timesteps"] = training["timesteps"]
        return shown

    @_reloads
    def compare(self, *config_ids):
        """The named configs side by side, keeping fields that differ.

        Score summaries stay even when they match. A setting that is the same
        on every named config is dropped.
        """
        if len(config_ids) < 2:
            raise TypeError("Pass at least two config ids to compare.")
        ids = [self._entry(config_id)[0] for config_id in config_ids]
        frame = self._config_frame()
        pieces = [frame[frame["config_id"] == pid] for pid in ids]
        view = pd.concat(pieces, ignore_index=True)
        keep = [c for c in _CONFIG_LEAD if c in view.columns]
        for column in view.columns:
            if column in keep:
                continue
            if _varies(view[column]):
                keep.append(column)
        return view[keep].reset_index(drop=True)

    @_reloads
    def filter(self, *conditions):
        """The runs that match every condition, as another ``Results``.

        A condition compares a config setting or a field of the run:

            from rlmine import config, run

            results.filter(config.training.timesteps <= 50_000)
            results.filter(run.score > 200, config.runtime == "gpu")
            results.filter((config.agent.gamma >= 0.99) | (run.minutes < 5))

        A condition can also be a string, evaluated like ``DataFrame.query``
        over the column names ``table`` shows, so ``and``, ``or``, ``not``,
        ``in``, and ``@variable`` work:

            results.filter("timesteps <= 50_000 and score > 200")
            results.filter("training_timesteps <= 50_000")

        Where two sections share a name, such as ``gamma``, the short name is
        the agent's. ``training_timesteps`` and ``evaluation_gamma`` name a
        section's own. Unlike the objects, a string comparison with ``!=``
        matches runs that have no value.

        Config settings are named by the sections they are stored under.
        ``run`` fields are ``score``, ``mean``, ``std_dev``, ``minutes``,
        ``date``, ``status``, and the package versions. Conditions
        passed together must all hold. Join them with ``&``, ``|``, and ``~``,
        each comparison in parentheses. A run with no value for a setting does
        not match a comparison on it; ``.isnull()`` finds those runs.

        Each run keeps the ``status`` it has in the whole file, so a
        ``latest`` view of the result never promotes an older run that a
        condition happened to keep. ``configs``, ``get_config``, and ``compare``
        summarise only the runs that matched. A name that no run has, such as
        ``config.trainig``, raises ``KeyError`` instead of matching nothing.
        """
        if not conditions:
            raise TypeError(
                "Pass at least one condition, such as config.training.timesteps <= 50_000 "
                "or 'timesteps <= 50_000'."
            )
        for condition in conditions:
            if not isinstance(condition, (Expr, str)):
                raise TypeError(
                    f"filter takes conditions built from config and run, or strings, "
                    f"not {type(condition).__name__}."
                )
        marked = [
            (pid, entry, self._marked(entry))
            for pid, entry in sorted(self.document.items(), key=_set_sort_key)
        ]
        for condition in conditions:
            if isinstance(condition, Expr):
                for ref in condition.refs():
                    self._check_ref(ref, marked)
        caller = sys._getframe(2)  # past the _reloads wrapper
        scope = {**caller.f_globals, **caller.f_locals}
        keep = _filter_mask(conditions, marked, scope)
        document = {}
        position = 0
        for pid, entry, runs in marked:
            kept = []
            for item in runs:
                if keep[position]:
                    kept.append(item)
                position += 1
            if kept:
                document[pid] = {**entry, "runs": kept}
        narrowed = object.__new__(type(self))
        narrowed.__dict__.update(self.__dict__)
        narrowed.document = document
        narrowed._history = None
        narrowed._configs = None
        narrowed._keep_status = True
        narrowed._filtered = True
        return narrowed

    def _marked(self, entry):
        """Copies of a config's runs with ``status``. A filtered file keeps its own."""
        runs = list(entry.get("runs") or [])
        if self._keep_status:
            return [dict(run) for run in runs]
        return _mark_runs(runs)

    @staticmethod
    def _check_ref(ref, marked):
        """Raise when no run has the field, which is almost always a typo."""
        if not any(runs for _, _, runs in marked):
            return
        for pid, entry, runs in marked:
            settings = entry.get("config") or {}
            for item in runs:
                if ref.value(settings, {**item, "config_id": format_config_id(pid)}) is not _MISSING:
                    return
        _, first, runs = next(item for item in marked if item[2])
        parent = (first.get("config") or {}) if ref._root == "config" else runs[0]
        known = None
        for key in ref._path:
            if isinstance(parent, dict) and key in parent:
                parent = parent[key]
            else:
                known = ", ".join(map(str, parent)) if isinstance(parent, dict) else None
                break
        hint = f" Available here: {known}." if known else ""
        raise KeyError(f"No run has {ref!r}.{hint}")

    def _config_frame(self):
        if self._configs is None:
            self._configs = _config_frame(self.document)
        return self._configs

    def _param_fields(self, params):
        self._check_params(params)
        frame = self._config_frame()
        fields = [c for c in frame.columns if c not in _CONFIG_LEAD]
        if params == "none":
            return []
        if params == "all":
            return fields
        return [c for c in fields if _varies(frame[c]) or self._differs(frame[c])]

    def _differs(self, series):
        """True when the reference names this field and any config disagrees."""
        ref = self._reference
        if ref is None or series.name not in ref:
            return False
        return any(not _same(value, ref[series.name]) for value in series)

    def _entry(self, config_id):
        pid = format_config_id(config_id)
        if pid is None or pid not in self.document:
            raise KeyError(f"No config {config_id!r} in {self.path.name}.")
        return pid, self.document[pid]

    @staticmethod
    def _check_params(params):
        if params not in _PARAMS:
            names = ", ".join(repr(name) for name in _PARAMS)
            raise ValueError(f"params must be one of {names}.")


def _query_frame(marked):
    """One row per run, in ``marked`` order, with the names a string condition may use.

    Each setting appears as its short name and, when it sits in a section, as
    ``section_name``. Run fields keep their own names.
    """
    rows = []
    for pid, entry, runs in marked:
        settings = entry.get("config") or {}
        flat = _flatten_config(settings)
        qualified = {}
        for section in ("environment", "agent", "training", "evaluation"):
            body = settings.get(section)
            if isinstance(body, dict):
                for key, value in body.items():
                    qualified[f"{section}_{key}"] = value
        for item in runs:
            rows.append({**qualified, **flat, **item, "config_id": format_config_id(pid)})
    frame = pd.DataFrame(rows)
    for column in _NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _filter_mask(conditions, marked, scope):
    """For each run in ``marked`` order, whether every condition holds."""
    total = sum(len(runs) for _, _, runs in marked)
    keep = [True] * total
    if not total:
        return keep
    queries = [c for c in conditions if isinstance(c, str)]
    if queries:
        frame = _query_frame(marked)
        for text in queries:
            try:
                result = frame.eval(text, engine="python", local_dict=scope)
            except NameError as error:
                known = ", ".join(str(c) for c in frame.columns)
                raise KeyError(f"{error} in {text!r}. Names you can use: {known}.") from None
            if not (isinstance(result, pd.Series) and result.dtype == bool):
                raise TypeError(f"{text!r} is not a true/false condition.")
            keep = [k and bool(v) for k, v in zip(keep, result)]
    exprs = [c for c in conditions if isinstance(c, Expr)]
    if exprs:
        position = 0
        for pid, entry, runs in marked:
            settings = entry.get("config") or {}
            for item in runs:
                record = {**item, "config_id": format_config_id(pid)}
                keep[position] = keep[position] and all(c.test(settings, record) for c in exprs)
                position += 1
    return keep


def _load_config(config):
    """The reference config: a dict, or a JSON file holding one."""
    if config is None:
        return None
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"No config file at {path}.")
        config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("config must be a dict or the path of a JSON file.")
    return copy.deepcopy(config)


def _frame_look(styler):
    """Make a Styler look like a plain DataFrame table, which Colab and Jupyter style well.

    A bare Styler table lacks the ``dataframe`` class their CSS targets, so cells
    sit left-aligned with no padding and the headers run together.
    """
    return styler.set_table_attributes('class="dataframe"').set_table_styles(
        [
            {"selector": "th, td", "props": [("text-align", "right"), ("padding", "0.5em 0.9em")]},
            {"selector": "th", "props": [("font-weight", "bold")]},
        ],
        overwrite=False,
    )


def _tint_score(styler):
    """Give the score column a light red background."""
    if "score" in styler.data.columns:
        styler = styler.set_properties(subset=["score"], **{"background-color": "#ffd6d6", "color": "#000000"})
    return styler


def _trim(value):
    """A float without trailing zeros: 3.0 shows as 3 and 0.50 as 0.5."""
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return np.format_float_positional(value, trim="-")
    return str(value)


def _steps(value):
    """Timesteps with ``_`` as the thousands separator: 1000000 shows as 1_000_000."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if np.isfinite(value) and float(value) == int(value):
            return f"{int(value):_}"
    return _trim(value)


def _dash(value):
    """Like ``_trim``, with an em dash for a missing value."""
    return "—" if _blank_value(value) else _trim(value)


def _dash_steps(value):
    return "—" if _blank_value(value) else _steps(value)


def _formats(columns, plain=_trim, steps=_steps):
    """One formatter per column: ``timesteps`` gets separators, the rest ``plain``."""
    return {c: (steps if c == "timesteps" else plain) for c in columns}


class _TrimmedFrame(pd.DataFrame):
    """A results table that shows floats without trailing zeros.

    It is a DataFrame in every other way. Only the display changes; the
    stored values are untouched.
    """

    @property
    def _constructor(self):
        return _TrimmedFrame

    def styled(self):
        """The table as a pandas Styler, with trailing zeros trimmed."""
        plain = pd.DataFrame(self)
        return _tint_score(_frame_look(plain.style.format(_formats(plain.columns))))

    def _repr_html_(self):
        try:
            return self.styled()._repr_html_()
        except ImportError:  # Styler needs jinja2
            return super()._repr_html_()

    def __repr__(self):
        plain = pd.DataFrame(self)
        formatters = {c: _trim for c in plain.columns if pd.api.types.is_float_dtype(plain[c])}
        if "timesteps" in plain.columns:
            formatters["timesteps"] = _steps
        return plain.to_string(formatters=formatters, max_rows=pd.get_option("display.max_rows"))


class _RuntimeFrame(_TrimmedFrame):
    """The ``runtimes`` table: per-runtime scores in light red, the overall score in light orange."""

    _metadata = ["_reference"]

    @property
    def _constructor(self):
        return _RuntimeFrame

    def __repr__(self):
        # Format each cell first: to_string passes a missing value through as NaN.
        plain = pd.DataFrame(self).astype(object)
        shown = plain.copy()
        for column, fmt in _formats(plain.columns, _dash, _dash_steps).items():
            shown[column] = plain[column].map(fmt)
        return shown.to_string(max_rows=pd.get_option("display.max_rows"))

    def styled(self):
        plain = pd.DataFrame(self)
        styler = _frame_look(plain.style.format(_formats(plain.columns, _dash, _dash_steps)))
        ref = getattr(self, "_reference", None) or {}
        if ref:
            styles = pd.DataFrame("", index=plain.index, columns=plain.columns)
            for position, name in enumerate(plain.columns):
                if name not in ref or name in ("score", "timesteps"):
                    continue
                for row, value in enumerate(plain.iloc[:, position]):
                    if not _same(value, ref[name]):
                        styles.iat[row, position] = _SHADE
            styler = styler.apply(lambda _: styles, axis=None)
        per_runtime = [c for c in self.columns if c.endswith("_score")]
        if per_runtime:
            styler = styler.set_properties(
                subset=per_runtime, **{"background-color": "#ffd6d6", "color": "#000000"}
            )
        if "score" in self.columns:
            styler = styler.set_properties(
                subset=["score"], **{"background-color": "#ffe0b3", "color": "#000000"}
            )
        # A dark vertical border on each side of every runtime's score/date/id group.
        edge = "2px solid #000000"
        first = [c for c in self.columns if c.endswith("_score")]
        last = [c for c in self.columns if c.endswith("_id")]
        if first:
            styler = styler.set_properties(subset=first, **{"border-left": edge})
        if last:
            styler = styler.set_properties(subset=last, **{"border-right": edge})
        styler = styler.set_table_styles(
            [
                {"selector": f"th.col{i}", "props": [("border-left", edge)]}
                for i, c in enumerate(self.columns)
                if c in first
            ]
            + [
                {"selector": f"th.col{i}", "props": [("border-right", edge)]}
                for i, c in enumerate(self.columns)
                if c in last
            ],
            overwrite=False,
        )
        return styler


class _ShadedFrame(_TrimmedFrame):
    """A results table that shades settings differing from the reference config.

    It is a DataFrame in every other way. Only the notebook display changes.
    """

    _metadata = ["_reference"]

    @property
    def _constructor(self):
        return _ShadedFrame

    def _shade_styles(self):
        ref = getattr(self, "_reference", None) or {}
        styles = pd.DataFrame("", index=self.index, columns=self.columns)
        for position, name in enumerate(self.columns):
            if name not in ref or name in _LEAD or name in _EXTRA:
                continue
            for row, value in enumerate(self.iloc[:, position]):
                if not _same(value, ref[name]):
                    styles.iat[row, position] = _SHADE
        return styles

    def shaded(self):
        """The table as a pandas Styler, with differing settings shaded."""
        plain = pd.DataFrame(self)
        styler = plain.style.apply(lambda _: self._shade_styles(), axis=None).format(_formats(plain.columns))
        return _tint_score(_frame_look(styler))

    def _repr_html_(self):
        try:
            return self.shaded()._repr_html_()
        except ImportError:  # Styler needs jinja2
            return super()._repr_html_()


def _core_first(fields):
    """``runtime`` then ``timesteps`` ahead of the other settings, always present."""
    return ["runtime", "timesteps"] + [c for c in fields if c not in ("runtime", "timesteps")]


def _config_diff(config, reference):
    """The settings of ``config`` that differ from ``reference``, nested as given.

    Sections and other dicts are walked; a setting the reference lacks is kept
    out, and a section with no differences is dropped.
    """
    out = {}
    for key, value in config.items():
        if key not in reference:
            continue
        ref = reference[key]
        if isinstance(value, dict) and isinstance(ref, dict):
            inner = _config_diff(value, ref)
            if inner:
                out[key] = inner
        elif not _same(value, ref):
            out[key] = value
    return out


def _same(value, reference):
    """Equal settings. Floats compare to rounding error; a missing value is not equal."""
    if isinstance(value, (dict, list, tuple)) or isinstance(reference, (dict, list, tuple)):
        return _group_key(value) == _group_key(reference)
    if _blank_value(value) or _blank_value(reference):
        return _blank_value(value) and _blank_value(reference)
    if isinstance(value, (bool, np.bool_)) or isinstance(reference, (bool, np.bool_)):
        return (
            isinstance(value, (bool, np.bool_))
            and isinstance(reference, (bool, np.bool_))
            and bool(value) == bool(reference)
        )
    if isinstance(value, (int, float, np.number)) and isinstance(reference, (int, float, np.number)):
        return math.isclose(float(value), float(reference), rel_tol=1e-9, abs_tol=1e-12)
    return value == reference


def _blank_value(value):
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _flatten_ids(args):
    """``(1, 2)`` and ``([1, 2],)`` both mean ids 1 and 2."""
    ids = []
    for item in args:
        if isinstance(item, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
            ids.extend(item)
        else:
            ids.append(item)
    return ids


def _load_document(path):
    """The study document, migrated in memory. The file is left as it was."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and _is_study_document(parsed):
        doc, _ = _migrate_document(parsed)
    else:
        doc = group_flat_records(_flat_records(text, parsed, path))
    doc, _ = _normalize_config_ids(doc)
    return doc


def _runs_frame(document):
    rows = []
    for pid, entry in sorted(document.items(), key=_set_sort_key):
        flat = _flatten_config(entry.get("config") or {})
        for run in entry.get("runs") or []:
            row = dict(flat)
            row.update(run)
            row["config_id"] = format_config_id(pid)
            rows.append(row)
    rows.sort(key=lambda row: row.get("timestamp") if isinstance(row.get("timestamp"), str) else "")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["config_id", *_LEAD[1:], "timestamp"])
    for column in _NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "config_id" in frame.columns:
        frame["config_id"] = frame["config_id"].map(format_config_id)
    return frame


def _config_frame(document):
    rows = []
    columns = list(_CONFIG_LEAD)
    ordered = sorted(document.items(), key=_set_sort_key)
    for pid, entry in ordered:
        runs = list(entry.get("runs") or [])
        scores = [s for s in (_number(run.get("score")) for run in runs) if s is not None]
        minutes = [m for m in (_number(run.get("minutes")) for run in runs) if m is not None]
        latest = _latest_run(runs)
        row = {
            "config_id": format_config_id(pid),
            "n": len(runs),
            "score": _number(latest.get("score")) if latest else None,
            "best": max(scores) if scores else None,
            "mean": float(np.mean(scores)) if scores else None,
            "worst": min(scores) if scores else None,
            "minutes": float(np.sum(minutes)) if minutes else 0.0,
        }
        for key, value in _flatten_config(entry.get("config") or {}).items():
            if key in row or key in RUN_FIELDS:
                continue
            row[key] = value
            if key not in columns:
                columns.append(key)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        "score", ascending=False, na_position="last", kind="mergesort"
    ).reset_index(drop=True)


def _annotate_status(frame, keep=False):
    """Mark each run latest, current, or outdated.

    ``latest`` is the most recent run of a config. ``current`` is an earlier
    run with the same result. ``outdated`` is an earlier run with a different
    result. ``keep=True`` trusts the ``status`` already on each run, which a
    filtered file carries from the whole file.
    """
    frame = frame.copy()
    if frame.empty or "config_id" not in frame.columns:
        frame["status"] = pd.Series(dtype=object)
        return frame
    if keep and "status" in frame.columns:
        status = frame["status"]
    else:
        status = pd.Series("outdated", index=frame.index, dtype=object)
        for _, index in frame.groupby("config_id", sort=False).groups.items():
            positions = list(index)
            latest = positions[-1]
            status.loc[latest] = "latest"
            latest_key = _outcome_key(frame.loc[latest])
            for position in positions[:-1]:
                if _outcome_key(frame.loc[position]) == latest_key:
                    status.loc[position] = "current"
    frame["status"] = status
    front = [name for name in ("config_id", "status") if name in frame.columns]
    rest = [name for name in frame.columns if name not in front]
    return frame[front + rest]


def _latest_runs(frame):
    """One row per config: the run with status ``latest``."""
    if frame.empty or "config_id" not in frame.columns:
        return frame
    return frame[frame["status"] == "latest"].reset_index(drop=True)


def _latest_run(runs):
    """The last run by timestamp. A missing timestamp is earlier than any stamp.

    Two runs with the same timestamp keep file order, so the later one is the latest.
    """
    if not runs:
        return None

    def key(item):
        index, run = item
        stamp = run.get("timestamp")
        return (stamp if isinstance(stamp, str) else "", index)

    return max(enumerate(runs), key=key)[1]


def _mark_runs(runs):
    """Copies of ``runs`` with ``status`` set from the latest result."""
    runs = [dict(run) for run in runs]
    latest = _latest_run(runs)
    if latest is None:
        return runs
    latest_key = _outcome_key(latest)
    for run in runs:
        if run is latest:
            run["status"] = "latest"
        elif _outcome_key(run) == latest_key:
            run["status"] = "current"
        else:
            run["status"] = "outdated"
    return runs


def _outcome_key(run):
    if isinstance(run, dict):
        values = (run.get(name) for name in _OUTCOME)
    else:
        values = ((run[name] if name in getattr(run, "index", ()) else None) for name in _OUTCOME)
    return tuple(_number(value) for value in values)


def _column(frame, name):
    if name not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def _delta(new, old):
    new, old = _number(new), _number(old)
    if new is None or old is None:
        return None
    return new - old


def _group_key(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        return str(value)
    return value


def _varies(series):
    return len({_group_key(value) for value in series}) > 1
