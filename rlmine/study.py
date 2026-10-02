"""The Study object: run one trial from a config."""

from __future__ import annotations

import copy
import inspect
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from .results import Results
from .samplers import resolve_samplers
from .scoring import resolve_score_fn, round_to_digits, score_digits, score_fn_name
from .stores import (
    _BUCKETS,
    _flatten_config,
    _order_config,
    config_key,
    format_config_id,
    new_config_id,
    resolve_store,
)
from .utils import (
    Pruned,
    _quiet_third_party_warnings,
    env_info,
    now_iso,
    today,
)

__all__ = ["Study"]

METRIC_COLUMNS = ["score", "mean", "std_dev", "minutes", "mean_length", "success_rate"]

_TABLE_LEAD = ["config_id", "score", "mean", "std_dev", "minutes", "date", "timestamp"]
_TABLE_HIDE = {
    "error",
    "python",
    "numpy",
    "gymnasium",
    "stable_baselines3",
    "torch",
    "config",
}


class Study:
    """Runs one trial from a config and records the result.

    ``config`` is the default configuration, in the same shape as the config
    stored for each result: ``environment``, ``agent``, ``training``, and
    ``evaluation``. ``runtime`` is not part of that default. It is recorded
    from the machine when the trial runs.

    ``trial`` is a callable taking that config and returning an
    ``evaluate``-style stats dict with ``mean_return`` and ``stdev_return``.
    It may take a second argument, a context dict carrying ``config_id``.

        study.run()
        study.run(agent={"gamma": 0.95}, training={"timesteps": 50_000})
        study.run(n=10, agent={"initial_lr": s.loguniform(1e-2, 1e-4, sig=2)})
    """

    def __init__(self, name, trial, config, store=None, mirror=None, score_fn=None, verbose=True):
        self.name = name
        self.trial = trial
        self.config = _copy_config(config)
        self.store = resolve_store(store, name)
        self.mirrors = _as_list(mirror)
        self._score_fn_given = score_fn is not None
        self.score_fn = resolve_score_fn(score_fn)
        self.verbose = verbose
        self._trial_wants_context = _accepts_two_args(trial)
        _quiet_third_party_warnings()

    def __repr__(self):
        return (
            f"Study(name={self.name!r}, runs={len(self.history())}, "
            f"store={self.store.location!r})"
        )

    def run(
        self,
        n=1,
        *,
        config_id=None,
        parent_sampling="uniform",
        seed=None,
        prune=None,
        prune_minutes=None,
        **overrides,
    ):
        """Run ``n`` trials.

        With no arguments, the study's config is used as written. Pass a
        section to replace fields inside it. Nested dicts are merged, so
        ``run(agent={"gamma": 0.95})`` changes gamma and leaves the rest of
        the agent as it is.

        ``config_id`` starts from a stored config instead. Its environment,
        agent, training, and evaluation replace the study's ``config``
        entirely. The id's ``runtime`` does not carry over: the trial records
        the machine it runs on, so replaying a GPU config on a CPU is a
        different config. Overrides apply on top, exactly as they do to the
        study's own config.

        ``config_id`` can be a list of ids. Each is run in turn (``n`` trials
        apiece) from its stored config. Each id's own ``runtime`` is dropped,
        so the ids may come from different runtimes; each trial records the
        machine it runs on.

            study.run(config_id=[927170, 415882])

        ``config_id`` can also be a :class:`~rlmine.results.Results`, usually
        filtered. Each trial draws its own parent from the configs in it and
        starts from that config, so ``n=20`` mutates up to 20 different
        parents. ``parent_sampling`` sets the draw: ``"uniform"`` (default)
        gives every config the same chance; ``"roulette"`` weights each config
        by its latest score, shifted so the lowest-scoring config has weight
        zero (equal scores fall back to uniform). Configs with no score are
        never drawn by roulette. The header shows which parent each trial used.

            best = Results(path).filter(run.score > 200)
            study.run(n=20, config_id=best, parent_sampling="roulette",
                      agent={"initial_lr": m.proportional(0.2, sig=2)})

        Each override field can be:

        - a value: ``agent={"gamma": 0.95}``
        - a sampler from :mod:`rlmine.samplers`, which draws a new value:
          ``agent={"initial_lr": s.loguniform(1e-4, 1e-2, sig=2)}``
        - a mutation from :mod:`rlmine.mutations`, which changes the value
          the config already has: ``agent={"initial_lr": m.proportional(0.2)}``

        Samplers and mutations are not values yet. Each trial resolves its
        own, always from the base config and never from the previous trial,
        and the resulting number is what is trained and stored. ``seed``
        seeds those draws only. An agent seed inside the config is separate.

        ``prune`` and ``prune_minutes`` abandon a trial that is not doing well
        enough. Each is a list of ``(when, min_reward)`` pairs, where ``when``
        is a timestep count or minutes of training. They go to the trial in
        its context dict, not into the config, so they are not part of the
        key. A pruned trial is not recorded. ``sb3_trial`` honours them; a
        custom trial can read ``context["prune"]`` or raise
        :class:`~rlmine.Pruned` itself.

            study.run(n=20, agent={"initial_lr": s.loguniform(1e-4, 1e-2, sig=2)},
                      prune=[(200_000, 0)], prune_minutes=[(20, 0)])

        With ``verbose`` on, each trial prints its id, the values drawn or
        mutated, and the score. The returned rows stay available to code.
        A notebook does not render them as a table.

            study.run()
            study.run(training={"timesteps": 50_000})
            study.run(agent={"gamma": 0.95, "n_steps": 8})
            study.run(n=10, seed=0, agent={"initial_lr": s.loguniform(1e-2, 1e-4, sig=2)})
            study.run(n=10, config_id=927170, agent={"initial_lr": m.proportional(0.2, sig=2)})
        """
        count = _run_count(n)
        rng = np.random.default_rng(seed)
        if parent_sampling not in ("uniform", "roulette"):
            raise ValueError("parent_sampling must be 'uniform' or 'roulette'")
        pool = None
        if isinstance(config_id, Results):
            pool = _ParentPool(config_id, parent_sampling)
            bases = [(None, None)]
        elif isinstance(config_id, (list, tuple)):
            if not config_id:
                raise ValueError("config_id list is empty")
            bases = [
                (self._stored_config(cid), format_config_id(cid)) for cid in config_id
            ]
        elif config_id is None:
            bases = [(self.config, None)]
        else:
            bases = [(self._stored_config(config_id), format_config_id(config_id))]
        records = []
        logs = []
        for base, source in bases:
            for _ in range(count):
                if pool is not None:
                    pid = pool.draw(rng)
                    base, source = pool.config(pid), pid
                config, score_fn, digits, info, runtime, drawn = self._prepare(
                    base, overrides, rng
                )
                if self.verbose and records:
                    print()
                record, text = self._execute(
                    config, score_fn, digits, info, runtime, drawn, source,
                    prune=prune, prune_minutes=prune_minutes,
                )
                records.append(record)
                logs.append(text)
        return _RunFrame(records, log="\n\n".join(logs), echoed=self.verbose)

    def rerun(self, config_id=None, top=None):
        """Replay stored configs on this runtime and report whether they drifted.

        Pass ``config_id`` to replay one config, or ``top`` to replay the ``k``
        best configs for the current runtime, ranked by latest score. Only
        configs recorded on this runtime are replayed: a config id from another
        runtime is reported and nothing runs.

        Each replay is a new run under the same id. It is compared with the
        config's latest score from before the replay. Returns one row per
        replay with ``previous``, ``new``, ``delta``, and ``drifted``.

            study.rerun(config_id=927170)
            study.rerun(top=5)
        """
        if (config_id is None) == (top is None):
            raise ValueError("Pass exactly one of config_id or top.")
        if top is not None:
            if isinstance(top, bool) or not isinstance(top, (int, np.integer)) or int(top) < 1:
                raise ValueError(f"top must be a positive integer, got {top!r}")

        latest = self._latest_by_config()
        runtime = env_info()["runtime"]

        if config_id is not None:
            pid = format_config_id(config_id)
            if pid is None:
                raise ValueError(
                    f"config_id must be an integer from 100000 to 999999, got {config_id!r}"
                )
            if pid not in latest.index:
                raise ValueError(f"No scored config {pid} in {self.store.location}")
            stored = latest.loc[pid, "runtime"]
            if stored != runtime:
                print(f"Config {pid} was recorded on runtime {stored!r}, not the current "
                      f"runtime {runtime!r}. Nothing was run.")
                return pd.DataFrame(columns=_RERUN_COLUMNS)
            ids = [pid]
        else:
            here = latest[latest["runtime"] == runtime]
            if here.empty:
                print(f"No configs recorded on the current runtime {runtime!r}. Nothing was run.")
                return pd.DataFrame(columns=_RERUN_COLUMNS)
            ids = list(here.sort_values("score", ascending=False, kind="mergesort").index[: int(top)])

        rows = []
        for pid in ids:
            before = latest.loc[pid]
            frame = self.run(config_id=pid)
            record = frame.iloc[0]
            new = _maybe_float(record.get("score"))
            old = _maybe_float(before["score"])
            delta = _safe_delta(new, old)
            rows.append(
                {
                    "config_id": pid,
                    "previous": old,
                    "previous_date": before.get("date"),
                    "new": new,
                    "date": record.get("date"),
                    "delta": delta,
                    "drifted": None if delta is None else delta != 0,
                }
            )
        report = pd.DataFrame(rows, columns=_RERUN_COLUMNS)
        print(_format_rerun(report))
        return report

    def _latest_by_config(self):
        """The latest scored run of each config, indexed by config id."""
        history = self.history(ok_only=True)
        if history.empty or "config_id" not in history.columns:
            return pd.DataFrame(columns=["score", "date", "runtime"])
        if "timestamp" in history.columns:
            history = history.sort_values("timestamp", kind="mergesort", na_position="first")
        if "runtime" not in history.columns:
            history = history.assign(runtime=None)
        return history.groupby("config_id", sort=False).tail(1).set_index("config_id")

    def _stored_config(self, config_id):
        """The config stored under ``config_id``, without ``runtime``."""
        pid = format_config_id(config_id)
        if pid is None:
            raise ValueError(
                f"config_id must be an integer from 100000 to 999999, got {config_id!r}"
            )
        finder = getattr(self.store, "config_for_id", None)
        if callable(finder):
            stored = finder(pid)
        else:
            stored = self._config_from_history(pid)
        if not isinstance(stored, dict):
            raise ValueError(f"No config {pid} in {self.store.location}")
        return {
            name: copy.deepcopy(stored.get(name) or {}) for name in _BUCKETS
        }

    def _config_from_history(self, pid):
        history = self.history()
        if history.empty or "config_id" not in history.columns or "config" not in history.columns:
            return None
        for _, row in history.iterrows():
            stored = row["config"]
            if row["config_id"] == pid and isinstance(stored, dict):
                return stored
        return None

    def _prepare(self, base, overrides, rng):
        drawn = []
        config = resolve_samplers(
            merge_config(base, overrides), rng, drawn=drawn, base=base
        )
        evaluation = dict(config.get("evaluation") or {})
        if self._score_fn_given:
            evaluation["score"] = score_fn_name(self.score_fn)
            score_fn = self.score_fn
        else:
            evaluation.setdefault("score", score_fn_name(self.score_fn))
            score_fn = resolve_score_fn(evaluation["score"])
        config["evaluation"] = evaluation
        digits = score_digits(evaluation)

        info = env_info()
        runtime = info.pop("runtime")
        config = _order_config({**config, "runtime": runtime})
        return config, score_fn, digits, info, runtime, drawn

    def _execute(
        self, config, score_fn, digits, info, runtime, drawn, source=None,
        prune=None, prune_minutes=None,
    ):
        _quiet_third_party_warnings()
        config_id = self._config_id_for(config)
        lines = [_format_header(config_id, drawn, source)]
        if self.verbose:
            print("\n".join(lines))

        context = {"config_id": config_id}
        if prune:
            context["prune"] = prune
        if prune_minutes:
            context["prune_minutes"] = prune_minutes
        started = time.time()
        error, pruned, stats = None, None, {}
        try:
            supplied = copy.deepcopy(config)
            if self._trial_wants_context:
                stats = self.trial(supplied, context) or {}
            else:
                stats = self.trial(supplied) or {}
        except KeyboardInterrupt:
            raise
        except Pruned as exc:
            error = pruned = str(exc) or "pruned"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                traceback.print_exc()

        minutes = round((time.time() - started) / 60, 2)

        score = None
        if error is None:
            try:
                # Round every element of the stats first, then score those
                # values, so the stored score is the formula applied to the
                # numbers that are stored with it.
                stats = round_to_digits(stats, digits)
                score = round_to_digits(float(score_fn(stats)), digits)
            except Exception as exc:
                error = f"score_fn failed: {type(exc).__name__}: {exc}"

        record = {
            "config_id": config_id,
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

        record.update(_jsonable(_flatten_config(config)))
        record.update(info)
        record["runtime"] = runtime
        record["config"] = config

        if error is None:
            self._write(record)
            lines.append(_format_score(record))
        elif pruned is not None:
            lines.append(f"  PRUNED after {minutes} min: {pruned}")
        else:
            lines.append(f"  FAILED after {minutes} min: {error}")
        if self.verbose:
            print(lines[-1])

        return record, "\n".join(lines)

    def _write(self, record, mirror=True):
        self.store.append(record)
        if not mirror:
            return
        # The sheet stays a flat table. The nested config is the JSON document's
        # identity and would otherwise land in one unreadable cell.
        flat = {key: value for key, value in record.items() if key != "config"}
        for target in self.mirrors:
            try:
                target.append(flat)
            except Exception as exc:
                warnings.warn(f"Mirror write failed ({exc}).", stacklevel=2)

    def _config_id_for(self, config):
        """The integer id of this config, from 100000 to 999999.

        Two configs match when every section matches, including fixed settings
        such as ``evaluation.episodes``, and ``runtime`` matches. The same
        agent on CPU and on a GPU are different configs. A repeat reuses the id.
        """
        finder = getattr(self.store, "id_for_config", None)
        if callable(finder):
            return finder(config)

        history = self.history()
        target = config_key(config)
        taken = []
        if not history.empty and "config_id" in history.columns:
            seen = set()
            for _, row in history.iterrows():
                pid = format_config_id(row["config_id"])
                if pid is None or pid in seen:
                    continue
                seen.add(pid)
                taken.append(pid)
                stored = row.get("config") if "config" in row.index else None
                if isinstance(stored, dict) and config_key(stored) == target:
                    return pid
        return new_config_id(taken)

    def history(self, ok_only=False):
        """Every recorded run, in the order it was stored.

        A ``config_id`` from 100000 to 999999 names the config. Repeated scores
        for that config are separate rows, with their timestamps and package
        versions.
        """
        frame = self.store.load()
        if frame.empty:
            return frame

        frame = frame.copy()
        if "config_id" in frame.columns:
            frame["config_id"] = frame["config_id"].map(format_config_id)
        if "score" in frame.columns:
            frame["score"] = pd.to_numeric(frame["score"], errors="coerce")

        if ok_only:
            if "error" in frame.columns:
                failed = frame["error"].notna() & (frame["error"].astype(str).str.strip() != "")
                frame = frame[~failed]
            frame = frame[frame["score"].notna()]
        return frame

    def table(self, n=None, sort="score", ascending=False, params=True, extra=()):
        """A compact, readable view of results."""
        frame = self.history()
        if frame.empty:
            print(f"No results yet in {self.store.location}")
            return frame

        columns = [c for c in _TABLE_LEAD if c in frame.columns]
        if params:
            columns += [
                c for c in frame.columns if c not in columns and c not in _TABLE_HIDE
            ]
        columns += [c for c in extra if c in frame.columns and c not in columns]

        view = frame[columns]
        if sort in view.columns:
            view = view.sort_values(sort, ascending=ascending, na_position="last")
        if n is not None:
            view = view.head(int(n))

        # Round the metrics only. Config values stay as stored, so a learning
        # rate of 1e-5 is not flattened to zero.
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

    def drift_report(self):
        """Compare each later score of a config with the one before it.

        A config with a single score is omitted. Package versions are included
        so a move in the score can be lined up with a library change.
        """
        history = self.history()
        if history.empty or "config_id" not in history.columns:
            return pd.DataFrame()

        rows = []
        for config_id, group in history.groupby("config_id", sort=False):
            ordered = group.reset_index(drop=True)
            if len(ordered) < 2:
                continue
            for i in range(1, len(ordered)):
                earlier = ordered.iloc[i - 1]
                later = ordered.iloc[i]
                rows.append(
                    {
                        "config_id": config_id,
                        "earlier_score": earlier.get("score"),
                        "later_score": later.get("score"),
                        "delta": _safe_delta(later.get("score"), earlier.get("score")),
                        "earlier_date": earlier.get("date"),
                        "later_date": later.get("date"),
                        "sb3_then": earlier.get("stable_baselines3"),
                        "sb3_now": later.get("stable_baselines3"),
                        "gym_then": earlier.get("gymnasium"),
                        "gym_now": later.get("gymnasium"),
                    }
                )
        return pd.DataFrame(rows).round(3)


def _run_count(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return int(n)


def merge_config(config, overrides):
    """Copy ``config`` and fold section overrides into it.

    ``overrides`` names sections (``agent={"gamma": 0.95}``). Dict values
    merge recursively. Anything else replaces the field.
    """
    unknown = [key for key in overrides if key not in _BUCKETS]
    if unknown:
        names = ", ".join(_BUCKETS)
        raise TypeError(
            f"Unknown config section {unknown[0]!r}. "
            f"Pass one of: {names}."
        )
    for key, value in overrides.items():
        if not isinstance(value, dict):
            raise TypeError(
                f"{key} override must be a dict of values, "
                f"got {type(value).__name__}."
            )

    merged = copy.deepcopy(config)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _copy_config(config):
    if not isinstance(config, dict):
        raise TypeError(
            "config must be a dict with environment, agent, training, "
            "and evaluation sections."
        )
    unknown = [key for key in config if key not in _BUCKETS and key != "runtime"]
    if unknown:
        names = ", ".join(_BUCKETS)
        raise TypeError(
            f"Unknown config section {unknown[0]!r}. Use one of: {names}."
        )
    for name in _BUCKETS:
        if name in config and config[name] is not None and not isinstance(config[name], dict):
            raise TypeError(f"config {name} must be a dict.")
    copied = copy.deepcopy(config)
    copied.pop("runtime", None)
    return copied


def _deep_merge(base, patch):
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


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


class _RunFrame(pd.DataFrame):
    """Rows from one ``run`` call.

    The live log is printed as each trial finishes. Showing the object in a
    notebook prints that same text, and does not draw a table.
    """

    _metadata = ["_log", "_echoed"]

    def __init__(self, data, log, echoed):
        super().__init__(data)
        self._log = log
        self._echoed = echoed

    def __repr__(self):
        return self._log

    def _repr_html_(self):
        return None

    def _ipython_display_(self):
        if self._echoed:
            self._echoed = False
            return
        print(self._log)


class _ParentPool:
    """Configs of a ``Results`` that ``run`` draws parents from."""

    def __init__(self, results, how):
        frame = results._config_frame()
        if frame.empty:
            raise ValueError("The Results has no configs to draw a parent from.")
        self.results = results
        self.ids = [format_config_id(pid) for pid in frame["config_id"]]
        scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=float)
        if how == "uniform":
            self.p = None
            return
        ok = np.isfinite(scores)
        if not ok.any():
            raise ValueError("No config in the Results has a score for roulette sampling.")
        weights = np.where(ok, scores - scores[ok].min(), 0.0)
        if weights.sum() <= 0:
            weights = ok.astype(float)
        self.p = weights / weights.sum()

    def draw(self, rng):
        return self.ids[int(rng.choice(len(self.ids), p=self.p))]

    def config(self, pid):
        stored = self.results.get_config(pid)
        return {name: copy.deepcopy(stored.get(name) or {}) for name in _BUCKETS}


_SCORE = "\033[1;31m"
_RESET = "\033[0m"


def _format_header(config_id, drawn, source=None):
    """``[id]`` and, on the same line, where it came from and what changed."""
    parts = [f"[{config_id}]"]
    if source is not None:
        parts.append(f"(from {source})")
    sampled = _format_sampled(drawn)
    if sampled:
        parts.append(sampled)
    return "  ".join(parts)


def _format_sampled(drawn):
    if not drawn:
        return ""
    leaves = [path[-1] for path, _ in drawn]
    return "  ".join(
        f"{_sample_name(path, leaves)}={_fmt_value(value)}"
        for path, value in drawn
    )


def _sample_name(path, leaves):
    leaf = path[-1]
    if leaves.count(leaf) > 1:
        return ".".join(str(part) for part in path)
    if len(path) > 1:
        return ".".join(str(part) for part in path[1:])
    return str(leaf)


def _fmt_value(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), "g")
    return str(value)


_RERUN_COLUMNS = ["config_id", "previous", "previous_date", "new", "date", "delta", "drifted"]


def _format_rerun(report):
    lines = []
    for _, row in report.iterrows():
        if row["delta"] is None or pd.isna(row["delta"]):
            lines.append(f"[{row['config_id']}] could not be compared (the replay failed or has no score).")
        elif row["drifted"]:
            lines.append(
                f"[{row['config_id']}] DRIFTED: {_fmt_metric(row['previous'])} -> "
                f"{_fmt_metric(row['new'])} ({row['delta']:+.2f})"
            )
        else:
            lines.append(f"[{row['config_id']}] no drift: {_fmt_metric(row['new'])}")
    return "\n".join(lines)


def _format_score(record):
    score = _fmt_metric(record.get("score"))
    parts = [f"score={_SCORE}{score}{_RESET}"]
    parts += [
        f"{key}={_fmt_metric(record.get(key))}"
        for key in ("mean", "std_dev", "minutes")
    ]
    return "  " + "  ".join(parts)


def _fmt_metric(value):
    number = _maybe_float(value)
    if number is None:
        return "nan"
    return f"{round(number, 2):.2f}"
