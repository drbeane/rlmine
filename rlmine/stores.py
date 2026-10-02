"""Result stores.

The recommended source of truth is :class:`JSONStore`: one JSON document
keyed by a 6-digit config id. Each entry holds the full config — environment,
agent, training, evaluation, and runtime — and the history of runs of that
exact config. Because JSON preserves types, values come back exactly as they
went in, so a ``net_arch`` of ``[256, 256]`` is a list of ints rather than
the string ``'[256, 256]'``, and a boolean is a boolean rather than ``'TRUE'``.

A file left over from the older line-per-run format is grouped into this
document the first time it is read.

A Google Sheet is still the nicest way to watch a long run, so
:class:`SheetMirror` can be attached as a read-only-ish view. Mirror failures
are warnings, never errors: losing the view must not lose the results.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import random
import stat
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Store", "JSONStore", "JSONLStore", "CSVStore", "SheetMirror", "MemoryStore", "resolve_store"]


class Store:
    """Append-only record storage."""

    def append(self, record):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @property
    def location(self):
        return type(self).__name__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# Keys that describe a run rather than the config it used.
# Anything else on a flat record is part of the config's identity.
# Retired names stay here so an old file does not treat them as parameters,
# and _order_run drops them instead of writing them again.
RUN_FIELDS = {
    "idx",
    "config_id",
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
    "timestamp",
    "mean_length",
    "success_rate",
    "error",
    "python",
    "gymnasium",
    "stable_baselines3",
    "torch",
    "numpy",
    "time",
    "sb3_version",
    "gym_version",
}

_RETIRED_RUN_FIELDS = {"run_id", "study", "origin", "parent_id", "error"}

_RUN_ORDER = [
    "idx",
    "score",
    "mean",
    "std_dev",
    "minutes",
    "date",
    "timestamp",
    "mean_length",
    "success_rate",
    "python",
    "gymnasium",
    "stable_baselines3",
    "torch",
    "numpy",
]


def format_config_id(value):
    """A parameter-set id as an int from 100000 to 999999.

    ``None`` stays ``None``. The id is always six digits and never starts
    with 0, so a zero-padded string such as ``"009131"`` is not an id.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            return None
        value = int(value)
    if isinstance(value, (int, np.integer)):
        number = int(value)
        if 100_000 <= number <= 999_999:
            return number
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if len(text) == 6 and text.isdigit() and not text.startswith("0"):
        return int(text)
    return None


def new_config_id(taken, rng=None):
    """A random integer from 100000 to 999999 that is not already in ``taken``."""
    rng = rng or random.Random()
    used = set()
    for item in taken:
        pid = format_config_id(item)
        if pid is not None:
            used.add(pid)
    for _ in range(10_000):
        pid = rng.randrange(100_000, 1_000_000)
        if pid not in used:
            return pid
    raise RuntimeError("Could not allocate a 6-digit parameter-set id")


def _remap_config_id(value, used, rng):
    """Keep a valid id, or assign one in 100000..999999 that is still free."""
    current = format_config_id(value)
    if current is not None and current not in used:
        return current
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        candidate = 100_000 + int(text)
        if 100_000 <= candidate <= 999_999 and candidate not in used:
            return candidate
    return new_config_id(used, rng)


def _normalize_config_ids(doc, rng=None):
    """Config ids are ints from 100000 to 999999.

    An id that starts with 0 is replaced.
    """
    rng = rng or random.Random()
    changed = False
    id_map = {}
    used = set()
    pending = []
    for pid in doc:
        current = format_config_id(pid)
        if current is not None and current not in used:
            id_map[pid] = current
            used.add(current)
        else:
            pending.append(pid)

    for pid in pending:
        new_id = _remap_config_id(pid, used, rng)
        id_map[pid] = new_id
        used.add(new_id)
        changed = True

    normalized = {}
    for pid, entry in doc.items():
        new_pid = id_map[pid]
        if str(new_pid) != str(pid):
            changed = True
        normalized[new_pid] = {
            "config": entry.get("config") or {},
            "runs": [dict(run) for run in (entry.get("runs") or [])],
        }
    return normalized, changed


def _canonicalize(value):
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def canonical_params(params):
    """A stable string for comparing parameter sets, independent of key order.

    Values pass through JSON so ``7e-4`` and ``0.0007`` compare equal after
    they have been stored and loaded.
    """
    normalized = json.loads(
        json.dumps(_canonicalize(params), sort_keys=True, default=_json_default)
    )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


_BUCKETS = ("environment", "agent", "training", "evaluation")


def _blank(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return value == ""


def _empty_buckets():
    return {name: {} for name in _BUCKETS}


def _is_bucket_config(config):
    return isinstance(config, dict) and any(name in config for name in _BUCKETS)


def _complete_buckets(config):
    """The four buckets, plus ``runtime`` when it was set.

    Missing buckets are empty dicts so two writers describe the same config
    the same way. Nested values are canonicalized.
    """
    config = _canonicalize(config or {})
    completed = _empty_buckets()
    for name in _BUCKETS:
        section = config.get(name) or {}
        completed[name] = section if isinstance(section, dict) else section
    if not _blank(config.get("runtime")):
        completed["runtime"] = config["runtime"]
    for key, value in config.items():
        if key not in completed:
            completed[key] = value
    return completed


def _order_config(config):
    config = _complete_buckets(config)
    ordered = {name: config[name] for name in _BUCKETS}
    if "runtime" in config:
        ordered["runtime"] = config["runtime"]
    for key, value in config.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def config_key(config):
    """Identity of a config. Key order does not matter; values do."""
    return canonical_params(_order_config(config))


def _flatten_config(config):
    """One flat dict for the history table.

    Agent fields win over training, then evaluation, then environment, so a
    searched ``gamma`` or ``seed`` is not replaced by the evaluation copy.
    The score formula is not a column: the run's numeric score owns that name.
    """
    row = {}
    for name in ("agent", "training", "evaluation", "environment"):
        section = (config or {}).get(name) or {}
        if not isinstance(section, dict):
            if name not in row:
                row[name] = section
            continue
        for key, value in section.items():
            if key == "score" or key in row:
                continue
            row[key] = value
    if not _blank((config or {}).get("runtime")):
        row["runtime"] = config["runtime"]
    return row


def _restore_id_columns(frame):
    """``config_id`` stays a Python int."""
    if frame.empty:
        return frame
    frame = frame.copy()
    if "param_id" in frame.columns and "config_id" not in frame.columns:
        frame = frame.rename(columns={"param_id": "config_id"})
    if "config_id" in frame.columns:
        frame["config_id"] = frame["config_id"].map(format_config_id)
    return frame


def _legacy_id_name(record):
    """Rows written before the rename call the id ``param_id``."""
    if isinstance(record, dict) and "param_id" in record and "config_id" not in record:
        record = {("config_id" if k == "param_id" else k): v for k, v in record.items()}
    return record


def _is_failed_run(run):
    """A run that recorded an error message. ``null`` is not a failure."""
    if not isinstance(run, dict) or "error" not in run:
        return False
    error = run.get("error")
    if error is None:
        return False
    try:
        if pd.isna(error):
            return False
    except (TypeError, ValueError):
        pass
    return not (isinstance(error, str) and not error.strip())


def _order_run(run):
    run = {key: value for key, value in run.items() if key not in _RETIRED_RUN_FIELDS}
    ordered = {}
    for key in _RUN_ORDER:
        if key in run:
            ordered[key] = run[key]
    for key, value in run.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _run_stamp(run):
    for key in ("timestamp", "date"):
        if isinstance(run.get(key), str):
            return run[key]
    return ""


def _assign_idx(doc):
    """Give every run a unique integer ``idx``, 1-based and consecutive.

    Existing indexes are kept when they are all unique integers; runs without
    one are numbered after the highest, oldest first (file order breaks ties).
    If the existing indexes collide, every run is renumbered the same way.
    Returns True if anything changed.
    """
    runs = [run for entry in doc.values() for run in (entry.get("runs") or [])]
    seen = [run["idx"] for run in runs if "idx" in run]
    valid = all(type(i) is int for i in seen) and len(set(seen)) == len(seen)
    changed = False
    if not valid:
        for run in runs:
            run.pop("idx", None)
        changed = True
    todo = [run for run in runs if "idx" not in run]
    if not todo:
        return changed
    top = max((run["idx"] for run in runs if "idx" in run), default=0)
    for offset, run in enumerate(sorted(todo, key=_run_stamp), start=1):
        run["idx"] = top + offset
    return True


def _next_idx(doc):
    return max(
        (run["idx"] for entry in doc.values() for run in (entry.get("runs") or []) if type(run.get("idx")) is int),
        default=0,
    ) + 1


def _next_frame_idx(frame):
    """One more than the highest ``idx`` in a flat table of runs."""
    if frame is None or frame.empty or "idx" not in frame.columns:
        return 1
    top = pd.to_numeric(frame["idx"], errors="coerce").max()
    return 1 if pd.isna(top) else int(top) + 1


def _has_retired_fields(runs):
    return any(key in run for run in runs for key in _RETIRED_RUN_FIELDS)


def split_record(record, param_names=None):
    """Separate a flat row into a bucket config and one run.

    A row that already carries a bucket ``config`` uses that as the identity.
    Otherwise every non-run field is placed on ``agent``. ``runtime`` is part
    of the config in either case, never a field of the run. The same agent on
    CPU and on a GPU are different configs.
    """
    supplied = record.get("config")
    if _is_bucket_config(supplied):
        config = _complete_buckets(supplied)
        if not _blank(record.get("runtime")) and "runtime" not in config:
            config["runtime"] = _canonicalize(record["runtime"])
        skip = set(_flatten_config(config)) - RUN_FIELDS
        skip.update(("config", "config_id", "status", "runtime"))
        run = _order_run(
            {key: _canonicalize(value) for key, value in record.items() if key not in skip}
        )
        return config, run

    if param_names is None:
        names = [
            key
            for key in record
            if key not in RUN_FIELDS and key not in ("config_id", "status", "runtime", "config")
        ]
    else:
        names = [key for key in param_names if key in record]
    config = _empty_buckets()
    config["agent"] = {name: _canonicalize(record[name]) for name in names}
    if not _blank(record.get("runtime")):
        config["runtime"] = _canonicalize(record["runtime"])
    skip = set(names)
    skip.update(("config_id", "status", "runtime", "config"))
    run = _order_run(
        {key: _canonicalize(value) for key, value in record.items() if key not in skip}
    )
    return _order_config(config), run


def _is_study_document(doc):
    if not isinstance(doc, dict):
        return False
    if not doc:
        return True
    return all(
        isinstance(entry, dict) and "runs" in entry and ("config" in entry or "params" in entry)
        for entry in doc.values()
    )


def _set_sort_key(item):
    """Earliest timestamp, so configs stay in the order they were first run."""
    runs = item[1].get("runs") or []
    stamps = [run.get("timestamp") for run in runs if isinstance(run.get("timestamp"), str)]
    return min(stamps) if stamps else ""


def _flat_records(text, parsed, path):
    """Rows from a legacy file: one JSON object, or one object per line."""
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]

    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.warn(
                f"Skipping unreadable line {number} of {path}",
                stacklevel=2,
            )
    return records


def _migrate_entry(entry):
    """One old entry may become several configs when its runs used different machines.

    The first piece keeps the caller's id. A document that is already in the
    bucket shape, with runtime on the config, is returned unchanged.
    """
    import copy

    raw_config = entry.get("config") or {}
    runs_in = [dict(run) for run in (entry.get("runs") or [])]
    already = (
        _is_bucket_config(raw_config)
        and "params" not in entry
        and all("runtime" not in run for run in runs_in)
    )
    if already:
        kept = [_order_run(run) for run in runs_in if not _is_failed_run(run)]
        dropped = len(kept) != len(runs_in)
        return [(raw_config, kept)], _has_retired_fields(runs_in) or dropped

    if _is_bucket_config(raw_config):
        template = _complete_buckets(raw_config)
        agent = dict(template.get("agent") or {})
        for key, value in (entry.get("params") or {}).items():
            agent.setdefault(key, _canonicalize(value))
        template["agent"] = agent
    else:
        template = _empty_buckets()
        template["agent"] = {
            key: _canonicalize(value) for key, value in (entry.get("params") or {}).items()
        }
        if not _blank(raw_config.get("runtime")):
            template["runtime"] = _canonicalize(raw_config["runtime"])

    groups = {}
    for run in runs_in:
        if _is_failed_run(run):
            continue
        runtime = run.pop("runtime", None)
        if _blank(runtime):
            runtime = template.get("runtime")
        else:
            runtime = _canonicalize(runtime)
        groups.setdefault(runtime, []).append(_order_run(run))

    if not groups:
        return [(template, [])], True

    keys = list(groups)
    primary = template.get("runtime")
    if primary in keys:
        keys.remove(primary)
        keys.insert(0, primary)
    pieces = []
    for runtime in keys:
        config = copy.deepcopy(template)
        if _blank(runtime):
            config.pop("runtime", None)
        else:
            config["runtime"] = runtime
        pieces.append((config, groups[runtime]))
    return pieces, True


def _migrate_document(doc, rng=None):
    """Rewrite ``{params, config: {runtime}}`` documents into the bucket shape."""
    rng = rng or random.Random()
    migrated = {}
    changed = False
    for pid, entry in doc.items():
        pieces, piece_changed = _migrate_entry(entry)
        changed = changed or piece_changed
        kept = [(config, runs) for config, runs in pieces if runs]
        if len(kept) != len(pieces):
            changed = True
        if not kept:
            continue
        config, runs = kept[0]
        migrated[pid] = {"config": config, "runs": runs}
        for config, runs in kept[1:]:
            migrated[new_config_id(list(migrated), rng)] = {"config": config, "runs": runs}
            changed = True
    return migrated, changed


def _find_config(doc, config, requested=None):
    """The id of ``config`` in ``doc``, or ``None``."""
    key = config_key(config)
    if (
        requested is not None
        and requested in doc
        and config_key(doc[requested].get("config")) == key
    ):
        return requested
    for pid, entry in doc.items():
        if config_key(entry.get("config")) == key:
            return pid
    return None


def group_flat_records(records, param_names=None, rng=None):
    """Collapse flat rows into ``{config_id: {config, runs}}``.

    Rows with the same config, including runtime, share one id. Ids already
    stored on a row are kept when they do not collide.
    """
    rng = rng or random.Random()
    doc = {}
    for record in records:
        if _is_failed_run(record):
            continue
        record = _legacy_id_name(record)
        config, run = split_record(record, param_names)
        requested = format_config_id(record.get("config_id"))
        pid = _find_config(doc, config, requested)
        if pid is None:
            pid = requested if requested and requested not in doc else new_config_id(doc, rng)
            doc[pid] = {"config": _order_config(config), "runs": []}
        doc[pid]["runs"].append(run)
    return doc


def _resolve_path(path, name, suffix, extra_suffixes=()):
    """Expand a Colab Drive shorthand and mount Drive if needed."""
    text = str(path)

    if text.startswith("drive/"):
        text = "/content/" + text

    if text.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception:
            warnings.warn(
                "Could not mount Google Drive; results will be written to the "
                "ephemeral Colab disk and lost when the runtime restarts.",
                stacklevel=3,
            )

    resolved = Path(text)
    if resolved.suffix not in ((suffix,) + tuple(extra_suffixes)):
        resolved = resolved / f"{name}{suffix}"
        for old in extra_suffixes:
            legacy = resolved.with_suffix(old)
            if not resolved.exists() and legacy.exists():
                resolved = legacy
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class JSONStore(Store):
    """Configs keyed by a random 6-digit id.

    The id is an integer from 100000 to 999999, so it never starts with 0.
    JSON object keys are strings, so the file stores the decimal form of that
    integer.

    The file is one JSON object. Each key is one config::

        {
          "184203": {
            "config": {
              "environment": {"id": "BipedalWalker-v3", "n_envs": 16},
              "agent": {"algo": "A2C", "gamma": 0.99, "seed": 1},
              "training": {"timesteps": 1000000, "eval_freq": 1000},
              "evaluation": {"episodes": 50, "gamma": 1.0, "score": "mean_minus_std", "digits": 4},
              "runtime": "Tesla T4"
            },
            "runs": [
                {"score": 12.0, "timestamp": "2026-08-26T01:06:27", "torch": "2.11.0"}
            ]
          }
        }

    The id covers the whole config. Fixed settings such as ``eval_episodes``
    are part of it, and so is ``runtime``: the same agent on CPU and on a GPU
    are different configs. Package versions stay on the run. A second run of
    the same config is appended to ``runs``. ``load`` flattens the document
    back to one row per run so the rest of the study can keep treating history
    as a table.

    ``path`` may be a directory (the file is then ``<path>/<name>.json``) or a
    full ``.json`` path. An existing ``<name>.jsonl`` from an earlier version
    is still found and used as is, and a full ``.jsonl`` path is accepted.
    The Colab shorthand ``'drive/MyDrive/rl_mining'`` is
    expanded and Drive is mounted automatically if necessary.

    An older file with one flat JSON object per line is rewritten into this
    shape the first time it is read. The rewrite goes to a temporary file and
    then replaces the original, so a crash mid-write leaves the previous
    document in place.
    """

    groups_param_sets = True

    def __init__(self, path, name="study"):
        self.file = _resolve_path(path, name, ".json", extra_suffixes=(".jsonl",))

    @property
    def location(self):
        return str(self.file)

    def id_for_config(self, config):
        """The id of ``config``, allocating one from 100000 to 999999 if it is new.

        Does not write. The caller passes the id back on the record so the
        following append stores the run under the same key.
        """
        config = _order_config(config)
        doc = self._read_document()
        found = _find_config(doc, config)
        if found is not None:
            return format_config_id(found)
        return new_config_id(doc)

    def config_for_id(self, config_id):
        """A copy of the config stored under ``config_id``, or ``None``."""
        import copy

        pid = format_config_id(config_id)
        if pid is None:
            return None
        doc = self._read_document()
        entry = doc.get(pid) or doc.get(str(pid))
        if not isinstance(entry, dict):
            return None
        return copy.deepcopy(entry.get("config") or {})

    def append(self, record, param_names=None):
        if _is_failed_run(record):
            return format_config_id(record.get("config_id"))
        doc = self._read_document()
        config, run = split_record(record, param_names)
        requested = format_config_id(record.get("config_id"))
        pid = _find_config(doc, config, requested)
        if pid is None:
            pid = requested if requested and requested not in doc else new_config_id(doc)
            doc[pid] = {"config": _order_config(config), "runs": []}
        run["idx"] = _next_idx(doc)
        doc[pid]["runs"].append(_order_run(run))
        self._write_document(doc)
        record["idx"] = run["idx"]
        record["config_id"] = pid
        return pid

    def delete_runs(self, run_ids):
        """Remove runs by ``idx``. A config left with no runs is removed too.

        Returns ``{"runs": [...], "configs": [...], "missing": [...]}``: the run
        ids deleted, the config ids removed because they became empty, and the
        requested run ids that do not exist. Nothing is written if none matched.
        """
        wanted = _as_int_set(run_ids)
        doc = self._read_document()
        deleted, emptied = [], []
        for pid in list(doc):
            runs = doc[pid].get("runs") or []
            kept = [run for run in runs if run.get("idx") not in wanted]
            if len(kept) == len(runs):
                continue
            deleted.extend(run["idx"] for run in runs if run.get("idx") in wanted)
            if kept:
                doc[pid]["runs"] = kept
            else:
                del doc[pid]
                emptied.append(format_config_id(pid))
        if deleted:
            self._write_document(doc)
        return {
            "runs": sorted(deleted),
            "configs": emptied,
            "missing": sorted(wanted - set(deleted)),
        }

    def delete_duplicates(self, dry_run=False):
        """Remove runs that repeat an earlier one, keeping the most recent.

        Two runs of a config are duplicates when everything but ``idx``,
        ``date``, ``timestamp`` and ``minutes`` matches: the score and its
        components, and the package versions. With ``dry_run`` nothing is
        deleted. Returns what :meth:`delete_runs` returns.
        """
        ignore = {"idx", "date", "timestamp", "minutes"}
        doomed = []
        for entry in self._read_document().values():
            newest = {}
            for run in sorted(entry.get("runs") or [], key=lambda r: (_run_stamp(r), r.get("idx", 0))):
                key = canonical_params({k: v for k, v in run.items() if k not in ignore})
                if key in newest:
                    doomed.append(newest[key]["idx"])
                newest[key] = run
        if dry_run:
            return {"runs": sorted(doomed), "configs": [], "missing": []}
        return self.delete_runs(doomed) if doomed else {"runs": [], "configs": [], "missing": []}

    def delete_configs(self, config_ids):
        """Remove whole configs, with all their runs.

        Returns ``{"configs": [...], "runs": [...], "missing": [...]}``.
        """
        wanted = _as_int_set(config_ids)
        doc = self._read_document()
        removed, runs = [], []
        for pid in list(doc):
            if format_config_id(pid) in wanted:
                runs.extend(run["idx"] for run in doc[pid].get("runs") or [] if "idx" in run)
                removed.append(format_config_id(pid))
                del doc[pid]
        if removed:
            self._write_document(doc)
        return {
            "configs": sorted(removed),
            "runs": sorted(runs),
            "missing": sorted(wanted - set(removed)),
        }

    def load(self):
        doc = self._read_document()
        if not doc:
            return pd.DataFrame()

        rows = []
        for pid, entry in doc.items():
            flat = _flatten_config(entry.get("config") or {})
            for run in entry.get("runs") or []:
                row = dict(flat)
                row.update(run)
                row["config_id"] = format_config_id(pid)
                rows.append(row)
        rows.sort(key=lambda row: row.get("timestamp") if isinstance(row.get("timestamp"), str) else "")
        frame = pd.DataFrame(rows)
        return _restore_id_columns(frame)

    def _read_document(self):
        if not self.file.exists():
            return {}
        text = self.file.read_text(encoding="utf-8").strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict) and _is_study_document(parsed):
            doc, changed = _migrate_document(parsed)
            doc, ids_changed = _normalize_config_ids(doc)
            idx_changed = _assign_idx(doc)
            if changed or ids_changed or idx_changed:
                self._write_document(doc)
            return doc

        records = _flat_records(text, parsed, self.file)
        doc = group_flat_records(records)
        doc, _ = _normalize_config_ids(doc)
        _assign_idx(doc)
        self._write_document(doc)
        return doc

    def _write_document(self, doc):
        ordered = {}
        for pid, entry in sorted(doc.items(), key=_set_sort_key):
            ordered[pid] = {
                "config": _order_config(entry.get("config")),
                "runs": [_order_run(run) for run in (entry.get("runs") or [])],
            }
        payload = json.dumps(ordered, indent=2, default=_json_default) + "\n"
        temporary = self.file.with_name(self.file.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        _replace_file(temporary, self.file)


def _as_int_set(values):
    """Ids from one int or an iterable of ints (numpy ints and ``"123456"`` are fine)."""
    if isinstance(values, (int, np.integer, str)):
        values = [values]
    out = set()
    for value in values:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            raise ValueError(f"Not an integer id: {value!r}") from None
    return out


def _replace_file(temporary, dest):
    """Move ``temporary`` onto ``dest``.

    The replace is atomic. On Windows a file inside Dropbox, or one open in
    an editor, is often locked for a moment, and ``os.replace`` raises
    ``PermissionError`` (WinError 5). Retry, then overwrite the destination
    in place so a finished trial is still recorded.
    """
    delay = 0.05
    for attempt in range(6):
        try:
            os.replace(temporary, dest)
            return
        except PermissionError:
            if dest.exists():
                with contextlib.suppress(OSError):
                    os.chmod(dest, stat.S_IWRITE | stat.S_IREAD)
            if attempt == 5:
                break
            time.sleep(delay)
            delay = min(delay * 2, 0.5)
    dest.write_bytes(temporary.read_bytes())
    with contextlib.suppress(OSError):
        temporary.unlink()


# Old name, from when the file was called .jsonl. Kept so existing notebooks run.
JSONLStore = JSONStore


class CSVStore(Store):
    """A flat CSV, for when opening the file directly in Excel matters more
    than type fidelity. Lists and booleans come back as strings, which
    ``Space.parse_row`` knows how to undo."""

    def __init__(self, path, name="study"):
        self.file = _resolve_path(path, name, ".csv")

    @property
    def location(self):
        return str(self.file)

    def append(self, record):
        existing = self.load()
        record["idx"] = _next_frame_idx(existing)
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in record.items()}
        frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        frame.to_csv(self.file, index=False, quoting=csv.QUOTE_MINIMAL)

    def load(self):
        if not self.file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.file)


class SheetMirror(Store):
    """Best-effort mirror into a Google Sheet, for live viewing in Colab.

    Construction opens the sheet and issues a no-op write so Colab's OAuth
    prompt happens immediately, not after the first (possibly hours-long)
    trial. Every later failure is a warning, never an error, so a flaky
    Sheets call cannot interrupt a mining run.
    """

    def __init__(self, url, worksheet_id=None):
        self.url = url
        self.worksheet_id = worksheet_id
        self._handle = None
        self.connect()

    @property
    def location(self):
        return self.url

    def connect(self):
        """Authenticate and open the sheet now, before any trial runs."""
        try:
            sheet = self._sheet()
            self._probe_write(sheet)
        except ImportError:
            return
        except Exception as exc:
            warnings.warn(
                f"Sheet mirror could not connect ({exc}); "
                "you may be asked to authenticate after the first trial.",
                stacklevel=2,
            )

    def _sheet(self):
        if self._handle is None:
            from google.colab import sheets

            kwargs = {"url": self.url, "backend": "pandas", "display": False}
            if self.worksheet_id is not None:
                kwargs["worksheet_id"] = str(self.worksheet_id)
            # InteractiveSheet always prints the spreadsheet URL on
            # construction, even with display=False.
            with contextlib.redirect_stdout(io.StringIO()):
                self._handle = sheets.InteractiveSheet(**kwargs)
        return self._handle

    def _probe_write(self, sheet):
        """No-op write so Colab requests write scope before the first trial."""
        worksheet = sheet.worksheet
        try:
            cell = worksheet.acell("A1", value_render_option="FORMULA")
        except TypeError:
            cell = worksheet.acell("A1")
        current = "" if getattr(cell, "value", None) is None else cell.value
        worksheet.update(range_name="A1", values=[[current]])

    @staticmethod
    def _to_cell(value):
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return value

    def _push(self, sheet, frame):
        """Write ``frame`` through gspread with named arguments.

        Colab's ``InteractiveSheet.update`` still calls
        ``worksheet.update(location, data)``, which gspread 6 warns about on
        every append. Named arguments work on both gspread 5 and 6.
        """
        values = [list(frame.columns)]
        for _, row in frame.iterrows():
            values.append([self._to_cell(v) for v in row.tolist()])
        sheet.worksheet.clear()
        sheet.worksheet.update(range_name="A1", values=values)

    def append(self, record):
        try:
            sheet = self._sheet()
            frame = sheet.as_df()
            for column in record:
                if column not in frame.columns:
                    frame[column] = None
            position = len(frame)
            frame.loc[position] = None
            for column, value in record.items():
                frame.loc[position, column] = self._to_cell(value)
            self._push(sheet, frame)
        except Exception as exc:
            warnings.warn(f"Sheet mirror failed ({exc}); results are still saved.", stacklevel=2)

    def load(self):
        try:
            return self._sheet().as_df()
        except Exception as exc:
            warnings.warn(f"Could not read sheet ({exc}).", stacklevel=2)
            return pd.DataFrame()


class MemoryStore(Store):
    """In-process storage, used by the tests and for dry runs."""

    def __init__(self, records=None):
        self.records = list(records or [])

    def append(self, record):
        record["idx"] = _next_frame_idx(pd.DataFrame(self.records))
        self.records.append(dict(record))

    def load(self):
        return pd.DataFrame(self.records)


def _is_store(obj):
    """True for a Store, including one built by a previous import of rlmine."""
    return callable(getattr(obj, "append", None)) and callable(getattr(obj, "load", None))


def resolve_store(store, name):
    """Accept a Store, a path, or None (meaning a local file next to the notebook)."""
    if store is None:
        return JSONStore("results", name)
    if isinstance(store, (str, Path)):
        return JSONStore(store, name)
    if _is_store(store):
        return store
    raise TypeError(
        f"Cannot interpret {store!r} as a store. Pass a JSONStore, a path, "
        "or None. If you just reloaded rlmine, re-run the cell that constructs "
        "JSONStore / SheetMirror."
    )
