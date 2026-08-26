"""Result stores.

The recommended source of truth is :class:`JSONLStore`: one JSON object per
line, appended after every trial. Because JSON preserves types, parameters come
back exactly as they went in, so a ``net_arch`` of ``[256, 256]`` is a list of
ints rather than the string ``'[256, 256]'``, and a boolean is a boolean rather
than ``'TRUE'``.

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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Store", "JSONLStore", "CSVStore", "SheetMirror", "MemoryStore", "resolve_store"]


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


def _resolve_path(path, name, suffix):
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
    if resolved.suffix != suffix:
        resolved = resolved / f"{name}{suffix}"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


class JSONLStore(Store):
    """One JSON object per line. Appending never rewrites earlier rows.

    ``path`` may be a directory (the file is then ``<path>/<name>.jsonl``) or a
    full ``.jsonl`` path. The Colab shorthand ``'drive/MyDrive/rl_mining'`` is
    expanded and Drive is mounted automatically if necessary.
    """

    def __init__(self, path, name="study"):
        self.file = _resolve_path(path, name, ".jsonl")

    @property
    def location(self):
        return str(self.file)

    def append(self, record):
        with open(self.file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    def load(self):
        if not self.file.exists():
            return pd.DataFrame()

        records = []
        with open(self.file, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    warnings.warn(
                        f"Skipping unreadable line {number} of {self.file}",
                        stacklevel=2,
                    )
        return pd.DataFrame(records)


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
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in record.items()}
        frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        frame.to_csv(self.file, index=False, quoting=csv.QUOTE_MINIMAL)

    def load(self):
        if not self.file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.file)


class SheetMirror(Store):
    """Best-effort mirror into a Google Sheet, for live viewing in Colab.

    Every failure is downgraded to a warning so a flaky Sheets call can never
    interrupt a multi-hour mining run.
    """

    def __init__(self, url, worksheet_id=None):
        self.url = url
        self.worksheet_id = worksheet_id
        self._handle = None

    @property
    def location(self):
        return self.url

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
        self.records.append(dict(record))

    def load(self):
        return pd.DataFrame(self.records)


def resolve_store(store, name):
    """Accept a Store, a path, or None (meaning a local file next to the notebook)."""
    if store is None:
        return JSONLStore("results", name)
    if isinstance(store, Store):
        return store
    if isinstance(store, (str, Path)):
        return JSONLStore(store, name)
    raise TypeError(f"Cannot interpret {store!r} as a store")
