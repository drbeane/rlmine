"""Small shared helpers: rounding, run ids, environment metadata, display."""

from __future__ import annotations

import os
import random
import sys
import warnings
from datetime import datetime
from math import floor, log10


def round_sig(x, n):
    """Round to n significant digits. Returns x unchanged if x is zero."""
    if x == 0:
        return x
    k = -floor(log10(abs(x))) + n - 1
    return round(x, k)


def new_run_id(prefix=None):
    """Time-ordered, collision-resistant id, e.g. '20260825-171530-a3f2'.

    Sortable by creation time so history stays readable, but unique enough that
    two Colab sessions mining the same study will not clash.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"{random.getrandbits(16):04x}"
    base = f"{stamp}-{suffix}"
    return f"{prefix}-{base}" if prefix else base


def _version(module_name):
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def is_colab():
    return "google.colab" in sys.modules or os.path.exists("/content")


def detect_runtime():
    """Best-effort description of the accelerator, for drift diagnosis."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"
    except Exception:
        return "unknown"


def env_info():
    """Package versions and runtime, recorded on every result row.

    This is what makes a drift check interpretable rather than mysterious.
    """
    return {
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "gymnasium": _version("gymnasium"),
        "stable_baselines3": _version("stable_baselines3"),
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "runtime": detect_runtime(),
    }


def today():
    return datetime.now().strftime("%Y-%m-%d")


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _quiet_jupyter_utcnow():
    """jupyter_client still calls ``datetime.utcnow()``, which Python 3.12+
    deprecates. IPython shows that warning on every ``display()``, which
    drowns the trial output. Filter only that deprecation."""
    warnings.filterwarnings(
        "ignore",
        message=r".*utcnow\(\) is deprecated",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"jupyter_client(\.|$)",
    )


def display_obj(obj):
    """Render in a notebook when possible, fall back to print elsewhere."""
    try:
        from IPython.display import display

        _quiet_jupyter_utcnow()
        display(obj)
    except Exception:
        print(obj)


def display_header(text, level=3):
    try:
        from IPython.display import HTML, display

        _quiet_jupyter_utcnow()
        display(HTML(f"<h{level}>{text}</h{level}>"))
    except Exception:
        print(f"\n=== {text} ===")


def fmt_number(value, sig=4):
    if value is None:
        return ""
    try:
        if isinstance(value, bool):
            return str(value)
        return f"{round_sig(float(value), sig):g}"
    except (TypeError, ValueError):
        return str(value)
