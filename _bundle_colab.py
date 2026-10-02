"""Build a single-file Colab drop-in of the rlmine package. Not part of the package."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "rlmine"
OUT = Path(__file__).resolve().parent.parent / "rlmine.py"

MODULES = [
    ("utils", ROOT / "utils.py"),
    ("params", ROOT / "params.py"),
    ("space", ROOT / "space.py"),
    ("scoring", ROOT / "scoring.py"),
    ("stores", ROOT / "stores.py"),
    ("trials.sb3", ROOT / "trials" / "sb3.py"),
    ("trials.tabular", ROOT / "trials" / "tabular.py"),
    ("study", ROOT / "study.py"),
]

HEADER = '''\
"""rlmine — single-file Colab drop-in (temporary).

Flattened copy of the package so you can test without pushing to git.

Colab
-----
1. Comment out ``!pip install git+https://github.com/drbeane/rlmine.git``.
2. Upload this file to the VM as ``/content/rlmine.py`` (Files pane, drag and drop).
3. Keep the same imports as the package::

       from rlmine import Study, params as P
       from rlmine.stores import JSONStore, SheetMirror
       from rlmine.trials import sb3_trial

4. After re-uploading, unload the old copy first::

       import sys
       for k in list(sys.modules):
           if k == "rlmine" or k.startswith("rlmine."):
               del sys.modules[k]
       import rlmine

``rlmine.__dev_bundle__`` is True on this file, False on the git package.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import csv
import inspect
import io
import json
import os
import random
import shutil
import sys
import time
import traceback
import types
import warnings
from datetime import datetime
from math import floor, log10
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0-colab"
__dev_bundle__ = True

'''

FOOTER = '''

# ---------------------------------------------------------------------------
# Package-shaped imports (so notebook cells do not change)
# ---------------------------------------------------------------------------


def _submodule(fullname, **attrs):
    mod = types.ModuleType(fullname)
    mod.__package__ = fullname.rpartition(".")[0] or fullname
    mod.__dict__.update(attrs)
    sys.modules[fullname] = mod
    return mod


params = _submodule(
    "rlmine.params",
    Param=Param,
    Int=Int,
    Float=Float,
    Bool=Bool,
    Choice=Choice,
    Action=Action,
    DEFAULT=DEFAULT,
    SAMPLE=SAMPLE,
    MUTATE=MUTATE,
    PARENT=PARENT,
    scale=scale,
    times=times,
    shift=shift,
    pick=pick,
    resample=resample,
    flip=flip,
    fixed=fixed,
    uniform=uniform,
    loguniform=loguniform,
    choice=choice,
)

utils = _submodule(
    "rlmine.utils",
    round_sig=round_sig,
    is_colab=is_colab,
    detect_runtime=detect_runtime,
    env_info=env_info,
    today=today,
    now_iso=now_iso,
    _quiet_third_party_warnings=_quiet_third_party_warnings,
    display_obj=display_obj,
    display_header=display_header,
    fmt_number=fmt_number,
)

space = _submodule("rlmine.space", Space=Space)

scoring = _submodule(
    "rlmine.scoring",
    mean_minus_std=mean_minus_std,
    mean_return=mean_return,
    success_rate=success_rate,
    resolve_score_fn=resolve_score_fn,
    score_digits=score_digits,
    round_to_digits=round_to_digits,
    with_score_digits=with_score_digits,
)

stores = _submodule(
    "rlmine.stores",
    Store=Store,
    JSONStore=JSONStore,
    JSONLStore=JSONStore,
    CSVStore=CSVStore,
    SheetMirror=SheetMirror,
    MemoryStore=MemoryStore,
    resolve_store=resolve_store,
)

trials_sb3 = _submodule(
    "rlmine.trials.sb3",
    sb3_trial=sb3_trial,
    linear_schedule=linear_schedule,
    route_params=route_params,
    POLICY_PARAMS=POLICY_PARAMS,
)

trials_tabular = _submodule(
    "rlmine.trials.tabular",
    tabular_trial=tabular_trial,
    mc_trial=mc_trial,
    q_learning_trial=q_learning_trial,
)

trials = _submodule(
    "rlmine.trials",
    sb3_trial=sb3_trial,
    linear_schedule=linear_schedule,
    tabular_trial=tabular_trial,
    mc_trial=mc_trial,
    q_learning_trial=q_learning_trial,
    sb3=trials_sb3,
    tabular=trials_tabular,
)

study = _submodule("rlmine.study", Study=Study)

# samplers.py is executed apart from this file so its Sampler / uniform /
# loguniform names do not replace the parameter-space versions above.
_sampler_ns = dict(globals())
_sampler_ns["__name__"] = "rlmine.samplers"
exec(compile(_SAMPLERS_SRC, "samplers.py", "exec"), _sampler_ns)
resolve_samplers = _sampler_ns["resolve_samplers"]
samplers = _submodule(
    "rlmine.samplers",
    Sampler=_sampler_ns["Sampler"],
    loguniform=_sampler_ns["loguniform"],
    uniform=_sampler_ns["uniform"],
    normal=_sampler_ns["normal"],
    resolve_samplers=resolve_samplers,
)

# Answer to `import rlmine` even if this file was uploaded under another name.
sys.modules["rlmine"] = sys.modules[__name__]

_quiet_third_party_warnings()

__all__ = [
    "Study",
    "Space",
    "params",
    "samplers",
    "Int",
    "Float",
    "Bool",
    "Choice",
    "JSONStore",
    "JSONLStore",
    "CSVStore",
    "SheetMirror",
    "MemoryStore",
    "mean_minus_std",
    "mean_return",
    "success_rate",
    "__version__",
    "__dev_bundle__",
]
'''


def strip_top_level_imports(source: str) -> str:
    """Drop future imports, stdlib/third-party imports, and relative imports.

    Lazy imports inside functions (gymnasium, sb3, colab, rltools) are kept.
    """
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module == "__future__":
                continue
            continue
        if isinstance(node, ast.Import):
            continue
        keep.append(node)
    tree.body = keep
    return ast.unparse(tree)


def strip_relative_imports_in_functions(source: str) -> str:
    """sb3_trial lazily does `from ..utils import ...`; that is already in scope."""
    tree = ast.parse(source)

    class RelStrip(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.level and node.level > 0:
                return None
            return node

    tree = RelStrip().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def main():
    sampler_path = ROOT / "samplers.py"
    sampler_body = strip_top_level_imports(sampler_path.read_text(encoding="utf-8"))
    sampler_body = strip_relative_imports_in_functions(sampler_body)
    parts = [HEADER, f"_SAMPLERS_SRC = {json.dumps(sampler_body)}\n\n"]
    for label, path in MODULES:
        raw = path.read_text(encoding="utf-8")
        body = strip_top_level_imports(raw)
        body = strip_relative_imports_in_functions(body)
        parts.append(f"\n# === {label} ({path.name}) ===\n\n")
        parts.append(body)
        parts.append("\n")
    parts.append(FOOTER)
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
