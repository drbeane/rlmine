"""Trial factories.

A trial is any callable ``params -> stats``. These factories build the two
shapes that cover the labs, but writing your own function is always an option
when an environment needs special handling.
"""

from .sb3 import linear_schedule, sb3_trial
from .tabular import mc_trial, q_learning_trial, tabular_trial

__all__ = [
    "sb3_trial",
    "linear_schedule",
    "tabular_trial",
    "mc_trial",
    "q_learning_trial",
]
