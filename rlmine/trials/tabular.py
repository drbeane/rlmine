"""Trial factory for the tabular ``rltools`` agents.

Covers the Monte Carlo and Q-learning notebooks. The trial reads a config in
the same shape as the stored config key. Training fields (``episodes``,
``max_steps``, ``updates``, ``eval_eps``) and the remaining agent fields go to
the training method. Evaluation fields score the trained agent.

The environment factory is still code: it is called once per trial so no
state leaks between runs. The ``environment`` section of the config records
the choices that built it.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import partial

__all__ = ["tabular_trial", "mc_trial", "q_learning_trial", "read_tabular_config"]


def read_tabular_config(config):
    """The values a tabular trial reads from a config."""
    agent = dict((config or {}).get("agent") or {})
    training = dict((config or {}).get("training") or {})
    evaluation = dict((config or {}).get("evaluation") or {})

    algo_name = agent.pop("algo", None)
    method = agent.pop("method", None)
    agent_gamma = agent.pop("gamma", evaluation.get("gamma", 1.0))
    call_kwargs = dict(training)
    call_kwargs.update(agent)

    return {
        "algo_name": algo_name,
        "method": method,
        "agent_gamma": agent_gamma,
        "call_kwargs": call_kwargs,
        "eval_episodes": int(evaluation.get("episodes", 500)),
        "eval_max_steps": int(evaluation.get("max_steps", 1000)),
        "eval_seed": int(evaluation.get("seed", 1)),
        "eval_gamma": float(evaluation.get("gamma", 1.0)),
        "check_success": bool(evaluation.get("check_success", False)),
    }


@contextmanager
def _transient_tqdm():
    """Make the tqdm bars the agents draw clear themselves when they finish.

    The rltools agents import ``tqdm`` inside the training method and keep the
    bar on screen. Swapping the name for the duration of the call avoids
    editing rltools. The bar is plain text on stdout, not a notebook widget:
    a closed widget leaves an empty output row behind, while a text bar is
    overwritten in place and leaves nothing.
    """
    try:
        import tqdm.auto as auto
        from tqdm import tqdm
    except ImportError:
        yield
        return

    original = auto.tqdm
    auto.tqdm = partial(tqdm, leave=False, file=sys.stdout)
    try:
        yield
    finally:
        auto.tqdm = original


def tabular_trial(agent_cls, method, env_factory, *, progress_bar=True, show_report=False):
    """Build a trial function for a tabular agent.

    Args:
        agent_cls: ``MCAgent`` or ``TDAgent`` from ``rltools``.
        method: Name of the training method, e.g. ``'control'`` or ``'q_learning'``.
        env_factory: Zero-argument callable returning a fresh environment. It is
            called once per trial so no state leaks between runs.
        progress_bar: Show a training bar that clears when training ends. It
            sets the agent's ``show_progress``, overriding any value in the
            config. Not part of the config key.
        show_report: Print the evaluation report. Not part of the config.
    """

    def trial(config, context=None):
        from rltools.utils import evaluate

        settings = read_tabular_config(config)
        _check_agent(agent_cls, method, settings)

        env = env_factory()
        agent = agent_cls(env, gamma=settings["agent_gamma"])
        call_kwargs = dict(settings["call_kwargs"], show_progress=progress_bar)
        with _transient_tqdm():
            getattr(agent, method)(**call_kwargs)

        return evaluate(
            env,
            agent,
            gamma=settings["eval_gamma"],
            episodes=settings["eval_episodes"],
            max_steps=settings["eval_max_steps"],
            seed=settings["eval_seed"],
            check_success=settings["check_success"],
            show_report=show_report,
        )

    return trial


def _check_agent(agent_cls, method, settings):
    actual = getattr(agent_cls, "__name__", None)
    if settings["algo_name"] and actual and settings["algo_name"] != actual:
        raise ValueError(
            f"config agent.algo is {settings['algo_name']!r}, "
            f"but this trial runs {actual!r}."
        )
    if settings["method"] and settings["method"] != method:
        raise ValueError(
            f"config agent.method is {settings['method']!r}, "
            f"but this trial calls {method!r}."
        )


def mc_trial(env_factory, **kwargs):
    """Monte Carlo control, as in the Week 03 ``MC-Control`` notebooks."""
    from rltools.monte_carlo import MCAgent

    return tabular_trial(MCAgent, "control", env_factory, **kwargs)


def q_learning_trial(env_factory, **kwargs):
    """Q-learning, as in the Week 03 ``Q-Learning`` notebooks."""
    from rltools.temp_diff import TDAgent

    return tabular_trial(TDAgent, "q_learning", env_factory, **kwargs)
