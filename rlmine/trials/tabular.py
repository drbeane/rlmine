"""Trial factory for the tabular ``rltools`` agents.

Covers the Monte Carlo and Q-learning notebooks. The searched parameters go
straight to ``MCAgent.control`` or ``TDAgent.q_learning``; anything held fixed
for the whole study belongs in ``train_kwargs``.

Note that ``episodes`` in the search space means *training* episodes. The
number of evaluation episodes is a property of the study, not of the search, so
it is named ``eval_episodes`` here.
"""

from __future__ import annotations

__all__ = ["tabular_trial", "mc_trial", "q_learning_trial"]


def tabular_trial(
    agent_cls,
    method,
    env_factory,
    *,
    gamma=1.0,
    eval_episodes=500,
    eval_max_steps=1000,
    eval_seed=1,
    check_success=False,
    train_kwargs=None,
    show_report=False,
):
    """Build a trial function for a tabular agent.

    Args:
        agent_cls: ``MCAgent`` or ``TDAgent`` from ``rltools``.
        method: Name of the training method, e.g. ``'control'`` or ``'q_learning'``.
        env_factory: Zero-argument callable returning a fresh environment. It is
            called once per trial so no state leaks between runs.
        train_kwargs: Fixed training arguments, e.g.
            ``dict(max_steps=500, updates=1000, eval_eps=1000)``.
    """
    base_train_kwargs = dict(train_kwargs or {})

    def trial(params):
        from rltools.utils import evaluate

        env = env_factory()
        agent = agent_cls(env, gamma=gamma)

        call_kwargs = dict(base_train_kwargs)
        call_kwargs.update(params)
        getattr(agent, method)(**call_kwargs)

        return evaluate(
            env,
            agent,
            gamma=gamma,
            episodes=eval_episodes,
            max_steps=eval_max_steps,
            seed=eval_seed,
            check_success=check_success,
            show_report=show_report,
        )

    return trial


def mc_trial(env_factory, **kwargs):
    """Monte Carlo control, as in the Week 03 ``MC-Control`` notebooks."""
    from rltools.monte_carlo import MCAgent

    return tabular_trial(MCAgent, "control", env_factory, **kwargs)


def q_learning_trial(env_factory, **kwargs):
    """Q-learning, as in the Week 03 ``Q-Learning`` notebooks."""
    from rltools.temp_diff import TDAgent

    return tabular_trial(TDAgent, "q_learning", env_factory, **kwargs)
