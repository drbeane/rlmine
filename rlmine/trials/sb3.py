"""Trial factory for Stable-Baselines3 algorithms.

Covers the DQN and A2C notebooks: build vectorised environments, train with an
``EvalCallback``, reload the best checkpoint, and evaluate it with
``rltools.utils.evaluate``.

The trial reads a config in the same shape as the stored config key:

``environment``   id, ``n_envs``, seed, ``normalize``, optional ``kwargs``
``agent``         algorithm fields. ``initial_lr`` / ``final_lr`` become a
                  linear schedule. ``net_arch``, ``log_std_init``, and
                  ``ortho_init`` go into ``policy_kwargs``. The rest go to
                  the algorithm constructor.
``training``      ``timesteps``, ``eval_freq``, ``n_eval_episodes``
``evaluation``    the final scoring run
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["sb3_trial", "linear_schedule", "route_params", "read_sb3_config", "POLICY_PARAMS"]

POLICY_PARAMS = {
    "net_arch",
    "log_std_init",
    "ortho_init",
    "activation_fn",
    "optimizer_class",
    "share_features_extractor",
}


def linear_schedule(initial_lr, final_lr):
    """Learning rate falling linearly from ``initial_lr`` to ``final_lr``."""

    def schedule(progress_remaining):
        return final_lr + progress_remaining * (initial_lr - final_lr)

    return schedule


def route_params(params, policy_kwargs=None, model_kwargs=None, default_timesteps=100_000):
    """Split a flat parameter dict into the pieces SB3 expects.

    Returns ``(timesteps, constructor_kwargs)`` where ``constructor_kwargs``
    already contains ``policy_kwargs`` and, when ``initial_lr`` is present,
    a ``learning_rate`` schedule.
    """
    params = dict(params)
    timesteps = int(params.pop("timesteps", default_timesteps))

    run_policy_kwargs = dict(policy_kwargs or {})
    for name in list(params):
        if name in POLICY_PARAMS:
            run_policy_kwargs[name] = params.pop(name)

    learning_rate = None
    if "initial_lr" in params:
        initial = params.pop("initial_lr")
        learning_rate = linear_schedule(initial, params.pop("final_lr", initial))
    else:
        params.pop("final_lr", None)

    constructor_kwargs = dict(model_kwargs or {})
    constructor_kwargs.update(params)
    if learning_rate is not None:
        constructor_kwargs["learning_rate"] = learning_rate
    if run_policy_kwargs:
        constructor_kwargs["policy_kwargs"] = run_policy_kwargs

    return timesteps, constructor_kwargs


def read_sb3_config(config):
    """The values an SB3 trial reads from a config.

    ``environment.normalize`` is ``False`` or the ``VecNormalize`` keyword
    arguments. ``agent.initial_lr`` and ``agent.final_lr`` are turned into a
    ``learning_rate`` schedule on the constructor kwargs.
    """
    environment = dict((config or {}).get("environment") or {})
    agent = dict((config or {}).get("agent") or {})
    training = dict((config or {}).get("training") or {})
    evaluation = dict((config or {}).get("evaluation") or {})

    if "id" not in environment:
        raise ValueError("config environment.id is required.")
    if "timesteps" not in training:
        raise ValueError("config training.timesteps is required.")

    normalize = environment.get("normalize", False)
    if normalize is True:
        normalize = {"norm_obs": True, "norm_reward": False, "clip_obs": 10.0}
    elif not normalize:
        normalize = False
    elif not isinstance(normalize, dict):
        raise TypeError("config environment.normalize must be a dict or false.")

    algo_name = agent.pop("algo", None)
    policy = agent.pop("policy", "MlpPolicy")
    agent.pop("lr_schedule", None)
    _, constructor_kwargs = route_params(agent)

    return {
        "env_id": environment["id"],
        "n_envs": int(environment.get("n_envs", 1)),
        "env_seed": int(environment.get("seed", 0)),
        "env_kwargs": dict(environment.get("kwargs") or {}),
        "normalize": normalize,
        "algo_name": algo_name,
        "policy": policy,
        "timesteps": int(training["timesteps"]),
        "eval_freq": int(training.get("eval_freq", 1000)),
        "n_eval_episodes": int(training.get("n_eval_episodes", 20)),
        "callback_deterministic": bool(training.get("deterministic", True)),
        "constructor_kwargs": constructor_kwargs,
        "eval_episodes": int(evaluation.get("episodes", 50)),
        "eval_max_steps": int(evaluation.get("max_steps", 1000)),
        "eval_seed": int(evaluation.get("seed", 1)),
        "eval_gamma": float(evaluation.get("gamma", 1.0)),
        "eval_deterministic": bool(evaluation.get("deterministic", True)),
        "check_success": bool(evaluation.get("check_success", False)),
    }


def sb3_trial(
    algo=None,
    *,
    eval_dir="evaluation",
    progress_bar=True,
    verbose=0,
    show_report=False,
    save_best_to=None,
    prune=None,
    prune_minutes=None,
):
    """Build a trial function for an SB3 algorithm.

    The values that can change the score come from the config passed to the
    trial. ``algo`` is the class to construct, such as ``A2C``. When the
    config names ``agent.algo``, that name has to match. Omit ``algo`` to
    look the class up from ``agent.algo`` on ``stable_baselines3``.

    ``verbose``, ``progress_bar``, and ``show_report`` are logging flags.
    They are not part of the config. ``eval_dir`` is scratch space for
    ``EvalCallback`` and is cleared each trial. ``save_best_to`` keeps each
    run's best checkpoint, named by run id.

    ``prune`` and ``prune_minutes`` abandon a run that is not doing well
    enough. Each is a list of ``(when, min_reward)`` pairs, where ``when`` is
    a timestep count or a number of minutes of training. At the first
    in-training evaluation at or after ``when``, a mean evaluation reward
    below ``min_reward`` stops the run. A pruned run is not logged. Like the
    logging flags, these are not part of the config, so they never change the
    key or the score of a run that completes. The reward compared is the
    ``EvalCallback`` mean, not the final score. Rules given here are
    defaults; ``Study.run(prune=..., prune_minutes=...)`` overrides them for
    that call.

        sb3_trial(A2C, prune=[(200_000, 0), (500_000, 100)], prune_minutes=[(30, 50)])
    """
    step_rules = _prune_rules(prune, "prune")
    minute_rules = _prune_rules(prune_minutes, "prune_minutes")

    def trial(config, context=None):
        from ..utils import _quiet_third_party_warnings

        _quiet_third_party_warnings()

        import gymnasium
        from rltools.utils import SB3Agent, evaluate
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize

        settings = read_sb3_config(config)
        # Rules passed to Study.run take precedence over the factory defaults.
        ctx = context or {}
        run_steps = _prune_rules(ctx["prune"], "prune") if ctx.get("prune") else step_rules
        run_minutes = (
            _prune_rules(ctx["prune_minutes"], "prune_minutes")
            if ctx.get("prune_minutes")
            else minute_rules
        )
        algo_cls = _algo_class(algo, settings["algo_name"])
        normalize = settings["normalize"]

        train_env = make_vec_env(
            settings["env_id"],
            n_envs=settings["n_envs"],
            seed=settings["env_seed"],
            env_kwargs=settings["env_kwargs"],
        )
        eval_env = make_vec_env(
            settings["env_id"],
            n_envs=settings["n_envs"],
            seed=settings["env_seed"],
            env_kwargs=settings["env_kwargs"],
        )
        if normalize:
            train_env = VecNormalize(train_env, **normalize)
            eval_env = VecNormalize(eval_env, **normalize)
        test_env = gymnasium.make(
            settings["env_id"], render_mode="rgb_array", **settings["env_kwargs"]
        )

        if os.path.exists(eval_dir):
            shutil.rmtree(eval_dir)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=eval_dir,
            log_path=eval_dir,
            n_eval_episodes=settings["n_eval_episodes"],
            eval_freq=settings["eval_freq"],
            deterministic=settings["callback_deterministic"],
            callback_after_eval=_pruner(run_steps, run_minutes),
            warn=False,
            verbose=verbose,
        )

        try:
            model = algo_cls(
                policy=settings["policy"],
                env=train_env,
                verbose=verbose,
                **settings["constructor_kwargs"],
            )

            callbacks = [callback]
            bar = _transient_bar(settings["timesteps"]) if progress_bar else None
            if bar is not None:
                callbacks.append(bar)
            try:
                model.learn(
                    total_timesteps=settings["timesteps"],
                    callback=callbacks,
                    progress_bar=False,
                )
            finally:
                if bar is not None:
                    bar.close()

            best_model = _load_best(algo_cls, eval_dir, test_env, model)

            if save_best_to and context and context.get("config_id") is not None:
                _keep_checkpoint(eval_dir, save_best_to, context["config_id"])

            normalizer = train_env.normalize_obs if normalize else None
            agent = SB3Agent(
                best_model,
                deterministic=settings["eval_deterministic"],
                normalizer=normalizer,
            )

            return evaluate(
                test_env,
                agent,
                gamma=settings["eval_gamma"],
                episodes=settings["eval_episodes"],
                max_steps=settings["eval_max_steps"],
                seed=settings["eval_seed"],
                check_success=settings["check_success"],
                show_report=show_report,
            )
        finally:
            for env in (train_env, eval_env, test_env):
                try:
                    env.close()
                except Exception:
                    pass

    return trial


def _transient_bar(total):
    """A training progress callback whose bar is cleared when training ends.

    Returns ``None`` when tqdm is missing, so training carries on without it.
    The bar is plain text on stdout, not a notebook widget: a closed widget
    leaves an empty output row behind, while a text bar is overwritten in
    place and leaves nothing.
    """
    try:
        import sys

        from stable_baselines3.common.callbacks import BaseCallback
        from tqdm import tqdm
    except ImportError:
        return None

    class TransientBar(BaseCallback):
        def __init__(self):
            super().__init__()
            self.bar = None

        def _on_training_start(self):
            self.bar = tqdm(total=total, leave=False, unit="step", file=sys.stdout)

        def _on_step(self):
            if self.bar is not None:
                self.bar.update(min(self.num_timesteps, total) - self.bar.n)
            return True

        def close(self):
            if self.bar is not None:
                self.bar.close()
                self.bar = None

    return TransientBar()


def _prune_rules(rules, name):
    """Validated ``(when, min_reward)`` pairs, earliest first."""
    if not rules:
        return []
    checked = []
    for rule in rules:
        try:
            when, minimum = rule
            when, minimum = float(when), float(minimum)
        except (TypeError, ValueError):
            raise TypeError(f"{name} takes (when, min_reward) pairs, got {rule!r}.") from None
        if when <= 0:
            raise ValueError(f"{name} thresholds must be positive, got {rule!r}.")
        checked.append((when, minimum))
    return sorted(checked)


def _pruner(step_rules, minute_rules):
    """An ``EvalCallback`` ``callback_after_eval`` that raises ``Pruned``.

    Returns ``None`` when there are no rules. It only reads the evaluation
    result and the clock, so a run that survives is unchanged.
    """
    if not step_rules and not minute_rules:
        return None

    import time

    from stable_baselines3.common.callbacks import BaseCallback

    from ..utils import Pruned

    class Pruner(BaseCallback):
        def __init__(self):
            super().__init__()
            self.started = None
            self.steps_left = list(step_rules)
            self.minutes_left = list(minute_rules)

        def _on_training_start(self):
            self.started = time.time()

        def _on_step(self):
            reward = getattr(self.parent, "last_mean_reward", None)
            if reward is None:
                return True
            if self.started is None:
                self.started = time.time()
            minutes = (time.time() - self.started) / 60
            reward = float(reward)

            # Every rule that has come due is settled at this evaluation.
            for left, now, unit in (
                (self.steps_left, self.num_timesteps, "steps"),
                (self.minutes_left, minutes, "min"),
            ):
                while left and now >= left[0][0]:
                    when, minimum = left.pop(0)
                    if reward < minimum:
                        raise Pruned(
                            f"eval reward {reward:.2f} < {minimum:g} "
                            f"after {when:,.0f} {unit} (at {self.num_timesteps:,} steps)"
                        )
            return True

    return Pruner()


def _algo_class(algo, name):
    if algo is not None:
        actual = getattr(algo, "__name__", None)
        if name and actual and name != actual:
            raise ValueError(
                f"config agent.algo is {name!r}, but this trial runs {actual!r}."
            )
        return algo
    if not name:
        raise ValueError(
            "config agent.algo is required, or pass the algorithm class to sb3_trial."
        )
    import stable_baselines3 as sb3

    found = getattr(sb3, name, None)
    if found is None:
        raise ValueError(f"Unknown Stable-Baselines3 algorithm {name!r}.")
    return found


def _load_best(algo, eval_dir, test_env, fallback):
    """Prefer the best checkpoint; fall back to the final model."""
    best_path = Path(eval_dir) / "best_model.zip"
    if not best_path.exists():
        return fallback
    try:
        return algo.load(best_path, env=test_env)
    except Exception:
        # Observation-space mismatches are common with wrapped envs; the model
        # only needs to predict, so loading without an env is enough.
        return algo.load(best_path)


def _keep_checkpoint(eval_dir, destination, config_id):
    source = Path(eval_dir) / "best_model.zip"
    if not source.exists():
        return
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target_dir / f"{config_id}.zip")
