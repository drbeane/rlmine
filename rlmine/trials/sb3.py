"""Trial factory for Stable-Baselines3 algorithms.

Covers the DQN and A2C notebooks: build vectorised environments, train with an
``EvalCallback``, reload the best checkpoint, and evaluate it with
``rltools.utils.evaluate``.

Parameter routing is by name, so a space can stay flat:

``timesteps``                                  drives ``learn(total_timesteps=...)``
``initial_lr`` / ``final_lr``                  become a linear learning-rate schedule
``net_arch``, ``log_std_init``, ``ortho_init`` go into ``policy_kwargs``
everything else                                goes to the algorithm constructor
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = ["sb3_trial", "linear_schedule", "route_params", "POLICY_PARAMS"]

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
    already contains ``policy_kwargs`` and, if the space defines ``initial_lr``,
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


def prefer_cpu_device(algo, policy, constructor_kwargs):
    """Default A2C/PPO + MLP to CPU.

    Stable-Baselines3 warns that on-policy MLP policies train more slowly on
    GPU than on CPU. Honour an explicit ``device`` if the caller already set
    one, and leave CNN policies on the default device so they can use the GPU.
    """
    if "device" in constructor_kwargs:
        return constructor_kwargs
    name = getattr(algo, "__name__", "")
    if name in {"A2C", "PPO"} and "Cnn" not in str(policy):
        constructor_kwargs["device"] = "cpu"
    return constructor_kwargs


def sb3_trial(
    algo,
    env_id,
    *,
    policy="MlpPolicy",
    n_envs=1,
    env_kwargs=None,
    env_seed=0,
    normalize=False,
    norm_kwargs=None,
    eval_freq=1000,
    n_eval_episodes=20,
    eval_dir="evaluation",
    eval_episodes=50,
    eval_max_steps=1000,
    eval_seed=1,
    gamma=1.0,
    deterministic=True,
    check_success=False,
    default_timesteps=100_000,
    policy_kwargs=None,
    model_kwargs=None,
    progress_bar=True,
    verbose=0,
    show_report=False,
    save_best_to=None,
):
    """Build a trial function for an SB3 algorithm.

    Args:
        algo: An SB3 class such as ``DQN`` or ``A2C``.
        env_id: Gymnasium id, e.g. ``'LunarLander-v3'``.
        n_envs: Parallel training environments.
        normalize: Wrap train and eval envs in ``VecNormalize``. The running
            observation statistics are then also applied at evaluation time.
        eval_dir: Scratch directory for ``EvalCallback``; cleared each trial.
        save_best_to: Optional directory to keep each run's best checkpoint,
            named by run id.
        model_kwargs: Fixed constructor arguments that are not part of the
            search space. A2C and PPO with an MLP policy default to
            ``device='cpu'`` (faster than GPU for these algorithms); pass
            ``device`` here to override.
    """
    env_kwargs = dict(env_kwargs or {})
    norm_kwargs = dict(norm_kwargs or {"norm_obs": True, "norm_reward": False, "clip_obs": 10.0})
    base_policy_kwargs = dict(policy_kwargs or {})
    base_model_kwargs = dict(model_kwargs or {})

    def trial(params, context=None):
        from ..utils import _quiet_third_party_warnings

        _quiet_third_party_warnings()

        import gymnasium
        from rltools.utils import SB3Agent, evaluate
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecNormalize

        timesteps, constructor_kwargs = route_params(
            params, base_policy_kwargs, base_model_kwargs, default_timesteps
        )
        prefer_cpu_device(algo, policy, constructor_kwargs)

        train_env = make_vec_env(env_id, n_envs=n_envs, seed=env_seed, env_kwargs=env_kwargs)
        eval_env = make_vec_env(env_id, n_envs=n_envs, seed=env_seed, env_kwargs=env_kwargs)
        if normalize:
            train_env = VecNormalize(train_env, **norm_kwargs)
            eval_env = VecNormalize(eval_env, **norm_kwargs)
        test_env = gymnasium.make(env_id, render_mode="rgb_array", **env_kwargs)

        if os.path.exists(eval_dir):
            shutil.rmtree(eval_dir)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=eval_dir,
            log_path=eval_dir,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            warn=False,
            verbose=verbose,
        )

        try:
            model = algo(policy=policy, env=train_env, verbose=verbose, **constructor_kwargs)

            try:
                model.learn(
                    total_timesteps=timesteps, callback=callback, progress_bar=progress_bar
                )
            except ImportError:
                # progress_bar needs tqdm and rich; carry on without it.
                model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)

            best_model = _load_best(algo, eval_dir, test_env, model)

            if save_best_to and context:
                _keep_checkpoint(eval_dir, save_best_to, context["run_id"])

            normalizer = train_env.normalize_obs if normalize else None
            agent = SB3Agent(best_model, deterministic=deterministic, normalizer=normalizer)

            return evaluate(
                test_env,
                agent,
                gamma=gamma,
                episodes=eval_episodes,
                max_steps=eval_max_steps,
                seed=eval_seed,
                check_success=check_success,
                show_report=show_report,
            )
        finally:
            for env in (train_env, eval_env, test_env):
                try:
                    env.close()
                except Exception:
                    pass

    return trial


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


def _keep_checkpoint(eval_dir, destination, run_id):
    source = Path(eval_dir) / "best_model.zip"
    if not source.exists():
        return
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target_dir / f"{run_id}.zip")
