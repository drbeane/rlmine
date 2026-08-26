"""rlmine: hyperparameter mining for reinforcement learning labs.

    from rlmine import Study, params as P
    from rlmine.trials import sb3_trial
    from stable_baselines3 import DQN

    study = Study(
        name  = 'lunar-lander-dqn',
        trial = sb3_trial(DQN, 'LunarLander-v3', n_envs=8),
        space = dict(
            timesteps     = P.Int(300_000, mutate='fixed'),
            learning_rate = P.Float(7e-4, sig=2, bounds=(0, None), mutate=P.scale(0.2)),
            net_arch      = P.Choice([[128, 128], [256, 256]], default=[256, 256]),
        ),
        store = 'drive/MyDrive/rl_mining',
    )

    study.run(learning_rate=1e-3)                       # specific values
    study.mine(n=5, mutate=['learning_rate'])           # roulette + perturb
    study.recheck(top=3)                                # drift check
    study.table()                                       # readable results
"""

from . import params
from .params import Bool, Choice, Float, Int
from .scoring import mean_minus_std, mean_return, success_rate
from .space import Space
from .stores import CSVStore, JSONLStore, MemoryStore, SheetMirror
from .study import Study

__version__ = "0.1.0"

__all__ = [
    "Study",
    "Space",
    "params",
    "Int",
    "Float",
    "Bool",
    "Choice",
    "JSONLStore",
    "CSVStore",
    "SheetMirror",
    "MemoryStore",
    "mean_minus_std",
    "mean_return",
    "success_rate",
    "__version__",
]
