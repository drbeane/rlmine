"""rlmine: hyperparameter mining for reinforcement learning labs.

    from rlmine import Study
    from rlmine.trials import sb3_trial
    from stable_baselines3 import A2C

    study = Study(
        name  = 'bipedal-walker-a2c',
        trial = sb3_trial(A2C),
        config = {
            'environment': {'id': 'BipedalWalker-v3', 'n_envs': 16},
            'agent':       {'algo': 'A2C', 'gamma': 0.99, 'seed': 1},
            'training':    {'timesteps': 1_000_000},
            'evaluation':  {'episodes': 50, 'gamma': 1.0, 'score': 'mean_minus_std'},
        },
    )

    study.run()                                 # the config as written
    study.run(agent={'gamma': 0.95})            # change one section for this trial

    from rlmine import samplers as s, mutations as m

    study.run(agent={'initial_lr': s.loguniform(1e-4, 1e-2, sig=2)})   # draw a value
    study.run(config_id=184203,                                        # start from a stored config
              agent={'initial_lr': m.proportional(0.2, sig=2)})        # and nudge a value
    study.table()                               # readable results

    from rlmine import Results

    Results('results/bipedal-walker-a2c.json').best(5)

    from rlmine import config, run

    Results('results/bipedal-walker-a2c.json').filter(config.training.timesteps <= 50_000)
"""

from . import mutations, params, samplers
from .params import Bool, Choice, Float, Int
from .query import config, run
from .results import Results
from .scoring import mean_minus_std, mean_return, success_rate
from .space import Space
from .stores import CSVStore, JSONLStore, JSONStore, MemoryStore, SheetMirror
from .study import Study
from .utils import Pruned, _quiet_third_party_warnings

_quiet_third_party_warnings()

__version__ = "0.1.0"

__all__ = [
    "Study",
    "Results",
    "config",
    "run",
    "Space",
    "params",
    "samplers",
    "mutations",
    "Int",
    "Float",
    "Bool",
    "Choice",
    "JSONStore",
    "JSONLStore",
    "CSVStore",
    "SheetMirror",
    "MemoryStore",
    "Pruned",
    "mean_minus_std",
    "mean_return",
    "success_rate",
    "__version__",
]
