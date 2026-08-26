# rlmine

Hyperparameter mining for reinforcement learning labs.

Train many agents, score them, log every result, and use the good ones to find
better ones. Works with any environment and any algorithm; ships with helpers
for Stable-Baselines3 and for the tabular agents in
[`rltools`](https://github.com/drbeane/rltools).

```python
from rlmine import Study, params as P
from rlmine.trials import sb3_trial
from stable_baselines3 import DQN

lander = Study(
    name  = 'lunar-lander-dqn',
    trial = sb3_trial(DQN, 'LunarLander-v3', n_envs=8),
    space = dict(
        timesteps     = P.Int(300_000, mutate='fixed'),
        learning_rate = P.Float(7e-4, sig=2, bounds=(0, None), mutate=P.scale(0.2)),
        gamma         = P.Float(0.99, digits=3, bounds=(0, 1)),
        net_arch      = P.Choice([[128, 128], [256, 256]], default=[256, 256]),
    ),
    store = 'drive/MyDrive/rl_mining',
)

lander.run(learning_rate=1e-3)                     # a specific configuration
lander.mine(n=5, mutate=['learning_rate'])         # explore near good results
lander.recheck(top=3)                              # has anything drifted?
lander.table()                                     # read the results
```

## Install

```
!pip install git+https://github.com/drbeane/rlmine.git
```

Only `numpy` and `pandas` are required. Install the extras you need for your
trials (`stable-baselines3`, `gymnasium`, `rltools`) as usual.

## The three operations

All three do the same thing — run a trial, score it, append a row — and differ
only in where the parameters come from.

| Call | Parameters come from |
| --- | --- |
| `study.run(**values)` | Space defaults, with your explicit values on top |
| `study.mine(n, mutate=[...])` | A roulette-selected parent, with the named parameters perturbed |
| `study.recheck(top=k)` | A stored row, replayed verbatim |

Every row records `origin` and `parent_id`, so you can trace which ancestor a
good configuration descended from.

### Explore

```python
study.mine(n=10, mutate=['learning_rate', 'gamma'])   # roulette-select a parent
study.mine(n=5, mutate='all', parent=16)              # pin a parent by index
study.mine(n=5, mutate=['gamma'], timesteps=50_000)   # pin a value while mining
```

A parent is selected for each model, so a good result found early in a call can
seed later ones. Selection is weighted by score among runs that scored above
zero; failed runs are never selected. With no history at all, values are drawn
fresh from each parameter's sampler.

### Check for drift

```python
study.recheck(top=3)        # the three best configurations
study.recheck(latest=5)     # the five most recent
study.recheck(rows=[0, 4])  # specific runs, by index or run id
study.drift_report()        # original vs recheck score, side by side
```

Every row stores the Python, Gymnasium, Stable-Baselines3, and Torch versions
it ran under, plus the accelerator, so a score that has moved can be traced to
a library change rather than guessed at.

## Defining a space

A parameter declares its type, default, bounds, how to perturb it, and
optionally how to sample it fresh.

```python
space = dict(
    timesteps     = P.Int(1_000_000, mutate='fixed'),
    n_steps       = P.Int(5, bounds=(1, None), mutate=P.times([0.5, 1, 2])),
    learning_rate = P.Float(7e-4, sig=2, bounds=(0, None), mutate=P.scale(0.2)),
    gamma         = P.Float(0.99, digits=3, bounds=(0, 1)),
    net_arch      = P.Choice([[128, 128], [256, 256], [512, 512]]),
    use_sde       = P.Bool(False),
)
```

**Types.** `P.Int`, `P.Float`, `P.Bool`, `P.Choice`. Floats take `sig` (round to
significant digits, right for learning rates) or `digits` (decimal places,
right for gamma). All take `bounds=(low, high)`, where `None` means unbounded.

**Mutations** apply when a parameter is named in `mutate=[...]`.

| Mutation | Effect |
| --- | --- |
| `P.scale(0.2)` | Multiply by a uniform factor in `[0.8, 1.2]` |
| `P.times([0.5, 1, 2])` | Multiply by a factor from an explicit list |
| `P.shift(1)` | Add a uniform offset in `[-1, 1]` |
| `P.pick([...])` | Choose from an explicit list |
| `P.flip(0.5)` | Flip a boolean |
| `P.resample()` | Ignore the parent, draw fresh from the sampler |
| `'fixed'` | Never change, even when named in `mutate` |

Defaults if you do not specify one: floats jitter by 10%, integers halve or
double, booleans flip, and choices pick a new option. Use `mutate='fixed'` for
things like `timesteps` and `seed` that you want to hold constant.

**Samplers** produce a value when there is no parent to inherit from:
`P.uniform(a, b)`, `P.loguniform(a, b)`, `P.choice([...])`. Without one, the
default is used.

**Constraints** express rules that span parameters:

```python
def constraints(p):
    p['final_lr'] = min(p['initial_lr'], p['final_lr'])
    return p

Study(..., space=space, constraints=constraints)
```

Call `study.space.describe()` for a table of the whole space.

## Trial functions

A trial is any callable taking a parameter dict and returning the stats dict
from `rltools.utils.evaluate` (anything with `mean_return` and `stdev_return`).

### Stable-Baselines3

```python
from rlmine.trials import sb3_trial
from stable_baselines3 import A2C

trial = sb3_trial(
    A2C, 'BipedalWalker-v3',
    n_envs=16, normalize=True,
    eval_freq=1000, n_eval_episodes=20,
    eval_episodes=50, eval_max_steps=1600,
)
```

It builds the vectorised environments, attaches an `EvalCallback`, trains,
reloads the best checkpoint, and evaluates it. Parameters are routed by name so
the space can stay flat:

| Parameter | Goes to |
| --- | --- |
| `timesteps` | `learn(total_timesteps=...)` |
| `initial_lr`, `final_lr` | A linear learning-rate schedule |
| `net_arch`, `log_std_init`, `ortho_init` | `policy_kwargs` |
| everything else | The algorithm constructor |

With `normalize=True` the running observation statistics from training are also
applied at evaluation time. Pass `model_kwargs=` for constructor arguments that
are fixed rather than searched, and `save_best_to=` to keep each run's
checkpoint.

### Tabular agents

```python
from rlmine.trials import q_learning_trial
import rltools.gym as gym

trial = q_learning_trial(
    lambda: gym.make('CartPole-v1', num_bins=25, render_mode='rgb_array'),
    eval_episodes=500,
    train_kwargs=dict(max_steps=1000, updates=1000, eval_eps=100),
)
```

`mc_trial` and `q_learning_trial` wrap `MCAgent.control` and
`TDAgent.q_learning`; `tabular_trial` takes any agent class and method name. The
environment factory is called once per trial so no state leaks between runs.

Note that `episodes` in the space means *training* episodes and is passed to the
agent. Evaluation episodes are a property of the study, named `eval_episodes`.

### Your own

When an environment needs special handling, just write the function:

```python
def trial(params):
    env = build_my_env(params)
    agent = train_somehow(env, params)
    return evaluate(env, agent, gamma=1.0, episodes=50)

Study(name='custom', trial=trial, space=space, store='results')
```

Add a second argument to receive a context dict with `run_id`, `study`,
`origin`, and `parent_id`.

## Scoring

The default score is mean return minus one standard deviation: it prefers
agents that are reliably good over agents that are occasionally brilliant. Pass
`score_fn=` to change it.

```python
from rlmine import mean_return, success_rate

Study(..., score_fn=success_rate)     # needs check_success=True in the trial
Study(..., score_fn=lambda s: s['mean_return'] - 2 * s['stdev_return'])
```

## Where results go

The recommended source of truth is a JSONL file: one JSON object per line,
appended after each trial.

```python
store = 'drive/MyDrive/rl_mining'          # shorthand; mounts Drive if needed
store = JSONLStore('results', 'my-study')  # explicit
```

JSONL is preferred over a spreadsheet for three reasons. Types survive the
round trip, so `net_arch` comes back as `[256, 256]` rather than the string
`'[256, 256]'` and a boolean comes back as a boolean rather than `'TRUE'`.
Appending never rewrites earlier rows, so a crash cannot corrupt history and
two notebooks can mine the same study at once. And because the file is
append-only, a run's index is stable, which makes `parent=6` mean the same
thing tomorrow as it does today.

To keep watching results live in a Google Sheet, attach a mirror. Mirror
failures are warnings, never errors, so a flaky Sheets call cannot interrupt a
multi-hour run.

```python
from rlmine.stores import JSONLStore, SheetMirror

Study(
    ...,
    store  = JSONLStore('drive/MyDrive/rl_mining', 'my-study'),
    mirror = SheetMirror('https://docs.google.com/spreadsheets/d/...'),
)
```

`CSVStore` is available if opening the file directly in Excel matters more than
type fidelity.

### Reading results

```python
study.table(n=20)      # tidy, sorted by score
study.best(3)          # the three best runs
study.history()        # everything, unmodified
study.params_of(6)     # one run's parameters, ready to pass back into run()
study.drift_report()   # rechecks against their originals
```

## Migrating existing Google Sheets

Import accumulated history once; the string cells are converted to real types
on the way in.

```python
from google.colab import sheets

old = sheets.InteractiveSheet(url=SHEET_URL, backend='pandas', display=False).as_df()
study.import_rows(old)
```

Imported rows are immediately usable as mining parents and as recheck targets,
which is the point: drift checking needs the history. Mirroring is off for
imports, so this will not duplicate rows back into the sheet it just read.

## Notes

Trials are wrapped: if one raises, the row is written with `status='failed'` and
the error message, and the loop continues. An overnight run of ten models will
not be lost to a single out-of-memory error.

Colab runtimes are ephemeral, so write results to Drive or to a sheet. The
default store is a local `results/` directory, which is fine for testing and
will vanish when the runtime restarts.

## Example

[`examples/Bipedal Walker A2C - rlmine.ipynb`](examples) is the Lab 7 mining
notebook rewritten to use this package.

## Tests

```
pytest tests
```
