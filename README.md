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
        timesteps     = P.Int(300_000),
        learning_rate = P.Float(7e-4, sig=2, bounds=(0, None),
                                sample=P.loguniform(1e-5, 1e-2),
                                mutate=P.scale(0.2)),
        gamma         = P.Float(0.99, digits=3, bounds=(0, 1),
                                sample=(0.9, 0.999),
                                mutate=P.scale(0.02)),
        net_arch      = P.Choice([[128, 128], [256, 256]], default=[256, 256]),
    ),
    store = 'drive/MyDrive/rl_mining',
)

lander.run(n=10, sample=['learning_rate', 'gamma'])     # sample some, default the rest
lander.run(learning_rate=1e-3)                          # a specific configuration
lander.mine(n=5, mutate=['learning_rate'])              # explore near good results
lander.recheck(top=3)                                   # has anything drifted?
lander.table()                                          # read the results
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
| `study.run(...)` | Per parameter: a given value, the default, a sample, or a mutation of a parent |
| `study.mine(n, mutate=[...])` | A roulette-selected parent, with the named parameters perturbed |
| `study.recheck(top=k)` | A stored row, replayed verbatim |

### Run: pick a behaviour for each parameter

Each call states, for every variable, one of four things:

- a **given value** — pass it as a keyword argument
- the **default** — omit it, or pass `P.DEFAULT`
- **sample** from the declared range — `sample=['...']` or `P.SAMPLE`
- **mutate** from a parent — `mutate=['...']` or `P.MUTATE` (needs `parent=`)

```python
# Sample three parameters, use the default for everything else
study.run(n=10, sample=['learning_rate', 'gamma', 'n_steps'])

# Pin two values, sample one, default the rest
study.run(n=10, learning_rate=1e-3, gamma=0.995, sample=['n_steps'])

# One exact configuration (everything else is the default)
study.run(n=1, learning_rate=1e-3, gamma=0.995)

# Mutate two parameters from a known config; inherit the rest.
# parent is the integer id of that config.
study.run(n=10, parent=184203, mutate=['learning_rate', 'gamma'])

# Same ideas as sentinels, if you prefer them next to the values
study.run(n=10, learning_rate=P.SAMPLE, gamma=P.DEFAULT, n_steps=P.SAMPLE)
study.run(n=10, parent=184203, learning_rate=P.MUTATE, gamma=P.MUTATE)
```

You can also state a range or transformation for this run only:

```python
study.run(n=10, sample={'learning_rate': P.loguniform(1e-5, 1e-3)})
study.run(n=10, parent=184203, mutate={'gamma': P.scale(0.05)})
study.run(n=10, learning_rate=P.SAMPLE(P.loguniform(1e-5, 1e-3)))
```

### Explore

```python
study.mine(n=10, mutate=['learning_rate', 'gamma'])   # roulette-select a parent
study.mine(n=5, mutate='all', parent=184203)          # pin a parent config
study.mine(n=5, mutate=['gamma'], timesteps=50_000)   # pin a value while mining
```

A parent is selected for each model, so a good result found early in a call can
seed later ones. Selection is weighted by score among runs that scored above
zero; failed runs are never selected. With no history at all, mutated
parameters are sampled from their declared range (if they have one) and the
rest use their defaults.

### Check for drift

```python
study.recheck(top=3)             # the three best configurations
study.recheck(latest=5)          # the five most recently run
study.recheck(rows=[184203])     # one config, by its id
study.drift_report()             # earlier score vs later score, per config
```

Every row stores the Python, Gymnasium, Stable-Baselines3, and Torch versions
it ran under, plus the accelerator, so a score that has moved can be traced to
a library change rather than guessed at.

## Defining a space

Each parameter declares a **default** (used when you are not sampling or
mutating it) and **hard bounds** (always applied). Next to those you can
state the **sample range** and the **mutation** used when evolving.

```python
space = dict(
    timesteps     = P.Int(1_000_000, bounds=(1, None)),
    n_steps       = P.Int(5, bounds=(1, None),
                          sample=(2, 32),
                          mutate=P.times([0.5, 1, 2])),
    learning_rate = P.Float(7e-4, sig=2, bounds=(0, None),
                            sample=P.loguniform(1e-5, 1e-2),
                            mutate=P.scale(0.2)),
    gamma         = P.Float(0.99, digits=3, bounds=(0, 1),
                            sample=(0.9, 0.999),
                            mutate=P.scale(0.02)),
    net_arch      = P.Choice([[128, 128], [256, 256], [512, 512]]),
    use_sde       = P.Bool(False),
)
```

**Types.** `P.Int`, `P.Float`, `P.Bool`, `P.Choice`. Floats take `sig` (round to
significant digits, right for learning rates) or `digits` (decimal places,
right for gamma). All take `bounds=(low, high)`, where `None` means unbounded.
Sampled and mutated values are clipped to these bounds.

**Sample range** is what `sample=` / `P.SAMPLE` draws from. For numeric
parameters a two-number range `(low, high)` is uniform; use `P.loguniform(a, b)`
for log scale, or `P.choice([...])` for an explicit list. `P.Choice` already
samples from its options. Asking to sample a parameter with no range is an
error.

**Mutations** apply when a parameter is named in `mutate=[...]` or passed as
`P.MUTATE`.

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
things like `timesteps` and `seed` that you want to hold constant when
inheriting from a parent.

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
    digits=4,
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

Add a second argument to receive a context dict with `config_id`.

## Scoring

The default score is mean return minus one standard deviation: it prefers
agents that are reliably good over agents that are occasionally brilliant. Pass
`score_fn=` to change it.

```python
from rlmine import mean_return, success_rate

Study(..., score_fn=success_rate)     # needs check_success=True in the trial
Study(..., score_fn=lambda s: s['mean_return'] - 2 * s['stdev_return'])
```

`digits` on the evaluation block sets how many decimal places are kept.
Every numeric element of the stats is rounded first, including each item of a
list or array, and the score is then computed from those rounded values.
`sb3_trial` and `tabular_trial` take `digits=` and store it on that block.
The A2C example uses 4. Omit it to keep full precision. A custom trial can
set the same key in `config_for`:

```python
"evaluation": {"episodes": 50, "score": score, "digits": 4}
```

## Where results go

The recommended source of truth is a JSON document, one per study. Each
config is a key: a random integer from 100000 to 999999, so the id never
starts with 0. JSON object keys are strings, so the file stores the decimal
form of that integer. Under the key, `config` is everything you set that can
change the score: environment, agent, training, evaluation, and `runtime`.
`runs` holds every trial of that exact config: the score, when it was
recorded, and the package versions.

```json
{
  "184203": {
    "config": {
      "environment": {"id": "BipedalWalker-v3", "n_envs": 16},
      "agent": {"algo": "A2C", "initial_lr": 0.0007, "gamma": 0.99, "seed": 1},
      "training": {"timesteps": 1000000, "n_eval_episodes": 20},
      "evaluation": {"episodes": 50, "gamma": 1.0, "score": "mean_minus_std", "digits": 4},
      "runtime": "Tesla T4"
    },
    "runs": [
      {
        "score": -110.3439,
        "timestamp": "2026-08-26T01:06:27",
        "python": "3.13.15",
        "torch": "2.11.0"
      }
    ]
  }
}
```

A mined hyperparameter and a fixed setting such as `evaluation.episodes` both
belong on the config. Seed 1 and seed 2 are different configs, and so are the
same agent on CPU and on a Tesla T4. Running that same config again appends
another object to `runs` instead of creating a new id, which is what makes a
recheck comparable to the earlier scores. Library versions stay on each run,
so a later recheck can show whether a score moved because Torch or Gymnasium
changed.

```python
store = 'drive/MyDrive/rl_mining'          # shorthand; mounts Drive if needed
store = JSONStore('results', 'my-study')  # explicit
```

JSON is preferred over a spreadsheet because types survive the round trip, so
`net_arch` comes back as `[256, 256]` rather than the string `'[256, 256]'`
and a boolean comes back as a boolean rather than `'TRUE'`. Each write
replaces the file atomically, so a crash mid-write keeps the previous
document. `parent=184203` means that config.

An older file with one flat JSON object per line is grouped into this shape
the first time it is read. `history()` and `table()` still show one row per
run, with a `config_id` column.

To keep watching results live in a Google Sheet, attach a mirror. Constructing
``SheetMirror`` opens the sheet and probes a write immediately, so Colab's
OAuth prompt happens before any trial starts. Later mirror failures are
warnings, never errors, so a flaky Sheets call cannot interrupt a multi-hour
run.

```python
from rlmine.stores import JSONStore, SheetMirror

Study(
    ...,
    store  = JSONStore('drive/MyDrive/rl_mining', 'my-study'),
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
study.drift_report()   # later scores against the one before, per config
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

Trials are wrapped: if one raises, that run is not written, and the loop
continues. An overnight run of ten models will not be lost to a single
out-of-memory error.

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
