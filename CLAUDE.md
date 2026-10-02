# rlmine

Hyperparameter mining for RL labs. Train many agents, score them, log every result, and use good ones to find better ones. Algorithm- and environment-agnostic; helpers ship for Stable-Baselines3 and for tabular agents in [rltools](https://github.com/drbeane/rltools).

Python >= 3.9. Required: `numpy`, `pandas`. Optional: `stable-baselines3`, `gymnasium`, `rltools`. Tests: `pytest tests`.

## Layout

- `study.py` — `Study`: the public API (`run`, `mine`, `recheck`, `table`, `best`, `history`, `drift_report`)
- `params.py` / `space.py` — parameter types (`P.Int`, `P.Float`, `P.Bool`, `P.Choice`) and the search space
- `scoring.py` — default score is mean return minus one standard deviation
- `stores.py` — `JSONStore` (source of truth; one JSON document, `JSONLStore` is the old name), `SheetMirror`, `CSVStore`, `MemoryStore`
- `trials/` — `sb3_trial`, `q_learning_trial`, `mc_trial`, `tabular_trial`

## The three operations

All three run a trial, score it, and append a row. They differ only in where parameters come from.

| Call | Parameters |
| --- | --- |
| `run(...)` | given value, default, `sample=`, or `mutate=` from a parent |
| `mine(n, mutate=[...])` | roulette-selected parent, named parameters perturbed |
| `recheck(...)` | a stored row, replayed verbatim |

Each run records its score, when it was recorded, and the package versions. A trial that raises is not written; the loop continues.

## Config key

One JSON document per study. Each key is a random integer (100000–999999) for one config. Repeats of that config append to its `runs`. Writes are atomic. `history()` / `table()` flatten to one row per run.

The key is every user-set choice that can change the score, from all four buckets below, plus `runtime`. A mined hyperparameter and a fixed setting such as `eval_episodes` both belong on it. Anything left off the key will make incomparable runs look like the same config. Switching CPU and GPU is a different config, since that choice is under your control.

Scores for one key may still drift. The only acceptable cause is a package change, recorded on the run (`python`, `gymnasium`, `stable_baselines3`, `torch`). Logging flags (`verbose`, `progress_bar`, `show_report`) are not part of the key either.

A trial is any callable `(params) -> stats` with `mean_return` and `stdev_return`. An optional second argument receives `{config_id}`.

Colab runtimes are ephemeral. Persist to Drive or a sheet; the default local `results/` directory is for tests only.

### Only the latest score matters

For a config, the latest run is its result. Older runs are kept for posterity and for comparison, but they can no longer be reproduced, so they are not important. Rank, compare, and report on the latest run. Never average a config's scores: the mean over its runs is meaningless. Result views show the latest score and its date, not run counts or package versions.

`Results.runtimes(k)` compares the top setups. Configs that differ only in `runtime` form one setup, ranked by the best latest score among its runtimes. Each row shows the overall `score`, `timesteps`, and, per runtime checked, the config id, its latest score, the date of that run, and its minutes. Nothing else.

### Buckets

Most fields are fixed when the study is built. `timesteps` and `eval_episodes` are typical examples. Only fields placed in the search space are sampled or mutated. Fixed fields are still stored on the key; a manual change creates a different config, not a search.

| Bucket | What it holds | Old notebooks |
| --- | --- | --- |
| Environment | Id, constructor kwargs, vectorization, wrappers | `BipedalWalker-v3` with `n_envs` and `VecNormalize`; `LunarLander-v3` with `n_envs`; FrozenLake `prob`, `rew_struct`, map; CartPole `num_bins` |
| Agent | Algorithm, policy, and its hyperparameters, including its seed and its discount | A2C / DQN constructor and `policy_kwargs` (learning rate or schedule, `gamma`, `net_arch`, `n_steps`, exploration, buffer); `MCAgent.control` / `TDAgent.q_learning` (`alpha`, `epsilon`, decays, `exploring_starts`, agent `gamma`) |
| Training | How long to train, and any in-training evaluation that chooses the weights | `timesteps` or training `episodes`; SB3 `EvalCallback` (`eval_freq`, `n_eval_episodes`, `deterministic`); tabular `max_steps`, `updates`, `eval_eps` |
| Evaluation | The final scoring run, including the score formula and rounding | `evaluate` episodes, `max_steps`, seed, `gamma`, `deterministic`, `check_success`; score = mean minus one standard deviation; `digits` rounds the score and every element of its components before they are stored |

The callback is part of training. It decides which checkpoint is scored, and the tabular labs have no callback; `eval_eps` is the same stage. Agent `gamma` and evaluation `gamma` stay separate: Bipedal learns at 0.99 and scores at 1.0. `runtime` sits beside the buckets, not inside one: it is not a pipeline stage.

## Prior work

Earlier attempts live in `temp/Old Versions`: Bipedal Walker mining, Lunar Lander DQN (CPU, T4 GPU, and parameter mining), CartPole Q-learning and Monte Carlo control, and FrozenLake Q-learning and Monte Carlo control. The finished app must cover every use case in those notebooks.
