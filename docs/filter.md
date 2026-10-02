# Filtering results

`Results.filter` keeps the runs that match a condition and returns another `Results`, so everything else (`best`, `table`, `by`, `configs`) works on the narrowed set. The file is never changed.

```python
from rlmine import Results, config, run

results = Results("results/bipedal-walker-a2c.json")
```

## Strings

A string is evaluated like `DataFrame.query` over the column names that `table()` shows.

```python
results.filter("timesteps <= 50_000")
results.filter("timesteps <= 50_000 and score > 200")
results.filter("runtime == 'gpu' or minutes < 5")
results.filter("not runtime == 'gpu'")
results.filter("learning_rate in [0.001, 0.003]")
results.filter("n_steps.isnull()")            # runs with no value for n_steps
```

Use a Python variable with `@`:

```python
limit = 50_000
results.filter("timesteps <= @limit")
```

Where two sections share a name, the short name is the agent's. Use `section_name` to be explicit:

```python
results.filter("training_timesteps <= 50_000")
results.filter("evaluation_gamma == 1.0")
results.filter("agent_gamma >= 0.99")
```

## Objects

`config` reads a setting by the path it is stored under. `run` reads a field of the recorded run.

```python
results.filter(config.training.timesteps <= 50_000)
results.filter(config.runtime == "gpu")
results.filter(run.score > 200)
results.filter(run.minutes < 5)
results.filter(run.date >= "2026-09-01")
```

Helpers:

```python
results.filter(config.training.timesteps.between(100_000, 500_000))
results.filter(config.training.timesteps.isin([250_000, 500_000]))
results.filter(config.agent.n_steps.isnull())
results.filter(config.agent.n_steps.notnull())
```

Join conditions with `&`, `|`, and `~`. Each comparison needs its own parentheses.

```python
results.filter((config.agent.gamma >= 0.99) | (run.minutes < 5))
results.filter(~(config.runtime == "gpu"))
```

Conditions passed as separate arguments must all hold:

```python
results.filter(config.training.timesteps <= 50_000, run.score > 200)
```

If `config` clashes with a variable in your notebook, import it under another name:

```python
from rlmine import config as c
results.filter(c.training.timesteps <= 50_000)
```

## Mixing and chaining

```python
results.filter("score > 200", config.runtime == "gpu")
results.filter("timesteps <= 50_000").filter(run.minutes < 5)
```

## Common uses

The best short runs:

```python
results.filter("timesteps <= 50_000").best(5)
```

Compare one setting among runs with the same training budget:

```python
results.filter("timesteps == 500_000").by("initial_lr")
```

Everything that ran on a T4:

```python
results.filter(config.runtime == "gpu").table()
```

Only the runs that are still the current result of their config:

```python
results.filter("status == 'latest'").best(10)
```

## Rules

- A run with no value for a setting never matches an object comparison, including `!=`. Use `.isnull()` to find those runs. A string `!=` does match them.
- A name that no run has raises a `KeyError` listing the names you can use, so a typo does not silently return nothing.
- Each run keeps the `status` (`latest`, `current`, `outdated`) it has in the whole file. Filtering on an old score will not make that run look like its config's latest result.
- `configs`, `get`, and `compare` summarise only the runs that matched.
