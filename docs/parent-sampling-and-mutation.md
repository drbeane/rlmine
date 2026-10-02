# Parent sampling and mutation

A parent is a stored config. `study.run` starts from it, applies your mutations to the fields you name, trains, scores, and stores the child as its own config.

The parent comes from `config_id`: one id, or a `Results` object (usually filtered) that `run` draws from on every trial.

```python
from rlmine import Results, run, config
from rlmine import mutations as m
```

## Mutating one config

Ten children of config 927170, each with its learning rate scaled by a random factor between 0.8 and 1.2:

```python
study.run(n=10, config_id=927170,
          agent={"learning_rate": m.proportional(0.2, sig=2)})
```

Several fields at once, with bounds and rounding:

```python
study.run(n=10, config_id=927170, agent={
    "learning_rate": m.log_proportional(2, sig=2),   # between half and double
    "gamma":         m.shift(0.01, digits=3, high=1.0),
    "n_steps":       m.factor([0.5, 2]),             # halve or double
    "use_sde":       m.flip(0.2),                    # 20% chance to flip
    "net_arch":      m.choice([[64, 64], [128, 128]]),
})
```

Each trial mutates the parent again. A mutated child is never the parent of the next trial in the same call. Fields you don't name stay as they are in the parent.

## Sampling parents from Results

Filter a results file down to the configs worth building on, then pass it as `config_id`. Every trial draws its own parent.

```python
good = Results("results/walker.json").filter(run.score > 200)
```

**Uniform** (the default): every config in `good` is equally likely.

```python
study.run(n=20, config_id=good,
          agent={"learning_rate": m.proportional(0.2, sig=2)})
```

**Roulette**: higher-scoring configs are drawn more often.

```python
study.run(n=20, config_id=good, parent_sampling="roulette",
          agent={"learning_rate": m.proportional(0.2, sig=2)})
```

Filters can combine settings and run fields:

```python
gpu_best = Results(path).filter(config.runtime == "gpu", run.score > 150)
long_runs = Results(path).filter("timesteps >= 500_000 and score > 200")

study.run(n=10, config_id=gpu_best, parent_sampling="roulette", seed=0,
          agent={"gamma": m.shift(0.01, digits=3, high=1.0)})
```

With `verbose` on, each trial's header names its parent, for example `[412908]  (from 927170)  learning_rate=0.00061`.

## How roulette weighs configs

- Sampling is by config, not by run. A config rechecked many times is not favored.
- A config's weight is its latest score minus the lowest score in the `Results`. The lowest-scoring config therefore has weight zero and is never drawn.
- Configs with no score are never drawn by roulette.
- If every score is equal, roulette falls back to uniform.
- `seed` seeds both the parent draws and the mutations.

## Errors

- `parent_sampling` other than `"uniform"` or `"roulette"` raises `ValueError`.
- An empty `Results` raises `ValueError`.
- Roulette on a `Results` where no config has a score raises `ValueError`.
