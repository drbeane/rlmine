# Results methods

`Results` opens a results file and answers questions about it. It never runs anything and never rewrites the file.

```python
from rlmine import Results

results = Results("results/bipedal-walker-a2c.json")
```

Pass `config=` (a dict, or the path of a JSON file) to shade, in `table` and `best`, every setting that differs from that reference config.

## Methods

| Method | Returns |
| --- | --- |
| `summary()` | One table: best and worst latest score (with their config ids), mean score, number of configs and runs, total minutes, first and last timestamp |
| `table(n=None, params="varying", latest=False)` | One row per run, sorted by score, with the settings that vary |
| `best(k=10, params="varying", latest=True)` | The `k` highest-scoring runs, one per config by default |
| `configs(params="varying")` | One row per config: latest score, `best`, `mean`, `worst`, runs `n`, total `minutes` |
| `by(field, latest=False)` | Runs, best score, and mean score for each value of one column, such as `"initial_lr"` |
| `runs(config_id)` | Every run of one config, with its result, date, and package versions |
| `get(config_id)` | One config as a dict: its settings, its runs, and score summaries |
| `compare(*config_ids)` | The named configs side by side, keeping only the settings that differ |
| `drift()` | Each repeat of a config against the run before it, with package versions, so a score change can be traced to a library change |
| `filter(*conditions)` | Another `Results` holding only the matching runs. See [filter.md](filter.md) |
| `len(results)` | Number of runs |

## Options

- **`params`**: `"varying"` shows only settings that differ between configs, `"all"` shows every setting, and `"none"` shows none.
- **`latest`**: `True` keeps one row per config, the run with status `latest`.

## Status

Every run is marked relative to the most recent run of its config:

- `latest`: the most recent run.
- `current`: an earlier run with the same result as the latest.
- `outdated`: an earlier run with a different result.

`best` uses the latest run of each config, so an old score cannot outrank what that config scores now.

## Examples

```python
results.summary()
results.best(5)
results.table(20)
results.configs()
results.by("initial_lr")
results.runs(323629)
results.compare(323629, 844803)
results.drift()
results.filter("timesteps <= 50_000").best(5)
```
