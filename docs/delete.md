# Deleting runs and configs

`Results` can delete from the results file. Open the file and call one of three methods. Each one rewrites the file, reloads the `Results` so `best`, `table` and the rest show the new state, and prints what it did. None of them returns anything.

```python
from rlmine import Results

results = Results("results/bipedal-walker-a2c.json")
```

Deleting cannot be undone. Copy the file first if you are unsure.

## Delete runs by run id

The run id is the `idx` column in `table()` and `runs()`. Pass one id, several, or a list.

```python
results.table()                       # find the idx values
results.runs(184203)                  # every run of one config

results.delete_runs(4)
results.delete_runs(4, 7, 9)
results.delete_runs([4, 7, 9])
```

```
Deleted 3 run(s) and 0 config(s).
```

If you delete the last run of a config, the config is removed too, and the message counts it:

```
Deleted 1 run(s) and 1 config(s).
```

An id that does not exist is skipped and reported. The others are still deleted:

```python
results.delete_runs(4, 9999)
```

```
Deleted 1 run(s) and 0 config(s).
Not found: [9999]
```

Run ids are reused. A new run takes the highest `idx` plus one, so after you delete the highest-numbered run, the next run gets that number again.

## Delete whole configs

The config id is the six-digit `config_id`. This removes the config and every run it has.

```python
results.delete_configs(184203)
results.delete_configs(184203, 551920)
results.delete_configs([184203, 551920])
```

Combine it with a query to remove a group of configs. Read the ids off the filtered view:

```python
slow = results.filter("minutes > 120").configs()
results.delete_configs(slow["config_id"])
```

## Delete duplicates

Two runs of the same config are duplicates when everything matches except `idx`, `date`, `timestamp` and `minutes`. That means the same score and score components, and the same package versions. The most recent run in each group is kept.

A run with a different score, or recorded under a different package version, is not a duplicate and stays.

Preview first:

```python
results.delete_duplicates(dry_run=True)
```

```
Would delete 5 run(s): [2, 3, 8, 11, 12]
```

Then delete:

```python
results.delete_duplicates()
```

```
Deleted 5 run(s) and 0 config(s).
```

## Filtered results

A `Results` from `filter(...)` refuses to delete and raises `ValueError`. Its view is only part of the file, and a delete would reload the whole file. Use the filter to find ids, then delete from the `Results` you opened:

```python
bad = results.filter("score < 0")
ids = list(bad.table()["idx"])        # run ids

results.delete_runs(ids)
```

## Notes

- Only a `.json` results file is changed. A `CSVStore` or `MemoryStore` cannot delete.
- A `SheetMirror` is not updated. Deleted rows stay in the sheet.
- The same deletes exist on `JSONStore` as `delete_runs(ids)`, `delete_configs(ids)` and `delete_duplicates(dry_run=False)`. They take a list and return a dict of what was deleted, for use in code.
