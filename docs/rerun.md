# Rerunning for drift

`Study.rerun` replays stored configs on the current runtime and reports whether the score changed. Use it after a package update, or on a new Colab session, to see whether old results still hold.

```python
walker.rerun(config_id=927170)   # one config
walker.rerun(top=5)              # the 5 best configs for this runtime
```

Pass exactly one of `config_id` or `top`.

## One config

If the config was recorded on the current runtime, it is replayed as a new run under the same id and compared with its latest score from before the replay.

If it was recorded on a different runtime, `rerun` says so and stops. Nothing is run, because the same agent on another runtime is a different config.

## Top k

`top=k` takes the latest score of every config recorded on the current runtime, ranks them, and replays the best `k`. The ranking is fixed before any replay, so a replay cannot change which configs are picked. If there are no configs for this runtime, nothing is run.

## The report

One line is printed per config:

```
[623469] DRIFTED: 4.48 -> 4.19 (-0.29)
[517622] no drift: 2.11
```

The same information is returned as a DataFrame:

| Column | Meaning |
| --- | --- |
| `config_id` | The config replayed |
| `previous`, `previous_date` | Its latest score and date before the replay |
| `new`, `date` | The replay's score and date |
| `delta` | `new - previous` |
| `drifted` | `True` if `delta` is not zero |

Drift is any difference in the stored (rounded) score; there is no tolerance. A replay that fails is not recorded and shows as "could not be compared".

Each replay is an ordinary run, so it becomes the config's latest result. The earlier scores stay in the file.
