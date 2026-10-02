"""Tests for the mining engine, using a cheap synthetic trial function."""

import warnings

import numpy as np
import pandas as pd
import pytest

from rlmine import JSONLStore, MemoryStore, SheetMirror, Space, Study
from rlmine import params as P
from rlmine import samplers as s
from rlmine.trials.sb3 import route_params
from rlmine.utils import round_sig


def quadratic_trial(config):
    """A fast stand-in for training: score peaks at learning_rate == 0.005."""
    lr = config["agent"]["learning_rate"]
    mean = 100.0 - 500_000 * (lr - 0.005) ** 2
    return {"mean_return": mean, "stdev_return": 5.0}


def make_space():
    return dict(
        timesteps=P.Int(100_000, mutate="fixed"),
        learning_rate=P.Float(
            7e-4, sig=2, bounds=(1e-6, 1.0),
            sample=P.loguniform(1e-5, 1e-1), mutate=P.scale(0.2),
        ),
        gamma=P.Float(0.99, digits=3, bounds=(0, 1), sample=(0.9, 0.999)),
        net_arch=P.Choice([[64, 64], [128, 128], [256, 256]], default=[128, 128]),
        use_sde=P.Bool(False),
    )


def make_config(**overrides):
    config = {
        "environment": {"id": "toy"},
        "agent": {
            "learning_rate": 7e-4,
            "gamma": 0.99,
            "net_arch": [128, 128],
            "use_sde": False,
        },
        "training": {"timesteps": 100_000},
        "evaluation": {"episodes": 50, "score": "mean_minus_std"},
    }
    for section, values in overrides.items():
        config[section].update(values)
    return config


def make_study(tmp_path, **kwargs):
    kwargs.setdefault("verbose", False)
    config = kwargs.pop("config", None) or make_config()
    return Study(
        name="test-study",
        trial=quadratic_trial,
        config=config,
        store=JSONLStore(tmp_path, "test-study"),
        **kwargs,
    )


# ---------------------------------------------------------------- parameters


def test_float_rounds_to_significant_digits():
    param = P.Float(7e-4, sig=2)
    assert param.clean(0.00067891) == 0.00068


def test_float_clips_to_bounds():
    param = P.Float(0.99, digits=3, bounds=(0, 1))
    assert param.clean(1.4) == 1.0
    assert param.clean(-0.2) == 0.0


def test_int_mutation_halves_or_doubles():
    param = P.Int(4, bounds=(1, None), mutate=P.times([0.5, 1, 2]))
    rng = np.random.default_rng(0)
    values = {param.perturb(4, rng) for _ in range(50)}
    assert values <= {2, 4, 8}


def test_fixed_mutation_never_changes_the_value():
    param = P.Int(300_000, mutate="fixed")
    rng = np.random.default_rng(0)
    assert all(param.perturb(300_000, rng) == 300_000 for _ in range(20))


def test_choice_mutation_stays_within_options():
    options = [[64, 64], [128, 128], [256, 256]]
    param = P.Choice(options, default=[128, 128])
    rng = np.random.default_rng(0)
    assert all(param.perturb([64, 64], rng) in options for _ in range(30))


def test_scale_mutation_respects_its_range():
    param = P.Float(1.0, mutate=P.scale(0.2))
    rng = np.random.default_rng(0)
    values = [param.perturb(1.0, rng) for _ in range(200)]
    assert all(0.8 <= v <= 1.2 for v in values)
    assert len(set(values)) > 1


def test_numeric_tuple_sample_is_uniform():
    param = P.Float(0.5, digits=3, bounds=(0, 1), sample=(0.1, 0.2))
    rng = np.random.default_rng(0)
    values = [param.sample_value(rng) for _ in range(50)]
    assert all(0.1 <= v <= 0.2 for v in values)


def test_sampling_without_a_range_is_an_error():
    param = P.Float(0.5)
    with pytest.raises(ValueError, match="no sample range"):
        param.sample_value(np.random.default_rng(0), name="gamma")


@pytest.mark.parametrize(
    "param,raw,expected",
    [
        (P.Int(0), "1,000,000", 1_000_000),
        (P.Int(0), "2_000_000", 2_000_000),
        (P.Float(0.0, sig=3), "7.0e-04", 0.0007),
        (P.Bool(False), "TRUE", True),
        (P.Bool(True), "FALSE", False),
        (P.Choice([[256, 256]], default=[256, 256]), "[256, 256]", [256, 256]),
    ],
)
def test_parsing_spreadsheet_cells(param, raw, expected):
    assert param.clean(param.parse(raw)) == expected


# -------------------------------------------------------------------- space


def test_space_rejects_plain_values():
    with pytest.raises(TypeError, match="Param objects"):
        Space({"learning_rate": 0.001})


def test_space_accepts_params_from_another_module():
    """Reloading rlmine must not invalidate an already-built space dict."""

    class Int:
        def __init__(self, default):
            self.default = default

        def clean(self, value):
            return int(value)

    space = Space({"timesteps": Int(1_000_000)})
    assert space.names == ["timesteps"]
    assert space["timesteps"].default == 1_000_000


def test_resolve_store_accepts_store_from_another_module(tmp_path):
    """Reloading rlmine must not reject a JSONLStore built by the previous import."""

    class JSONLStore:
        def __init__(self):
            self.records = []

        def append(self, record):
            self.records.append(record)

        def load(self):
            return pd.DataFrame(self.records)

        @property
        def location(self):
            return "foreign"

    foreign = JSONLStore()
    study = Study(
        name="reload-store",
        trial=quadratic_trial,
        config=make_config(),
        store=foreign,
        verbose=False,
    )
    study.run()
    assert len(foreign.records) == 1


def test_resolve_store_still_rejects_garbage():
    with pytest.raises(TypeError, match="as a store"):
        Study(
            name="bad-store",
            trial=quadratic_trial,
            config=make_config(),
            store=object(),
            verbose=False,
        )


def test_unknown_override_is_an_error():
    space = Space(make_space())
    with pytest.raises(KeyError, match="learnig_rate"):
        space.derive(overrides={"learnig_rate": 0.1})


def test_unknown_mutate_name_is_an_error():
    space = Space(make_space())
    with pytest.raises(KeyError, match="lr"):
        space.derive(mutate=["lr"])


def test_override_beats_mutation():
    space = Space(make_space())
    parent = space.defaults()
    values = space.derive(
        parent=parent,
        mutate=["learning_rate"],
        overrides={"learning_rate": 0.003},
        rng=np.random.default_rng(0),
    )
    assert values["learning_rate"] == 0.003


def test_unmutated_parameters_are_inherited():
    space = Space(make_space())
    parent = dict(space.defaults(), gamma=0.95, net_arch=[256, 256])
    values = space.derive(
        parent=parent, mutate=["learning_rate"], rng=np.random.default_rng(0)
    )
    assert values["gamma"] == 0.95
    assert values["net_arch"] == [256, 256]
    assert values["learning_rate"] != parent["learning_rate"]


def test_constraints_are_applied():
    space = Space(
        {"initial_lr": P.Float(1e-3, sig=2), "final_lr": P.Float(1e-2, sig=2)},
        constraints=lambda p: {**p, "final_lr": min(p["initial_lr"], p["final_lr"])},
    )
    values = space.derive()
    assert values["final_lr"] == values["initial_lr"] == 1e-3


def test_parse_row_recovers_types_from_strings():
    space = Space(make_space())
    row = {
        "timesteps": "1,000,000",
        "learning_rate": "7.0e-04",
        "gamma": "0.95",
        "net_arch": "[256, 256]",
        "use_sde": "TRUE",
    }
    values = space.parse_row(row)
    assert values == {
        "timesteps": 1_000_000,
        "learning_rate": 0.0007,
        "gamma": 0.95,
        "net_arch": [256, 256],
        "use_sde": True,
    }


def test_derive_samples_named_parameters_and_defaults_the_rest():
    space = Space(make_space())
    rng = np.random.default_rng(0)
    values = space.derive(sample=["learning_rate", "gamma"], rng=rng)
    assert values["timesteps"] == 100_000
    assert values["net_arch"] == [128, 128]
    assert values["use_sde"] is False
    assert 1e-5 <= values["learning_rate"] <= 1e-1
    assert 0.9 <= values["gamma"] <= 0.999


def test_derive_lifts_sample_out_of_overrides():
    """Older Study.run(n=1, **overrides) leaked sample= into the param dict."""
    space = Space(make_space())
    rng = np.random.default_rng(0)
    values = space.derive(
        overrides={"sample": ["gamma"], "learning_rate": 0.005},
        rng=rng,
    )
    assert values["learning_rate"] == 0.005
    assert values["timesteps"] == 100_000
    assert 0.9 <= values["gamma"] <= 0.999


def test_derive_uses_a_given_value():
    space = Space(make_space())
    values = space.derive(overrides={"learning_rate": 0.003, "gamma": P.DEFAULT})
    assert values["learning_rate"] == 0.003
    assert values["gamma"] == 0.99


def test_derive_sample_sentinel_and_call_time_range():
    space = Space(make_space())
    rng = np.random.default_rng(1)
    values = space.derive(
        overrides={"learning_rate": P.SAMPLE(P.loguniform(0.01, 0.02)), "gamma": P.SAMPLE},
        rng=rng,
    )
    assert 0.01 <= values["learning_rate"] <= 0.02
    assert 0.9 <= values["gamma"] <= 0.999


def test_derive_mutates_from_parent_and_can_force_default():
    space = Space(make_space())
    parent = dict(space.defaults(), learning_rate=0.005, gamma=0.9)
    rng = np.random.default_rng(0)
    values = space.derive(
        parent=parent,
        mutate=["learning_rate"],
        overrides={"gamma": P.DEFAULT, "net_arch": P.PARENT},
        rng=rng,
    )
    assert values["learning_rate"] != 0.005
    assert values["gamma"] == 0.99
    assert values["net_arch"] == [128, 128]
    assert values["timesteps"] == 100_000


def test_derive_call_time_mutation():
    space = Space(make_space())
    parent = space.defaults()
    rng = np.random.default_rng(0)
    values = space.derive(
        parent=parent,
        mutate={"gamma": P.shift(0.05)},
        rng=rng,
    )
    assert values["gamma"] != parent["gamma"]
    assert 0.0 <= values["gamma"] <= 1.0


def test_sample_without_declared_range_errors():
    space = Space({"n_steps": P.Int(5, bounds=(1, None))})
    with pytest.raises(ValueError, match="n_steps"):
        space.derive(sample=["n_steps"], rng=np.random.default_rng(0))


# ------------------------------------------------------------------- stores


def test_jsonl_roundtrip_preserves_types(tmp_path):
    import json

    store = JSONLStore(tmp_path, "types")
    store.append({"net_arch": [256, 256], "use_sde": True, "timesteps": 300_000})

    # The stored representation keeps real types, so no string parsing is
    # needed on the way back in. This is the whole reason for preferring JSON
    # over a spreadsheet as the source of truth.
    stored = json.loads((tmp_path / "types.json").read_text())
    assert len(stored) == 1
    pid, entry = next(iter(stored.items()))
    assert pid.isdigit() and len(pid) == 6 and not pid.startswith("0")
    assert "params" not in entry
    assert entry["config"]["agent"]["net_arch"] == [256, 256]
    assert entry["config"]["agent"]["use_sde"] is True
    assert entry["config"]["agent"]["timesteps"] == 300_000
    assert "net_arch" not in entry["runs"][0]

    row = store.load().iloc[0]
    assert row["config_id"] == int(pid)
    assert row["net_arch"] == [256, 256]
    assert bool(row["use_sde"]) is True


def test_loading_a_missing_file_is_empty(tmp_path):
    assert JSONLStore(tmp_path, "nothing").load().empty


def test_jsonl_replace_retries_when_the_rename_is_denied(tmp_path, monkeypatch):
    import os

    store = JSONLStore(tmp_path, "locked")
    calls = {"n": 0}
    real = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr("rlmine.stores.os.replace", flaky)
    monkeypatch.setattr("rlmine.stores.time.sleep", lambda _seconds: None)
    store.append({"learning_rate": 0.005, "score": 1.0})
    assert calls["n"] == 3
    assert (tmp_path / "locked.json").is_file()
    assert not (tmp_path / "locked.json.tmp").exists()
    assert store.load().iloc[0]["score"] == 1.0


def test_jsonl_overwrites_in_place_when_replace_stays_denied(tmp_path, monkeypatch):
    path = tmp_path / "locked.json"
    path.write_text("{}\n", encoding="utf-8")
    store = JSONLStore(path)

    def denied(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr("rlmine.stores.os.replace", denied)
    monkeypatch.setattr("rlmine.stores.time.sleep", lambda _seconds: None)
    store.append({"learning_rate": 0.005, "score": 1.0})
    assert path.read_text(encoding="utf-8").strip() not in ("", "{}")
    assert not path.with_name(path.name + ".tmp").exists()
    assert store.load().iloc[0]["learning_rate"] == 0.005


def test_same_parameters_share_a_set_and_keep_earlier_runs(tmp_path):
    store = JSONLStore(tmp_path, "append")
    store.append({"learning_rate": 0.005, "score": 1.0})
    store.append({"learning_rate": 0.001, "score": 2.0})
    store.append({"learning_rate": 0.005, "score": 3.0})

    stored = __import__("json").loads((tmp_path / "append.json").read_text())
    assert len(stored) == 2
    repeated = next(entry for entry in stored.values() if entry["config"]["agent"]["learning_rate"] == 0.005)
    assert [run["score"] for run in repeated["runs"]] == [1.0, 3.0]
    assert "learning_rate" not in repeated["runs"][0]

    loaded = store.load()
    assert loaded["config_id"].nunique() == 2
    same = loaded[loaded["learning_rate"] == 0.005]
    assert same["config_id"].nunique() == 1
    assert same["score"].tolist() == [1.0, 3.0]


def test_legacy_jsonl_is_grouped_on_load(tmp_path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"run_id": "a", "score": 1, "gamma": 0.99, "seed": 1}',
                '{"run_id": "b", "score": 2, "gamma": 0.99, "seed": 1}',
                '{"run_id": "c", "score": 3, "gamma": 0.9, "seed": 2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    store = JSONLStore(path)
    loaded = store.load()
    assert loaded["score"].tolist() == [1, 2, 3]
    assert "run_id" not in loaded.columns
    assert loaded.iloc[0]["config_id"] == loaded.iloc[1]["config_id"]
    assert loaded.iloc[0]["config_id"] != loaded.iloc[2]["config_id"]

    stored = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert all("config" in entry and "runs" in entry and "params" not in entry for entry in stored.values())
    shared = next(entry for entry in stored.values() if entry["config"]["agent"]["seed"] == 1)
    assert [run["score"] for run in shared["runs"]] == [1, 2]
    assert "run_id" not in shared["runs"][0]
    assert "gamma" not in shared["runs"][0]


def test_runtime_splits_otherwise_identical_sets(tmp_path):
    import json

    store = JSONLStore(tmp_path, "runtime")
    store.append({"gamma": 0.99, "seed": 1, "runtime": "cpu", "score": 1})
    store.append({"gamma": 0.99, "seed": 1, "runtime": "Tesla T4", "score": 2})
    store.append({"gamma": 0.99, "seed": 1, "runtime": "cpu", "score": 3})

    stored = json.loads((tmp_path / "runtime.json").read_text())
    assert len(stored) == 2
    cpu = next(entry for entry in stored.values() if entry["config"]["runtime"] == "cpu")
    gpu = next(entry for entry in stored.values() if entry["config"]["runtime"] == "Tesla T4")
    assert [run["score"] for run in cpu["runs"]] == [1, 3]
    assert [run["score"] for run in gpu["runs"]] == [2]
    assert all("runtime" not in run for entry in stored.values() for run in entry["runs"])

    loaded = store.load()
    cpu_rows = loaded[loaded["runtime"] == "cpu"]
    gpu_rows = loaded[loaded["runtime"] == "Tesla T4"]
    assert cpu_rows["config_id"].nunique() == 1
    assert gpu_rows["config_id"].nunique() == 1
    assert cpu_rows["config_id"].iloc[0] != gpu_rows["config_id"].iloc[0]
    assert cpu_rows["score"].tolist() == [1, 3]
    assert gpu_rows["score"].tolist() == [2]


def test_leading_zero_ids_are_rewritten(tmp_path):
    import json

    path = tmp_path / "legacy.jsonl"
    path.write_text(
        json.dumps(
            {
                "009131": {
                    "params": {"gamma": 0.99},
                    "config": {},
                    "runs": [
                        {
                            "idx": 0,
                            "run_id": "run-a",
                            "origin": "manual",
                            "parent_id": None,
                            "score": 1,
                        }
                    ],
                },
                "184203": {
                    "params": {"gamma": 0.9},
                    "config": {},
                    "runs": [
                        {
                            "idx": 1,
                            "run_id": "run-b",
                            "origin": "mine",
                            "parent_id": "run-a",
                            "score": 2,
                        }
                    ],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    store = JSONLStore(path)
    loaded = store.load()
    assert int(loaded.iloc[0]["config_id"]) == 109131
    assert 100_000 <= int(loaded.iloc[1]["config_id"]) <= 999_999
    assert "parent_id" not in loaded.columns

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "009131" not in stored
    assert "109131" in stored
    assert all(key.isdigit() and len(key) == 6 and not key.startswith("0") for key in stored)
    child = stored["184203"]
    assert "parent_id" not in child["runs"][0]
    assert "run_id" not in child["runs"][0]
    assert child["config"]["agent"]["gamma"] == 0.9
    assert "params" not in child


def test_bucket_document_is_kept_as_read(tmp_path):
    import json

    path = tmp_path / "bipedal.jsonl"
    original = {
        "916583": {
            "config": {
                "environment": {"id": "BipedalWalker-v3", "n_envs": 16},
                "agent": {"algo": "A2C", "gamma": 0.99, "seed": 1},
                "training": {"timesteps": 5000, "n_eval_episodes": 20},
                "evaluation": {"episodes": 50, "gamma": 1.0, "score": "mean_minus_std"},
                "runtime": "Tesla T4",
            },
            "runs": [
                {
                    "idx": 0,
                    "run_id": "run-a",
                    "origin": "mine",
                    "parent_id": None,
                    "score": -110.34,
                }
            ],
        }
    }
    path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

    loaded = JSONLStore(path).load()
    assert loaded.iloc[0]["config_id"] == 916583
    assert loaded.iloc[0]["gamma"] == 0.99
    assert loaded.iloc[0]["timesteps"] == 5000
    assert loaded.iloc[0]["runtime"] == "Tesla T4"
    assert loaded.iloc[0]["score"] == -110.34

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "params" not in stored["916583"]
    assert stored["916583"]["config"]["runtime"] == "Tesla T4"
    assert stored["916583"]["config"]["evaluation"]["gamma"] == 1.0
    assert "runtime" not in stored["916583"]["runs"][0]
    for retired in ("run_id", "study", "origin", "parent_id"):
        assert retired not in stored["916583"]["runs"][0]


def test_sb3_config_is_what_the_trial_reads():
    from rlmine.trials.sb3 import read_sb3_config

    class A2C:
        pass

    settings = read_sb3_config(
        {
            "environment": {
                "id": "BipedalWalker-v3",
                "n_envs": 16,
                "seed": 0,
                "normalize": {"norm_obs": True, "norm_reward": False, "clip_obs": 10.0},
            },
            "agent": {
                "algo": "A2C",
                "policy": "MlpPolicy",
                "lr_schedule": "linear",
                "initial_lr": 7e-4,
                "final_lr": 7e-4,
                "gamma": 0.99,
                "log_std_init": 0,
                "ortho_init": True,
                "seed": 1,
            },
            "training": {
                "timesteps": 5000,
                "eval_freq": 1000,
                "n_eval_episodes": 20,
                "deterministic": True,
            },
            "evaluation": {
                "episodes": 50,
                "max_steps": 1600,
                "seed": 1,
                "gamma": 1.0,
                "deterministic": True,
                "check_success": False,
                "score": "mean_minus_std",
                "digits": 4,
            },
        }
    )
    assert settings["env_id"] == "BipedalWalker-v3"
    assert settings["n_envs"] == 16
    assert settings["normalize"]["norm_obs"] is True
    assert settings["algo_name"] == "A2C"
    assert settings["policy"] == "MlpPolicy"
    assert settings["timesteps"] == 5000
    assert settings["n_eval_episodes"] == 20
    assert settings["eval_episodes"] == 50
    assert settings["eval_max_steps"] == 1600
    assert settings["eval_gamma"] == 1.0
    assert "timesteps" not in settings["constructor_kwargs"]
    assert "lr_schedule" not in settings["constructor_kwargs"]
    assert "initial_lr" not in settings["constructor_kwargs"]
    schedule = settings["constructor_kwargs"]["learning_rate"]
    assert schedule(1.0) == pytest.approx(7e-4)
    assert schedule(0.0) == pytest.approx(7e-4)
    assert settings["constructor_kwargs"]["gamma"] == 0.99
    assert settings["constructor_kwargs"]["policy_kwargs"]["ortho_init"] is True
    assert settings["constructor_kwargs"]["seed"] == 1

    with pytest.raises(ValueError, match="environment.id"):
        read_sb3_config({})
    with pytest.raises(ValueError, match="timesteps"):
        read_sb3_config({"environment": {"id": "BipedalWalker-v3"}})


def test_tabular_config_keeps_training_episodes_out_of_the_agent():
    from rlmine.trials.tabular import read_tabular_config

    settings = read_tabular_config(
        {
            "environment": {"id": "FrozenLake-v1", "prob": 0.9},
            "agent": {
                "algo": "MCAgent",
                "method": "control",
                "gamma": 1.0,
                "alpha": 0.1,
                "seed": 1,
            },
            "training": {
                "episodes": 30_000,
                "max_steps": 500,
                "updates": 1000,
                "eval_eps": 1000,
            },
            "evaluation": {
                "episodes": 5000,
                "gamma": 1.0,
                "check_success": True,
                "score": "mean_minus_std",
            },
        }
    )
    assert settings["algo_name"] == "MCAgent"
    assert settings["method"] == "control"
    assert settings["agent_gamma"] == 1.0
    assert settings["call_kwargs"]["alpha"] == 0.1
    assert settings["call_kwargs"]["episodes"] == 30_000
    assert settings["call_kwargs"]["eval_eps"] == 1000
    assert "gamma" not in settings["call_kwargs"]
    assert "algo" not in settings["call_kwargs"]
    assert settings["eval_episodes"] == 5000
    assert settings["check_success"] is True


def test_study_stores_the_config_it_ran(tmp_path):
    import json

    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 0.005})

    stored = json.loads((tmp_path / "test-study.json").read_text(encoding="utf-8"))
    entry = next(iter(stored.values()))
    assert entry["config"]["environment"]["id"] == "toy"
    assert entry["config"]["training"]["timesteps"] == 100_000
    assert entry["config"]["agent"]["learning_rate"] == 0.005
    assert entry["config"]["agent"]["gamma"] == 0.99
    assert entry["config"]["evaluation"]["score"] == "mean_minus_std"
    assert entry["config"]["runtime"]
    assert "runtime" not in entry["runs"][0]
    assert "learning_rate" not in entry["runs"][0]

    row = study.history().iloc[0]
    assert row["timesteps"] == 100_000
    assert row["learning_rate"] == 0.005
    assert row["score"] == pytest.approx(95.0)
    assert row["runtime"] == entry["config"]["runtime"]
    assert study.config["agent"]["learning_rate"] == 7e-4


def test_run_merges_a_nested_section(tmp_path):
    import json

    study = make_study(
        tmp_path,
        config=make_config(
            environment={
                "normalize": {"norm_obs": True, "norm_reward": False, "clip_obs": 10.0}
            }
        ),
    )
    study.run(environment={"normalize": {"clip_obs": 5.0}})

    stored = json.loads((tmp_path / "test-study.json").read_text(encoding="utf-8"))
    normalize = next(iter(stored.values()))["config"]["environment"]["normalize"]
    assert normalize["clip_obs"] == 5.0
    assert normalize["norm_obs"] is True
    assert normalize["norm_reward"] is False
    assert study.config["environment"]["normalize"]["clip_obs"] == 10.0


def test_sampler_construction_does_not_draw():
    first = s.loguniform(1e-2, 1e-4, sig=2)
    second = s.loguniform(1e-4, 1e-2, sig=2)
    assert isinstance(first, s.Sampler)
    assert repr(first) == "loguniform(0.0001, 0.01, sig=2)"
    assert first.draw(np.random.default_rng(0)) == second.draw(np.random.default_rng(0))


def test_loguniform_stays_inside_its_range_and_rounds():
    sampler = s.loguniform(1e-2, 1e-4, sig=2)
    values = [sampler.draw(np.random.default_rng(i)) for i in range(40)]
    assert min(values) >= 1e-4
    assert max(values) <= 1e-2
    assert len(set(values)) > 1
    for value in values:
        assert value == pytest.approx(round_sig(value, 2))


def test_uniform_and_normal_draw_rounded_values():
    uniform = s.uniform(0.9, 0.999, digits=3)
    values = [uniform.draw(np.random.default_rng(i)) for i in range(40)]
    assert min(values) >= 0.9
    assert max(values) <= 0.999
    assert all(value == round(value, 3) for value in values)

    clipped = s.normal(0.99, 0.05, low=0.9, high=1.0, digits=3)
    normal_values = [clipped.draw(np.random.default_rng(i)) for i in range(40)]
    assert min(normal_values) >= 0.9
    assert max(normal_values) <= 1.0
    assert s.normal(0.5, 0).draw(np.random.default_rng(0)) == 0.5


def test_sampler_bounds_are_checked_up_front():
    with pytest.raises(ValueError, match="positive"):
        s.loguniform(-1, 1)
    with pytest.raises(ValueError, match="std"):
        s.normal(0.0, -1)
    with pytest.raises(ValueError, match="sig"):
        s.uniform(0, 1, sig=0)


def test_run_draws_a_fresh_sample_for_each_trial(tmp_path):
    sampler = s.loguniform(1e-4, 1e-2, sig=2)
    study = make_study(tmp_path)
    agent = {"learning_rate": sampler}
    frame = study.run(n=3, seed=0, agent=agent)

    rng = np.random.default_rng(0)
    expected = [sampler.draw(rng) for _ in range(3)]
    assert frame["learning_rate"].tolist() == expected
    assert len(set(expected)) > 1
    assert agent["learning_rate"] is sampler
    assert study.config["agent"]["learning_rate"] == 7e-4

    stored = __import__("json").loads((tmp_path / "test-study.json").read_text())
    assert len(stored) == len(set(expected))
    for entry in stored.values():
        assert isinstance(entry["config"]["agent"]["learning_rate"], float)


def test_run_seed_repeats_the_same_draws(tmp_path):
    sampler = s.uniform(0.0, 1.0, digits=4)
    first = make_study(tmp_path)
    second = make_study(tmp_path)
    left = first.run(n=2, seed=7, agent={"learning_rate": sampler})
    right = second.run(n=2, seed=7, agent={"learning_rate": sampler})
    assert left["learning_rate"].tolist() == right["learning_rate"].tolist()


def test_run_prints_drawn_values_as_text(tmp_path, capsys):
    sampler = s.loguniform(1e-4, 1e-2, sig=2)
    study = make_study(tmp_path, verbose=True)
    frame = study.run(n=2, seed=0, agent={"learning_rate": sampler})

    rng = np.random.default_rng(0)
    drawn = [sampler.draw(rng) for _ in range(2)]
    out = capsys.readouterr().out
    for value in drawn:
        assert f"learning_rate={format(value, 'g')}" in out
    assert out.count("score=") == 2
    assert "\033[1;31m" in out
    assert "\n\n" in out
    assert out.splitlines()[0].startswith("[")
    for hidden in ("net_arch", "timesteps", "gamma", "use_sde", "<table"):
        assert hidden not in out
    assert frame["learning_rate"].tolist() == drawn
    assert repr(frame) == out.strip()


def test_run_text_skips_values_that_were_not_sampled(tmp_path, capsys):
    study = make_study(tmp_path, verbose=True)
    study.run(agent={"learning_rate": 0.005})
    out = capsys.readouterr().out
    assert "learning_rate" not in out
    assert "score=" in out
    assert out.startswith("[")
    assert "test-study" not in out


def test_run_prints_a_nested_sample_by_its_path(tmp_path, capsys):
    study = make_study(
        tmp_path,
        verbose=True,
        config=make_config(
            environment={"normalize": {"norm_obs": True, "clip_obs": 10.0}}
        ),
    )
    study.run(
        n=1,
        seed=1,
        agent={"gamma": s.uniform(0.9, 0.99, digits=2)},
        evaluation={"gamma": s.uniform(0.9, 0.99, digits=2)},
        environment={"normalize": {"clip_obs": s.uniform(1.0, 5.0, digits=1)}},
    )
    out = capsys.readouterr().out
    assert "agent.gamma=" in out
    assert "evaluation.gamma=" in out
    assert "normalize.clip_obs=" in out
    assert "norm_obs" not in out


def test_run_result_does_not_render_as_a_table(tmp_path):
    from IPython.core.formatters import DisplayFormatter

    study = make_study(tmp_path, verbose=True)
    frame = study.run()
    formatter = DisplayFormatter()
    formatted, _ = formatter.format(frame)
    assert formatted == {}
    formatted, _ = formatter.format(frame)
    assert "text/html" not in formatted
    assert "score=" in repr(frame)


def test_run_samples_a_nested_value(tmp_path):
    study = make_study(
        tmp_path,
        config=make_config(
            environment={"normalize": {"norm_obs": True, "clip_obs": 10.0}}
        ),
    )
    sampler = s.uniform(1.0, 5.0, digits=1)
    study.run(n=1, seed=1, environment={"normalize": {"clip_obs": sampler}})
    stored = __import__("json").loads((tmp_path / "test-study.json").read_text())
    normalize = next(iter(stored.values()))["config"]["environment"]["normalize"]
    expected = sampler.draw(np.random.default_rng(1))
    assert normalize["clip_obs"] == expected
    assert normalize["norm_obs"] is True
    assert isinstance(sampler, s.Sampler)


def test_param_sampler_is_not_a_config_value(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(TypeError, match="rlmine.samplers"):
        study.run(agent={"learning_rate": P.loguniform(1e-4, 1e-2)})
    with pytest.raises(TypeError, match="rlmine.samplers"):
        study.run(agent={"net_arch": P.choice([[64, 64], [128, 128]])})


def test_run_rejects_a_bad_count(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        study.run(n=0)
    with pytest.raises(ValueError, match="positive integer"):
        study.run(n=1.5)


def test_unknown_config_section_is_an_error(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(TypeError, match="agent"):
        study.run(learning_rate=0.005)


def test_evaluation_digits_round_score_components_before_json(tmp_path):
    """Round every element first, then score those values, then store them."""
    import json

    def trial(config):
        return {
            "mean_return": -110.21626281738281,
            "stdev_return": 0.12764158844947815,
            "mean_length": 65.86421,
            "sr": 0.33336,
            "parts": [1.23456, 2.34567],
            "matrix": np.array([[1.23456, 2.34567]]),
        }

    def from_parts(stats):
        # Uses the list and the array, so each element has to be rounded
        # before this runs. The stored score is still mean minus std.
        assert stats["parts"] == [1.2346, 2.3457]
        assert stats["matrix"].tolist() == [[1.2346, 2.3457]]
        return stats["mean_return"] - stats["stdev_return"]

    study = make_study(
        tmp_path,
        config=make_config(evaluation={"digits": 4}),
        score_fn=from_parts,
    )
    study.trial = trial
    study.run()

    row = study.history().iloc[0]
    assert row["mean"] == -110.2163
    assert row["std_dev"] == 0.1276
    assert row["mean_length"] == 65.8642
    assert row["success_rate"] == 0.3334
    assert row["score"] == -110.3439
    assert row["score"] == round(row["mean"] - row["std_dev"], 4)

    text = (tmp_path / "test-study.json").read_text(encoding="utf-8")
    stored = json.loads(text)
    run = next(iter(stored.values()))["runs"][0]
    assert run["score"] == -110.3439
    assert run["mean"] == -110.2163
    assert run["std_dev"] == 0.1276
    assert "-110.21626281738281" not in text
    assert stored[next(iter(stored))]["config"]["evaluation"]["digits"] == 4


def test_score_uses_rounded_components_not_the_raw_difference(tmp_path):
    def trial(config):
        # 1.23456 - 0.00004 = 1.23452, which rounds to 1.2345.
        # Rounded components are 1.2346 and 0.0000, so the stored score is 1.2346.
        return {"mean_return": 1.23456, "stdev_return": 0.00004}

    study = make_study(tmp_path, config=make_config(evaluation={"digits": 4}))
    study.trial = trial
    study.run()
    row = study.history().iloc[0]
    assert row["mean"] == 1.2346
    assert row["std_dev"] == 0.0
    assert row["score"] == 1.2346


def test_omitted_digits_keep_full_precision(tmp_path):
    raw_mean = -110.21626281738281
    raw_std = 0.12764158844947815

    def trial(config):
        return {"mean_return": raw_mean, "stdev_return": raw_std}

    study = make_study(tmp_path)
    study.trial = trial
    study.run()
    row = study.history().iloc[0]
    assert row["mean"] == raw_mean
    assert row["std_dev"] == raw_std
    assert row["score"] == pytest.approx(raw_mean - raw_std)


def test_bad_digits_fail_before_the_trial(tmp_path):
    called = []

    def trial(config):
        called.append(True)
        return {"mean_return": 1.0, "stdev_return": 0.0}

    study = make_study(tmp_path, config=make_config(evaluation={"digits": -1}))
    study.trial = trial
    with pytest.raises(ValueError, match="digits"):
        study.run()
    assert called == []


def test_run_uses_defaults_plus_overrides(tmp_path):
    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 0.005})
    row = study.history().iloc[0]
    assert row["learning_rate"] == 0.005
    assert row["gamma"] == 0.99
    assert "origin" not in row.index
    assert "error" not in row.index
    assert row["score"] == pytest.approx(95.0)


def test_repeated_configuration_reuses_its_config_id(tmp_path):
    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 0.005})
    study.run(agent={"learning_rate": 0.005})
    history = study.history()
    assert history.iloc[0]["config_id"] == history.iloc[1]["config_id"]
    config_id = int(history.iloc[0]["config_id"])
    assert 100_000 <= config_id <= 999_999

    stored = __import__("json").loads((tmp_path / "test-study.json").read_text())
    assert len(stored) == 1
    entry = next(iter(stored.values()))
    assert len(entry["runs"]) == 2
    assert "learning_rate" not in entry["runs"][0]


def test_cpu_and_gpu_runs_of_the_same_parameters_are_different_sets(tmp_path, monkeypatch):
    runtime = {"name": "Tesla T4"}
    monkeypatch.setattr("rlmine.utils.detect_runtime", lambda: runtime["name"])
    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 0.005})
    runtime["name"] = "cpu"
    study.run(agent={"learning_rate": 0.005})
    runtime["name"] = "Tesla T4"
    study.run(agent={"learning_rate": 0.005})

    history = study.history()
    assert history.iloc[0]["config_id"] == history.iloc[2]["config_id"]
    assert history.iloc[0]["config_id"] != history.iloc[1]["config_id"]
    assert history.iloc[1]["runtime"] == "cpu"

    stored = __import__("json").loads((tmp_path / "test-study.json").read_text())
    gpu = stored[str(int(history.iloc[0]["config_id"]))]
    assert gpu["config"]["runtime"] == "Tesla T4"
    assert gpu["config"]["agent"]["learning_rate"] == 0.005
    assert gpu["config"]["evaluation"]["score"] == "mean_minus_std"
    assert len(gpu["runs"]) == 2
    assert "runtime" not in gpu["runs"][0]
    assert "learning_rate" not in gpu["runs"][0]


def test_seed_makes_a_different_parameter_set(tmp_path):
    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 0.005, "seed": 1})
    study.run(agent={"learning_rate": 0.005, "seed": 2})
    study.run(agent={"learning_rate": 0.005, "seed": 1})
    history = study.history()
    assert history.iloc[0]["config_id"] == history.iloc[2]["config_id"]
    assert history.iloc[0]["config_id"] != history.iloc[1]["config_id"]


def test_drift_report_detects_a_changed_result(tmp_path):
    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 0.005})

    # Simulate a library upgrade that degrades performance.
    study.trial = lambda config: {"mean_return": 80.0, "stdev_return": 5.0}
    study.run(agent={"learning_rate": 0.005})

    report = study.drift_report()
    assert len(report) == 1
    assert report.iloc[0]["config_id"] == study.history().iloc[0]["config_id"]
    assert report.iloc[0]["earlier_score"] == pytest.approx(95.0)
    assert report.iloc[0]["later_score"] == pytest.approx(75.0)
    assert report.iloc[0]["delta"] == pytest.approx(-20.0)


def test_failed_trial_is_not_logged_and_a_later_run_continues(tmp_path):
    import json

    def flaky(config):
        raise RuntimeError("CUDA out of memory")

    study = make_study(tmp_path)
    study.trial = flaky
    returned = study.run()
    assert returned.iloc[0]["score"] is None or pd.isna(returned.iloc[0]["score"])
    assert "error" not in returned.columns
    assert study.history().empty
    assert not (tmp_path / "test-study.json").exists()

    study.trial = quadratic_trial
    study.run(agent={"learning_rate": 0.005})
    study.trial = flaky
    study.run(agent={"learning_rate": 0.005})

    history = study.history()
    assert len(history) == 1
    assert "error" not in history.columns
    assert history.iloc[0]["score"] == pytest.approx(95.0)
    stored = json.loads((tmp_path / "test-study.json").read_text(encoding="utf-8"))
    assert "error" not in json.dumps(stored)
    assert len(next(iter(stored.values()))["runs"]) == 1


def test_every_row_records_versions_for_diagnosis(tmp_path):
    study = make_study(tmp_path)
    study.run()
    row = study.history().iloc[0]
    for column in ("python", "numpy", "runtime", "date", "timestamp", "minutes"):
        assert column in row


def test_table_is_sorted_by_score(tmp_path):
    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 0.001})
    study.run(agent={"learning_rate": 0.005})
    table = study.table()
    assert table.iloc[0]["learning_rate"] == 0.005
    assert "config_id" in table.columns
    assert "run_id" not in table.columns


def test_table_keeps_full_precision_for_small_parameters(tmp_path):
    """Blanket rounding would show a 1e-5 learning rate as 0.0."""
    study = make_study(tmp_path)
    study.run(agent={"learning_rate": 1.2e-5})

    table = study.table()
    assert table.iloc[0]["learning_rate"] == pytest.approx(1.2e-5)
    assert table.iloc[0]["score"] == round(table.iloc[0]["score"], 2)


def test_run_writes_one_row_to_the_mirror(tmp_path):
    mirror = MemoryStore()
    study = make_study(tmp_path, mirror=mirror)
    study.run()
    assert len(mirror.load()) == 1
    assert "config" not in mirror.load().columns


def test_context_is_passed_to_two_argument_trials(tmp_path):
    seen = {}

    def trial_with_context(config, context):
        seen.update(context)
        return quadratic_trial(config)

    study = make_study(tmp_path)
    study.trial = trial_with_context
    study._trial_wants_context = True
    study.run()
    assert seen["config_id"] == study.history().iloc[0]["config_id"]
    assert "run_id" not in seen
    assert "origin" not in seen



# ------------------------------------------------- SB3 parameter routing


def test_timesteps_is_separated_from_constructor_arguments():
    timesteps, kwargs = route_params({"timesteps": 300_000, "gamma": 0.99})
    assert timesteps == 300_000
    assert "timesteps" not in kwargs
    assert kwargs["gamma"] == 0.99


def test_policy_parameters_are_nested():
    _, kwargs = route_params({"net_arch": [256, 256], "ortho_init": True, "gamma": 0.99})
    assert kwargs["policy_kwargs"] == {"net_arch": [256, 256], "ortho_init": True}
    assert "net_arch" not in kwargs
    assert kwargs["gamma"] == 0.99


def test_learning_rate_bounds_become_a_linear_schedule():
    _, kwargs = route_params({"initial_lr": 1e-3, "final_lr": 1e-5})
    schedule = kwargs["learning_rate"]
    assert schedule(1.0) == pytest.approx(1e-3)
    assert schedule(0.0) == pytest.approx(1e-5)
    assert schedule(0.5) == pytest.approx(5.05e-4)


def test_initial_lr_alone_gives_a_constant_schedule():
    _, kwargs = route_params({"initial_lr": 7e-4})
    schedule = kwargs["learning_rate"]
    assert schedule(1.0) == pytest.approx(7e-4)
    assert schedule(0.0) == pytest.approx(7e-4)


def test_plain_learning_rate_is_passed_straight_through():
    _, kwargs = route_params({"learning_rate": 7e-4})
    assert kwargs["learning_rate"] == 7e-4


def test_searched_parameters_override_fixed_model_kwargs():
    _, kwargs = route_params(
        {"gamma": 0.95}, model_kwargs={"gamma": 0.99, "buffer_size": 50_000}
    )
    assert kwargs["gamma"] == 0.95
    assert kwargs["buffer_size"] == 50_000


def test_routing_does_not_mutate_the_caller_dict():
    params = {"timesteps": 1000, "net_arch": [64, 64]}
    route_params(params)
    assert params == {"timesteps": 1000, "net_arch": [64, 64]}


def test_routing_does_not_force_device():
    _, kwargs = route_params({"gamma": 0.99})
    assert "device" not in kwargs


def test_model_kwargs_can_set_device():
    _, kwargs = route_params({"gamma": 0.99}, model_kwargs={"device": "auto"})
    assert kwargs["device"] == "auto"


def test_memory_store_supports_dry_runs():
    study = Study(
        name="memory",
        trial=quadratic_trial,
        config=make_config(),
        store=MemoryStore(),
        verbose=False,
    )
    study.run()
    study.run(agent={"learning_rate": 0.005})
    assert len(study.history()) == 2


def test_swig_and_utcnow_deprecations_are_filtered():
    from rlmine.utils import _quiet_third_party_warnings

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _quiet_third_party_warnings()
        warnings.warn(
            "builtin type SwigPyPacked has no __module__ attribute",
            DeprecationWarning,
        )
        warnings.warn(
            "builtin type SwigPyObject has no __module__ attribute",
            DeprecationWarning,
        )
        warnings.warn(
            "builtin type swigvarlink has no __module__ attribute",
            DeprecationWarning,
        )
        warnings.warn(
            "datetime.datetime.utcnow() is deprecated and scheduled for removal "
            "in a future version. Use timezone-aware objects to represent "
            "datetimes in UTC: datetime.datetime.now(datetime.UTC).",
            DeprecationWarning,
        )
        warnings.warn(
            "\nKernel._parent_header is deprecated in ipykernel 6. Use .get_parent()",
            DeprecationWarning,
        )
        warnings.warn(
            "The order of arguments in worksheet.update() has changed. Please "
            "pass values first and range_name secondor used named arguments "
            "(range_name=, values=)",
            DeprecationWarning,
        )
        noise = [
            str(w.message)
            for w in recorded
            if "SwigPy" in str(w.message)
            or "swigvarlink" in str(w.message)
            or "utcnow" in str(w.message)
            or "_parent_header" in str(w.message)
            or "worksheet.update" in str(w.message)
        ]
        assert noise == []


class _FakeCell:
    def __init__(self, value):
        self.value = value


class _FakeWorksheet:
    def __init__(self, a1="score"):
        self.cleared = False
        self.updated = None
        self.a1 = a1

    def acell(self, addr, value_render_option=None):
        return _FakeCell(self.a1)

    def clear(self):
        self.cleared = True

    def update(self, **kwargs):
        self.updated = kwargs


class _FakeSheet:
    def __init__(self):
        self.worksheet = _FakeWorksheet()


def test_sheet_mirror_writes_with_named_gspread_arguments():
    mirror = SheetMirror("https://example.invalid")
    frame = pd.DataFrame([{"score": 1.5, "net_arch": [64, 64], "empty": np.nan}])
    sheet = _FakeSheet()
    mirror._push(sheet, frame)
    assert sheet.worksheet.cleared
    assert sheet.worksheet.updated["range_name"] == "A1"
    header, row = sheet.worksheet.updated["values"]
    assert header == ["score", "net_arch", "empty"]
    assert row == [1.5, "[64, 64]", ""]


def test_sheet_mirror_connect_is_silent_without_colab():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        SheetMirror("https://example.invalid")
    assert not any("Sheet mirror" in str(w.message) for w in recorded)


def test_sheet_mirror_connects_eagerly(monkeypatch):
    calls = {"sheet": 0, "probe": 0}

    def fake_sheet(self):
        calls["sheet"] += 1
        return _FakeSheet()

    def fake_probe(self, sheet):
        calls["probe"] += 1

    monkeypatch.setattr(SheetMirror, "_sheet", fake_sheet)
    monkeypatch.setattr(SheetMirror, "_probe_write", fake_probe)
    SheetMirror("https://example.invalid")
    assert calls == {"sheet": 1, "probe": 1}


def test_sheet_mirror_probe_write_rewrites_a1_unchanged():
    mirror = SheetMirror("https://example.invalid")
    sheet = _FakeSheet()
    sheet.worksheet.a1 = "run_id"
    mirror._probe_write(sheet)
    assert sheet.worksheet.updated == {"range_name": "A1", "values": [["run_id"]]}


def test_runs_get_consecutive_idx(tmp_path):
    import json

    path = tmp_path / "idx.jsonl"
    path.write_text(
        json.dumps(
            {
                "111111": {"config": {}, "runs": [{"score": 1, "date": "2025-10-14"}, {"score": 1, "date": "2025-11-15"}]},
                "222222": {"config": {}, "runs": [{"score": 2, "date": "2025-10-15"}]},
            }
        ),
        encoding="utf-8",
    )
    store = JSONLStore(path)
    store.load()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert [r["idx"] for r in stored["111111"]["runs"]] == [1, 3]
    assert [r["idx"] for r in stored["222222"]["runs"]] == [2]
    record = {"config_id": "222222", "score": 5}
    store.append(record)
    assert record["idx"] == 4
    assert JSONLStore(path).load()["idx"].sort_values().tolist() == [1, 2, 3, 4]


def test_every_store_assigns_idx(tmp_path):
    from rlmine.stores import CSVStore, MemoryStore

    for store in (MemoryStore(), CSVStore(tmp_path / "s.csv"), JSONLStore(tmp_path / "s.jsonl")):
        records = [{"config_id": "123456", "score": i} for i in range(3)]
        for record in records:
            store.append(record)
        assert [r["idx"] for r in records] == [1, 2, 3]
        assert sorted(store.load()["idx"].tolist()) == [1, 2, 3]


def test_rows_with_the_old_param_id_name_are_still_read():
    from rlmine.stores import group_flat_records

    doc = group_flat_records([{"param_id": "222222", "score": 5}])
    assert list(doc) == [222222]


def test_pruned_trial_is_not_logged_and_the_loop_continues(tmp_path):
    from rlmine import Pruned

    def picky(config):
        if config["agent"]["learning_rate"] < 0.004:
            raise Pruned("eval reward 3.00 < 10 after 1,000 steps")
        return quadratic_trial(config)

    study = make_study(tmp_path)
    study.trial = picky
    study.run(agent={"learning_rate": 0.001})
    assert study.history().empty
    assert not (tmp_path / "test-study.json").exists()

    returned = study.run(n=2, agent={"learning_rate": 0.005})
    assert len(returned) == 2
    assert len(study.history()) == 2


def test_pruner_stops_a_weak_run_and_leaves_a_good_one(tmp_path):
    from types import SimpleNamespace

    from rlmine import Pruned
    from rlmine.trials.sb3 import _prune_rules, _pruner

    assert _pruner([], []) is None
    with pytest.raises(ValueError):
        _prune_rules([(0, 1)], "prune")
    with pytest.raises(TypeError):
        _prune_rules([5], "prune")

    def evaluate_at(pruner, steps, reward):
        pruner.parent = SimpleNamespace(last_mean_reward=reward)
        pruner.num_timesteps = steps
        return pruner._on_step()

    pruner = _pruner(_prune_rules([(1000, 50), (2000, 100)], "prune"), [])
    assert evaluate_at(pruner, 500, -5)       # no rule is due yet
    assert evaluate_at(pruner, 1200, 60)      # passes the first
    with pytest.raises(Pruned, match="< 100"):
        evaluate_at(pruner, 2500, 80)

    pruner = _pruner(_prune_rules([(1000, 50)], "prune"), [])
    with pytest.raises(Pruned):
        evaluate_at(pruner, 1000, 10)


def test_run_passes_prune_rules_to_the_trial_but_not_into_the_key(tmp_path):
    import json

    seen = []

    def spy(config, context):
        seen.append(dict(context))
        return quadratic_trial(config)

    study = make_study(tmp_path)
    study.trial = spy
    study._trial_wants_context = True
    study.run(agent={"learning_rate": 0.005}, prune=[(1000, 5)], prune_minutes=[(2, 1)])
    study.run(agent={"learning_rate": 0.005})

    assert seen[0]["prune"] == [(1000, 5)]
    assert seen[0]["prune_minutes"] == [(2, 1)]
    assert "prune" not in seen[1]
    history = study.history()
    assert history["config_id"].nunique() == 1
    assert "prune" not in (tmp_path / "test-study.json").read_text(encoding="utf-8")
