"""Tests for the mining engine, using a cheap synthetic trial function."""

import numpy as np
import pandas as pd
import pytest

from rlmine import JSONLStore, MemoryStore, Space, Study
from rlmine import params as P
from rlmine.trials.sb3 import route_params


def quadratic_trial(params):
    """A fast stand-in for training: score peaks at learning_rate == 0.005."""
    lr = params["learning_rate"]
    mean = 100.0 - 500_000 * (lr - 0.005) ** 2
    return {"mean_return": mean, "stdev_return": 5.0}


def make_space():
    return dict(
        timesteps=P.Int(100_000, mutate="fixed"),
        learning_rate=P.Float(7e-4, sig=2, bounds=(1e-6, 1.0), mutate=P.scale(0.2)),
        gamma=P.Float(0.99, digits=3, bounds=(0, 1)),
        net_arch=P.Choice([[64, 64], [128, 128], [256, 256]], default=[128, 128]),
        use_sde=P.Bool(False),
    )


def make_study(tmp_path, **kwargs):
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("verbose", False)
    return Study(
        name="test-study",
        trial=quadratic_trial,
        space=make_space(),
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


def test_loguniform_sampling_stays_in_range():
    param = P.Float(1e-3, sig=3, sample=P.loguniform(1e-5, 1e-1))
    rng = np.random.default_rng(0)
    values = [param.draw(rng) for _ in range(100)]
    assert all(1e-5 <= v <= 1e-1 for v in values)


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


# ------------------------------------------------------------------- stores


def test_jsonl_roundtrip_preserves_types(tmp_path):
    import json

    store = JSONLStore(tmp_path, "types")
    store.append({"net_arch": [256, 256], "use_sde": True, "timesteps": 300_000})

    # The stored representation keeps real types, so no string parsing is
    # needed on the way back in. This is the whole reason for preferring JSONL
    # over a spreadsheet as the source of truth.
    stored = json.loads((tmp_path / "types.jsonl").read_text().strip())
    assert stored["net_arch"] == [256, 256]
    assert stored["use_sde"] is True
    assert stored["timesteps"] == 300_000

    row = store.load().iloc[0]
    assert row["net_arch"] == [256, 256]
    assert bool(row["use_sde"]) is True


def test_loading_a_missing_file_is_empty(tmp_path):
    assert JSONLStore(tmp_path, "nothing").load().empty


def test_appending_never_rewrites_earlier_rows(tmp_path):
    store = JSONLStore(tmp_path, "append")
    for i in range(3):
        store.append({"i": i})
    lines = (tmp_path / "append.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    assert store.load()["i"].tolist() == [0, 1, 2]


# -------------------------------------------------------------------- study


def test_run_uses_defaults_plus_overrides(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.005)
    row = study.history().iloc[0]
    assert row["learning_rate"] == 0.005
    assert row["gamma"] == 0.99
    assert row["origin"] == "manual"
    assert row["status"] == "ok"
    assert row["score"] == pytest.approx(95.0)


def test_mine_without_history_falls_back_to_fresh_values(tmp_path):
    study = make_study(tmp_path)
    study.mine(n=1, mutate=["learning_rate"])
    assert len(study.history()) == 1
    assert study.history().iloc[0]["parent_id"] is None


def test_mine_records_parent_lineage(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.005)
    seed_id = study.history().iloc[0]["run_id"]

    study.mine(n=3, mutate=["learning_rate"])
    history = study.history()
    children = history.iloc[1:]

    assert (children["origin"] == "mine").all()
    assert children["parent_id"].isin(history["run_id"]).all()
    assert children.iloc[0]["parent_id"] == seed_id


def test_mine_reselects_a_parent_each_iteration(tmp_path):
    """A child produced early in a call can parent a later one.

    This matches the original notebooks, which re-read the sheet inside the
    loop, and it is what lets a single call make progressive headway.
    """
    study = make_study(tmp_path)
    study.run(learning_rate=0.001)
    study.mine(n=8, mutate=["learning_rate"])

    history = study.history()
    parents = set(history.iloc[1:]["parent_id"])
    assert len(parents) > 1
    assert parents <= set(history["run_id"])


def test_mine_only_perturbs_named_parameters(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.005, gamma=0.9, net_arch=[64, 64])
    study.mine(n=4, mutate=["learning_rate"])

    children = study.history().iloc[1:]
    assert (children["gamma"] == 0.9).all()
    assert children["net_arch"].apply(lambda a: a == [64, 64]).all()
    assert (children["learning_rate"] != 0.005).any()


def test_mine_improves_on_a_smooth_objective(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.001)
    for _ in range(12):
        study.mine(n=1, mutate=["learning_rate"])
    scores = study.history()["score"]
    assert scores.max() > scores.iloc[0]


def test_pinned_parent_by_index(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.001)
    study.run(learning_rate=0.005)
    study.mine(n=1, mutate=[], parent=0)

    child = study.history().iloc[-1]
    assert child["learning_rate"] == 0.001


def test_recheck_replays_parameters_and_links_back(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.005)
    original = study.history().iloc[0]

    study.recheck(top=1)
    recheck = study.history().iloc[1]
    assert recheck["origin"] == "recheck"
    assert recheck["parent_id"] == original["run_id"]
    assert recheck["learning_rate"] == original["learning_rate"]
    assert recheck["score"] == pytest.approx(original["score"])


def test_recheck_needs_a_selection(tmp_path):
    study = make_study(tmp_path)
    study.run()
    with pytest.raises(ValueError, match="rows=|top=|latest="):
        study.recheck()


def test_recheck_on_empty_history_is_an_error(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(ValueError, match="No completed runs"):
        study.recheck(top=1)


def test_drift_report_detects_a_changed_result(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.005)

    # Simulate a library upgrade that degrades performance.
    study.trial = lambda params: {"mean_return": 80.0, "stdev_return": 5.0}
    study.recheck(top=1)

    report = study.drift_report()
    assert len(report) == 1
    assert report.iloc[0]["original_score"] == pytest.approx(95.0)
    assert report.iloc[0]["recheck_score"] == pytest.approx(75.0)
    assert report.iloc[0]["delta"] == pytest.approx(-20.0)


def test_failed_trial_is_recorded_and_the_loop_continues(tmp_path):
    calls = {"n": 0}

    def flaky(params):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("CUDA out of memory")
        return quadratic_trial(params)

    study = make_study(tmp_path)
    study.trial = flaky
    study.run(n=3)

    history = study.history()
    assert len(history) == 3
    assert history["status"].tolist() == ["ok", "failed", "ok"]
    assert "CUDA out of memory" in history.iloc[1]["error"]
    assert pd.isna(history.iloc[1]["score"])


def test_failed_runs_are_never_selected_as_parents(tmp_path):
    study = make_study(tmp_path)
    study.trial = lambda params: (_ for _ in ()).throw(RuntimeError("boom"))
    study.run()

    study.trial = quadratic_trial
    study.mine(n=1, mutate=["learning_rate"])
    assert study.history().iloc[1]["parent_id"] is None


def test_roulette_falls_back_when_no_positive_scores(tmp_path, recwarn):
    study = make_study(tmp_path)
    study.trial = lambda params: {"mean_return": -100.0, "stdev_return": 5.0}
    study.run()

    study.mine(n=3, mutate=["learning_rate"])
    assert not any("positive score" in str(w.message) for w in recwarn)
    assert study.history().iloc[1]["parent_id"] is not None


def test_no_positive_score_message_prints_once(tmp_path, capsys):
    study = make_study(tmp_path, verbose=True)
    study.trial = lambda params: {"mean_return": -100.0, "stdev_return": 5.0}
    study.run()
    capsys.readouterr()

    study.mine(n=3, mutate=["learning_rate"])
    out = capsys.readouterr().out
    assert out.count("No runs with a positive score yet") == 1


def test_every_row_records_versions_for_diagnosis(tmp_path):
    study = make_study(tmp_path)
    study.run()
    row = study.history().iloc[0]
    for column in ("python", "numpy", "runtime", "date", "timestamp", "minutes"):
        assert column in row


def test_table_is_sorted_by_score(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.001)
    study.run(learning_rate=0.005)
    table = study.table()
    assert table.iloc[0]["learning_rate"] == 0.005
    assert "run_id" in table.columns


def test_table_keeps_full_precision_for_small_parameters(tmp_path):
    """Blanket rounding would show a 1e-5 learning rate as 0.0."""
    study = make_study(tmp_path)
    study.run(learning_rate=1.2e-5)

    table = study.table()
    assert table.iloc[0]["learning_rate"] == pytest.approx(1.2e-5)
    assert table.iloc[0]["score"] == round(table.iloc[0]["score"], 2)


def test_params_of_round_trips_into_run(tmp_path):
    study = make_study(tmp_path)
    study.run(learning_rate=0.005, net_arch=[256, 256])
    params = study.params_of(0)
    assert params["net_arch"] == [256, 256]

    study.run(**params)
    assert study.history().iloc[1]["learning_rate"] == 0.005


def test_import_rows_migrates_string_cells(tmp_path):
    study = make_study(tmp_path)
    old_sheet = pd.DataFrame(
        [
            {
                "timesteps": "1,000,000",
                "learning_rate": "7.0e-04",
                "gamma": "0.99",
                "net_arch": "[256, 256]",
                "use_sde": "FALSE",
                "score": "42.5",
                "mean": "50",
                "std_dev": "7.5",
                "date": "2025-11-14",
            }
        ]
    )
    study.import_rows(old_sheet)

    row = study.history().iloc[0]
    assert row["net_arch"] == [256, 256]
    assert bool(row["use_sde"]) is False
    assert row["timesteps"] == 1_000_000
    assert row["score"] == 42.5
    assert row["origin"] == "imported"

    # Imported rows are immediately usable as mining parents.
    study.mine(n=1, mutate=["learning_rate"])
    assert study.history().iloc[1]["parent_id"] == row["run_id"]


def test_import_does_not_echo_rows_back_to_the_mirror(tmp_path):
    mirror = MemoryStore()
    study = make_study(tmp_path, mirror=mirror)

    study.import_rows(pd.DataFrame([{"learning_rate": "0.005"}]))
    assert mirror.load().empty

    study.run()
    assert len(mirror.load()) == 1


def test_context_is_passed_to_two_argument_trials(tmp_path):
    seen = {}

    def trial_with_context(params, context):
        seen.update(context)
        return quadratic_trial(params)

    study = make_study(tmp_path)
    study.trial = trial_with_context
    study._trial_wants_context = True
    study.run()
    assert seen["study"] == "test-study"
    assert seen["origin"] == "manual"


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


def test_memory_store_supports_dry_runs():
    study = Study(
        name="memory",
        trial=quadratic_trial,
        space=make_space(),
        store=MemoryStore(),
        verbose=False,
        seed=0,
    )
    study.run(n=2)
    assert len(study.history()) == 2
