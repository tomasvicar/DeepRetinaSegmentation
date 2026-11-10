"""Optuna5 native rung membership, demonstrated in memory; no live study access."""

import optuna
import pytest


@pytest.mark.parametrize("native_rungs,expected_prune", [(False, False), (True, True)])
def test_intermediate_curve_alone_does_not_enter_native_asha_competition(
    native_rungs, expected_prune
):
    assert optuna.__version__ == "5.0.0"
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3),
    )
    baseline = optuna.trial.create_trial(
        value=0.9,
        intermediate_values={3: 0.9, 9: 0.9, 27: 0.9, 60: 0.9},
        system_attrs={f"completed_rung_{i}": 0.9 for i in range(3)} if native_rungs else {},
        user_attrs={"origin": "synthetic_reference"},
    )
    study.add_trial(baseline)
    trial = study.ask()
    trial.report(0.1, 3)
    assert trial.should_prune() is expected_prune
