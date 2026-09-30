import json

import numpy as np
import pandas as pd
import pytest

from banditlab.bandit import EpsilonGreedy, ThompsonSampling
from banditlab.data import clean_data
from banditlab.experiment import build_environment, simulate_once


def test_only_observed_arm_is_updated():
    policy = ThompsonSampling(7)
    policy.update(0, 2, 1)
    policy.update(1, 0, 0)
    np.testing.assert_array_equal(policy.alpha, [[1, 1, 2], [1, 1, 1]])
    np.testing.assert_array_equal(policy.beta, [[1, 1, 1], [2, 1, 1]])
    with pytest.raises(ValueError, match="binary"):
        policy.update(0, 0, 2)


def test_contextual_learning_favors_each_contexts_winning_arm():
    policy = ThompsonSampling(7, alpha=[[90, 1, 1], [1, 90, 1]], beta=[[10, 50, 50], [50, 10, 50]])
    assert sum(policy.select(0) == 0 for _ in range(100)) > 95
    assert sum(policy.select(1) == 1 for _ in range(100)) > 95


def test_epsilon_one_visits_all_actions():
    policy = EpsilonGreedy(7, epsilon=1)
    assert set(policy.select(0) for _ in range(50)) == {0, 1, 2}


def example_frame():
    return pd.DataFrame(
        {
            "contact": ["cellular", "telephone"] * 10,
            "month": ["may"] * 20,
            "day_of_week": ["mon"] * 20,
            "previous": [0] * 20,
            "poutcome": ["nonexistent"] * 20,
            "y": ["yes", "no"] * 10,
            "duration": [9999] * 20,
            "income": [1000] * 20,
            "gender": ["x"] * 20,
        }
    )


def test_minimization_removes_leakage_and_personal_columns():
    clean = clean_data(example_frame())
    assert set(clean.columns) == {"contact", "month", "day_of_week", "previous", "poutcome", "converted"}
    assert clean.converted.tolist() == [1, 0] * 10
    assert len(clean) == 20  # Indistinguishable minimized rows are not entity duplicates.
    invalid = example_frame()
    invalid.loc[0, "y"] = "unknown"
    with pytest.raises(ValueError, match="yes/no"):
        clean_data(invalid)


def test_holdout_labels_cannot_change_simulator():
    from banditlab.data import ROOT

    config = json.loads((ROOT / "config.json").read_text())
    frame = clean_data(example_frame())
    before = build_environment(frame, config)
    frame.loc[12:, "converted"] = 1 - frame.loc[12:, "converted"]
    after = build_environment(frame, config)
    assert before == after


def test_reproducible_simulation_and_oracle_expected_regret():
    config = {"steps": 300, "pilot_steps": 60, "epsilon": 0.1}
    probabilities = np.array([[0.6, 0.1, 0.3], [0.1, 0.6, 0.3]])
    first = simulate_once(probabilities, [0.5, 0.5], config, 42)
    assert first == simulate_once(probabilities, [0.5, 0.5], config, 42)
    metrics = pd.DataFrame(first[0]).set_index("policy")
    assert metrics.loc["oracle", "expected_regret"] == 0
    assert (metrics.expected_regret >= 0).all()
    assert 0 <= metrics.loc["thompson", "conversion"] <= 1


def test_equal_arms_have_zero_expected_policy_advantage():
    config = {"steps": 300, "pilot_steps": 60, "epsilon": 0.1}
    metrics, *_ = simulate_once(np.array([[0.1] * 3, [0.2] * 3]), [0.5, 0.5], config, 42)
    assert all(row["expected_regret"] == 0 for row in metrics)
    assert len({round(row["expected_conversion"], 12) for row in metrics}) == 1
