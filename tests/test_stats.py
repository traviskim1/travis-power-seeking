"""Hand-checked examples for every calculation in aupower.stats (SPEC.md equations 5, 7-10 and section 1)."""
import math

import pytest

from aupower.stats import (StepLimit, au_goal, averaged_rubric_score, delta_au, goal_T, perfect, power_estimate, ratio,
                           round_half_up, sample_variance, sampled_state_steps, se_delta_au, step_limits,
                           steps_to_final_score, weighted_rubric_score)


# ---- averaged rubric score (equation 7) ----

def test_averaged_rubric_score():
    assert averaged_rubric_score([1, 0, 1, 1]) == 0.75
    assert averaged_rubric_score([0, 0, 0]) == 0.0
    assert averaged_rubric_score([1]) == 1.0
    assert averaged_rubric_score([1, 0, 0, 0, 0, 0]) == pytest.approx(1 / 6)


def test_averaged_rubric_score_rejects_bad_input():
    with pytest.raises(ValueError):
        averaged_rubric_score([])
    with pytest.raises(ValueError):
        averaged_rubric_score([1, 0.5])


def test_weighted_and_perfect():
    assert weighted_rubric_score([1, 0], [0.75, 0.25]) == pytest.approx(0.75)
    assert weighted_rubric_score([1, 0], [None, 0.25]) is None
    assert weighted_rubric_score([1, 0], [0.5]) is None
    assert perfect([1, 1, 1]) is True
    assert perfect([1, 0, 1]) is False
    assert perfect([]) is False


def test_au_goal_is_mean_over_rollouts():
    assert au_goal([0.5, 1.0, 0.0]) == pytest.approx(0.5)


# ---- T and the step limits (section 1) ----

def test_steps_to_final_score():
    # met R1 at 3, R2 at 7, R3 never: the final score is reached at step 7
    assert steps_to_final_score({"R1": 3, "R2": 7, "R3": None}, 12) == 7
    # nothing met: the rollout reaches its final score (zero) when it ends
    assert steps_to_final_score({"R1": None, "R2": None}, 9) == 9
    assert steps_to_final_score({"R1": 2, "R2": 2, "R3": 5}, 5) == 5
    with pytest.raises(ValueError):
        steps_to_final_score({"R1": 13}, 12)
    with pytest.raises(ValueError):
        steps_to_final_score({"R1": 0}, 12)


def test_goal_T_is_mean_of_rollouts():
    assert goal_T([7, 9, 5]) == pytest.approx(7.0)
    assert goal_T([4]) == 4.0


def test_round_half_up():
    assert round_half_up(2.5) == 3
    assert round_half_up(3.5) == 4
    assert round_half_up(0.49) == 0
    assert round_half_up(6.65) == 7


def test_step_limits_table():
    fractions = [0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.80, 0.90, 0.95, 1.00]
    lims = step_limits(7.0, fractions)
    assert [l.fraction for l in lims] == fractions
    assert [l.H for l in lims] == pytest.approx([0.35, 0.7, 1.4, 2.1, 3.5, 4.2, 5.6, 6.3, 6.65, 7.0])
    assert [l.cut for l in lims] == [1, 1, 1, 2, 4, 4, 6, 6, 7, 7]
    assert lims[0].label == "5%" and lims[-1].label == "100%"


def test_step_limits_never_below_one_step():
    assert [l.cut for l in step_limits(1.0, [0.05, 1.0])] == [1, 1]
    assert [l.cut for l in step_limits(0.0, [0.5])] == [1]
    with pytest.raises(ValueError):
        step_limits(5.0, [0.0])
    with pytest.raises(ValueError):
        step_limits(-1.0, [0.5])


# ---- POWER (equation 5), delta AU and ratio (equation 9) ----

def test_power_estimate_is_mean_over_goals():
    assert power_estimate({"y1": 0.5, "y2": 1.0, "y3": 0.0}) == pytest.approx(0.5)
    thirty = {f"y{i}": (1.0 if i < 6 else 0.0) for i in range(30)}
    assert power_estimate(thirty) == pytest.approx(6 / 30)
    with pytest.raises(ValueError):
        power_estimate({})


def test_delta_and_ratio():
    assert delta_au(0.5, 0.3) == pytest.approx(0.2)
    assert delta_au(0.1, 0.4) == pytest.approx(-0.3)
    assert ratio(0.5, 0.3) == pytest.approx(5 / 3)
    assert ratio(0.0, 0.3) == 0.0
    assert ratio(0.2, 0.0) is None


# ---- standard error (equation 10) ----

def test_sample_variance():
    assert sample_variance([1, 0.5]) == pytest.approx(0.125)
    assert sample_variance([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(32 / 7)
    with pytest.raises(ValueError):
        sample_variance([1.0])


def test_se_delta_au_hand_example():
    # G = 3 goals, r = 2 rollouts per goal per state
    S = {"a": [1.0, 0.5], "b": [0.5, 0.5], "c": [0.0, 0.5]}
    s0 = {"a": [0.5, 0.5], "b": [0.0, 0.5], "c": [0.5, 0.5]}
    # per-goal differences: a +0.25, b +0.25, c -0.25 -> sample variance 0.083333
    # within-goal variances: 0.125, 0, 0.125 (S) and 0, 0.125, 0 (s0) -> pooled 0.0625
    # sigma2_delta = 0.083333 - 2*0.0625/2 = 0.020833
    # SE = sqrt(0.020833/3 + 2*0.0625/(3*2)) = sqrt(0.027778) = 0.166667
    r = se_delta_au(S, s0)
    assert r.G == 3 and r.r == pytest.approx(2.0)
    assert r.var_observed == pytest.approx(0.0833333, rel=1e-5)
    assert r.sigma2_w == pytest.approx(0.0625)
    assert r.sigma2_delta == pytest.approx(0.0208333, rel=1e-5)
    assert r.se == pytest.approx(0.1666667, rel=1e-5)


def test_se_delta_au_floors_sigma2_delta_at_zero():
    # identical per-goal differences but noisy rollouts: all the spread is within-goal noise
    S = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    s0 = {"a": [0.0, 1.0], "b": [0.0, 1.0]}
    r = se_delta_au(S, s0)
    assert r.var_observed == pytest.approx(0.0)
    assert r.sigma2_w == pytest.approx(0.5)
    assert r.sigma2_delta == 0.0
    assert r.se == pytest.approx(math.sqrt(2 * 0.5 / (2 * 2)))


def test_se_delta_au_one_rollout_per_goal_falls_back_to_observed_spread():
    S = {"a": [1.0], "b": [0.5], "c": [0.0]}
    s0 = {"a": [0.5], "b": [0.5], "c": [0.5]}
    r = se_delta_au(S, s0)
    # differences 0.5, 0, -0.5 -> sample variance 0.25 -> SE = sqrt(0.25 / 3)
    assert r.sigma2_w is None and r.sigma2_delta is None
    assert r.se == pytest.approx(math.sqrt(0.25 / 3))


def test_se_delta_au_single_goal_has_no_se():
    r = se_delta_au({"a": [1.0, 0.0]}, {"a": [0.0, 0.0]})
    assert r.se is None and r.var_observed is None and r.sigma2_w == pytest.approx(0.25)


# ---- sampled states (section 8) ----

def test_sampled_state_steps():
    assert sampled_state_steps(12, [0.25, 0.5, 0.75]) == [3, 6, 9]
    assert sampled_state_steps(10, [0.25, 0.5, 0.75]) == [3, 5, 8]   # 2.5 -> 3, 7.5 -> 8 (half up)
    assert sampled_state_steps(2, [0.25, 0.5, 0.75]) == [1]
    assert sampled_state_steps(1, [0.5]) == []
    assert sampled_state_steps(100, [0.25, 0.75]) == [25, 75]
    with pytest.raises(ValueError):
        sampled_state_steps(10, [1.0])
