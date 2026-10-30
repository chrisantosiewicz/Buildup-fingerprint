import numpy as np
import pandas as pd

from buildup.value import add_option_value, decisions, shrink


def _opts():
    # One decision, three options: a safe sideways pass, a forward pass, and a risky through ball.
    return pd.DataFrame({
        "decision_id": ["d1"] * 3, "chosen": [False, True, False],
        "xpass_completion": [0.95, 0.80, 0.40], "xthreat": [0.010, 0.020, 0.080],
        "P": [0.005, 0.010, 0.020], "cost": [0.01, 0.02, 0.05],
        "player_in_possession_position": ["CM"] * 3,
    })


def test_option_value_is_reward_plus_lookahead_minus_cost():
    o = add_option_value(_opts(), gamma=1.0, kappa=1.0)
    expected = o["xpass_completion"] * (o["xthreat"] + o["P"]) - (1 - o["xpass_completion"]) * o["cost"]
    assert np.allclose(o["V"], expected)
    assert o["is_best"].sum() == 1


def test_gamma_zero_drops_lookahead():
    o = add_option_value(_opts(), gamma=0.0)
    assert np.allclose(o["V_progression"], 0)


def test_decision_gap_and_bva():
    o = add_option_value(_opts(), gamma=1.0)
    d = decisions(o)
    assert len(d) == 1
    row = d.iloc[0]
    assert np.isclose(row["decision_gap"], row["V"] - o["V"].max())
    assert np.isclose(row["bva"], row["V"] - o["V"].mean())
    assert row["decision_gap"] <= 0
    assert row["pos_group"] == "MF"


def test_shrinkage_pulls_small_samples_harder():
    rng = np.random.default_rng(0)
    rows = [{"player": "big", "grp": "MF", "v": x} for x in rng.normal(1.0, 1.0, 400)]
    rows += [{"player": "small", "grp": "MF", "v": x} for x in rng.normal(1.0, 1.0, 5)]
    rows += [{"player": f"p{i}", "grp": "MF", "v": x} for i in range(20) for x in rng.normal(0.0, 1.0, 50)]
    s = shrink(pd.DataFrame(rows), "v", "player", "grp")
    gm = pd.DataFrame(rows)["v"].mean()
    pull = lambda p: abs(s.at[p, "mean"] - s.at[p, "shrunk"]) / abs(s.at[p, "mean"] - gm)
    assert pull("small") > pull("big")


def test_decision_quality_runs_from_worst_to_best():
    o = add_option_value(_opts(), gamma=1.0)
    row = decisions(o).iloc[0]
    assert np.isclose(row["decision_quality"], (row["V"] - o["V"].min()) / (o["V"].max() - o["V"].min()))
    assert 0 <= row["decision_quality"] <= 1
