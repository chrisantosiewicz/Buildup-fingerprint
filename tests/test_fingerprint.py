import numpy as np
import pandas as pd

from buildup.fingerprint import posterior, tier


def _df(true_sd, n_players=40, matches=20, per_match=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        true = rng.normal(0, true_sd)
        for m in range(matches):
            for _ in range(per_match):
                rows.append({"player": f"p{p}", "pos_group": "MF", "match_id": m, "v": true + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_pure_noise_is_shrunk_to_the_position_mean():
    s = posterior(_df(true_sd=0.0), "v")
    assert s["B"].max() < 0.2
    assert (s["p_above_position"].map(tier) == "around average").mean() > 0.9


def test_real_differences_survive_and_rank_correctly():
    df = _df(true_sd=0.5)
    s = posterior(df, "v")
    assert s["B"].min() > 0.8
    true_order = df.groupby("player")["v"].mean().rank()
    assert np.corrcoef(true_order, s["shrunk"].rank())[0, 1] > 0.95


def test_tier_cutoffs():
    assert tier(0.9) == "above average" and tier(0.1) == "below average" and tier(0.5) == "around average"
