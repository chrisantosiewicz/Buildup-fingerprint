"""Week 3 pipeline: setup (before loss) vs reaction (after loss) from tracking.

Usage:
    python scripts/week3.py --data path/to/opendata/data --out outputs [--workers 4]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from buildup.features import turnover_features
from buildup.io import list_match_ids, load_all_events, load_events
from buildup.models import (
    DANGER_TARGET, LOSS_CATEGORICAL, LOSS_NUMERIC, grouped_cv, prepare_turnovers,
)
from buildup.outcomes import add_outcomes
from buildup.tracking import MatchTracking
from buildup.turnovers import build_turnovers

SHAPE_COLS = ["n_behind", "n_last_line", "n_screen", "n_central_behind", "n_opp_beyond",
              "balance", "last_line_height", "depth", "rest_width"]
SETUP_NUMERIC = ["su_" + c for c in SHAPE_COLS]      # 2 s before the loss
AT_LOSS_NUMERIC = ["at_" + c for c in SHAPE_COLS]    # at the moment of the loss
REACTION_NUMERIC = ["re_n_caught_ahead", "re_recovery_speed_mean", "re_n_recovering_fast",
                    "re_gain_behind", "re_n_within10_at_loss", "re_n_within10_at_regain"]


def _match_features(args):
    data_dir, match_id = args
    to = build_turnovers(load_events(data_dir, match_id))
    return turnover_features(MatchTracking(data_dir, match_id), to)


def bootstrap_gain(y, p_a, p_b, groups, n=500, seed=0):
    """Match-level bootstrap 95% interval of AUC(p_b) - AUC(p_a)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ids}
    gains = []
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(ids, len(ids))])
        if y[idx].min() != y[idx].max():
            gains.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    return [round(float(np.percentile(gains, 2.5)), 3), round(float(np.percentile(gains, 97.5)), 3)]


def rate_table(df, row, col):
    t = df.pivot_table(index=row, columns=col, values=DANGER_TARGET, aggfunc=["mean", "size"], observed=False)
    return {"rate": t["mean"].round(3).to_dict(orient="index"), "n": t["size"].to_dict(orient="index")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    events = load_all_events(args.data)
    to = prepare_turnovers(add_outcomes(build_turnovers(events), events))
    ids = list_match_ids(args.data)
    with ProcessPoolExecutor(args.workers) as ex:
        feats = pd.concat(list(ex.map(_match_features, [(args.data, m) for m in ids])), ignore_index=True)
    df = to.merge(feats, on="turnover_id", how="left")
    df.to_csv(out / "turnover_features.csv", index=False)
    res = {"n_turnovers": len(df), "tracking_ok": int(df["tracking_ok"].sum())}
    df = df[df["tracking_ok"] == True].reset_index(drop=True)  # noqa: E712

    # --- Models: location -> + setup -> + setup + reaction ----------------------
    y, g = df[DANGER_TARGET].astype(int).to_numpy(), df["match_id"].to_numpy()
    m_loc = grouped_cv(df, DANGER_TARGET, LOSS_NUMERIC, LOSS_CATEGORICAL)
    m_su = grouped_cv(df, DANGER_TARGET, LOSS_NUMERIC + SETUP_NUMERIC, LOSS_CATEGORICAL)
    m_at = grouped_cv(df, DANGER_TARGET, LOSS_NUMERIC + SETUP_NUMERIC + AT_LOSS_NUMERIC, LOSS_CATEGORICAL)
    m_all = grouped_cv(df, DANGER_TARGET, LOSS_NUMERIC + SETUP_NUMERIC + AT_LOSS_NUMERIC + REACTION_NUMERIC, LOSS_CATEGORICAL)
    m_re_only = grouped_cv(df, DANGER_TARGET, LOSS_NUMERIC + REACTION_NUMERIC, LOSS_CATEGORICAL)
    m_all_gbm = grouped_cv(df, DANGER_TARGET, LOSS_NUMERIC + SETUP_NUMERIC + AT_LOSS_NUMERIC + REACTION_NUMERIC,
                           LOSS_CATEGORICAL, kind="gbm")
    res["models"] = {
        "location": m_loc.summary(),
        "+setup(loss-2s)": m_su.summary(),
        "+shape_at_loss": m_at.summary(),
        "+reaction(loss->regain)": m_all.summary(),
        "location+reaction_only": m_re_only.summary(),
        "all_gbm": m_all_gbm.summary(),
        "gain_setup_over_location_95ci": bootstrap_gain(y, m_loc.oof, m_su.oof, g),
        "gain_at_loss_over_setup_95ci": bootstrap_gain(y, m_su.oof, m_at.oof, g),
        "gain_reaction_over_shape_95ci": bootstrap_gain(y, m_at.oof, m_all.oof, g),
    }
    # Sensitivity: only turnovers where most rest defenders were on camera at setup.
    det = df[df["su_behind_detected_share"] >= 0.75].reset_index(drop=True)
    if det[DANGER_TARGET].sum() > 20:
        res["models"]["detected_only"] = {
            "n": len(det),
            "location": grouped_cv(det, DANGER_TARGET, LOSS_NUMERIC, LOSS_CATEGORICAL).summary(),
            "+setup": grouped_cv(det, DANGER_TARGET, LOSS_NUMERIC + SETUP_NUMERIC, LOSS_CATEGORICAL).summary(),
            "+shape_at_loss": grouped_cv(det, DANGER_TARGET, LOSS_NUMERIC + SETUP_NUMERIC + AT_LOSS_NUMERIC, LOSS_CATEGORICAL).summary(),
        }

    # --- Coach-language tables ---------------------------------------------------
    df["behind_band"] = pd.cut(df["su_n_behind"], [-1, 3, 5, 7, 10], labels=["0-3", "4-5", "6-7", "8-10"])
    res["danger_by_loss_third_x_behind_at_setup"] = rate_table(df, "loss_third", "behind_band")
    df["at_behind_band"] = pd.cut(df["at_n_behind"], [-1, 3, 5, 7, 10], labels=["0-3", "4-5", "6-7", "8-10"])
    res["danger_by_loss_third_x_behind_at_loss"] = rate_table(df, "loss_third", "at_behind_band")
    mid = df[df["loss_third"] == "middle_third"]
    shapes = mid.groupby("su_shape")[DANGER_TARGET].agg(["mean", "size"])
    res["middle_third_shapes"] = shapes[shapes["size"] >= 25].sort_values("mean").round(3).to_dict(orient="index")
    df["balance_band"] = pd.cut(df["su_balance"], [-10, 0, 2, 4, 10], labels=["<=0", "1-2", "3-4", "5+"])
    res["danger_by_balance"] = df.groupby("balance_band", observed=False)[DANGER_TARGET].agg(["mean", "size"]).round(3).to_dict(orient="index")
    df["reaction_band"] = pd.cut(df["re_recovery_speed_mean"], [-20, 0, 2, 20], labels=["not recovering", "0-2 m/s", "2+ m/s"])
    res["danger_by_shape_at_loss_x_recovery"] = rate_table(df, "at_behind_band", "reaction_band")

    with open(out / "week3_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
