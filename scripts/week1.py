"""Week 1 pipeline: turnovers, danger outcomes, models, exposure (events only).

Usage:
    python scripts/week1.py --data path/to/opendata/data --out outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from buildup.io import load_all_events
from buildup.models import (
    DANGER_TARGET, LOSS_CATEGORICAL, LOSS_NUMERIC, PLOSS_CATEGORICAL, PLOSS_NUMERIC,
    SHAPE_CATEGORICAL, SHAPE_NUMERIC, add_exposure, fit_full, grouped_cv,
    prepare_possessions, prepare_turnovers,
)
from buildup.outcomes import add_outcomes
from buildup.turnovers import build_turnovers


def bootstrap_auc_gain(y, p_base, p_full, groups, n=500, seed=0):
    """Match-level bootstrap of the AUC gain from adding shape features."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    gains = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        gains.append(roc_auc_score(y[idx], p_full[idx]) - roc_auc_score(y[idx], p_base[idx]))
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="SkillCorner opendata data/ folder")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    events = load_all_events(args.data)
    to = prepare_turnovers(add_outcomes(build_turnovers(events), events))
    to.to_csv(out / "turnovers.csv", index=False)

    # --- 1. Describe and validate the outcomes --------------------------------
    res["n_matches"] = int(to["match_id"].nunique())
    res["n_turnovers"] = len(to)
    res["lost_by"] = to["lost_by"].value_counts().to_dict()
    res["outcome_rates"] = to[["final_third_10s", "box_10s", "shot_10s", "shot_15s", "goal_15s"]].mean().round(4).to_dict()
    res["danger_level_counts"] = {int(k): int(v) for k, v in to["danger_level"].value_counts().sort_index().items()}
    # Does the model-free outcome line up with SkillCorner EPV? (independent check)
    v = to.dropna(subset=["epv_gain_10s"])
    res["validation"] = {
        "auc_epv_gain_vs_shot15": round(roc_auc_score(v["shot_15s"], v["epv_gain_10s"]), 3),
        "auc_progress_m_vs_shot15": round(roc_auc_score(to["shot_15s"], to["progress_m_10s"]), 3),
        "mean_epv_gain_by_danger_level": v.groupby("danger_level")["epv_gain_10s"].mean().round(4).to_dict(),
    }
    res["danger_by_loss_third"] = to.groupby("loss_third")[DANGER_TARGET].mean().round(3).to_dict()
    res["danger_by_goal_side_count"] = (
        to.assign(goal_side=to["rd_n_goal_side"].clip(upper=10))
        .groupby("goal_side")[DANGER_TARGET].agg(["mean", "size"]).round(3).to_dict(orient="index")
    )

    # --- 2. Danger if lost: location/context baseline vs + shape proxies --------
    y = to[DANGER_TARGET].astype(int).to_numpy()
    g = to["match_id"].to_numpy()
    base = grouped_cv(to, DANGER_TARGET, LOSS_NUMERIC, LOSS_CATEGORICAL)
    full = grouped_cv(to, DANGER_TARGET, LOSS_NUMERIC + SHAPE_NUMERIC, LOSS_CATEGORICAL + SHAPE_CATEGORICAL)
    base_gbm = grouped_cv(to, DANGER_TARGET, LOSS_NUMERIC, LOSS_CATEGORICAL, kind="gbm")
    full_gbm = grouped_cv(to, DANGER_TARGET, LOSS_NUMERIC + SHAPE_NUMERIC, LOSS_CATEGORICAL + SHAPE_CATEGORICAL, kind="gbm")
    lo, hi = bootstrap_auc_gain(y, base.oof, full.oof, g)
    res["danger_model"] = {
        "baseline_logit": base.summary(), "shape_logit": full.summary(),
        "baseline_gbm": base_gbm.summary(), "shape_gbm": full_gbm.summary(),
        "auc_gain_logit_95ci": [round(lo, 3), round(hi, 3)],
    }

    # --- 3. Probability of loss on every possession ----------------------------
    pp = prepare_possessions(events)
    res["n_possessions"] = len(pp)
    res["possession_loss_rate"] = round(float(pp["lost"].mean()), 4)
    pl = grouped_cv(pp, "lost", PLOSS_NUMERIC, PLOSS_CATEGORICAL)
    pl_gbm = grouped_cv(pp, "lost", PLOSS_NUMERIC, PLOSS_CATEGORICAL, kind="gbm")
    res["ploss_model"] = {"logit": pl.summary(), "gbm": pl_gbm.summary()}

    # --- 4. Exposure v0 and a sanity check at team-match level -----------------
    # In-sample models here; the CV numbers above are the honest performance.
    ploss_m = fit_full(pp, "lost", PLOSS_NUMERIC, PLOSS_CATEGORICAL)
    danger_m = fit_full(to, DANGER_TARGET, LOSS_NUMERIC, LOSS_CATEGORICAL)
    ex = add_exposure(pp, ploss_m, danger_m)
    ex[["match_id", "event_id", "team_id", "team_shortname", "player_name", "period", "frame_start",
        "x_start", "y_start", "p_loss", "p_danger_if_lost", "exposure", "lost"]].to_csv(out / "possession_exposure.csv", index=False)
    tm = ex.groupby(["match_id", "team_id"]).agg(expected=("exposure", "sum")).reset_index()
    actual = to.groupby(["match_id", "losing_team_id"])[DANGER_TARGET].sum().rename("actual").reset_index()
    tm = tm.merge(actual, left_on=["match_id", "team_id"], right_on=["match_id", "losing_team_id"], how="left").fillna({"actual": 0})
    res["exposure_check"] = {
        "team_matches": len(tm),
        "spearman_expected_vs_actual_dangerous_transitions": round(tm["expected"].corr(tm["actual"], method="spearman"), 3),
        "total_expected": round(float(tm["expected"].sum()), 1),
        "total_actual": int(tm["actual"].sum()),
    }
    with open(out / "week1_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
