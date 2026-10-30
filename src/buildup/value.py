"""Option value, decisions and player profiles.

    V(o) = xPass(o) * [ xThreat(o) + gamma * P(o) ] - (1 - xPass(o)) * kappa * C(o)

All terms are in goals: xThreat and P are scoring values; C is the probability
that a cut-out becomes a dangerous counter times the goal rate of dangerous
counters (measured from the turnover table).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .models import DANGER_TARGET, _logit
from .progression import DEFAULT_GBM, _gbm, _X

# Turnover-cost model: where the ball is lost and where it's won (losing team's frame).
COST_NUMERIC = ["loss_x", "loss_abs_y", "won_x", "won_abs_y", "loss_pressure_id", "score_diff_losing", "minute"]
COST_CATEGORICAL = ["loss_phase_type"]

POSITION_GROUPS = {
    "GK": "GK", "LCB": "CB", "RCB": "CB", "CB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "DM": "MF", "LDM": "MF", "RDM": "MF", "CM": "MF", "LM": "MF", "RM": "MF",
    "AM": "AM", "LW": "AM", "RW": "AM", "LF": "FW", "RF": "FW", "CF": "FW",
}


def fit_cost_model(turnovers: pd.DataFrame):
    """Logistic model of a dangerous counter from loss and regain spots."""
    t = turnovers.copy()
    t["loss_abs_y"] = t["loss_y"].abs()
    t["won_x"] = -t["regain_x"]           # regain x is in the regaining team's frame: flip it
    t["won_abs_y"] = t["regain_y"].abs()
    model = _logit(COST_NUMERIC, COST_CATEGORICAL).fit(t[COST_NUMERIC + COST_CATEGORICAL], t[DANGER_TARGET].astype(int))
    dangerous = t[DANGER_TARGET].astype(bool)
    goals_per_danger = float(t.loc[dangerous, "goal_15s"].sum() / max(dangerous.sum(), 1))
    return model, goals_per_danger


def option_cost_inputs(opts: pd.DataFrame) -> pd.DataFrame:
    """Cost-model inputs for each option: lost at the passer, won halfway along the pass."""
    X = pd.DataFrame(index=opts.index)
    X["loss_x"] = opts["passer_x"]
    X["loss_abs_y"] = opts["passer_abs_y"]
    X["won_x"] = (opts["passer_x"] + opts["opt_x"]) / 2
    X["won_abs_y"] = ((opts["passer_y"] + opts["opt_y"]) / 2).abs()
    X["loss_pressure_id"] = opts["passer_pressure_id"]
    X["score_diff_losing"] = opts["team_score"] - opts["opponent_team_score"]
    X["minute"] = opts["minute_start"]
    X["loss_phase_type"] = opts["phase"]
    return X


def oof_progression(opts: pd.DataFrame, train: pd.DataFrame, target: str, params: dict | None = None,
                    task: str = "reg", n_splits: int = 5) -> np.ndarray:
    """Progression potential for every option, predicted by a model that never saw that match."""
    params = params or DEFAULT_GBM
    pred = np.zeros(len(opts))
    matches = opts["match_id"].to_numpy()
    for tr_m, te_m in GroupKFold(n_splits=n_splits).split(opts, groups=matches):
        test_ids = np.unique(matches[te_m])
        tr = train[~train["match_id"].isin(test_ids)]
        m = _gbm(task, params).fit(_X(tr), tr[target].astype(float if task == "reg" else int))
        Xte = _X(opts.iloc[te_m])
        pred[te_m] = m.predict(Xte) if task == "reg" else m.predict_proba(Xte)[:, 1]
    return pred


def oof_completion(opts: pd.DataFrame, numeric: list[str], params: dict | None = None,
                   n_splits: int = 5) -> np.ndarray:
    """Completion probability for every option from a model trained on the passes actually
    played (received or not), each match predicted by a model that never saw it."""
    params = params or DEFAULT_GBM
    pred = np.zeros(len(opts))
    matches = opts["match_id"].to_numpy()
    played = opts[opts["chosen"] == True]  # noqa: E712
    for _, te_m in GroupKFold(n_splits=n_splits).split(opts, groups=matches):
        test_ids = np.unique(matches[te_m])
        tr = played[~played["match_id"].isin(test_ids)]
        m = _gbm("clf", params).fit(_X(tr, numeric), (tr["received"] == True).astype(int))  # noqa: E712
        pred[te_m] = m.predict_proba(_X(opts.iloc[te_m], numeric))[:, 1]
    return pred


def add_option_value(opts: pd.DataFrame, gamma: float = 1.0, kappa: float = 1.0,
                     completion: str = "xpass_completion") -> pd.DataFrame:
    """V and its decision-level comparisons. Needs columns P (progression) and cost."""
    o = opts.copy()
    xp = o[completion]
    o["V_reward"] = xp * o["xthreat"]
    o["V_progression"] = xp * gamma * o["P"]
    o["V_cost"] = (1 - xp) * kappa * o["cost"]
    o["V"] = o["V_reward"] + o["V_progression"] - o["V_cost"]
    g = o.groupby("decision_id")["V"]
    o["V_best"] = g.transform("max")
    o["V_mean"] = g.transform("mean")
    o["V_worst"] = g.transform("min")
    o["is_best"] = o["V"] == o["V_best"]
    return o


def decisions(o: pd.DataFrame) -> pd.DataFrame:
    """One row per passing decision: the chosen option against the alternatives."""
    ch = o[o["chosen"]].copy()
    ch["n_options"] = o.groupby("decision_id")["V"].transform("size")[ch.index]
    ch["decision_gap"] = ch["V"] - ch["V_best"]            # <= 0; 0 = picked the best option
    ch["bva"] = ch["V"] - ch["V_mean"]                      # value above the average choice
    ch["picked_best"] = ch["is_best"]
    # Where the choice sits between the worst (0) and best (1) option. Unlike the gap, it doesn't
    # grow just because a player's options are spread wider (e.g. higher up the pitch).
    spread = (ch["V_best"] - ch["V_worst"]).replace(0, np.nan)
    ch["decision_quality"] = (ch["V"] - ch["V_worst"]) / spread
    ch["pos_group"] = ch["player_in_possession_position"].map(POSITION_GROUPS).fillna("Other")
    return ch


def shrink(df: pd.DataFrame, value: str, by: str, group: str) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of per-player means towards their position-group mean.

    k = within-player variance / between-player variance (method of moments),
    shrunk = (n * mean + k * group_mean) / (n + k).
    """
    s = df.groupby(by).agg(n=(value, "size"), mean=(value, "mean"), var=(value, "var"), grp=(group, "first"))
    within = df[value].var() if s["var"].isna().all() else np.nanmean(s["var"])
    between = max(np.var(s["mean"]) - within / s["n"].mean(), 1e-12)
    k = within / between
    gmean = df.groupby(group)[value].mean()
    s["shrunk"] = (s["n"] * s["mean"] + k * s["grp"].map(gmean)) / (s["n"] + k)
    s["k"] = k
    return s
