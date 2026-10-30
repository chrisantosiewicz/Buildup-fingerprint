"""Models: danger-if-lost, probability of loss, and turnover exposure.

All evaluation is out-of-sample with whole matches held out (GroupKFold on
match_id), so a model never sees other moments from the match it is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .turnovers import possessions, turnover_mask

# --- Feature sets ------------------------------------------------------------
# Danger-if-lost uses only what is known at the moment of loss, from the losing
# team's side, so the same model can score any possession ("what if we lost it here?").
LOSS_NUMERIC = ["loss_x", "loss_abs_y", "loss_pressure_id", "score_diff_losing", "minute"]
LOSS_CATEGORICAL = ["loss_phase_type", "lost_by"]

# Event-level proxies for the losing team's shape at t0 (seen from the ball winner).
# Caveat: measured at the regain, not before the loss; tracking replaces these.
SHAPE_NUMERIC = ["rd_n_goal_side", "rd_last_line_height"]
SHAPE_CATEGORICAL = ["rd_ball_inside_shape", "rd_organised"]

# Probability of losing the ball, from the start of each player possession
# (no end-of-possession information, to avoid leakage).
PLOSS_NUMERIC = [
    "x_start", "abs_y_start", "overall_pressure_start_id", "reception_difficulty_start_id",
    "space_constraint_start_id", "time_to_impact_start_id", "passing_option_ease_start_id",
    "n_passing_options_at_start", "n_passing_options_ahead_at_start", "score_diff", "minute_start",
]
PLOSS_CATEGORICAL = ["team_in_possession_phase_type", "start_type"]

DANGER_TARGET = "dangerous_10s"  # box entry or shot within 10 s of the regain


@dataclass
class CVResult:
    oof: np.ndarray
    auc: float
    logloss: float
    brier: float
    base_rate: float

    def summary(self) -> dict:
        return {"auc": round(self.auc, 3), "logloss": round(self.logloss, 4),
                "brier": round(self.brier, 4), "base_rate": round(self.base_rate, 3)}


def _logit(numeric: list[str], categorical: list[str]):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def _gbm():
    # Deliberately small trees and strong regularisation: ~2,750 turnovers, ~200 positives.
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def _prep_gbm(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    X = df[numeric + categorical].copy()
    for c in categorical:
        X[c] = X[c].astype(str).astype("category")
    return X


def grouped_cv(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str],
               kind: str = "logit", n_splits: int = 5) -> CVResult:
    """Out-of-fold probabilities with whole matches held out."""
    y = df[target].astype(int).to_numpy()
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=n_splits).split(df, y, groups=df["match_id"]):
        if kind == "logit":
            m = _logit(numeric, categorical).fit(df.iloc[tr][numeric + categorical], y[tr])
            oof[te] = m.predict_proba(df.iloc[te][numeric + categorical])[:, 1]
        else:
            X = _prep_gbm(df, numeric, categorical)
            m = _gbm().fit(X.iloc[tr], y[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    oof = np.clip(oof, 1e-4, 1 - 1e-4)
    return CVResult(oof, roc_auc_score(y, oof), log_loss(y, oof), brier_score_loss(y, oof), y.mean())


def fit_full(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str]):
    """Logistic model on all rows (for scoring new moments)."""
    return _logit(numeric, categorical).fit(df[numeric + categorical], df[target].astype(int))


# --- Tables ------------------------------------------------------------------

def prepare_turnovers(to: pd.DataFrame) -> pd.DataFrame:
    """Add model columns to the turnover table from add_outcomes()."""
    out = to.copy()
    out[DANGER_TARGET] = out["danger_level"] >= 2
    out["loss_abs_y"] = out["loss_y"].abs()
    for c in SHAPE_CATEGORICAL:
        out[c] = out[c].astype(str)
    return out


def prepare_possessions(events: pd.DataFrame) -> pd.DataFrame:
    """All player possessions with the loss label and model columns.

    Also carries the danger-model inputs, renamed as if the ball were lost at the
    start of this possession, so danger-if-lost can be scored for every possession.
    """
    pp = possessions(events)
    pp["lost"] = turnover_mask(pp)
    pp["abs_y_start"] = pp["y_start"].abs()
    pp["score_diff"] = pp["team_score"] - pp["opponent_team_score"]
    # Danger-if-lost inputs, from the possessing team's side.
    pp["loss_x"] = pp["x_start"]
    pp["loss_abs_y"] = pp["abs_y_start"]
    pp["loss_pressure_id"] = pp["overall_pressure_start_id"]
    pp["score_diff_losing"] = pp["score_diff"]
    pp["minute"] = pp["minute_start"]
    pp["loss_phase_type"] = pp["team_in_possession_phase_type"]
    pp["lost_by"] = "failed_pass"  # most common way of losing it; a neutral reference value
    return pp


def add_exposure(pp: pd.DataFrame, ploss_model, danger_model) -> pd.DataFrame:
    """Exposure = P(lose it in this possession) x P(dangerous transition | lost here)."""
    out = pp.copy()
    out["p_loss"] = ploss_model.predict_proba(out[PLOSS_NUMERIC + PLOSS_CATEGORICAL])[:, 1]
    out["p_danger_if_lost"] = danger_model.predict_proba(out[LOSS_NUMERIC + LOSS_CATEGORICAL])[:, 1]
    out["exposure"] = out["p_loss"] * out["p_danger_if_lost"]
    return out
