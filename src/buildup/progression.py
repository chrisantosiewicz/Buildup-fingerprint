"""Progression potential: how good the receiver's next situation will be.

Trained on chosen, completed passes (where the receiver's next options are
observed), then applied to every option. Evaluation and tuning are nested and
grouped by match: the outer folds score, the inner folds tune, and no match is
used for both.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .options import OPTION_CATEGORICAL, OPTION_NUMERIC

optuna.logging.set_verbosity(optuna.logging.WARNING)
# LightGBM warns on every eval_set with pandas categoricals; harmless here.
warnings.filterwarnings("ignore", message=".*_validate_eval_set_Xy.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

DEFAULT_GBM = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def _X(df: pd.DataFrame, numeric=OPTION_NUMERIC, categorical=OPTION_CATEGORICAL) -> pd.DataFrame:
    X = df[numeric + categorical].copy()
    for c in categorical:
        X[c] = X[c].astype(str).astype("category")
    return X


def _linear(task: str, numeric, categorical):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    est = Ridge(alpha=1.0) if task == "reg" else LogisticRegression(max_iter=3000)
    return make_pipeline(pre, est)


def _gbm(task: str, params: dict):
    return (lgb.LGBMRegressor if task == "reg" else lgb.LGBMClassifier)(**params)


def _predict(model, X, task):
    return model.predict(X) if task == "reg" else model.predict_proba(X)[:, 1]


def _score(y, p, task) -> dict:
    if task == "reg":
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        r2 = 1 - mean_squared_error(y, p) / np.var(y)
        return {"rmse": round(rmse, 5), "r2": round(float(r2), 3)}
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"auc": round(roc_auc_score(y, p), 3), "logloss": round(log_loss(y, p), 4)}


def tune_gbm(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, task: str, n_trials: int = 30,
             n_splits: int = 4, seed: int = 0) -> dict:
    """Optuna (TPE) search over a small LightGBM space, scored on grouped inner folds."""
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    def objective(trial):
        params = dict(
            n_estimators=1000, verbose=-1, subsample_freq=1,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 32),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
        )
        losses, iters = [], []
        for tr, va in folds:
            m = _gbm(task, params).fit(
                X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
            p = _predict(m, X.iloc[va], task)
            losses.append(mean_squared_error(y[va], p) if task == "reg" else log_loss(y[va], np.clip(p, 1e-4, 1 - 1e-4)))
            iters.append(m.best_iteration_ or params["n_estimators"])
        trial.set_user_attr("n_estimators", int(np.mean(iters)))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params, n_estimators=study.best_trial.user_attrs["n_estimators"],
                verbose=-1, subsample_freq=1)
    return best


def nested_cv(df: pd.DataFrame, target: str, task: str, n_trials: int = 30, n_outer: int = 5) -> dict:
    """Compare xThreat-only, linear, default GBM and tuned GBM on outer held-out matches.

    Returns scores per model and the out-of-fold predictions of the tuned GBM.
    """
    y = df[target].astype(float if task == "reg" else int).to_numpy()
    g = df["match_id"].to_numpy()
    X = _X(df)
    oof = {k: np.zeros(len(df)) for k in ["xthreat_only", "linear", "gbm_default", "gbm_tuned"]}
    params_per_fold = []
    for tr, te in GroupKFold(n_splits=n_outer).split(X, y, g):
        m0 = _linear(task, ["xthreat", "xpass_completion"], []).fit(df.iloc[tr][["xthreat", "xpass_completion"]], y[tr])
        oof["xthreat_only"][te] = _predict(m0, df.iloc[te][["xthreat", "xpass_completion"]], task)
        m1 = _linear(task, OPTION_NUMERIC, OPTION_CATEGORICAL).fit(df.iloc[tr][OPTION_NUMERIC + OPTION_CATEGORICAL], y[tr])
        oof["linear"][te] = _predict(m1, df.iloc[te][OPTION_NUMERIC + OPTION_CATEGORICAL], task)
        m2 = _gbm(task, DEFAULT_GBM).fit(X.iloc[tr], y[tr])
        oof["gbm_default"][te] = _predict(m2, X.iloc[te], task)
        best = tune_gbm(X.iloc[tr], y[tr], g[tr], task, n_trials=n_trials)
        params_per_fold.append(best)
        m3 = _gbm(task, best).fit(X.iloc[tr], y[tr])
        oof["gbm_tuned"][te] = _predict(m3, X.iloc[te], task)
    return {"scores": {k: _score(y, v, task) for k, v in oof.items()},
            "params_per_fold": params_per_fold, "oof": oof}


def fit_final(df: pd.DataFrame, target: str, task: str, n_trials: int = 30):
    """Tune on all matches (grouped folds), then fit on everything."""
    y = df[target].astype(float if task == "reg" else int).to_numpy()
    X = _X(df)
    best = tune_gbm(X, y, df["match_id"].to_numpy(), task, n_trials=n_trials)
    return _gbm(task, best).fit(X, y), best


def predict(model, df: pd.DataFrame, task: str) -> np.ndarray:
    return _predict(model, _X(df), task)
