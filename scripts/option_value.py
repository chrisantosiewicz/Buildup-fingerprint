"""Build-up value pipeline: option value, decisions, validation, player profiles.

Usage:
    python scripts/option_value.py --data path/to/opendata/data --out outputs [--trials 25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from concurrent.futures import ProcessPoolExecutor

from buildup.io import FPS, load_all_events
from buildup.options import OPTION_NUMERIC
from buildup.space import SPACE_NUMERIC, option_space
from buildup.tracking import MatchTracking
from buildup.models import prepare_turnovers
from buildup.options import _chain_progress, option_table, possession_option_value, receiver_targets
from buildup.outcomes import add_outcomes
from buildup.progression import fit_final
from buildup.turnovers import build_turnovers, possessions, turnover_mask
from buildup.value import (
    COST_CATEGORICAL, COST_NUMERIC, POSITION_GROUPS, add_option_value, decisions, fit_cost_model, oof_completion,
    oof_progression,
    option_cost_inputs, shrink,
)

BUILDUP_PHASES = ("build_up", "create")
NON_OPEN_PLAY = ("set_play", "disruption")
GAMMAS = [0, 0.5, 1, 1.5, 2, 3]
CREATED_THRESHOLD = 0.005   # goals: a better pass available at the pass than at the start
MIN_DECISIONS = 40
MIN_POSSESSIONS = 40
MIN_HALF = 15


def decision_outcomes(events: pd.DataFrame, turnovers: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    """What happened after each pass: progress within 10 s, or a dangerous counter conceded."""
    pp = possessions(events)
    pp["lost"] = turnover_mask(pp)
    groups = {k: g.reset_index(drop=True) for k, g in pp.groupby(["match_id", "period"])}
    loc = {(m, e): (k, i) for k, g in groups.items() for i, (m, e) in enumerate(zip(g["match_id"], g["event_id"]))}
    rows = []
    for r in dec[["match_id", "poss_event_id"]].itertuples(index=False):
        k, i = loc[(r.match_id, r.poss_event_id)]
        rows.append(_chain_progress(groups[k], i, 10 * FPS)["prog_10s"])
    out = dec[["decision_id", "match_id", "poss_event_id"]].copy()
    out["outcome_prog_10s"] = rows
    danger = turnovers.set_index(["match_id", "loss_event_id"])["danger_level"] >= 2
    out["outcome_conceded_danger"] = [bool(danger.get((m, e), False)) for m, e in zip(out["match_id"], out["poss_event_id"])]
    return out


def oof_cost(opts: pd.DataFrame, turnovers: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Turnover cost per option from a model that never saw that match."""
    cost = np.zeros(len(opts))
    m_ids = opts["match_id"].to_numpy()
    X_all = option_cost_inputs(opts)
    gpd_list = []
    for _, te in GroupKFold(5).split(opts, groups=m_ids):
        test = np.unique(m_ids[te])
        model, gpd = fit_cost_model(turnovers[~turnovers["match_id"].isin(test)])
        gpd_list.append(gpd)
        cost[te] = model.predict_proba(X_all.iloc[te][COST_NUMERIC + COST_CATEGORICAL])[:, 1] * gpd
    return cost, float(np.mean(gpd_list))


def creator_table(events: pd.DataFrame) -> pd.DataFrame:
    """Open-play possessions with whether the player made himself a better option than he started with.

    created = best option value at the pass moment exceeds the best at the start by at least
    CREATED_THRESHOLD. The raw difference is too noisy per player (split-half r near 0); the
    share of possessions where the player clearly improved his options is stable.
    """
    ovc = possession_option_value(events)
    pp = possessions(events)[["match_id", "event_id", "player_name", "player_position", "team_shortname",
                              "team_in_possession_phase_type", "carry"]]
    ovc = ovc.merge(pp, left_on=["match_id", "poss_event_id"], right_on=["match_id", "event_id"])
    ovc = ovc[~ovc["team_in_possession_phase_type"].isin(NON_OPEN_PLAY)].copy()
    ovc["player"] = ovc["player_name"]
    ovc["pos_group"] = ovc["player_position"].map(POSITION_GROUPS).fillna("Other")
    ovc["created"] = (ovc["on_ball_value_created"] >= CREATED_THRESHOLD).astype(float)
    return ovc


def _space(args):
    data_dir, match_id, opts = args
    r = option_space(MatchTracking(data_dir, match_id), opts)
    return r


def tracking_space(data_dir: str, opts: pd.DataFrame, workers: int) -> pd.DataFrame:
    cols = ["decision_id", "match_id", "pass_frame", "period", "team_id", "player_id", "passer_id"]
    jobs = [(data_dir, m, g[cols]) for m, g in opts.groupby("match_id")]
    with ProcessPoolExecutor(workers) as ex:
        return pd.concat(list(ex.map(_space, jobs)), ignore_index=True)


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--no-tracking", action="store_true",
                    help="use SkillCorner's xPass for completion instead of the tracking-based model")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    events = load_all_events(args.data)
    turnovers = prepare_turnovers(add_outcomes(build_turnovers(events), events))
    opts = option_table(events)
    targets = receiver_targets(events, opts)
    train = opts[opts["chosen"]].merge(targets[["decision_id", "next_best_ev", "prog_10s"]], on="decision_id")
    res["n_options"], res["n_decisions"], res["n_train"] = len(opts), int(opts["decision_id"].nunique()), len(train)

    # --- Layers 3 and 4: progression potential and turnover cost, out of fold by match -----------
    # Hyperparameters are tuned once on grouped folds; P for each option then comes from a model
    # that never saw that option's match.
    _, params = fit_final(train, "next_best_ev", "reg", n_trials=args.trials)
    res["progression_params"] = params
    opts["P"] = np.clip(oof_progression(opts, train, "next_best_ev", params), 0, None)
    opts["cost"], res["goals_per_dangerous_counter"] = oof_cost(opts, turnovers)

    # --- Layer 2: completion. SkillCorner's xPass, or (default) a model that adds the space
    # around each option from tracking 0.3 s before the pass. ---------------------------------
    if args.no_tracking:
        opts["completion"] = opts["xpass_completion"]
    else:
        sp = tracking_space(args.data, opts, args.workers)
        opts = opts.merge(sp, on=["decision_id", "player_id"], how="left")
        opts["completion"] = oof_completion(opts, OPTION_NUMERIC + SPACE_NUMERIC)
        played = opts[opts["chosen"] == True]  # noqa: E712
        y = (played["received"] == True).astype(int)  # noqa: E712
        res["completion_model"] = {
            "auc": {"skillcorner_xpass": auc(y, played["xpass_completion"]), "with_tracking": auc(y, played["completion"])},
            "options_with_tracking": round(float(opts["trk_nearest_opp"].notna().mean()), 3)}

    # --- Layer 5: gamma is fixed (goal units on both sides), sensitivity reported -------------
    # Tuning gamma on progress alone just pushes it up: more lookahead always predicts progress
    # better and conceded danger worse. So gamma is a stated choice, and the trade-off is shown.
    sens = []
    oc = None
    for gamma in GAMMAS:
        d = decisions(add_option_value(opts, gamma=gamma, completion="completion"))
        if oc is None:
            oc = decision_outcomes(events, turnovers, d)  # outcomes don't depend on gamma
        d = d.merge(oc[["decision_id", "outcome_prog_10s", "outcome_conceded_danger"]], on="decision_id")
        sens.append({"gamma": gamma, "progress_10s_auc": auc(d["outcome_prog_10s"], d["V"]),
                     "conceded_danger_auc": auc(d["outcome_conceded_danger"], -d["V"]),
                     "share_picked_best": round(float(d["picked_best"].mean()), 3)})
    res["gamma"], res["gamma_sensitivity"] = args.gamma, sens

    o = add_option_value(opts, gamma=args.gamma, completion="completion")
    d = decisions(o).merge(oc[["decision_id", "outcome_prog_10s", "outcome_conceded_danger"]], on="decision_id")
    o.to_csv(out / "option_values.csv", index=False)
    d.to_csv(out / "decisions.csv", index=False)

    # --- Validation (every layer is out of fold by match; nothing here is tuned on outcomes) ----
    res["validation"] = {
        "n_decisions": len(d),
        "progress_10s_auc": {"V": auc(d["outcome_prog_10s"], d["V"]),
                             "xThreat_only": auc(d["outcome_prog_10s"], d["xthreat"]),
                             "xPass_only": auc(d["outcome_prog_10s"], d["xpass_completion"]),
                             "completion_only": auc(d["outcome_prog_10s"], d["completion"]),
                             "progression_only": auc(d["outcome_prog_10s"], d["P"])},
        "conceded_danger_auc (higher V = safer)": {"V": auc(d["outcome_conceded_danger"], -d["V"]),
                                                   "xPass_only": auc(d["outcome_conceded_danger"], -d["xpass_completion"]),
                                                   "completion_only": auc(d["outcome_conceded_danger"], -d["completion"]),
                                                   "cost_term": auc(d["outcome_conceded_danger"], d["V_cost"])},
        "progress_rate_by_V_quintile": d.groupby(pd.qcut(d["V"], 5, labels=False, duplicates="drop"))["outcome_prog_10s"].mean().round(3).to_dict(),
        "conceded_danger_rate_by_V_quintile": d.groupby(pd.qcut(d["V"], 5, labels=False, duplicates="drop"))["outcome_conceded_danger"].mean().round(4).to_dict(),
    }
    res["share_picked_best_option"] = round(float(d["picked_best"].mean()), 3)
    res["share_picked_best_by_pressure"] = d.groupby("passer_pressure_id")["picked_best"].mean().round(3).to_dict()

    # Back passes under pressure: average value vs alternatives, and how often it was the best option.
    d["direction"] = np.select([d["pass_dx"] > 2, d["pass_dx"] < -2], ["forward", "backward"], "sideways")
    d["pressure"] = pd.cut(d["passer_pressure_id"], [0, 2, 3, 5], labels=["none/low", "medium", "high"])
    res["direction_by_pressure"] = {
        k: d.pivot_table(index="pressure", columns="direction", values=v, aggfunc=f, observed=False).round(r).to_dict(orient="index")
        for k, v, f, r in [("bva", "bva", "mean", 5), ("share_best", "picked_best", "mean", 3), ("n", "bva", "size", 0)]}

    # --- Player profiles: decision-maker, creator, provider -------------------------------------
    match_rank = {m: i for i, m in enumerate(sorted(d["match_id"].unique()))}
    bu = d[d["phase"].isin(BUILDUP_PHASES)].copy()
    bu["player"] = bu["player_in_possession_name"]
    bu["provided"] = bu["V_progression"]       # how good a situation the pass hands the receiver
    ovc = creator_table(events)

    def profile(df, value, nmin):
        s = shrink(df, value, "player", "pos_group")
        return s[s["n"] >= nmin]

    # Decision-maker: decision quality over all open play, relative to what's typical for the same
    # position, phase and number of options (the gap alone mostly measured position).
    dq = d[~d["phase"].isin(NON_OPEN_PLAY) & d["decision_quality"].notna()].copy()
    dq["player"] = dq["player_in_possession_name"]
    dq["dq_adj"] = dq["decision_quality"] - dq.groupby(["pos_group", "phase", "n_options"])["decision_quality"].transform("mean")
    dm = profile(dq, "dq_adj", MIN_DECISIONS).rename(columns={"shrunk": "decision_maker", "n": "n_decisions"})
    cr = profile(ovc, "created", MIN_POSSESSIONS).rename(columns={"shrunk": "creator", "n": "n_possessions"})
    pr = profile(bu, "provided", MIN_DECISIONS).rename(columns={"shrunk": "provider"})
    prof = dm[["n_decisions", "grp", "decision_maker"]].join(cr[["n_possessions", "creator"]], how="inner").join(pr[["provider"]], how="inner")
    for c in ["decision_maker", "creator", "provider"]:
        prof[c + "_pct"] = prof.groupby("grp")[c].rank(pct=True).round(2)
    team = bu.groupby("player")["team_shortname"].agg(lambda s: s.mode().iloc[0])
    prof["team"] = team.reindex(prof.index)
    prof.sort_values("decision_maker", ascending=False).to_csv(out / "player_profiles.csv")
    res["n_profiled_players"] = len(prof)
    res["top5_within_position"] = {
        c: prof[prof["grp"].isin(["CB", "FB", "MF", "AM", "FW"])].sort_values(c, ascending=False)
        .groupby("grp").head(1)[["team", "grp", c]].round(5).reset_index().to_dict(orient="records")
        for c in ["decision_maker", "creator", "provider"]}

    # Split-half stability (odd vs even matches). "within_position" removes the position-group mean
    # first, so a metric doesn't look stable just because centre-backs differ from wingers.
    stab = {}
    for name, df, val in [("decision_maker", dq, "dq_adj"), ("decision_gap", bu, "decision_gap"), ("decision_maker_bva", bu, "bva"),
                          ("creator", ovc, "created"), ("creator_raw_value", ovc, "on_ball_value_created"),
                          ("provider", bu, "provided")]:
        df = df.assign(half=df["match_id"].map(match_rank) % 2)
        df[val + "_w"] = df[val] - df.groupby("pos_group")[val].transform("mean")
        row = {}
        for tag, v in [("raw", val), ("within_position", val + "_w")]:
            a = df.groupby(["player", "half"])[v].agg(["mean", "size"]).unstack()
            a = a[(a[("size", 0)] >= MIN_HALF) & (a[("size", 1)] >= MIN_HALF)]
            row[tag] = round(float(a[("mean", 0)].corr(a[("mean", 1)])), 3)
        row["players"] = len(a)
        stab[name] = row
    res["split_half_stability"] = stab
    res["profile_correlations"] = prof[["decision_maker", "creator", "provider"]].corr().round(3).to_dict()

    # --- Team view: style of build-up -----------------------------------------------------------
    ovc_team = ovc.groupby("team_shortname")["created"].mean().rename("creator_share")
    tv = bu.groupby("team_shortname").agg(
        decisions=("decision_id", "size"), picked_best=("picked_best", "mean"),
        decision_gap=("decision_gap", "mean"), provided=("provided", "mean"),
        progress_10s=("outcome_prog_10s", "mean"), conceded_danger=("outcome_conceded_danger", "mean"))
    hb = bu[(bu["pressure"] == "high")]
    tv["back_pass_best_share_under_high_pressure"] = hb[hb["direction"] == "backward"].groupby("team_shortname")["picked_best"].mean()
    tv = tv.join(ovc_team).sort_values("picked_best", ascending=False)
    tv.round(5).to_csv(out / "team_buildup.csv")
    res["team_view"] = tv.round(4).reset_index().to_dict(orient="records")
    tstab = {}
    for name, df, val in [("picked_best", bu, "picked_best"), ("decision_gap", bu, "decision_gap"),
                          ("provided", bu, "provided"), ("creator_share", ovc, "created")]:
        a = df.assign(half=df["match_id"].map(match_rank) % 2).groupby(["team_shortname", "half"])[val].mean().unstack()
        tstab[name] = round(float(a[0].corr(a[1])), 3)
    res["team_split_half_stability"] = tstab

    with open(out / "buildup_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
