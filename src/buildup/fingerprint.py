"""Build-Up Fingerprint: style (what a player or team tends to do) and quality (how well).

Every metric carries its reliability: the split-half correlation between odd and even
matches, after removing the position-group mean, projected to the full sample with
Spearman-Brown. Quality metrics come with the probability that the player is above his
position average, from the empirical-Bayes posterior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .turnovers import possessions
from .value import POSITION_GROUPS

NON_OPEN_PLAY = ("set_play", "disruption")
CREATED_THRESHOLD = 0.005
OPTION_MEAN_COLS = ["xpass_completion", "completion", "pass_dx", "xthreat", "P", "V_cost"]


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    kind: str            # "style" or "quality"
    source: str          # "dec" (passing decisions) or "poss" (player possessions)
    column: str
    subset: str | None = None   # optional boolean column restricting the rows
    min_half: int = 15
    tiered: bool = True         # False: tiers don't hold up out of sample, show the estimate only


METRICS = [
    Metric("long_share", "Long passes (>30 m)", "style", "dec", "long"),
    Metric("forwardness", "Forwardness (vs average option)", "style", "dec", "forward"),
    Metric("risk", "Risk accepted (vs average option)", "style", "dec", "risk"),
    Metric("back_under_pressure", "Back passes under high pressure", "style", "dec", "back", "high_pressure", 8),
    Metric("carry_share", "Carries", "style", "poss", "carry"),
    Metric("quick_release", "Quick release (<=1 s)", "style", "poss", "quick"),
    Metric("tempo", "Time on the ball (s)", "style", "poss", "duration"),
    Metric("provider", "Provider", "quality", "dec", "provided"),
    Metric("decision_quality", "Decision quality", "quality", "dec", "dq_adj"),
    Metric("creator", "Creator", "quality", "poss", "created", tiered=False),
]


def decision_frame(o: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Open-play passing decisions with style columns measured against the decision's options."""
    if "completion" not in o:
        o = o.assign(completion=o["xpass_completion"])
        d = d.assign(completion=d["xpass_completion"])
    means = o.groupby("decision_id")[OPTION_MEAN_COLS].mean().add_suffix("_m")
    d = d.merge(means, left_on="decision_id", right_index=True)
    d = d[~d["phase"].isin(NON_OPEN_PLAY)].copy()
    d["player"] = d["player_in_possession_name"]
    # Risk as a style: how hard the chosen lane is by SkillCorner's xPass, vs the average option.
    # (The tracking-based completion is better at predicting outcomes, but as a style measure it
    # mostly picks up the moment; the xPass version is the steadier team trait: r 0.55 vs -0.09.)
    d["risk"] = d["xpass_completion_m"] - d["xpass_completion"]     # + = riskier than the average option
    d["forward"] = d["pass_dx"] - d["pass_dx_m"]
    d["cost_taken"] = d["V_cost"] - d["V_cost_m"]
    d["back"] = (d["pass_dx"] < -2).astype(float)
    d["long"] = (d["pass_len"] > 30).astype(float)
    d["high_pressure"] = d["passer_pressure_id"] >= 4
    d["provided"] = d["V_progression"]
    # Decision quality relative to what's typical for the same position, phase and option count.
    typical = d.groupby(["pos_group", "phase", "n_options"])["decision_quality"].transform("mean")
    d["dq_adj"] = d["decision_quality"] - typical
    return d


def possession_frame(events: pd.DataFrame, ovc: pd.DataFrame) -> pd.DataFrame:
    """Open-play player possessions with carry, tempo and option-creation columns."""
    p = possessions(events)
    p = p[~p["team_in_possession_phase_type"].isin(NON_OPEN_PLAY)].copy()
    p = p.merge(ovc[["match_id", "poss_event_id", "best_ev_start", "best_ev_pass", "on_ball_value_created"]],
                left_on=["match_id", "event_id"], right_on=["match_id", "poss_event_id"], how="left")
    p["player"] = p["player_name"]
    p["pos_group"] = p["player_position"].map(POSITION_GROUPS).fillna("Other")
    p["carry"] = (p["carry"] == True).astype(float)  # noqa: E712
    p["quick"] = (p["duration"] <= 1.0).astype(float)
    p["created"] = (p["on_ball_value_created"] >= CREATED_THRESHOLD).astype(float)
    return p


def _rows(m: Metric, dec: pd.DataFrame, poss: pd.DataFrame) -> pd.DataFrame:
    df = dec if m.source == "dec" else poss
    df = df[df[m.subset]] if m.subset else df
    return df[df[m.column].notna()]


def split_half(df: pd.DataFrame, col: str, by: str, match_rank: dict, min_half: int,
               within: str | None = "pos_group") -> dict:
    """Odd-vs-even-match correlation of per-`by` means (optionally within position)."""
    df = df.assign(half=df["match_id"].map(match_rank) % 2)
    v = col
    if within:
        df["_w"] = df[col] - df.groupby(within)[col].transform("mean")
        v = "_w"
    a = df.groupby([by, "half"])[v].agg(["mean", "size"]).unstack()
    a = a[(a[("size", 0)] >= min_half) & (a[("size", 1)] >= min_half)]
    r = float(a[("mean", 0)].corr(a[("mean", 1)])) if len(a) > 2 else np.nan
    rp = max(r, 0.0) if np.isfinite(r) else 0.0
    return {"split_half_r": round(r, 3), "full_sample_reliability": round(2 * rp / (1 + rp), 2), "n": len(a)}


def reliability_table(dec: pd.DataFrame, poss: pd.DataFrame) -> pd.DataFrame:
    match_rank = {m: i for i, m in enumerate(sorted(dec["match_id"].unique()))}
    rows = []
    for m in METRICS:
        df = _rows(m, dec, poss)
        raw = split_half(df, m.column, "player", match_rank, m.min_half, within=None)
        wp = split_half(df, m.column, "player", match_rank, m.min_half)
        team = split_half(df, m.column, "team_shortname", match_rank, 1, within=None)
        rows.append({"metric": m.key, "label": m.label, "kind": m.kind,
                     "player_r_raw": raw["split_half_r"], "player_r_within_position": wp["split_half_r"],
                     "player_reliability": wp["full_sample_reliability"], "players": wp["n"],
                     "team_r": team["split_half_r"]})
    return pd.DataFrame(rows)


def posterior(df: pd.DataFrame, value: str, by: str = "player", group: str = "pos_group",
              min_half: int = 15) -> pd.DataFrame:
    """Empirical-Bayes estimate per player, shrunk to his position mean, calibrated on split halves.

    Both variance components come from odd- vs even-match halves, so match-level shocks
    (opponent, game state) count as noise rather than as player signal:
      tau^2   = covariance of the two half-means (deviations from the position mean) across
                players: the variance of true player differences;
      sigma^2 = mean of (h0 - h1)^2 / (1/n0 + 1/n1): per-action noise, so se^2 = sigma^2 / n.
    Then shrunk deviation = B * mean deviation with B = tau^2 / (tau^2 + se^2), posterior
    sd = sqrt(B * se^2), and p_above_position = P(true deviation > 0).
    """
    match_rank = {m: i for i, m in enumerate(sorted(df["match_id"].unique()))}
    df = df.assign(half=df["match_id"].map(match_rank) % 2)
    gmean = df.groupby(group)[value].mean()
    df["dev"] = df[value] - df[group].map(gmean)
    h = df.groupby([by, "half"])["dev"].agg(["mean", "size"]).unstack()
    h = h[(h[("size", 0)] >= min_half) & (h[("size", 1)] >= min_half)]
    h0, h1, n0, n1 = h[("mean", 0)], h[("mean", 1)], h[("size", 0)], h[("size", 1)]
    tau2 = max(float(np.cov(h0, h1)[0, 1]), 1e-12)
    sigma2 = float(np.mean((h0 - h1) ** 2 / (1 / n0 + 1 / n1)))
    s = df.groupby(by).agg(n=("dev", "size"), dev=("dev", "mean"), grp=(group, "first"))
    s["group_mean"] = s["grp"].map(gmean)
    s["mean"] = s["group_mean"] + s["dev"]
    s["se2"] = sigma2 / s["n"]
    s["B"] = tau2 / (tau2 + s["se2"])
    s["shrunk"] = s["group_mean"] + s["B"] * s["dev"]
    s["sd"] = np.sqrt(s["B"] * s["se2"])
    s["p_above_position"] = norm.cdf(s["B"] * s["dev"] / s["sd"])
    s.attrs.update(tau2=tau2, sigma2=sigma2, players_in_calibration=len(h))
    return s


def tier(p: float) -> str:
    if p >= 0.8:
        return "above average"
    if p <= 0.2:
        return "below average"
    return "around average"


def player_fingerprint(dec: pd.DataFrame, poss: pd.DataFrame, min_decisions: int = 40,
                       min_possessions: int = 40) -> pd.DataFrame:
    """One row per player: style percentiles within position and quality estimates with tiers."""
    n_dec = dec.groupby("player").size()
    n_pos = poss.groupby("player").size()
    keep = n_dec[n_dec >= min_decisions].index.intersection(n_pos[n_pos >= min_possessions].index)
    pos = dec.groupby("player")["pos_group"].agg(lambda s: s.mode().iloc[0])
    team = dec.groupby("player")["team_shortname"].agg(lambda s: s.mode().iloc[0])
    fp = pd.DataFrame({"team": team, "pos_group": pos, "n_decisions": n_dec, "n_possessions": n_pos}).loc[keep]
    for m in METRICS:
        rows = _rows(m, dec, poss)
        rows = rows.assign(pos_group=rows["player"].map(pos).fillna(rows["pos_group"]))
        s = posterior(rows, m.column, min_half=m.min_half)
        fp[m.key] = s["shrunk"].reindex(fp.index)
        if m.kind == "quality" and m.tiered:
            fp[m.key + "_p_above"] = s["p_above_position"].reindex(fp.index).round(3)
            fp[m.key + "_tier"] = fp[m.key + "_p_above"].map(lambda p: tier(p) if pd.notna(p) else np.nan)
    for m in METRICS:
        fp[m.key + "_pct"] = fp.groupby("pos_group")[m.key].rank(pct=True).round(2)
    return fp[fp["pos_group"].isin(["CB", "FB", "MF", "AM", "FW"])]


def validate_tiers(dec: pd.DataFrame, poss: pd.DataFrame, min_actions: int = 20) -> dict:
    """Out-of-sample check of the quality tiers: estimate on odd matches, look at even matches.

    For each tier from the odd-match posterior: how many players, and the share whose
    even-match mean is above their position mean.
    """
    out = {}
    for m in [m for m in METRICS if m.kind == "quality"]:
        rows = _rows(m, dec, poss)
        rank = {mid: i for i, mid in enumerate(sorted(rows["match_id"].unique()))}
        half = rows["match_id"].map(rank) % 2
        a, b = rows[half == 1], rows[half == 0]
        s = posterior(a, m.column, min_half=max(m.min_half // 2, 5))
        s = s[s["n"] >= min_actions]
        gm = b.groupby("pos_group")[m.column].mean()
        bd = (b[m.column] - b["pos_group"].map(gm)).groupby(b["player"]).agg(["mean", "size"])
        bd = bd[bd["size"] >= min_actions]
        j = s.join(bd, how="inner", rsuffix="_b")
        j["tier"] = j["p_above_position"].map(tier)
        out[m.key] = {t: {"players": int(len(g)), "share_above_in_other_half": round(float((g["mean_b"] > 0).mean()), 2)}
                      for t, g in j.groupby("tier")}
    return out


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def team_style(dec: pd.DataFrame, poss: pd.DataFrame) -> pd.DataFrame:
    """Team build-up style on two axes plus outcomes.

    directness = mean z of long-pass share and forwardness
    risk       = mean z of risk accepted and (minus) back-pass share
    Only components that are stable at team level between odd and even matches are used
    (long 0.49, forwardness 0.54, risk 0.55, back-pass share 0.69; time on the ball 0.06,
    turnover cost accepted -0.2 and back passes under high pressure 0.12 are not).
    """
    t = dec.groupby("team_shortname").agg(
        decisions=("decision_id", "size"), long_share=("long", "mean"), forwardness=("forward", "mean"),
        risk=("risk", "mean"), cost_taken=("cost_taken", "mean"), provider=("provided", "mean"),
        decision_quality=("dq_adj", "mean"), picked_best=("picked_best", "mean"),
        progress_10s=("outcome_prog_10s", "mean"), conceded_danger=("outcome_conceded_danger", "mean"))
    hp = dec[dec["high_pressure"]]
    t["back_under_pressure"] = hp.groupby("team_shortname")["back"].mean()
    t["back_share"] = dec.groupby("team_shortname")["back"].mean()
    pt = poss.groupby("team_shortname").agg(tempo=("duration", "mean"), carry_share=("carry", "mean"),
                                            creator=("created", "mean"))
    t = t.join(pt)
    t["directness"] = (_z(t["long_share"]) + _z(t["forwardness"])) / 2
    t["risk_axis"] = (_z(t["risk"]) - _z(t["back_share"])) / 2
    return t


def team_outcomes(data_dir, events: pd.DataFrame, turnovers: pd.DataFrame) -> pd.DataFrame:
    """Per-match averages of what each team scored and allowed, over its matches in the data.

    goals from match.json; shots from possessions ending in a shot; dangerous counters are
    open-play turnovers that became a box entry or shot within 10 s (`dangerous_10s`).
    """
    from .io import load_match_meta

    rows = []
    for mid in events["match_id"].unique():
        m = load_match_meta(data_dir, mid)
        h, a = m["home_team"]["short_name"], m["away_team"]["short_name"]
        rows += [{"match_id": mid, "team_shortname": h, "goals_for": m["home_team_score"], "goals_against": m["away_team_score"]},
                 {"match_id": mid, "team_shortname": a, "goals_for": m["away_team_score"], "goals_against": m["home_team_score"]}]
    g = pd.DataFrame(rows)
    p = possessions(events)
    shots = p[p["end_type"] == "shot"].groupby(["match_id", "team_shortname"]).size().rename("shots_for")
    g = g.merge(shots.reset_index(), on=["match_id", "team_shortname"], how="left").fillna({"shots_for": 0})
    opp = g[["match_id", "team_shortname", "shots_for"]].rename(columns={"team_shortname": "opp", "shots_for": "shots_against"})
    g = g.merge(g[["match_id", "team_shortname"]].rename(columns={"team_shortname": "opp"}), on="match_id")
    g = g[g["opp"] != g["team_shortname"]].merge(opp, on=["match_id", "opp"]).drop(columns="opp")
    lost = turnovers.groupby(["match_id", "losing_team_id"]).agg(counters_conceded=("dangerous_10s", "sum"),
                                                                  turnovers=("dangerous_10s", "size")).reset_index()
    ids = p.groupby(["match_id", "team_id"])["team_shortname"].first().reset_index()
    lost = lost.merge(ids, left_on=["match_id", "losing_team_id"], right_on=["match_id", "team_id"])
    g = g.merge(lost[["match_id", "team_shortname", "counters_conceded", "turnovers"]], on=["match_id", "team_shortname"], how="left")
    out = g.groupby("team_shortname").agg(matches=("match_id", "nunique"), goals_for=("goals_for", "mean"),
                                          goals_against=("goals_against", "mean"), shots_for=("shots_for", "mean"),
                                          shots_against=("shots_against", "mean"),
                                          counters_conceded=("counters_conceded", "mean"), turnovers=("turnovers", "mean"))
    out["goal_diff"] = out["goals_for"] - out["goals_against"]
    return out
