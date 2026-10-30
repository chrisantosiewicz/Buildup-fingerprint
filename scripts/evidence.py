"""Evidence behind every headline number, with sample sizes, baselines and intervals.

Reads the outputs of option_value.py and fingerprint.py and writes evidence.json:
  value_vs_baselines  V against each of its parts alone (AUC, top-vs-bottom fifth, best-option picks),
                      with match-bootstrap intervals on the V - P difference
  tiers               odd -> even validation with players per tier, plus a continuous check
                      (odd-match estimate vs even-match mean, all players)
  provider_intervals  how far apart players really are: 80% intervals in percentile terms
  team_axes           odd vs even split-half r of the two team axes with a 95% interval
  counts              how the decision counts fit together

Usage: python scripts/evidence.py --results outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from buildup.fingerprint import METRICS, _rows, posterior, team_style, tier

RNG = np.random.default_rng(0)
N_BOOT = 500


def _auc(y, s):
    ok = np.isfinite(s) & np.isfinite(y)
    return float(roc_auc_score(y[ok], s[ok]))


def _fisher_ci(r: float, n: int) -> list[float]:
    if n <= 3 or not np.isfinite(r):
        return [np.nan, np.nan]
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return [round(float(np.tanh(z - 1.96 * se)), 2), round(float(np.tanh(z + 1.96 * se)), 2)]


def _boot_by_match(df: pd.DataFrame, stat) -> list[float]:
    matches = df["match_id"].unique()
    groups = {m: g for m, g in df.groupby("match_id")}
    vals = []
    for _ in range(N_BOOT):
        pick = RNG.choice(matches, len(matches), replace=True)
        vals.append(stat(pd.concat([groups[m] for m in pick])))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def value_vs_baselines(o: pd.DataFrame, d: pd.DataFrame) -> dict:
    """Does the full option value beat its parts? Three lenses, each against the same baselines."""
    d = d[d["outcome_prog_10s"].notna()].copy()
    d["P_only"] = d["P"]
    d["xThreat_only"] = d["xthreat"]
    d["completion_only"] = d["completion"]
    d["reward_only"] = d["V_reward"]
    d["cost_only"] = -d["V_cost"]
    scores = ["V", "P_only", "xThreat_only", "reward_only", "completion_only", "cost_only"]
    y_prog = d["outcome_prog_10s"].astype(float).values
    y_dang = d["outcome_conceded_danger"].astype(float).values
    out = {"n_decisions": int(len(d)), "n_matches": int(d["match_id"].nunique()),
           "base_rate_progress": round(float(y_prog.mean()), 3),
           "base_rate_conceded_danger": round(float(np.nanmean(y_dang)), 4)}
    # 1. AUC of the chosen option's score for each outcome (danger: higher score = safer)
    out["progress_auc"] = {s: round(_auc(y_prog, d[s].values), 3) for s in scores}
    out["conceded_danger_auc_higher_is_safer"] = {s: round(_auc(1 - y_dang, d[s].values), 3) for s in scores}
    # 2. progress rate, top fifth vs bottom fifth of each score
    spread = {}
    for s in scores:
        q = pd.qcut(d[s].rank(method="first"), 5, labels=False)
        r = d.groupby(q)["outcome_prog_10s"].mean()
        dz = d.groupby(q)["outcome_conceded_danger"].mean()
        spread[s] = {"progress_bottom_fifth": round(float(r.iloc[0]), 3), "progress_top_fifth": round(float(r.iloc[-1]), 3),
                     "danger_bottom_fifth": round(float(dz.iloc[0]), 4), "danger_top_fifth": round(float(dz.iloc[-1]), 4)}
    out["fifths"] = spread
    # 3. within the decision: when the player picked the option a score rates best, what happened?
    multi = o[o.groupby("decision_id")["decision_id"].transform("size") >= 2].copy()
    multi["P_only"], multi["xThreat_only"] = multi["P"], multi["xthreat"]
    multi["completion_only"], multi["reward_only"], multi["cost_only"] = multi["completion"], multi["V_reward"], -multi["V_cost"]
    picks = {}
    dd = d.set_index("decision_id")
    for s in scores:
        best = multi.loc[multi.groupby("decision_id")[s].idxmax(), ["decision_id", "chosen"]]
        best = best.assign(picked=best["chosen"].astype(bool)).set_index("decision_id")["picked"]
        j = dd.join(best, how="inner")
        g = j.groupby("picked")
        picks[s] = {"n": int(len(j)), "share_picked": round(float(j["picked"].mean()), 3),
                    "progress_if_picked": round(float(g["outcome_prog_10s"].mean().get(True, np.nan)), 3),
                    "progress_if_not": round(float(g["outcome_prog_10s"].mean().get(False, np.nan)), 3),
                    "danger_if_picked": round(float(g["outcome_conceded_danger"].mean().get(True, np.nan)), 4),
                    "danger_if_not": round(float(g["outcome_conceded_danger"].mean().get(False, np.nan)), 4)}
    out["picked_the_option_each_score_rates_best"] = picks
    # match-bootstrap intervals for V minus P-only
    out["V_minus_P_progress_auc_95ci"] = _boot_by_match(
        d, lambda x: _auc(x["outcome_prog_10s"].values.astype(float), x["V"].values)
        - _auc(x["outcome_prog_10s"].values.astype(float), x["P_only"].values))
    out["V_minus_P_danger_auc_95ci"] = _boot_by_match(
        d, lambda x: _auc(1 - x["outcome_conceded_danger"].values.astype(float), x["V"].values)
        - _auc(1 - x["outcome_conceded_danger"].values.astype(float), x["P_only"].values))
    return out


def tiers(dec: pd.DataFrame, poss: pd.DataFrame, min_actions: int = 20) -> dict:
    """Odd -> even check of the quality estimates, with counts and a continuous version."""
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
        r = float(np.corrcoef(j["B"] * j["dev"], j["mean_b"])[0, 1])
        out[m.key] = {
            "players_checked": int(len(j)),
            "by_tier": {t: {"players": int(len(g)), "above_in_other_half": int((g["mean_b"] > 0).sum()),
                            "share": round(float((g["mean_b"] > 0).mean()), 2)} for t, g in j.groupby("tier")},
            "continuous_r_odd_estimate_vs_even_mean": round(r, 2), "continuous_r_95ci": _fisher_ci(r, len(j))}
    return out


def provider_intervals(dec: pd.DataFrame, poss: pd.DataFrame, min_decisions: int = 40) -> dict:
    """Separation between players on the full sample: 80% intervals turned into position percentiles."""
    out = {}
    for m in [m for m in METRICS if m.kind == "quality"]:
        rows = _rows(m, dec, poss)
        pos = rows.groupby("player")["pos_group"].agg(lambda s: s.mode().iloc[0])
        rows = rows.assign(pos_group=rows["player"].map(pos))
        s = posterior(rows, m.column, min_half=m.min_half)
        s = s[(s["n"] >= min_decisions) & s["grp"].isin(["CB", "FB", "MF", "AM", "FW"])].copy()
        lo, hi = s["shrunk"] - 1.2816 * s["sd"], s["shrunk"] + 1.2816 * s["sd"]
        pct = lambda v, g: (s.loc[s["grp"] == g, "shrunk"].values[None, :] < v[:, None]).mean(1)  # noqa: E731
        lo_p = np.concatenate([pct(lo[s["grp"] == g].values, g) for g in s["grp"].unique()])
        hi_p = np.concatenate([pct(hi[s["grp"] == g].values, g) for g in s["grp"].unique()])
        tiers_ = s["p_above_position"].map(tier)
        out[m.key] = {"players": int(len(s)),
                      "median_80pct_interval_in_percentiles": [round(float(np.median(lo_p)) * 100), round(float(np.median(hi_p)) * 100)],
                      "median_interval_width_percentile_points": round(float(np.median(hi_p - lo_p)) * 100),
                      "tiered_above": int((tiers_ == "above average").sum()), "tiered_below": int((tiers_ == "below average").sum()),
                      "shrinkage_B_median": round(float(s["B"].median()), 2)}
    return out


def team_axes(dec: pd.DataFrame, poss: pd.DataFrame) -> dict:
    rank = {mid: i for i, mid in enumerate(sorted(dec["match_id"].unique()))}
    h = [team_style(dec[dec["match_id"].map(rank) % 2 == k], poss[poss["match_id"].map(rank) % 2 == k]) for k in (0, 1)]
    both = h[0].index.intersection(h[1].index)
    out = {"teams_in_both_halves": int(len(both)), "teams_total": int(dec["team_shortname"].nunique())}
    for c in ["directness", "risk_axis", "long_share", "forwardness", "risk", "back_share", "provider", "progress_10s", "creator", "on_ball_gain"]:
        if c in h[0]:
            r = float(h[0].loc[both, c].corr(h[1].loc[both, c]))
            out[c] = {"r": round(r, 2), "95ci": _fisher_ci(r, len(both))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs")
    args = ap.parse_args()
    src = Path(args.results)
    o = pd.read_csv(src / "option_values.csv", low_memory=False,
                    usecols=["decision_id", "chosen", "P", "xthreat", "completion", "V", "V_reward", "V_cost", "phase"])
    d_all = pd.read_csv(src / "decisions.csv", low_memory=False, usecols=["decision_id", "phase"])
    dec = pd.read_parquet(src / "decisions_style.parquet")
    poss = pd.read_parquet(src / "possessions_style.parquet")
    ev = {
        "counts": {"decisions_all_phases": int(len(d_all)), "decisions_open_play": int(len(dec)),
                   "options": int(len(o)), "note": "option_value.py validates on all decisions; the fingerprint uses open play only (set plays and disruptions removed)."},
        "value_vs_baselines": value_vs_baselines(o, dec),
        "tiers": tiers(dec, poss),
        "player_intervals": provider_intervals(dec, poss),
        "team_axes": team_axes(dec, poss),
    }
    (src / "evidence.json").write_text(json.dumps(ev, indent=1, default=str))
    print(json.dumps(ev, indent=1, default=str))


if __name__ == "__main__":
    main()
