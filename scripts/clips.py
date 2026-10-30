"""Moments worth showing: creators, missed options, and back passes that were right under pressure.

Usage:
    python scripts/clips.py --results outputs --out outputs [--n 15]

Writes clips.csv with match, period, minute:second and a one-line description per moment,
so they can be found in match footage or replayed in the Build-Up Explorer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clock(df: pd.DataFrame) -> pd.Series:
    return df["minute_start"].astype(int).astype(str) + ":" + df["second_start"].astype(int).astype(str).str.zfill(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    src, out = Path(args.results), Path(args.out)
    dec = pd.read_parquet(src / "decisions_style.parquet")
    poss = pd.read_parquet(src / "possessions_style.parquet")
    opts = pd.read_csv(src / "option_values.csv", low_memory=False,
                       usecols=["decision_id", "player_name", "V", "chosen"])
    best = opts.loc[opts.groupby("decision_id")["V"].idxmax(), ["decision_id", "player_name", "V"]]
    best = best.rename(columns={"player_name": "best_option", "V": "V_best_option"})
    dec = dec.merge(best, on="decision_id", how="left")
    n = args.n
    clips = []

    # 1. Creators: the biggest improvements in a player's own options before releasing the ball.
    c = poss[poss["on_ball_value_created"].notna()].nlargest(n, "on_ball_value_created")
    clips.append(pd.DataFrame({
        "kind": "creator", "match_id": c["match_id"], "period": c["period"], "clock": clock(c),
        "team": c["team_shortname"], "player": c["player"],
        "what": [f"{'carried' if k == 1 else 'held'} the ball: best option {a * 1000:.1f} -> {b * 1000:.1f} mG"
                 for k, a, b in zip(c["carry"], c["best_ev_start"], c["best_ev_pass"])],
        "value_mG": (c["on_ball_value_created"] * 1000).round(2)}))

    # 2. Missed options: the largest gaps between the pass played and the best option.
    m = dec.nsmallest(n, "decision_gap")
    clips.append(pd.DataFrame({
        "kind": "missed_option", "match_id": m["match_id"], "period": m["period"], "clock": clock(m),
        "team": m["team_shortname"], "player": m["player"],
        "what": "passed to " + m["player_name"] + " instead of " + m["best_option"],
        "value_mG": (m["decision_gap"] * 1000).round(2)}))

    # 3. Right back pass under high pressure: back passes that were the best option by the widest margin.
    opts_v = opts.groupby("decision_id")["V"]
    second = opts_v.apply(lambda v: v.nlargest(2).iloc[-1] if len(v) > 1 else v.iloc[0]).rename("V_second")
    b = dec[dec["high_pressure"] & (dec["back"] == 1) & dec["picked_best"]].merge(second, left_on="decision_id", right_index=True)
    b["margin"] = b["V"] - b["V_second"]
    b = b.nlargest(n, "margin")
    clips.append(pd.DataFrame({
        "kind": "right_back_pass", "match_id": b["match_id"], "period": b["period"], "clock": clock(b),
        "team": b["team_shortname"], "player": b["player"],
        "what": "under high pressure, went back to " + b["player_name"] + ", the best option by a clear margin",
        "value_mG": (b["margin"] * 1000).round(2)}))

    res = pd.concat(clips, ignore_index=True)
    res.to_csv(out / "clips.csv", index=False)
    print(res.groupby("kind").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
