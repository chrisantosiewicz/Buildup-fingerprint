"""Camera-coverage audit: are the losing team's rest defenders seen around a turnover?

For each turnover, look at the losing team's outfield players at several moments
relative to the loss, and report how many are detected (on camera) vs extrapolated.

Usage:
    python scripts/coverage_audit.py --data path/to/opendata/data --out outputs [--matches 1925299 1996435]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buildup.io import load_events
from buildup.tracking import MatchTracking, tracking_path
from buildup.turnovers import build_turnovers

# Offsets in frames (10 fps) relative to the loss frame; "t0" is the regain.
OFFSETS = {"loss-3s": -30, "loss-2s": -20, "loss": 0}


def audit_match(data_dir, match_id) -> pd.DataFrame:
    ev = load_events(data_dir, match_id)
    to = build_turnovers(ev)
    mt = MatchTracking(data_dir, match_id)
    rows = []
    for t in to.itertuples(index=False):
        moments = {k: t.loss_frame + v for k, v in OFFSETS.items()}
        moments["t0"] = t.t0_frame
        moments["t0+2s"] = t.t0_frame + 20
        for name, fr in moments.items():
            snap = mt.snapshot(int(fr), t.losing_team_id, t.period)
            if snap is None:
                rows.append({"turnover_id": t.turnover_id, "moment": name, "has_frame": False})
                continue
            own = snap[(snap["team"] == "own") & (~snap["is_gk"].fillna(False).astype(bool))]
            gs = own[own["goal_side"]]
            deepest4 = own.nsmallest(4, "x_rel")
            rows.append({
                "turnover_id": t.turnover_id, "match_id": match_id, "moment": name, "has_frame": True,
                "loss_third": t.loss_third,
                "n_outfield": len(own),
                "outfield_detected": own["is_detected"].mean(),
                "n_goal_side": len(gs),
                "goal_side_detected": gs["is_detected"].mean() if len(gs) else np.nan,
                "deepest4_detected": deepest4["is_detected"].mean() if len(deepest4) else np.nan,
                "deepest4_all_detected": bool(deepest4["is_detected"].all()) if len(deepest4) else False,
                "ball_detected": snap["ball_detected"].iloc[0] == True,  # noqa: E712 (values are True/False/None)
                "ball_position_known": bool(pd.notna(snap["ball_x_rel"].iloc[0])),
                "event_n_goal_side": t.rd_n_goal_side,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--matches", nargs="*", type=int)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ids = args.matches or [int(p.name) for p in (Path(args.data) / "matches").iterdir()
                           if tracking_path(args.data, int(p.name)).stat().st_size > 1000]
    df = pd.concat([audit_match(args.data, m) for m in ids], ignore_index=True)
    df.to_csv(out / "coverage_audit.csv", index=False)
    ok = df[df["has_frame"]]
    order = ["loss-3s", "loss-2s", "loss", "t0", "t0+2s"]
    summ = ok.groupby("moment")[["outfield_detected", "goal_side_detected", "deepest4_detected",
                                  "deepest4_all_detected", "ball_detected", "ball_position_known"]].mean().reindex(order).round(3)
    by_third = ok[ok["moment"] == "loss-2s"].groupby("loss_third")[["goal_side_detected", "deepest4_all_detected"]].mean().round(3)
    t0 = ok[ok["moment"] == "t0"]
    agree = {
        "corr_tracking_vs_event_goal_side_at_t0": round(t0["n_goal_side"].corr(t0["event_n_goal_side"]), 3),
        "mean_abs_diff": round((t0["n_goal_side"] - t0["event_n_goal_side"]).abs().mean(), 2),
    }
    res = {"matches": ids, "n_turnovers": int(df["turnover_id"].nunique()),
           "missing_frames": int((~df["has_frame"]).sum()),
           "by_moment": summ.to_dict(orient="index"),
           "loss-2s_by_loss_third": by_third.to_dict(orient="index"),
           "tracking_vs_events": agree}
    with open(out / "coverage_audit.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
