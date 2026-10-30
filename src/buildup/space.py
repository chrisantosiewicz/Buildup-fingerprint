"""Space around each passing option at the pass moment, from tracking.

For every option (chosen or not) the frame is just before the pass frame of the decision.
All geometry is in the passing team's frame (attacking +x).

Checked on 6,477 chosen passes (8 matches): the nearest-opponent distance separates completed
from failed passes with AUC 0.75 two seconds before the pass, 0.80 at -0.5 s and 0.81 at the
pass frame, rising smoothly, so the signal is the situation before the pass rather than
the interception itself. SkillCorner's event `separation_end` gives 0.72.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tracking import MatchTracking

SPACE_NUMERIC = ["trk_opp_5m", "trk_opp_10m", "trk_nearest_opp", "trk_nearest_closing", "trk_opp_cone",
                 "trk_receiver_vx", "trk_lane_min", "trk_lane_n2m"]
LOOKBACK = 10   # frames (1 s) for velocities
# Measured 0.3 s before the logged pass frame, so a few frames of event/tracking misalignment
# can't let the ball's flight (or the interception) leak into the features.
FRAME_OFFSET = -3


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from points (px, py) to the segment a-b."""
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / L2 if L2 > 0 else 0.0, 0, 1)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def option_space(mt: MatchTracking, opts: pd.DataFrame, frame_offset: int = FRAME_OFFSET) -> pd.DataFrame:
    """Space features for the options of one match. `opts` needs decision_id, pass_frame,
    period, team_id, player_id (the option) and passer_id."""
    rows = []
    for (did, frame, period, team), g in opts.groupby(["decision_id", "pass_frame", "period", "team_id"]):
        f0 = int(frame) + frame_offset
        now = mt.snapshot(f0, int(team), int(period))
        before = mt.snapshot(f0 - LOOKBACK, int(team), int(period))
        if now is None:
            continue
        now = now.set_index("player_id")
        opp = now[now["team"] == "opp"]
        prev = before.set_index("player_id") if before is not None else None
        passer = int(g["passer_id"].iloc[0])
        if passer not in now.index:
            continue
        ax, ay = now.at[passer, "x_rel"], now.at[passer, "y_rel"]
        for _, o in g.iterrows():
            pid = int(o["player_id"])
            if pid not in now.index:
                continue
            rx, ry = now.at[pid, "x_rel"], now.at[pid, "y_rel"]
            d = np.hypot(opp["x_rel"] - rx, opp["y_rel"] - ry)
            near = d.idxmin() if len(d) else None
            closing = np.nan
            vx = np.nan
            if prev is not None:
                if near in prev.index:
                    d_prev = np.hypot(prev.at[near, "x_rel"] - prev.at[pid, "x_rel"],
                                      prev.at[near, "y_rel"] - prev.at[pid, "y_rel"]) if pid in prev.index else np.nan
                    closing = (d_prev - d[near]) / (LOOKBACK / 10)       # m/s, + = defender closing in
                if pid in prev.index:
                    vx = (rx - prev.at[pid, "x_rel"]) / (LOOKBACK / 10)
            dx, dy = opp["x_rel"] - rx, opp["y_rel"] - ry
            cone = ((dx > 0) & (np.hypot(dx, dy) < 15) & (np.abs(dy) <= dx)).sum()
            lane = _seg_dist(opp["x_rel"].to_numpy(), opp["y_rel"].to_numpy(), ax, ay, rx, ry)
            rows.append({"decision_id": did, "player_id": pid,
                         "trk_opp_5m": int((d < 5).sum()), "trk_opp_10m": int((d < 10).sum()),
                         "trk_nearest_opp": float(d.min()) if len(d) else np.nan,
                         "trk_nearest_closing": closing, "trk_opp_cone": int(cone), "trk_receiver_vx": vx,
                         "trk_lane_min": float(lane.min()) if len(lane) else np.nan,
                         "trk_lane_n2m": int((lane < 2).sum()),
                         "trk_receiver_detected": bool(now.at[pid, "is_detected"])})
    return pd.DataFrame(rows)
