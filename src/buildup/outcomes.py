"""Model-free danger outcomes after a turnover.

The outcome window starts at t0 (the regain) and follows the regaining team's
possession chain: consecutive player possessions by the regaining team, stopping
at the first possession by the other team or at the end of the window.

Outcomes are deliberately model-free (box entry, shot, metres gained) so that
they are not built from the same defender positions we later use as predictors.
SkillCorner EPV is kept as a secondary outcome only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import FPS
from .turnovers import possessions

DANGER_LEVELS = {0: "contained", 1: "final third", 2: "box entry", 3: "shot"}


def _chain_after(pp_period: pd.DataFrame, start_idx: int, team_id, end_frame: int) -> pd.DataFrame:
    """Regaining team's consecutive possessions from start_idx up to end_frame."""
    rows = []
    for i in range(start_idx, len(pp_period)):
        r = pp_period.iloc[i]
        if r["team_id"] != team_id or r["frame_start"] > end_frame:
            break
        rows.append(i)
    return pp_period.iloc[rows]


def add_outcomes(turnovers: pd.DataFrame, events: pd.DataFrame, window_s: float = 10.0,
                 shot_window_s: float = 15.0) -> pd.DataFrame:
    """Attach danger outcomes for each turnover.

    Columns added (all for the regaining team, after t0):
      final_third_{w}s   reached the attacking third (x > pitch_length / 6)
      box_{w}s           a possession started or ended in the penalty area
      shot_{w}s          a possession ended in a shot within the window
      shot_{sw}s         same, with the longer shot window
      goal_{sw}s         that possession led to a goal (SkillCorner `lead_to_goal`)
      progress_m_{w}s    max metres gained towards goal vs the regain spot (>= 0)
      epv_gain_{w}s      max EPV reached minus EPV at the regain (secondary)
      chain_s            how long the regaining team kept the ball (capped at the window)
      danger_level       0 contained / 1 final third / 2 box entry / 3 shot (within window)
    """
    pp = possessions(events)
    w, sw = int(window_s), int(shot_window_s)
    by_period = {k: g.reset_index(drop=True) for k, g in pp.groupby(["match_id", "period"])}
    pos_lookup = {
        k: dict(zip(g["event_id"], range(len(g)))) for k, g in by_period.items()
    }

    records = []
    for t in turnovers.itertuples(index=False):
        key = (t.match_id, t.period)
        g = by_period[key]
        i0 = pos_lookup[key][t.regain_event_id]
        third_x = t.pitch_length / 6  # attacking third starts 1/6 of the length past halfway

        chain_long = _chain_after(g, i0, t.regaining_team_id, t.t0_frame + sw * FPS)
        chain = chain_long[chain_long["frame_start"] <= t.t0_frame + w * FPS]

        x_reach = np.nanmax(np.r_[chain["x_start"].to_numpy(), chain["x_end"].to_numpy()])
        box = bool(chain["penalty_area_start"].fillna(False).any() or chain["penalty_area_end"].fillna(False).any())
        shot_w = bool((chain["end_type"] == "shot").any())
        shots_long = chain_long[chain_long["end_type"] == "shot"]
        epv_max = np.nanmax(np.r_[chain["possession_epv_for_end"].to_numpy(dtype=float), np.nan]) \
            if chain["possession_epv_for_end"].notna().any() else np.nan

        chain_end = chain_long["frame_end"].max()
        records.append({
            "turnover_id": t.turnover_id,
            f"final_third_{w}s": bool(x_reach > third_x),
            f"box_{w}s": box,
            f"shot_{w}s": shot_w,
            f"shot_{sw}s": bool(len(shots_long)),
            f"goal_{sw}s": bool(shots_long["lead_to_goal"].fillna(False).any()) if len(shots_long) else False,
            f"progress_m_{w}s": max(0.0, float(x_reach - t.regain_x)),
            f"epv_gain_{w}s": epv_max - t.regain_epv_for_start if pd.notna(t.regain_epv_for_start) else np.nan,
            "chain_s": min(window_s, (chain_end - t.t0_frame) / FPS),
            "chain_n_possessions": len(chain),
        })

    out = turnovers.merge(pd.DataFrame(records), on="turnover_id", how="left")
    out["danger_level"] = np.select(
        [out[f"shot_{w}s"], out[f"box_{w}s"], out[f"final_third_{w}s"]], [3, 2, 1], default=0
    )
    return out
