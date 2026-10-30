"""Passing options: the decision table and the receiver's next situation.

SkillCorner logs, for every player possession, the teammates who were passing
options (event_type == "passing_option"). Checked on this data:
- `frame_end` of an option equals the pass frame (median 0 frames) for both the
  chosen option and the others alive at the pass, so the `*_end` columns are
  measured at the pass moment and are fair to compare across options.
- `passing_option_at_pass_moment` marks options available when the pass was
  played; `passing_option_at_player_possession_start` marks those available when
  the possession began (used for the receiver's situation on reception).
- Option x/y follow the event convention: the attacking team attacks +x.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import FPS
from .turnovers import possessions, turnover_mask

# Features known at the pass moment for every option (chosen or not).
OPTION_NUMERIC = [
    "opt_x", "opt_abs_y", "passer_x", "passer_abs_y", "pass_dx", "pass_len",
    "distance_to_player_in_possession_end", "separation_end", "delta_to_last_defensive_line_end",
    "last_defensive_line_height_end", "n_opponents_ahead_pass_reception", "n_opponents_bypassed",
    "n_simultaneous_passing_options", "passer_pressure_id", "xpass_completion", "xthreat",
]
OPTION_CATEGORICAL = ["inside_defensive_shape_end", "pass_ahead", "location_to_player_in_possession_end",
                      "phase", "trajectory_direction"]


def option_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per option in each passing decision (alive at the pass, or chosen)."""
    po = events[events["event_type"] == "passing_option"].copy()
    pp = possessions(events)
    pp["lost"] = turnover_mask(pp)
    # Option rows share many column names with possession rows, so prefix everything we take.
    poss_cols = pp[["match_id", "event_id", "player_id", "frame_start", "frame_end", "end_type",
                    "pass_outcome", "overall_pressure_end_id", "team_in_possession_phase_type", "lost"]].rename(columns={
        "event_id": "poss_event_id", "player_id": "passer_id", "frame_start": "poss_frame_start",
        "frame_end": "pass_frame", "end_type": "poss_end_type", "pass_outcome": "poss_pass_outcome",
        "overall_pressure_end_id": "passer_pressure_id", "team_in_possession_phase_type": "phase",
        "lost": "poss_lost"})
    po = po.merge(poss_cols, left_on=["match_id", "associated_player_possession_event_id"],
                  right_on=["match_id", "poss_event_id"], how="inner")
    # Decision set: options alive at the pass, plus the chosen one; only possessions ending in a pass.
    po = po[(po["poss_end_type"] == "pass") & ((po["passing_option_at_pass_moment"] == True) | (po["targeted"] == True))]  # noqa: E712
    po = po.copy()
    po["opt_x"], po["opt_y"] = po["x_end"], po["y_end"]
    po["opt_abs_y"] = po["opt_y"].abs()
    po["passer_x"], po["passer_y"] = po["player_in_possession_x_end"], po["player_in_possession_y_end"]
    po["passer_abs_y"] = po["passer_y"].abs()
    po["pass_dx"] = po["opt_x"] - po["passer_x"]
    po["pass_len"] = np.hypot(po["pass_dx"], po["opt_y"] - po["passer_y"])
    po["option_ev"] = po["xpass_completion"] * po["xthreat"]
    po["chosen"] = po["targeted"] == True  # noqa: E712
    po["decision_id"] = po["match_id"].astype(str) + "_" + po["poss_event_id"].astype(str)
    for c in OPTION_CATEGORICAL:
        po[c] = po[c].astype(str)
    # Keep decisions with exactly one chosen option (the pass that was played) and complete
    # SkillCorner values on every option (one option in this release has no xPass / xThreat).
    n_chosen = po.groupby("decision_id")["chosen"].transform("sum")
    complete = po[["xpass_completion", "xthreat"]].notna().all(axis=1).groupby(po["decision_id"]).transform("all")
    return po[(n_chosen == 1) & complete].reset_index(drop=True)


def _chain_progress(pp_g: pd.DataFrame, i0: int, window_frames: int) -> dict:
    """From possession i0, follow the same team's chain for `window_frames`."""
    team, f0 = pp_g.at[i0, "team_id"], pp_g.at[i0, "frame_start"]
    third0 = pp_g.at[i0, "third_id_start"]
    progressed = kept = False
    for i in range(i0, len(pp_g)):
        r = pp_g.iloc[i]
        if r["team_id"] != team:
            break
        if r["frame_start"] > f0 + window_frames:
            kept = True
            break
        if (r["third_id_end"] > third0) or (r["third_id_start"] > third0) or r["end_type"] == "shot" \
                or bool(r["penalty_area_end"]):
            progressed = True
        if r["lost"]:
            break
    else:
        kept = True
    return {"prog_10s": progressed, "kept_10s": kept}


def receiver_targets(events: pd.DataFrame, opts: pd.DataFrame, window_s: float = 10.0) -> pd.DataFrame:
    """For each chosen, completed option: the receiver's situation on reception.

    Targets:
      next_best_ev  best xPass x xThreat among the receiver's options at the start
                    of his possession (0 if he had none: a dead end)
      next_exp_ev   options weighted by SkillCorner's passing_option_score
      next_n_ahead  forward options available on reception
      prog_10s      team reaches a higher third / box / shot within 10 s of the
                    reception without losing the ball
      kept_10s      team still has the ball 10 s after the reception
    """
    pp = possessions(events)
    pp["lost"] = turnover_mask(pp)
    nxt = pp.groupby(["match_id", "period"]).shift(-1)
    pp["next_event_id"], pp["next_player_id"], pp["next_team_id"] = nxt["event_id"], nxt["player_id"], nxt["team_id"]

    ch = opts[opts["chosen"] & (opts["received"] == True)][["decision_id", "match_id", "poss_event_id", "player_id"]]  # noqa: E712
    ch = ch.merge(pp[["match_id", "event_id", "next_event_id", "next_player_id"]],
                  left_on=["match_id", "poss_event_id"], right_on=["match_id", "event_id"])
    ch = ch[ch["next_player_id"] == ch["player_id"]].drop(columns=["event_id", "next_player_id"])

    po = events[(events["event_type"] == "passing_option")
                & (events["passing_option_at_player_possession_start"] == True)].copy()  # noqa: E712
    po["ev"] = po["xpass_completion"] * po["xthreat"]
    w = po["passing_option_score"].clip(lower=0)
    po["w_ev"] = po["ev"] * w
    po["w"] = w
    agg = po.groupby(["match_id", "associated_player_possession_event_id"]).agg(
        next_best_ev=("ev", "max"), w_ev=("w_ev", "sum"), w=("w", "sum"),
        next_n_ahead=("pass_ahead", "sum"))
    agg["next_exp_ev"] = agg["w_ev"] / agg["w"].replace(0, np.nan)
    agg = agg.drop(columns=["w_ev", "w"]).reset_index().rename(
        columns={"associated_player_possession_event_id": "next_event_id"})
    ch = ch.merge(agg, on=["match_id", "next_event_id"], how="left")
    ch[["next_best_ev", "next_exp_ev", "next_n_ahead"]] = ch[["next_best_ev", "next_exp_ev", "next_n_ahead"]].fillna(0.0)

    groups = {k: g.reset_index(drop=True) for k, g in pp.groupby(["match_id", "period"])}
    loc = {(m, e): (k, i) for k, g in groups.items() for i, (m, e) in enumerate(zip(g["match_id"], g["event_id"]))}
    prog = []
    for r in ch.itertuples(index=False):
        k, i = loc[(r.match_id, r.next_event_id)]
        prog.append(_chain_progress(groups[k], i, int(window_s * FPS)))
    ch = pd.concat([ch.reset_index(drop=True), pd.DataFrame(prog)], axis=1)
    return ch


def possession_option_value(events: pd.DataFrame) -> pd.DataFrame:
    """Best option value at the start vs at the pass moment for each possession (for creators).

    on_ball_value_created = best EV at pass moment - best EV at possession start.
    """
    po = events[events["event_type"] == "passing_option"].copy()
    po["ev"] = po["xpass_completion"] * po["xthreat"]
    key = ["match_id", "associated_player_possession_event_id"]
    start = po[po["passing_option_at_player_possession_start"] == True].groupby(key)["ev"].max().rename("best_ev_start")  # noqa: E712
    end = po[po["passing_option_at_pass_moment"] == True].groupby(key)["ev"].max().rename("best_ev_pass")  # noqa: E712
    out = pd.concat([start, end], axis=1).fillna(0.0).reset_index().rename(
        columns={"associated_player_possession_event_id": "poss_event_id"})
    out["on_ball_value_created"] = out["best_ev_pass"] - out["best_ev_start"]
    return out
