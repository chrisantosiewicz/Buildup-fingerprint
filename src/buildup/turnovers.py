"""Open-play turnovers: one row per ball loss that the opponent wins in open play.

A turnover is a pair of consecutive player possessions (same match, same period):
  - the *losing* possession ends with a pass or a `possession_loss`
    (shots, clearances, fouls and offsides are excluded: different situations), and
  - the next possession belongs to the other team and starts with a `recovery`
    or `pass_interception` (open-play regains; restarts are excluded).
About 13% of these follow a pass SkillCorner marks `successful`: the receiver
lost it before a possession was logged (heavy touch, immediate tackle). They are
kept and flagged as lost_by = "lost_at_reception".

t0 is the regaining possession's `frame_start` (10 fps).
Coordinates follow SkillCorner event conventions: each row's x/y are oriented so
that the row's own team attacks towards +x. Loss columns are therefore in the
losing team's frame, regain columns in the regaining team's frame.
"""

from __future__ import annotations

import pandas as pd

from .io import FPS

OPEN_PLAY_REGAINS = ("recovery", "pass_interception")
MAX_GAP_S = 10.0  # loose balls can take a few seconds to be claimed

LOSS_COLS = {
    "event_id": "loss_event_id",
    "team_id": "losing_team_id",
    "team_shortname": "losing_team",
    "player_id": "loss_player_id",
    "player_name": "loss_player_name",
    "player_position": "loss_player_position",
    "frame_end": "loss_frame",
    "x_end": "loss_x",
    "y_end": "loss_y",
    "third_end": "loss_third",
    "channel_end": "loss_channel",
    "end_type": "loss_end_type",
    "pass_outcome": "loss_pass_outcome",
    "pass_range": "loss_pass_range",
    "overall_pressure_end_id": "loss_pressure_id",
    "team_in_possession_phase_type": "loss_phase_type",
    "n_player_possessions_in_phase": "loss_n_possessions_in_phase",
    "game_state": "loss_game_state",
    "team_score": "losing_team_score",
    "opponent_team_score": "regaining_team_score",
    "possession_epv_against_end": "loss_epv_against_end",
}

REGAIN_COLS = {
    "event_id": "regain_event_id",
    "team_id": "regaining_team_id",
    "team_shortname": "regaining_team",
    "player_id": "regain_player_id",
    "player_name": "regain_player_name",
    "frame_start": "t0_frame",
    "x_start": "regain_x",
    "y_start": "regain_y",
    "third_start": "regain_third",
    "channel_start": "regain_channel",
    "start_type": "regain_type",
    "overall_pressure_start_id": "regain_pressure_id",
    "possession_epv_for_start": "regain_epv_for_start",
    # Shape of the *losing* team at t0, seen from the ball winner.
    # "Opponents ahead" = losing-team players between the ball and their own goal.
    "n_opponents_ahead_start": "rd_n_goal_side",
    "last_defensive_line_height_start": "rd_last_line_height",
    "last_defensive_line_x_start": "rd_last_line_x",
    "inside_defensive_shape_start": "rd_ball_inside_shape",
    "organised_defense": "rd_organised",
    "n_passing_options_ahead": "regain_n_options_ahead",
}


def possessions(events: pd.DataFrame) -> pd.DataFrame:
    """Player possessions in time order within each match and period."""
    pp = events[events["event_type"] == "player_possession"].copy()
    return pp.sort_values(["match_id", "period", "frame_start", "index"]).reset_index(drop=True)


LOSS_END_TYPES = ("pass", "possession_loss")


def turnover_mask(pp: pd.DataFrame) -> pd.Series:
    """True where this possession is lost to an open-play regain by the opponent.

    `pp` must be in the order returned by `possessions()`.
    """
    nxt = pp.groupby(["match_id", "period"]).shift(-1)
    return (
        pp["end_type"].isin(LOSS_END_TYPES)
        & (pp["pass_outcome"].fillna("") != "offside")
        & nxt["team_id"].notna()
        & (nxt["team_id"] != pp["team_id"])
        & nxt["start_type"].isin(OPEN_PLAY_REGAINS)
        & ((nxt["frame_start"] - pp["frame_end"]) <= MAX_GAP_S * FPS)
    )


def build_turnovers(events: pd.DataFrame) -> pd.DataFrame:
    """One row per open-play turnover, with loss context, regain context and t0."""
    pp = possessions(events)
    nxt = pp.groupby(["match_id", "period"]).shift(-1)
    mask = turnover_mask(pp)

    loss = pp.loc[mask, ["match_id", "period", "minute_start", "pitch_length", "pitch_width", *LOSS_COLS]]
    loss = loss.rename(columns={**LOSS_COLS, "minute_start": "minute"})
    regain = nxt.loc[mask, list(REGAIN_COLS)].rename(columns=REGAIN_COLS)

    out = pd.concat([loss, regain], axis=1).reset_index(drop=True)
    out["gap_s"] = (out["t0_frame"] - out["loss_frame"]) / FPS
    out["score_diff_losing"] = out["losing_team_score"] - out["regaining_team_score"]
    out["lost_by"] = "dispossessed"
    out.loc[out["loss_end_type"] == "pass", "lost_by"] = "failed_pass"
    out.loc[(out["loss_end_type"] == "pass") & (out["loss_pass_outcome"] == "successful"), "lost_by"] = "lost_at_reception"
    # Regain x is in the regaining team's frame, where the goal they attack (the
    # losing team's goal) sits at x = +pitch_length / 2.
    out["regain_dist_to_losing_goal"] = out["pitch_length"] / 2 - out["regain_x"]
    out["turnover_id"] = out["match_id"].astype(str) + "_" + out["regain_event_id"].astype(str)
    return out
