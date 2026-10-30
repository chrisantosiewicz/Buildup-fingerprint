"""SkillCorner broadcast tracking: loading and team-relative snapshots.

Tracking quirks (unlike the dynamic events):
- Coordinates are absolute pitch coordinates in metres, origin at the centre spot,
  x along the length. They are NOT flipped per team: use `home_team_side` in
  match.json (one entry per period, "left_to_right" / "right_to_left") to know
  which way each team attacks.
- `player_id` in tracking = `players[].id` in match.json (not `trackable_object`).
- `is_detected` is False when the player is off-camera and his position is
  extrapolated. Broadcast cameras follow the ball, so deep players are often
  extrapolated: always check detection before trusting a distance.
- Frames are 10 fps and share their numbering with the dynamic events.
- The first frames of a file are empty (period = null) before kick-off.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_match_meta


def tracking_path(data_dir: str | Path, match_id: int) -> Path:
    return Path(data_dir) / "matches" / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl"


def load_tracking(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a tracking .jsonl file.

    Returns (players, frames):
      players: one row per (frame, player) with x, y, is_detected
      frames:  one row per frame with period, ball x/y/z, ball_detected, possession group
    """
    p_rows, f_rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["period"] is None:
                continue
            fr = d["frame"]
            b = d["ball_data"]
            f_rows.append((fr, d["period"], b["x"], b["y"], b["z"], b["is_detected"],
                           d["possession"]["group"], len(d["player_data"])))
            for pl in d["player_data"]:
                p_rows.append((fr, pl["player_id"], pl["x"], pl["y"], pl["is_detected"]))
    frames = pd.DataFrame(f_rows, columns=["frame", "period", "ball_x", "ball_y", "ball_z",
                                           "ball_detected", "possession_group", "n_players"])
    players = pd.DataFrame(p_rows, columns=["frame", "player_id", "x", "y", "is_detected"])
    return players, frames


def player_table(meta: dict) -> pd.DataFrame:
    """Player id -> team, role, goalkeeper flag."""
    rows = [{
        "player_id": p["id"], "team_id": p["team_id"], "number": p["number"],
        "short_name": p["short_name"], "role": p["player_role"]["acronym"],
        "position_group": p["player_role"]["position_group"],
        "is_gk": p["player_role"]["acronym"] == "GK",
    } for p in meta["players"]]
    return pd.DataFrame(rows)


def attack_sign(meta: dict, team_id: int, period: int) -> int:
    """+1 if the team attacks towards +x in this period, else -1."""
    home_side = meta["home_team_side"][period - 1]
    home_sign = 1 if home_side == "left_to_right" else -1
    return home_sign if team_id == meta["home_team"]["id"] else -home_sign


class MatchTracking:
    """Tracking for one match, indexed for fast frame lookups."""

    def __init__(self, data_dir: str | Path, match_id: int):
        self.match_id = match_id
        self.meta = load_match_meta(data_dir, match_id)
        players, frames = load_tracking(tracking_path(data_dir, match_id))
        self.players = players.merge(player_table(self.meta), on="player_id", how="left")
        self.frames = frames.set_index("frame")
        self._by_frame = {fr: g for fr, g in self.players.groupby("frame")}
        self.pitch_length = self.meta["pitch_length"]

    def snapshot(self, frame: int, team_id: int, period: int) -> pd.DataFrame | None:
        """Players at `frame` in the frame of reference of `team_id`.

        Returns x_rel / y_rel with `team_id` attacking towards +x (its own goal at
        x = -pitch_length/2), plus ball_x_rel / ball_y_rel, a `team` column ("own" / "opp") and
        `goal_side` (own player between the ball and own goal).
        """
        if frame not in self._by_frame or frame not in self.frames.index:
            return None
        s = attack_sign(self.meta, team_id, period)
        snap = self._by_frame[frame].copy()
        fr = self.frames.loc[frame]
        snap["x_rel"] = snap["x"] * s
        snap["y_rel"] = snap["y"] * s
        snap["ball_x_rel"] = fr["ball_x"] * s if pd.notna(fr["ball_x"]) else np.nan
        snap["ball_y_rel"] = fr["ball_y"] * s if pd.notna(fr["ball_y"]) else np.nan
        snap["ball_detected"] = fr["ball_detected"]
        snap["team"] = np.where(snap["team_id"] == team_id, "own", "opp")
        snap["goal_side"] = (snap["team"] == "own") & (snap["x_rel"] < snap["ball_x_rel"])
        return snap
