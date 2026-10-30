"""Rest-defence features from tracking, split into setup (before the loss) and
reaction (after the loss).

Everything is anchored to the **spot where the ball was lost** (ball position at
the loss frame), not to the moving ball. Anchoring to the moving ball leaks the
outcome: if the counter-attack carries the ball past defenders, a ball-relative
"players behind the ball" count drops *because* the attack is dangerous.
For the same reason the reaction window ends at the regain (t0), before the
danger outcome window starts.

All positions are in the losing team's frame (it attacks +x, its own goal is at
x = -pitch_length / 2), from MatchTracking.snapshot(). Goalkeepers are excluded.

Detection rule (see DECISIONS.md): counts and zones use every player, detected
or extrapolated; precise distances are only computed when the players involved
are detected, otherwise they are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import FPS
from .tracking import MatchTracking

# Players at least this far behind the ball form the "last line"; players
# goal-side but closer to the ball are the "screen" (coach language: 2+1, 3+2...).
LAST_LINE_GAP_M = 15.0
# Central lane: the width of the penalty box (40.32 m), where counters are most dangerous.
CENTRAL_HALF_WIDTH_M = 20.16
SETUP_OFFSET_S = 2.0      # setup is read 2 s before the loss
REACTION_MAX_S = 2.0      # reaction runs from the loss to the regain, capped at 2 s
REACTION_MIN_S = 0.3      # shorter windows are too short to measure movement
FAST_RECOVERY_MS = 4.0    # m/s back towards own goal counts as a recovery sprint


def _outfield(snap: pd.DataFrame, team: str) -> pd.DataFrame:
    return snap[(snap["team"] == team) & ~snap["is_gk"].fillna(False).astype(bool)]


def shape_label(n_last_line: int, n_screen: int) -> str:
    """Coach-language rest-defence shape, e.g. '3+1'. Capped at 4+3."""
    return f"{min(n_last_line, 4)}+{min(n_screen, 3)}"


def setup_features(snap: pd.DataFrame, anchor_x: float, pitch_length: float, prefix: str = "su_") -> dict:
    """Rest-defence shape relative to the loss spot `anchor_x`, at one frame."""
    own, opp = _outfield(snap, "own"), _outfield(snap, "opp")
    behind = own[own["x_rel"] < anchor_x]
    gap = anchor_x - behind["x_rel"]
    last_line = behind[gap >= LAST_LINE_GAP_M]
    screen = behind[gap < LAST_LINE_GAP_M]
    # Opponents already between the loss spot and our goal: who a counter would find.
    opp_beyond = opp[opp["x_rel"] < anchor_x]
    deepest = own.nsmallest(2, "x_rel")

    f = {
        "n_behind": len(behind),
        "n_last_line": len(last_line),
        "n_screen": len(screen),
        "shape": shape_label(len(last_line), len(screen)),
        "n_central_behind": int((behind["y_rel"].abs() < CENTRAL_HALF_WIDTH_M).sum()),
        "n_opp_beyond": len(opp_beyond),
        "balance": len(behind) - len(opp_beyond),
        "last_line_height": float(own["x_rel"].min() + pitch_length / 2) if len(own) else np.nan,
        "depth": float(anchor_x - own["x_rel"].min()) if len(own) else np.nan,
        "rest_width": float(behind["y_rel"].max() - behind["y_rel"].min()) if len(behind) >= 2 else np.nan,
        "behind_detected_share": float(behind["is_detected"].mean()) if len(behind) else np.nan,
        # Precise distance: only with detected players.
        "deepest_pair_spacing": (float(abs(deepest["y_rel"].iloc[0] - deepest["y_rel"].iloc[1]))
                                 if len(deepest) == 2 and deepest["is_detected"].all() else np.nan),
        # Space in behind: furthest-forward opponent vs our deepest player.
        "free_attacker_cushion": (float(opp_beyond["x_rel"].min() - own["x_rel"].min())
                                  if len(opp_beyond) and len(own) else np.nan),
    }
    return {prefix + k: v for k, v in f.items()}


def _near(snap: pd.DataFrame, x: float, y: float) -> tuple[float, int]:
    """Nearest own outfield player to (x, y) (NaN unless detected) and own players within 10 m."""
    own = _outfield(snap, "own")
    if not len(own) or pd.isna(x) or pd.isna(y):
        return np.nan, 0
    d = np.hypot(own["x_rel"] - x, own["y_rel"] - y)
    i = d.idxmin()
    return (float(d.min()) if bool(own.loc[i, "is_detected"]) else np.nan), int((d < 10).sum())


def reaction_features(at_loss: pd.DataFrame, at_end: pd.DataFrame, anchor_x: float, window_s: float) -> dict:
    """How the losing team moved between the loss and the end of the window.

    Players are matched by id. Recovery is measured on players caught upfield of
    the loss spot; pressure is measured around the ball at the end of the window
    (the ball winner, since the window ends at the regain).
    """
    own0 = _outfield(at_loss, "own").set_index("player_id")
    own1 = _outfield(at_end, "own").set_index("player_id")
    common = own0.index.intersection(own1.index)
    ahead = [p for p in common if own0.loc[p, "x_rel"] >= anchor_x]
    # Positive = moving back towards our own goal.
    rec = (own0.loc[ahead, "x_rel"] - own1.loc[ahead, "x_rel"]) / window_s if ahead else pd.Series(dtype=float)
    near0, n10_0 = _near(at_loss, at_loss["ball_x_rel"].iloc[0], at_loss["ball_y_rel"].iloc[0])
    near1, n10_1 = _near(at_end, at_end["ball_x_rel"].iloc[0], at_end["ball_y_rel"].iloc[0])
    return {
        "re_window_s": window_s,
        "re_n_caught_ahead": len(ahead),
        "re_recovery_speed_mean": float(rec.mean()) if len(rec) else np.nan,
        "re_n_recovering_fast": int((rec > FAST_RECOVERY_MS).sum()),
        "re_gain_behind": int((own1["x_rel"] < anchor_x).sum()) - int((own0["x_rel"] < anchor_x).sum()),
        "re_n_within10_at_loss": n10_0,
        "re_n_within10_at_regain": n10_1,
        "re_nearest_at_loss": near0,
        "re_nearest_at_regain": near1,
        "re_closing_speed": (near0 - near1) / window_s if pd.notna(near0) and pd.notna(near1) else np.nan,
    }


def turnover_features(mt: MatchTracking, turnovers: pd.DataFrame) -> pd.DataFrame:
    """Setup (loss - 2 s), shape at the loss, and reaction (loss -> regain) per turnover."""
    rows = []
    off = int(SETUP_OFFSET_S * FPS)
    for t in turnovers.itertuples(index=False):
        team, period, lf = t.losing_team_id, t.period, int(t.loss_frame)
        end = int(min(t.t0_frame, lf + REACTION_MAX_S * FPS))
        window_s = (end - lf) / FPS
        pre, at = mt.snapshot(lf - off, team, period), mt.snapshot(lf, team, period)
        aft = mt.snapshot(end, team, period)
        ok = all(s is not None for s in (pre, at, aft)) and pd.notna(at["ball_x_rel"].iloc[0]) if at is not None else False
        row = {"turnover_id": t.turnover_id, "tracking_ok": bool(ok)}
        if ok:
            anchor = float(at["ball_x_rel"].iloc[0])
            row["loss_spot_x_rel"] = anchor
            row.update(setup_features(pre, anchor, mt.pitch_length, "su_"))
            row.update(setup_features(at, anchor, mt.pitch_length, "at_"))
            if window_s >= REACTION_MIN_S:
                row.update(reaction_features(at, aft, anchor, window_s))
        rows.append(row)
    return pd.DataFrame(rows)
