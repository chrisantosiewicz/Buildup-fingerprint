import numpy as np
import pandas as pd

from buildup.outcomes import add_outcomes
from buildup.turnovers import build_turnovers


def _poss(i, team, f0, f1, start, end, outcome=None, x0=0.0, x1=0.0, pa=False):
    return {
        "match_id": 1, "period": 1, "index": i, "event_id": f"e{i}", "event_type": "player_possession",
        "team_id": team, "team_shortname": f"T{team}", "player_id": i, "player_name": f"p{i}",
        "player_position": "CM", "frame_start": f0, "frame_end": f1, "minute_start": 1,
        "start_type": start, "end_type": end, "pass_outcome": outcome, "pass_range": None,
        "x_start": x0, "y_start": 0.0, "x_end": x1, "y_end": 0.0, "third_start": "middle_third",
        "third_end": "middle_third", "channel_start": "center", "channel_end": "center",
        "penalty_area_start": pa, "penalty_area_end": pa, "overall_pressure_start_id": 2,
        "overall_pressure_end_id": 3, "team_in_possession_phase_type": "create",
        "n_player_possessions_in_phase": 3, "game_state": "drawing", "team_score": 0,
        "opponent_team_score": 0, "possession_epv_against_end": 0.01, "possession_epv_for_start": 0.02,
        "possession_epv_for_end": 0.03, "n_opponents_ahead_start": 6, "last_defensive_line_height_start": 30.0,
        "last_defensive_line_x_start": 20.0, "inside_defensive_shape_start": False, "organised_defense": True,
        "n_passing_options_ahead": 1, "lead_to_goal": False, "pitch_length": 105, "pitch_width": 68,
    }


def _events():
    return pd.DataFrame([
        _poss(0, 10, 0, 10, "unknown", "pass", "unsuccessful", x0=-5, x1=0),       # A loses it
        _poss(1, 20, 15, 30, "pass_interception", "pass", "successful", x0=0, x1=10),  # B wins it: turnover
        _poss(2, 20, 35, 50, "pass_reception", "shot", None, x0=20, x1=40, pa=True),   # B shoots at 3.5 s
        _poss(3, 10, 200, 210, "goal_kick_reception", "pass", "unsuccessful"),       # A loses it ...
        _poss(4, 20, 215, 220, "throw_in_reception", "pass", "successful"),          # ... but to a restart: not a turnover
    ])


def test_only_open_play_regains_count():
    to = build_turnovers(_events())
    assert len(to) == 1
    row = to.iloc[0]
    assert row["losing_team_id"] == 10 and row["regaining_team_id"] == 20
    assert row["t0_frame"] == 15 and np.isclose(row["gap_s"], 0.5)
    assert row["lost_by"] == "failed_pass"


def test_outcomes_follow_the_regaining_chain():
    ev = _events()
    to = add_outcomes(build_turnovers(ev), ev, window_s=10)
    row = to.iloc[0]
    assert row["shot_10s"] and row["box_10s"] and row["final_third_10s"]
    assert row["danger_level"] == 3
    assert np.isclose(row["progress_m_10s"], 40.0)
