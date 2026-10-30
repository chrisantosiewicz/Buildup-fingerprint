"""Loading SkillCorner open data (github.com/SkillCorner/opendata).

`data_dir` is the repo's `data/` folder. Layout:
    data/matches.json
    data/matches/{match_id}/{match_id}_dynamic_events.csv
    data/matches/{match_id}/{match_id}_match.json
    data/matches/{match_id}/{match_id}_phases_of_play.csv
    data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl   (git LFS)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Tracking and Game Intelligence frames are both at 10 fps.
FPS = 10


def list_match_ids(data_dir: str | Path) -> list[int]:
    """Match ids that have a folder under data/matches."""
    root = Path(data_dir) / "matches"
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def load_match_meta(data_dir: str | Path, match_id: int) -> dict:
    """Lineups, player roles, pitch size for one match."""
    path = Path(data_dir) / "matches" / str(match_id) / f"{match_id}_match.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_events(data_dir: str | Path, match_id: int) -> pd.DataFrame:
    """Dynamic events for one match.

    SkillCorner quirks:
    - `event_id` is unique within a match only, so key on (match_id, event_id).
    - Event x/y are in metres, origin at the centre spot, and are oriented so that
      the team of the event's `team_id` attacks towards +x (checked: attacking_third
      rows have x > ~17.5, penalty_area rows have x > ~35).
    - x/y are NOT rescaled to a standard pitch; pitch length is 104-106 m here.
    """
    path = Path(data_dir) / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv"
    # read_csv on this ~300-column file returns a fragmented frame; copy() consolidates it.
    ev = pd.read_csv(path, low_memory=False).copy()
    meta = load_match_meta(data_dir, match_id)
    return ev.assign(pitch_length=meta["pitch_length"], pitch_width=meta["pitch_width"])


def load_all_events(data_dir: str | Path, match_ids: list[int] | None = None) -> pd.DataFrame:
    """Dynamic events for every match (or the given ones), concatenated."""
    ids = match_ids if match_ids is not None else list_match_ids(data_dir)
    return pd.concat([load_events(data_dir, m) for m in ids], ignore_index=True)
