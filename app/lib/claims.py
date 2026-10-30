"""What the app is allowed to claim about each metric, read from its measured reliability.

Reliability is the within-position split-half correlation projected to all 20 matches
(metric_reliability.csv, RANKING_SYSTEM.md). The thresholds are the project's rule:

    >= 0.5      rank         sortable leaderboard, exact percentile
    0.3 to 0.5  tier         percentile with a range, tiers (top / middle / bottom third), no leaderboard
    < 0.3       low          shown per player only with a wide range and a "low confidence" label;
                             the claim is made at team level and with clips

Because the level is read from the data, a metric that gets more (or less) reliable when the models
change moves level by itself. tests/test_app.py pins today's levels so such a move is noticed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RANK_MIN, TIER_MIN = 0.5, 0.3
LEVEL_TEXT = {
    "rank": "Reliable enough to rank players",
    "tier": "Reliable enough for tiers, not exact ranks",
    "low": "Not reliable per player over 20 matches: read it at team level and in clips",
}


@dataclass(frozen=True)
class Claim:
    metric: str
    label: str
    kind: str
    reliability: float
    level: str
    team_r: float | None = None

    @property
    def can_rank(self) -> bool:
        return self.level == "rank"

    @property
    def text(self) -> str:
        return LEVEL_TEXT[self.level]


def level(reliability: float) -> str:
    if pd.isna(reliability):
        return "low"
    return "rank" if reliability >= RANK_MIN else "tier" if reliability >= TIER_MIN else "low"


def claims(rel: pd.DataFrame) -> dict[str, Claim]:
    return {r.metric: Claim(r.metric, r.label, r.kind, float(r.player_reliability), level(r.player_reliability),
                            None if pd.isna(getattr(r, "team_r", None)) else float(r.team_r))
            for r in rel.itertuples()}


def add(cl: dict[str, Claim], metric: str, label: str, kind: str, reliability: float) -> dict[str, Claim]:
    """Register a metric the app computes itself (execution), under the same rule."""
    return {**cl, metric: Claim(metric, label, kind, reliability, level(reliability))}
