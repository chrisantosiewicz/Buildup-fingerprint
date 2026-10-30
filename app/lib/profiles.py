"""Percentile profiles with honest ranges.

Each player's percentile within position is turned into an estimate and an 80% range using the metric's
reliability, adjusted for how many passes that player has (Kelley's true-score formula on the normal-score
scale, with Spearman-Brown for sample size):

    z      = Φ⁻¹(observed percentile)
    ρᵢ     = k·ρ / (1 + (k − 1)·ρ),   k = player's passes / typical player's passes
    true z ~ Normal(ρᵢ·z, ρᵢ·(1 − ρᵢ))

A reliable metric and a busy player give a narrow range near the observed percentile. An unreliable metric
pulls the estimate towards the 50th and widens the range, which is the honest picture with 20 matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

Z80 = norm.ppf(0.9)

# The coach's view: four questions about a passer, plus style dials.
GROUPS = {
    "Creation": ["provider", "creator"],
    "Decision": ["decision_quality"],
    "Execution": ["execution"],
    "Risk profile": ["forwardness", "risk"],
    "Style": ["long_share", "carry_share", "quick_release", "tempo", "back_under_pressure"],
}
EXPLAIN = {
    "provider": "How good a situation each pass hands the receiver (their next options), in goals.",
    "creator": "Value added on the ball: EPV gained plus passing options ahead gained, against similar possessions.",
    "decision_quality": "Where the chosen pass sits between the worst and best option available.",
    "execution": "Passes completed minus the completion the model expected, given the space around each option.",
    "forwardness": "How far forward the chosen pass goes compared with the average option available.",
    "risk": "How much harder (by SkillCorner xPass) the chosen pass is than the average option.",
    "long_share": "Share of passes longer than 30 m.",
    "carry_share": "Share of possessions where the player carries the ball.",
    "quick_release": "Share of possessions released within 1 second.",
    "tempo": "Time on the ball before releasing it, in seconds.",
    "back_under_pressure": "Share of passes under high pressure that go backwards.",
}


@dataclass(frozen=True)
class Estimate:
    observed: float     # observed percentile within position, 0..1
    estimate: float     # after allowing for reliability and sample size
    lo: float           # 80% range
    hi: float
    p_above: float      # chance the player is above the position's median
    reliability: float  # for this player's sample size


def player_reliability(rel: float, n: float, n_ref: float) -> float:
    k = max(n, 1) / max(n_ref, 1)
    rel = min(max(rel, 0.0), 0.99)
    return k * rel / (1 + (k - 1) * rel)


def estimate(pct: float, rel: float, n: float = 1, n_ref: float = 1) -> Estimate:
    if pd.isna(pct):
        return Estimate(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    r = player_reliability(rel, n, n_ref)
    z = norm.ppf(np.clip(pct, 0.01, 0.99))
    mu, sd = r * z, np.sqrt(max(r * (1 - r), 1e-9))
    return Estimate(float(pct), float(norm.cdf(mu)), float(norm.cdf(mu - Z80 * sd)), float(norm.cdf(mu + Z80 * sd)),
                    float(norm.cdf(mu / sd)), r)


def tier(e: Estimate, threshold: float = 0.8) -> str:
    if pd.isna(e.p_above):
        return "n/a"
    return "above average" if e.p_above >= threshold else "below average" if e.p_above <= 1 - threshold else "around average"


def within_position_pct(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values.groupby(groups).rank(pct=True)


def split_half_reliability(df: pd.DataFrame, a: str, b: str, group: str, min_n: int = 15,
                           na: str = "passes_odd", nb: str = "passes_even") -> tuple[float, int]:
    """Within-position split-half r projected to the full sample, the same check as RANKING_SYSTEM.md §1."""
    d = df[(df[na] >= min_n) & (df[nb] >= min_n)].dropna(subset=[a, b])
    if len(d) < 10:
        return np.nan, len(d)
    da = d[a] - d.groupby(group)[a].transform("mean")
    db = d[b] - d.groupby(group)[b].transform("mean")
    r = float(np.corrcoef(da, db)[0, 1])
    return (2 * r / (1 + r) if r > 0 else r), len(d)


RISK_QUADRANTS = {  # (forward, risky) -> name
    (True, True): "Bold progressor", (True, False): "Efficient progressor",
    (False, True): "Loose recycler", (False, False): "Safe recycler",
}


def risk_quadrant(forward: Estimate, risk: Estimate, sure: float = 0.8) -> tuple[str, str]:
    """Name the quadrant and say how sure we are of it."""
    if pd.isna(forward.estimate) or pd.isna(risk.estimate):
        return "n/a", ""
    name = RISK_QUADRANTS[(forward.estimate >= 0.5, risk.estimate >= 0.5)]
    pf = forward.p_above if forward.estimate >= 0.5 else 1 - forward.p_above
    pr = risk.p_above if risk.estimate >= 0.5 else 1 - risk.p_above
    p = pf * pr
    return name, "clear" if p >= sure else "leaning" if p >= 0.5 else "unclear"
