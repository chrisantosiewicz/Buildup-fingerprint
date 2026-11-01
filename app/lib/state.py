"""Everything the pages share, built once from a Results bundle (no Streamlit here, so it is testable)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import claims as cl
from . import data, profiles

EXECUTION_LABEL = "Execution (completion above expected)"


@dataclass
class AppState:
    res: data.Results
    claims: dict[str, cl.Claim]
    profiles: pd.DataFrame          # long: one row per player × metric
    execution: pd.DataFrame | None

    @property
    def players(self) -> pd.DataFrame:
        return self.res["players"]

    def profile(self, player: str) -> pd.DataFrame:
        return self.profiles[self.profiles["player"] == player].set_index("metric")


def build(res: data.Results) -> AppState:
    claims = cl.claims(res["reliability"]) if res.has("reliability") else {}
    execution = None
    players = res["players"].copy() if res.has("players") else None
    if res.has("options") and players is not None:
        execution = data.execution_table(res["options"])
        execution = execution.join(players["pos_group"], how="inner")
        players["execution"] = execution["execution"]
        players["execution_pct"] = profiles.within_position_pct(players["execution"], players["pos_group"])
        rel, _ = profiles.split_half_reliability(execution, "execution_odd", "execution_even", "pos_group")
        claims = cl.add(claims, "execution", EXECUTION_LABEL, "quality", rel)
    return AppState(res, claims, long_profiles(players, claims) if players is not None else pd.DataFrame(), execution)


def long_profiles(players: pd.DataFrame, claims: dict[str, cl.Claim]) -> pd.DataFrame:
    n_ref = float(players["n_decisions"].median())
    rows = []
    for metric in [m for g in profiles.GROUPS.values() for m in g]:
        c = claims.get(metric)
        if c is None or metric + "_pct" not in players:
            continue
        for name, r in players.iterrows():
            e = profiles.estimate(r[metric + "_pct"], c.reliability, r["n_decisions"], n_ref)
            given = r.get(metric + "_tier")
            p_above = r.get(metric + "_p_above", np.nan)  # the pipeline's shrinkage estimate, when it wrote one
            rows.append({"player": name, "team": r["team"], "pos_group": r["pos_group"], "n_decisions": r["n_decisions"],
                         "metric": metric, "label": c.label, "level": c.level, "value": r.get(metric, np.nan),
                         "observed": e.observed, "estimate": e.estimate, "lo": e.lo, "hi": e.hi,
                         "p_above": p_above if pd.notna(p_above) else e.p_above,
                         "player_reliability": e.reliability,
                         "tier": given if isinstance(given, str) else profiles.tier(e)})
    return pd.DataFrame(rows)


def team_context(state: AppState, team: str, metric: str) -> float | None:
    """The team's own value for a metric, shown next to any player number (part of a player's score is the team)."""
    ts = state.res["team_style"] if state.res.has("team_style") else None
    if ts is not None and team in ts.index and metric in ts.columns:
        return float(ts.at[team, metric])
    return None
