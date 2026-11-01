"""Data layer: find the pipeline outputs, check their columns, and derive the tables the pages need.

The app only reads files written by scripts/option_value.py, scripts/fingerprint.py and scripts/clips.py
(or the slim bundle from scripts/export_app_data.py). A missing file never crashes the app: it is recorded
as an issue and the pages that need it say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Columns of option_values.csv the app uses. Everything else is dropped on load to keep memory small.
OPTION_COLS = ["decision_id", "match_id", "period", "minute_start", "second_start", "player_name",
               "player_in_possession_name", "team_shortname", "opt_x", "opt_y", "passer_x", "passer_y",
               "xpass_completion", "completion", "xthreat", "P", "cost", "V", "V_reward", "V_progression", "V_cost",
               "chosen", "received", "phase", "passer_pressure_id", "is_best", "pitch_length", "pitch_width"]


@dataclass(frozen=True)
class Source:
    key: str
    names: tuple[str, ...]          # first one found wins (parquet before csv)
    required: tuple[str, ...]       # columns the pages rely on
    index_col: int | None = None
    usecols: tuple[str, ...] | None = None
    needed_for: str = ""


SOURCES = [
    Source("team_style", ("team_style.csv",), ("directness", "risk_axis", "provider", "progress_10s"), 0,
           needed_for="Team styles, team context on player cards"),
    Source("players", ("player_fingerprint.csv",), ("team", "pos_group", "n_decisions", "provider", "provider_pct"), 0,
           needed_for="Player profiles, rankings"),
    Source("reliability", ("metric_reliability.csv",), ("metric", "label", "kind", "player_reliability"),
           needed_for="Every claim the app makes"),
    Source("team_buildup", ("team_buildup.csv",), ("team_shortname",), needed_for="Situations"),
    Source("clips", ("clips.csv",), ("kind", "match_id", "period", "clock", "player"), needed_for="Clips"),
    Source("options", ("option_values.parquet", "option_values.csv"), ("decision_id", "V", "chosen"),
           usecols=tuple(OPTION_COLS), needed_for="Pass replay, execution quality, player moments"),
]
SUMMARIES = {"buildup": "buildup_results.json", "fingerprint": "fingerprint_results.json"}


def find(root: Path, name: str) -> Path | None:
    """Look in the results folder, then one level down (results/buildup, results/fingerprint, ...)."""
    if (root / name).exists():
        return root / name
    hits = sorted(root.glob(f"*/{name}"))
    return hits[0] if hits else None


def _read(path: Path, src: Source) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df[[c for c in df.columns if src.usecols is None or c in src.usecols]]
    keep = (lambda c: c in src.usecols) if src.usecols else None
    return pd.read_csv(path, index_col=src.index_col, usecols=keep, low_memory=False)


@dataclass
class Results:
    root: Path
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summaries: dict[str, dict] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def has(self, key: str) -> bool:
        return key in self.tables

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.tables[key]

    def summary(self, which: str, *keys, default=None):
        d = self.summaries.get(which, {})
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d


def load_results(root: str | Path) -> Results:
    root = Path(root)
    res = Results(root)
    if not root.exists():
        res.issues.append(f"Results folder not found: {root}")
        return res
    for src in SOURCES:
        path = next((p for p in (find(root, n) for n in src.names) if p), None)
        if path is None:
            res.issues.append(f"{src.names[-1]} not found (needed for: {src.needed_for})")
            continue
        df = _read(path, src)
        missing = [c for c in src.required if c not in df.columns]
        if missing:
            res.issues.append(f"{path.name} is missing columns {missing} (needed for: {src.needed_for})")
            continue
        res.tables[src.key], res.paths[src.key] = df, path
    for key, name in SUMMARIES.items():
        path = find(root, name)
        if path:
            res.summaries[key] = json.loads(path.read_text())
            res.paths[key] = path
    if res.has("options"):
        res.tables["options"] = _tidy_options(res["options"])
    return res


def _tidy_options(o: pd.DataFrame) -> pd.DataFrame:
    o = o.copy()
    for c in ("chosen", "is_best", "received"):
        if c in o:
            o[c] = o[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    return o


# ---------- option value with coach weights ----------

def revalue(opts: pd.DataFrame, gamma: float = 1.0, kappa: float = 1.0) -> pd.DataFrame:
    """Recompute option value for a lookahead weight γ and a risk weight κ.

    The pipeline stores the terms at γ = κ = 1 (V = V_reward + V_progression − V_cost),
    so V(γ, κ) = V_reward + γ·V_progression − κ·V_cost.
    """
    o = opts.copy()
    o["V"] = o["V_reward"] + gamma * o["V_progression"] - kappa * o["V_cost"]
    o["is_best"] = o["V"] == o.groupby("decision_id")["V"].transform("max")
    return o


def clock(minute: pd.Series, second: pd.Series) -> pd.Series:
    return minute.astype(int).astype(str) + ":" + second.astype(int).astype(str).str.zfill(2)


def decision_index(opts: pd.DataFrame) -> pd.DataFrame:
    """One row per decision (the option that was chosen), with a label for pickers."""
    ch = opts[opts["chosen"]].drop_duplicates("decision_id").copy()
    ch["clock"] = clock(ch["minute_start"], ch["second_start"])
    ch["label"] = (ch["clock"] + "  " + ch["player_in_possession_name"].fillna("?") + " → " + ch["player_name"].fillna("?")
                   + np.where(ch["is_best"], "  ✓ best", ""))
    return ch.sort_values(["match_id", "period", "minute_start", "second_start"]).reset_index(drop=True)


def link_clips(clips: pd.DataFrame, opts: pd.DataFrame | None, window_s: int = 20) -> pd.DataFrame:
    """Attach the decision a clip refers to, so it can open in the pass replay.

    Missed options and back passes are decisions: exact match on match, period, clock and passer.
    Creator clips are possessions (clock = start): take that player's first decision within `window_s`.
    """
    c = clips.copy()
    c["decision_id"] = pd.NA
    if opts is None or not len(opts):
        return c
    d = decision_index(opts)
    d["t"] = d["minute_start"].astype(int) * 60 + d["second_start"].astype(int)
    mm_ss = c["clock"].astype(str).str.split(":", expand=True).astype(int)
    c["t"] = mm_ss[0] * 60 + mm_ss[1]
    for i, r in c.iterrows():
        cand = d[(d["match_id"] == r["match_id"]) & (d["period"] == r["period"])
                 & (d["player_in_possession_name"] == r["player"]) & (d["t"] >= r["t"]) & (d["t"] <= r["t"] + window_s)]
        if len(cand):
            c.at[i, "decision_id"] = cand.sort_values("t")["decision_id"].iloc[0]
    return c.drop(columns="t")


# ---------- player-level measures the pipeline does not write ----------

def execution_table(opts: pd.DataFrame, min_passes: int = 30) -> pd.DataFrame:
    """Execution: passes completed minus the completion the model expected, per player.

    Also returns the odd/even-match split of the same number, so its reliability can be measured
    the same way as every other metric (see profiles.split_half_reliability).
    """
    ch = opts[opts["chosen"]].drop_duplicates("decision_id")
    ch = ch.assign(above=ch["received"].astype(float) - ch["completion"],
                   half=np.where(ch["match_id"].rank(method="dense").astype(int) % 2 == 1, "odd", "even"))
    g = ch.groupby("player_in_possession_name")
    out = pd.DataFrame({"team": g["team_shortname"].agg(lambda s: s.mode().iat[0]), "passes": g.size(),
                        "execution": g["above"].mean(), "expected_completion": g["completion"].mean(),
                        "completion": g["received"].mean()})
    halves = ch.groupby(["player_in_possession_name", "half"])["above"].agg(["mean", "size"]).unstack()
    for h in ("odd", "even"):
        out[f"execution_{h}"] = halves[("mean", h)] if ("mean", h) in halves else np.nan
        out[f"passes_{h}"] = halves[("size", h)] if ("size", h) in halves else 0
    out.index.name = "player"
    return out[out["passes"] >= min_passes]


def player_moments(opts: pd.DataFrame, player: str, n: int = 5) -> pd.DataFrame:
    """A player's best and worst passing decisions, as gap to the best option (0 = picked the best)."""
    o = opts[opts["player_in_possession_name"] == player]
    if not len(o):
        return o.iloc[0:0]
    best = o.groupby("decision_id")["V"].max().rename("V_best")
    d = decision_index(o).merge(best, left_on="decision_id", right_index=True)
    d["gap_mG"] = (d["V"] - d["V_best"]) * 1000
    d["added_mG"] = (d["V"] - o.groupby("decision_id")["V"].mean().reindex(d["decision_id"]).to_numpy()) * 1000
    top = d.nlargest(n, "added_mG").assign(moment="best")
    low = d.nsmallest(n, "gap_mG").assign(moment="missed")
    return pd.concat([top, low])
