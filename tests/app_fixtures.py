"""Synthetic pipeline outputs with the real file names and columns, for testing the Build-Up Explorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TEAMS = ["Alpha FC", "Beta United", "Gamma City"]
POS = ["CB", "FB", "MF", "AM"]
STYLE = ["long_share", "forwardness", "risk", "back_under_pressure", "carry_share", "quick_release", "tempo"]
QUALITY = ["provider", "decision_quality", "creator"]
REL = {"long_share": 0.68, "forwardness": 0.43, "risk": 0.23, "back_under_pressure": 0.59, "carry_share": 0.49,
       "quick_release": 0.42, "tempo": 0.53, "provider": 0.55, "decision_quality": 0.32, "creator": 0.16}


def write(root: Path, seed: int = 0, options: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    players = [f"P. {t[0]}{i:02d}" for t in TEAMS for i in range(12)]
    fp = pd.DataFrame({"team": [TEAMS[i // 12] for i in range(len(players))],
                       "pos_group": [POS[i % 4] for i in range(len(players))],
                       "n_decisions": rng.integers(40, 250, len(players)),
                       "n_possessions": rng.integers(50, 280, len(players))}, index=pd.Index(players, name="player"))
    for m in STYLE + QUALITY:
        fp[m] = rng.normal(0, 1, len(fp))
        fp[m + "_pct"] = fp.groupby("pos_group")[m].rank(pct=True)
    for m in ["provider", "decision_quality"]:
        fp[m + "_p_above"] = fp[m + "_pct"]
        fp[m + "_tier"] = np.where(fp[m + "_pct"] > 0.8, "above average",
                                   np.where(fp[m + "_pct"] < 0.2, "below average", "around average"))
    fp.to_csv(root / "player_fingerprint.csv")
    pd.DataFrame([{"metric": m, "label": m.replace("_", " ").capitalize(), "kind": "style" if m in STYLE else "quality",
                   "player_r_raw": r, "player_r_within_position": r / 2, "player_reliability": r, "players": 40,
                   "team_r": 0.4} for m, r in REL.items()]).to_csv(root / "metric_reliability.csv", index=False)
    ts = pd.DataFrame({c: rng.normal(0, 1, 3) for c in ["directness", "risk_axis", "long_share", "forwardness"]},
                      index=pd.Index(TEAMS, name="team_shortname"))
    ts["provider"], ts["progress_10s"], ts["conceded_danger"] = rng.uniform(0.005, 0.009, 3), 0.4, 0.005
    ts["decision_quality"], ts["creator"], ts["picked_best"] = 0.0, 0.06, 0.55
    ts.to_csv(root / "team_style.csv")
    pd.DataFrame({"team_shortname": TEAMS, "picked_best": 0.55, "progress_10s": 0.4, "conceded_danger": 0.005,
                  "back_pass_best_share_under_high_pressure": 0.6, "creator_share": 0.05}).to_csv(
        root / "team_buildup.csv", index=False)
    (root / "buildup_results.json").write_text(json.dumps({
        "n_decisions": 1000, "share_picked_best_option": 0.55,
        "completion_model": {"auc": {"skillcorner_xpass": 0.905, "with_tracking": 0.961}},
        "validation": {"progress_10s_auc": {"V": 0.65}, "progress_rate_by_V_quintile": {"0": 0.23, "4": 0.61}},
        "share_picked_best_by_pressure": {"1.0": 0.56, "4.0": 0.59},
        "direction_by_pressure": {"share_best": {"high": {"backward": 0.55, "forward": 0.6, "sideways": 0.5}},
                                  "bva": {"high": {"backward": 0.0005, "forward": 0.003, "sideways": 0.003}}},
        "gamma_sensitivity": [{"gamma": 1, "share_picked_best": 0.55}]}))
    rows, clips = [], []
    did = 0
    for match in range(1, 9):
        for k in range(150):
            passer = players[rng.integers(len(players))]
            team = fp.at[passer, "team"]
            mates = [p for p in players if fp.at[p, "team"] == team and p != passer]
            n_opt = rng.integers(2, 5)
            chosen = rng.integers(n_opt)
            minute, second = int(k * 0.6), int(rng.integers(60))
            for j in range(n_opt):
                comp = rng.uniform(0.5, 0.99)
                vr, vp, vc = comp * rng.uniform(0, 0.01), comp * rng.uniform(0, 0.02), (1 - comp) * rng.uniform(0, 0.02)
                rows.append(dict(decision_id=did, match_id=match, period=1 + (k >= 75), minute_start=minute,
                                 second_start=second, player_name=mates[j], player_in_possession_name=passer,
                                 team_shortname=team, opt_x=rng.uniform(-50, 50), opt_y=rng.uniform(-30, 30),
                                 passer_x=rng.uniform(-50, 0), passer_y=rng.uniform(-30, 30), xpass_completion=comp,
                                 completion=comp, xthreat=vr / comp, P=vp / comp, cost=vc / (1 - comp), V=vr + vp - vc,
                                 V_reward=vr, V_progression=vp, V_cost=vc, chosen=j == chosen,
                                 received=bool(rng.random() < comp) if j == chosen else False, phase="build_up",
                                 passer_pressure_id=int(rng.integers(1, 6)), pitch_length=105, pitch_width=68))
            if k == 10:
                clips.append(dict(kind="missed_option", match_id=match, period=1, clock=f"{minute}:{second:02d}",
                                  team=team, player=passer, what="passed to X instead of Y", value_mG=-5.0))
            did += 1
    o = pd.DataFrame(rows)
    o["is_best"] = o["V"] == o.groupby("decision_id")["V"].transform("max")
    if options:
        o.to_csv(root / "option_values.csv", index=False)
    clips.append(dict(kind="creator", match_id=99, period=1, clock="1:00", team=TEAMS[0], player=players[0],
                      what="carried the ball", value_mG=100.0))
    pd.DataFrame(clips).to_csv(root / "clips.csv", index=False)
    return root
