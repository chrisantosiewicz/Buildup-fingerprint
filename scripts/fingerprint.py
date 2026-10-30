"""Build-Up Fingerprint from the build-up pipeline's outputs.

Usage:
    python scripts/fingerprint.py --data path/to/opendata/data --buildup outputs --out outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buildup.fingerprint import (
    decision_frame, player_fingerprint, possession_frame, reliability_table, team_outcomes, team_style, validate_tiers,
)
from buildup.io import load_all_events
from buildup.models import prepare_turnovers
from buildup.outcomes import add_outcomes
from buildup.turnovers import build_turnovers
from buildup.options import possession_option_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--buildup", default="outputs", help="folder with option_values.csv and decisions.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = Path(args.buildup)

    events = load_all_events(args.data)
    o = pd.read_csv(src / "option_values.csv", low_memory=False)
    d = pd.read_csv(src / "decisions.csv", low_memory=False)
    dec = decision_frame(o, d)
    poss = possession_frame(events, possession_option_value(events))

    rel = reliability_table(dec, poss)
    fp = player_fingerprint(dec, poss)
    ts = team_style(dec, poss)
    turnovers = prepare_turnovers(add_outcomes(build_turnovers(events), events))
    ts = ts.join(team_outcomes(args.data, events, turnovers))
    rel.to_csv(out / "metric_reliability.csv", index=False)
    fp.round(5).to_csv(out / "player_fingerprint.csv")
    ts.round(5).to_csv(out / "team_style.csv")
    dec.to_parquet(out / "decisions_style.parquet")
    poss.to_parquet(out / "possessions_style.parquet")

    res = {"n_decisions": len(dec), "n_possessions": len(poss), "n_players": len(fp),
           "reliability": rel.to_dict(orient="records"),
           "team_style": ts[["directness", "risk_axis", "provider", "progress_10s", "conceded_danger"]].round(3).to_dict(orient="index"),
           "tier_validation_odd_to_even": validate_tiers(dec, poss),
           "quality_tiers": {m: fp[m + "_tier"].value_counts().to_dict() for m in ["provider", "decision_quality"]}}
    with open(out / "fingerprint_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
