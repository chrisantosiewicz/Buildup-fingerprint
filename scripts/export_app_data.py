"""Bundle the pipeline outputs the Build-Up Explorer reads into one small folder (for hosting the app).

Usage:
    python scripts/export_app_data.py --results outputs --out app_data
    streamlit run app/explorer.py -- --results app_data

Copies the small result files as they are and writes the option table as parquet with only the columns
the app uses (much smaller than option_values.csv). Reads outputs only; it never re-runs a model.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from lib.data import OPTION_COLS, SOURCES, SUMMARIES, find  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs")
    ap.add_argument("--out", default="app_data")
    args = ap.parse_args()
    src, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for s in SOURCES if s.key != "options" for n in s.names] + list(SUMMARIES.values())
    for name in names:
        p = find(src, name)
        if p:
            shutil.copy(p, out / name)
            print(f"copied {p}")
        else:
            print(f"missing {name}")
    p = find(src, "option_values.csv")
    if p:
        o = pd.read_csv(p, usecols=lambda c: c in OPTION_COLS, low_memory=False)
        o.to_parquet(out / "option_values.parquet", index=False)
        print(f"wrote option_values.parquet ({len(o):,} options)")
    else:
        print("missing option_values.csv: the pass replay and execution quality will be unavailable")


if __name__ == "__main__":
    main()
