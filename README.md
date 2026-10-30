# buildup (work in progress)

Build-up value and the Build-Up Fingerprint for SkillCorner open data (A-League 2024/25). The package began as a rest-defence study, which is kept as a documented null result. The submission README (`../README_DRAFT.md`) will replace this file before the deadline.

## Setup
```bash
git lfs install
git clone https://github.com/SkillCorner/opendata.git        # ~2 GB with tracking
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"
pytest
```

## Pipeline
```bash
python scripts/option_value.py     --data ../opendata/data --out outputs    # option value, decisions (~10 min)
python scripts/fingerprint.py --data ../opendata/data --buildup outputs --out outputs
python scripts/figures.py     --results outputs --out figures
streamlit run app/explorer.py -- --results outputs
```

`option_value.py` uses tracking by default for pass completion. Add `--no-tracking` to use SkillCorner's xPass only, which needs the event files alone.

Earlier studies:
- `scripts/week1.py`: turnover exposure, events only.
- `scripts/week3.py`: rest defence with tracking. The result is null (see `../REST_DEFENCE_FINDING.md`).

## Layout
- `src/buildup/io.py`, `tracking.py`: loading events, match metadata and tracking.
- `turnovers.py`, `outcomes.py`, `models.py`: open-play turnovers, danger outcomes, turnover-danger models.
- `options.py`: decision table from SkillCorner passing options, and the receiver's next situation.
- `space.py`: space around each option from tracking, 0.3 s before the pass.
- `progression.py`: progression-potential model (LightGBM, nested match-grouped CV, Optuna).
- `value.py`: turnover cost, completion model, option value, decisions.
- `fingerprint.py`: style and quality metrics, reliability, calibrated player estimates and tiers, team style axes.
- `features.py`: rest-defence features from tracking (null-result study).
- `scripts/`: runnable pipelines. `scripts/exploratory/`: the one-off analyses behind the decisions (they use scratch paths).
- `app/explorer.py`: the Build-Up Explorer (Streamlit).
- `tests/`: unit tests on synthetic data.

No data is committed. Licence: MIT (LICENSE.md).
