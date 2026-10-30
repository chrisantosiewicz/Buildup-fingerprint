# Exploratory analyses

These are the one-off studies behind the numbers in the write-up. They're kept for transparency, not as part of the pipeline.

- Run them from a working folder that has the SkillCorner open data at `dl/opendata/data` and pipeline outputs in `out10/` and `out11/`. Those are the outputs of `scripts/option_value.py` and `scripts/fingerprint.py`.
- Adjust those paths to match your own setup.
- The pipeline scripts in `scripts/` take their paths as arguments and don't depend on anything here.

| Script | What it checks |
|---|---|
| `style.py`, `comp.py`, `feat.py`, `dm2.py`, `creator2.py` | Early reliability and model-feature studies |
| `extval.py`, `extval10.py` | Our 20-match measures vs SkillCorner season aggregates |
| `space_eval.py`, `space_lag.py`, `space_leak.py`, `xpplus.py` | Tracking-based completion features and the leak check |
| `roles.py`, `robust.py` | Style clusters (dropped) and robustness to γ and κ |
| `bu_turnovers.py` | Chances conceded after build-up turnovers |
| `rel_up.py`, `rel_up2.py`, `rel_up3.py`, `risk_logit.py`, `risk_ext.py` | Making the risk and creator measures more reliable |
| `stories.py` | Safe pass vs back pass, dead-end receptions, and turning pressure into progress by team |
