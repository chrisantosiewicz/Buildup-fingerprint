"""How reliable is this? The claim each metric supports, and the checks behind the model."""

import pandas as pd
import streamlit as st

from lib import charts, ui
from lib.claims import RANK_MIN, TIER_MIN

ui.need("reliability")
s = ui.app()
st.title("How reliable is this?")
st.markdown(
    f"Each metric is computed on odd and even matches separately. **Reliability** is how well the two halves agree "
    f"after removing the position average, projected to all 20 matches. At {RANK_MIN} or above the app ranks players; "
    f"from {TIER_MIN} to {RANK_MIN} it shows tiers and ranges; below {TIER_MIN} it shows a wide range with a low "
    "confidence label and makes the claim at team level and with clips.")
st.plotly_chart(charts.reliability_chart(s.claims), width="stretch")
st.dataframe(pd.DataFrame([{"metric": c.label, "kind": c.kind, "reliability": c.reliability, "team r": c.team_r,
                            "what the app claims": c.text, "note": c.note} for c in s.claims.values()]),
             hide_index=True, width="stretch")
st.caption("Execution is measured inside the app from the option table, with the same odd/even check.")

st.subheader("Does option value predict what happens?")
val = s.res.summary("buildup", "validation")
if val:
    rows = [{"check": "Progress within 10 s (AUC)", **val.get("progress_10s_auc", {})}]
    st.dataframe(pd.DataFrame(rows).set_index("check"), width="stretch")
    q = val.get("progress_rate_by_V_quintile")
    if q:
        st.markdown("Progress rate by fifth of option value, bottom to top: "
                    + " → ".join(f"{v:.0%}" for v in q.values()))
comp = s.res.summary("buildup", "completion_model", "auc")
if comp:
    st.markdown(f"Completion model AUC on held-out matches: SkillCorner xPass {comp.get('skillcorner_xpass')}, "
                f"with tracking {comp.get('with_tracking')}.")
gs = s.res.summary("buildup", "gamma_sensitivity")
if gs:
    st.markdown("**Lookahead weight γ**: it trades progress against conceded danger.")
    st.dataframe(pd.DataFrame(gs), hide_index=True)
