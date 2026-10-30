"""Home: what the Build-Up Explorer answers, in one screen."""

import streamlit as st

from lib import ui

s = ui.app()
st.title("Build-Up Explorer")
st.markdown("**A completed pass is only as good as what the receiver can do next.** Every passing option in 20 "
            "A-League matches is valued in goals: completion × (threat + the receiver's next options) − interception "
            "× turnover cost.")

n = s.res.summary("buildup", "n_decisions")
best = s.res.summary("buildup", "share_picked_best_option")
auc = s.res.summary("buildup", "completion_model", "auc", "with_tracking")
q = s.res.summary("buildup", "validation", "progress_rate_by_V_quintile")
c = st.columns(4)
if n:
    c[0].metric("Passing decisions valued", f"{n:,}")
if best:
    c[1].metric("Picked the best option", f"{best:.0%}")
if auc:
    c[2].metric("Completion model AUC", f"{auc:.3f}")
if q:
    v = list(q.values())
    c[3].metric("Progress in 10 s, bottom → top fifth", f"{v[0]:.0%} → {v[-1]:.0%}")

st.markdown("""
**What you can do here**
- **Player profile:** creation, decision, execution and risk, as percentiles within position with honest ranges.
- **Squad overview:** a whole team on one screen, with each player's risk profile.
- **Team styles:** how each team builds up, and whether it works.
- **Rankings:** leaderboards for what the data can rank, tiers for the rest.
- **Pass replay:** any pass on the pitch, every option valued, with sliders for how much you value the next pass and fear counters.
- **Situations** and **Clips:** back passes under pressure, and moments to show in a video session.
- **How reliable is this?:** the evidence behind every claim.
""")
if s.res.issues:
    st.warning("Some results are missing, so some pages are limited. See *Data status* in the sidebar.")
