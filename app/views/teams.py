"""Team style map: how each team builds up, and whether it works."""

import streamlit as st

from lib import charts, ui

ui.need("team_style")
s = ui.app()
ts = s.res["team_style"]

st.title("Team styles")
st.caption("Directness = long-pass share + forwardness. Risk = how much harder the chosen pass is than the average "
           "option, minus back-pass share. Both axes agree between odd and even matches (r 0.74 and 0.71), so this "
           "is the most reliable layer in the app. Colour: provider, the quality of situations a team's passes create.")
team = st.selectbox("Highlight a team", ["None"] + sorted(ts.index))
st.plotly_chart(charts.team_map(ts, highlight=None if team == "None" else team), width="stretch")

cols = [c for c in ["directness", "risk_axis", "long_share", "forwardness", "back_under_pressure", "provider",
                    "decision_quality", "picked_best", "creator", "progress_10s", "conceded_danger"] if c in ts]
show = ts[cols].copy()
for c in ["provider"]:
    show[c] = show[c] * 1000
st.dataframe(show.sort_values("provider", ascending=False).round(3), width="stretch", column_config={
    "provider": st.column_config.NumberColumn("provider (mG/pass)", format="%.2f"),
    "creator": st.column_config.NumberColumn("creator", format="%.3f", help="Value added on the ball: EPV gained plus "
                                             "passing options ahead gained, against possessions of the same length, "
                                             "position and third."),
    "progress_10s": st.column_config.NumberColumn("progress in 10 s", format="%.2f"),
})
