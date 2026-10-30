"""Situations: what the right call looks like under pressure."""

import pandas as pd
import streamlit as st

from lib import charts, ui

s = ui.app()
st.title("Situations")
st.subheader("Back passes under pressure")
st.caption("Value is judged against the options available, not metres gained. Under high pressure the back pass is "
           "often the best option, and on average it is worth slightly more than the alternatives.")
dbp = s.res.summary("buildup", "direction_by_pressure")
if dbp:
    a, b = st.columns(2)
    if "share_best" in dbp:
        a.plotly_chart(charts.direction_by_pressure(dbp["share_best"], "share that were the best option", pct=True),
                       width="stretch")
    if "bva" in dbp:
        b.plotly_chart(charts.direction_by_pressure(dbp["bva"], "value vs average option (mG)", pct=False),
                       width="stretch")
else:
    st.info("Needs `buildup_results.json`.")

by_p = s.res.summary("buildup", "share_picked_best_by_pressure")
if by_p:
    names = {"1.0": "none", "2.0": "low", "3.0": "medium", "4.0": "high", "5.0": "very high"}
    st.markdown("**How often players pick the best option, by pressure**")
    st.dataframe(pd.DataFrame({"pressure": [names.get(k, k) for k in by_p], "picked the best option": list(by_p.values())}),
                 hide_index=True, column_config={"picked the best option": st.column_config.NumberColumn(format="percent")})

if s.res.has("team_buildup"):
    st.subheader("Which teams turn pressure into progress")
    tb = s.res["team_buildup"].set_index("team_shortname")
    cols = [c for c in ["back_pass_best_share_under_high_pressure", "picked_best", "progress_10s", "conceded_danger",
                        "creator_share"] if c in tb]
    st.dataframe(tb[cols].sort_values(cols[0], ascending=False), width="stretch", column_config={
        c: st.column_config.NumberColumn(c.replace("_", " "), format="percent") for c in cols})
