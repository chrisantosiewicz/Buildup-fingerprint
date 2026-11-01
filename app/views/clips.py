"""Clips: moments worth showing in a video session, each one replayable."""

import streamlit as st

from lib import data, ui

ui.need("clips")
s = ui.app()
st.title("Clips")
st.caption("Match, period and clock to find each moment in the footage. Creator moments are where the player "
           "improved their own options on the ball.")

clips = data.link_clips(s.res["clips"], s.res["options"] if s.res.has("options") else None)
kinds = {"creator": "Made their own option", "missed_option": "Missed a better option",
         "right_back_pass": "Right back pass under pressure"}
c1, c2 = st.columns(2)
kind = c1.selectbox("Kind", list(kinds), format_func=kinds.get)
teams = ["All"] + sorted(clips["team"].unique())
team = c2.selectbox("Team", teams)
c = clips[(clips["kind"] == kind) & ((clips["team"] == team) if team != "All" else True)]
ev = st.dataframe(c[["match_id", "period", "clock", "team", "player", "what", "value_mG", "decision_id"]],
                  hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                  column_config={"decision_id": None, "value_mG": st.column_config.NumberColumn("value (mG)", format="%.1f")})
if ev.selection.rows:
    did = c.iloc[ev.selection.rows[0]]["decision_id"]
    if did is not None and str(did) != "<NA>":
        st.session_state["replay_decision"] = did
        st.switch_page("views/replay.py")
    else:
        st.warning("This moment could not be matched to a pass in the option table.")
st.caption("Select a row to open it in the pass replay" + ("" if s.res.has("options") else " (needs the option table)") + ".")
