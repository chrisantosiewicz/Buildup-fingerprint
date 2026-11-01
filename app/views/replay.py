"""Pass replay: one decision on the pitch, every option valued, with coach weights."""

import streamlit as st

from lib import charts, data, ui

ui.need("options")
s = ui.app()
o_all = s.res["options"]

st.title("Pass replay")
st.caption("Option value = completion × (threat + γ · receiver's next options) − (1 − completion) × κ · turnover "
           "cost, in milli-goals. Attacking left to right.")

with st.sidebar:
    st.markdown("**Coach weights**")
    gamma = st.slider("γ: how much the next pass counts", 0.0, 2.0, 1.0, 0.25)
    kappa = st.slider("κ: how much we fear counters", 0.0, 3.0, 1.0, 0.25)
    st.caption("Reported results use γ = κ = 1. Player rankings barely move between 0.5 and 2 (rank r ≥ 0.92).")

o_all = data.revalue(o_all, gamma, kappa) if (gamma, kappa) != (1.0, 1.0) else o_all
idx = data.decision_index(o_all)
pre = st.session_state.pop("replay_decision", None) or st.query_params.get("decision")
if pre is not None:
    hit = idx[idx["decision_id"].astype(str) == str(pre)]
    pre = hit.iloc[0] if len(hit) else None

c1, c2, c3 = st.columns([1, 1, 2])
teams = sorted(idx["team_shortname"].dropna().unique())
team = c1.selectbox("Team", teams, index=teams.index(pre["team_shortname"]) if pre is not None else 0)
d1 = idx[idx["team_shortname"] == team]
passers = sorted(d1["player_in_possession_name"].dropna().unique())
pp = pre["player_in_possession_name"] if pre is not None and pre["player_in_possession_name"] in passers else None
passer = c2.selectbox("Passer", passers, index=passers.index(pp) if pp else 0)
d2 = d1[d1["player_in_possession_name"] == passer]
ids = d2["decision_id"].tolist()
labels = dict(zip(d2["decision_id"], d2["match_id"].astype(str) + " · P" + d2["period"].astype(str) + " · " + d2["label"]))
pick = c3.selectbox("Decision", ids, format_func=labels.get,
                    index=ids.index(pre["decision_id"]) if pre is not None and pre["decision_id"] in ids else 0)
st.query_params["decision"] = str(pick)

d = o_all[o_all["decision_id"] == pick]
if not len(d):
    st.stop()
ch = d[d["chosen"]].iloc[0]
best = d.loc[d["V"].idxmax()]
press = {1: "none", 2: "low", 3: "medium", 4: "high", 5: "very high"}.get(int(ch.get("passer_pressure_id", 0) or 0), "?")
m1, m2, m3 = st.columns(3)
m1.metric("Chosen", ch["player_name"])
m1.caption(f"Option value {ui.mg(ch['V'], signed=True)}")
m2.metric("Best option", best["player_name"])
m2.caption(f"Option value {ui.mg(best['V'], signed=True)}"
           + ("" if ch["decision_id"] is None or best["player_name"] != ch["player_name"] else " · the chosen pass"))
m3.metric("Pressure on the passer", press)
st.plotly_chart(charts.decision_figure(d), width="stretch")
st.markdown("**Why each option is worth what it is**")
st.plotly_chart(charts.value_terms(d), width="stretch")
