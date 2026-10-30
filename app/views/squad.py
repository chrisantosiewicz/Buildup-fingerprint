"""Squad view: every player in one team on one screen, for a coach preparing a session."""

import pandas as pd
import streamlit as st

from lib import charts, profiles, ui

ui.need("players", "reliability")
s = ui.app()
fp = s.players

st.title("Squad overview")
st.caption("Each player against their own position across the league. Ranges are 80%; tiers and quadrant names "
           "only appear where the data is clear enough.")
team = st.selectbox("Team", sorted(fp["team"].unique()))
squad = s.profiles[s.profiles["team"] == team]
if not len(squad):
    st.stop()


def cell(metric: str) -> pd.DataFrame:
    d = squad[squad["metric"] == metric].set_index("player")
    return d


rows = []
for p, r in fp[fp["team"] == team].sort_values(["pos_group", "n_decisions"], ascending=[True, False]).iterrows():
    out = {"player": p, "position": r["pos_group"], "decisions": int(r["n_decisions"])}
    get = {m: cell(m) for m in ["provider", "decision_quality", "execution", "forwardness", "risk"]}
    for m, col in [("provider", "Creation (provider)"), ("execution", "Execution")]:
        if p in get[m].index:
            e = get[m].loc[p]
            out[col] = e["estimate"] * 100
            out[col + " range"] = f"{e['lo']:.0%}–{e['hi']:.0%}"
    if p in get["decision_quality"].index:
        out["Decision quality"] = get["decision_quality"].at[p, "tier"]
    if p in get["forwardness"].index and p in get["risk"].index:
        f, k = get["forwardness"].loc[p], get["risk"].loc[p]
        E = lambda x: profiles.Estimate(x["observed"], x["estimate"], x["lo"], x["hi"], x["p_above"], x["player_reliability"])  # noqa: E731
        name, sure = profiles.risk_quadrant(E(f), E(k))
        out["Risk profile"] = f"{name} ({sure})"
    rows.append(out)
table = pd.DataFrame(rows)
cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.0f")
       for c in ["Creation (provider)", "Execution"] if c in table}
ev = st.dataframe(table, hide_index=True, width="stretch", column_config=cfg, on_select="rerun",
                  selection_mode="single-row")
if ev.selection.rows:
    st.query_params["player"] = table.iloc[ev.selection.rows[0]]["player"]
    st.switch_page("views/players.py")
st.caption("Select a player to open their full profile.")

st.markdown("**Risk profile of the squad**")
wide = squad[squad["metric"].isin(["forwardness", "risk"])].pivot(index="player", columns="metric",
                                                                   values=["estimate", "lo", "hi"])
if len(wide):
    wide.columns = [m if k == "estimate" else f"{m}_{k}" for k, m in wide.columns]
    fig = charts.risk_quadrant(wide.dropna(), player="")
    fig.data[0].update(mode="markers+text", textposition="top center", marker=dict(size=11, color=charts.BLUE))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Risk accepted has reliability {s.claims['risk'].reliability:.2f} per player, so positions on the "
               "vertical axis are tentative. Forwardness is steadier.")
