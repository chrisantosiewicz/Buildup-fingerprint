"""Player profile: creation, decision, execution and risk, as percentiles within position with honest ranges."""

import pandas as pd
import streamlit as st

from lib import charts, data, profiles, ui
from lib.state import team_context

ui.need("players", "reliability")
s = ui.app()
fp = s.players

st.title("Player profile")
st.caption("Every number is a percentile against players in the same position. The dot is the estimate after "
           "allowing for reliability and sample size, the bar is the 80% range, and the grey tick is the raw percentile.")

c1, c2, c3 = st.columns([1, 1, 2])
teams = sorted(fp["team"].unique())
pre = st.query_params.get("player")
pre_team = fp.at[pre, "team"] if pre in fp.index else None
team = c1.selectbox("Team", teams, index=teams.index(pre_team) if pre_team else 0)
positions = ["All"] + sorted(fp.loc[fp["team"] == team, "pos_group"].unique())
pos = c2.selectbox("Position", positions)
pool = fp[(fp["team"] == team) & ((fp["pos_group"] == pos) if pos != "All" else True)]
names = pool.sort_values("n_decisions", ascending=False).index.tolist()
player = c3.selectbox("Player", names, index=names.index(pre) if pre in names else 0)
if not player:
    st.stop()
st.query_params["player"] = player

r = fp.loc[player]
prof = s.profile(player)
peers_n = int((fp["pos_group"] == r["pos_group"]).sum())
st.subheader(f"{player}")
st.markdown(f"{r['team']} · {r['pos_group']} (compared with {peers_n} players) · "
            f"{int(r['n_decisions'])} passing decisions · {int(r.get('n_possessions', 0))} possessions")


def est(metric: str):
    return prof.loc[metric] if metric in prof.index else None


def pct(row) -> str:
    return "n/a" if row is None or pd.isna(row["estimate"]) else f"{row['estimate']:.0%}"


# ---- headline tiles ----
t1, t2, t3, t4 = st.columns(4)
prov, dq, ex = est("provider"), est("decision_quality"), est("execution")
TIER_TILE = {"above average": "Top tier", "around average": "Middle tier", "below average": "Bottom tier"}
prov_claim = s.claims.get("provider")
prov_value = pct(prov) if prov is None or (prov_claim and prov_claim.can_rank) else TIER_TILE.get(prov["tier"], prov["tier"])
t1.metric("Creation · provider", prov_value, help=profiles.EXPLAIN["provider"])
t1.markdown(ui.badge(prov_claim) + (f" range {prov['lo']:.0%}–{prov['hi']:.0%}" if prov is not None else ""))
t2.metric("Decision quality", TIER_TILE.get(dq["tier"], dq["tier"]) if dq is not None else "n/a", help=profiles.EXPLAIN["decision_quality"])
t2.markdown(ui.badge(s.claims.get("decision_quality"))
            + (f" {dq['p_above']:.0%} chance above average" if dq is not None else ""))
t3.metric("Execution", pct(ex), help=profiles.EXPLAIN["execution"])
t3.markdown(ui.badge(s.claims.get("execution")) if ex is not None else "Needs the option table")
fw, rk = est("forwardness"), est("risk")
if fw is not None and rk is not None:
    E = lambda x: profiles.Estimate(x["observed"], x["estimate"], x["lo"], x["hi"], x["p_above"], x["player_reliability"])  # noqa: E731
    name, sure = profiles.risk_quadrant(E(fw), E(rk))
    t4.metric("Risk profile", name if sure != "unclear" else "No clear lean",
              help="Forwardness against risk accepted, both against the average option available.")
    t4.markdown(f":gray-badge[{sure}]" + (f" nearest: {name}" if sure == "unclear" else ""))

# ---- groups ----
left, right = st.columns([1.25, 1])
with left:
    for group in ["Creation", "Decision", "Execution", "Style"]:
        rows = prof.loc[[m for m in profiles.GROUPS[group] if m in prof.index]].reset_index()
        if not len(rows):
            continue
        st.markdown(f"**{group}**")
        st.plotly_chart(charts.profile_ranges(rows), width="stretch", key=f"pr_{group}")
with right:
    st.markdown("**Risk profile**")
    peers = s.profiles[(s.profiles["pos_group"] == r["pos_group"]) & s.profiles["metric"].isin(["forwardness", "risk"])]
    if len(peers):
        wide = peers.pivot(index="player", columns="metric", values=["estimate", "lo", "hi"])
        wide.columns = [m if k == "estimate" else f"{m}_{k}" for k, m in wide.columns]
        st.plotly_chart(charts.risk_quadrant(wide.dropna(), player), width="stretch")
        rk_claim = s.claims["risk"]
        st.caption("Grey: other players in the same position. Whiskers: 80% range. "
                   f"Risk accepted has reliability {rk_claim.reliability:.2f} per player ({rk_claim.text.lower()}), so "
                   "many ranges cross the middle; the team style map is where risk reads most clearly.")

    st.markdown("**Team context**")
    st.caption("Part of what looks like player quality is the team around them, so the team's number sits next to the player's.")
    ctx = []
    for m, scale, unit in [("provider", 1000, "mG/pass"), ("decision_quality", 1, ""), ("creator", 1, "score")]:
        tv = team_context(s, r["team"], m)
        pv = r.get(m)
        if tv is not None and pd.notna(pv):
            ctx.append({"metric": s.claims[m].label if m in s.claims else m,
                        "player": f"{pv * scale:.2f} {unit}", "team": f"{tv * scale:.2f} {unit}"})
    if ctx:
        st.dataframe(pd.DataFrame(ctx), hide_index=True, width="stretch")
    if "creator" in s.claims and s.claims["creator"].level == "low":
        st.caption("Creator is not reliable per player over 20 matches. Use the team number above and the "
                   "player's creator clips on the Clips page.")

# ---- moments ----
if s.res.has("options"):
    st.markdown("**Moments to watch**: biggest value added and biggest missed options")
    mom = data.player_moments(s.res["options"], player)
    if len(mom):
        show = mom[["moment", "match_id", "period", "clock", "player_name", "added_mG", "gap_mG", "decision_id"]].rename(
            columns={"player_name": "passed to", "added_mG": "vs average option (mG)", "gap_mG": "vs best option (mG)"})
        ev = st.dataframe(show.round(2), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                          column_config={"decision_id": None})
        if ev.selection.rows:
            st.session_state["replay_decision"] = show.iloc[ev.selection.rows[0]]["decision_id"]
            st.switch_page("views/replay.py")
        st.caption("Select a row to open it in the pass replay.")

with st.expander("How to read this"):
    st.markdown(
        "- **Creation (provider):** the most reliable quality measure (reliability "
        f"{s.claims['provider'].reliability:.2f}, checked against SkillCorner season data), shown as tiers because "
        "player ranges are wide.\n"
        "- **Creation (creator):** value added on the ball. Its tiers are indicative only.\n"
        "- **Decision quality:** shown as a tier with the chance the player is above the position average. "
        "Exact ranks would over-claim at this reliability.\n"
        "- **Execution:** completion above what the model expected. Its reliability is measured here on odd vs even "
        "matches like every other metric" + (f" ({s.claims['execution'].reliability:.2f})." if "execution" in s.claims else ".") + "\n"
        "- **Ranges** come from each metric's reliability and the player's number of passes: fewer passes or a "
        "noisier metric means a wider range and an estimate pulled towards the middle.")
