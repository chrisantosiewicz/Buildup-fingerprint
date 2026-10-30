"""Leaderboards, only for what the data can rank. Everything else is shown as tiers."""

import streamlit as st

from lib import ui

ui.need("players", "reliability")
s = ui.app()
P = s.profiles

st.title("Rankings")
c1, c2 = st.columns(2)
pos = c1.selectbox("Position", sorted(s.players["pos_group"].unique()), index=0)
min_n = c2.slider("Minimum passing decisions", 40, 200, 60, step=10)
pool = P[(P["pos_group"] == pos) & (P["n_decisions"] >= min_n)]

rankable = [m for m, c in s.claims.items() if c.can_rank and c.kind == "quality"]
tiered = [m for m, c in s.claims.items() if c.level == "tier" and c.kind == "quality"]
low = [c.label for c in s.claims.values() if c.level == "low" and c.kind == "quality"]

for m in rankable:
    c = s.claims[m]
    st.subheader(f"{c.label} {ui.badge(c)}")
    st.caption(f"Reliability {c.reliability:.2f}: {c.text.lower()}. Ordered by the estimate; the range shows how "
               "much a rank could move.")
    t = pool[pool["metric"] == m].sort_values("estimate", ascending=False)
    t = t.assign(rank=range(1, len(t) + 1), estimate=t["estimate"] * 100,
                 range=[f"{a:.0%}–{b:.0%}" for a, b in zip(t["lo"], t["hi"])])
    st.dataframe(t[["rank", "player", "team", "n_decisions", "estimate", "range", "tier"]], hide_index=True,
                 width="stretch", column_config={"n_decisions": "decisions", "estimate": st.column_config.ProgressColumn(
                     "estimate (percentile)", min_value=0, max_value=100, format="%.0f")})

for m in tiered:
    c = s.claims[m]
    st.subheader(f"{c.label} {ui.badge(c)}")
    st.caption(f"Reliability {c.reliability:.2f}: {c.text.lower()}. Players are grouped, not ordered: "
               "'above' means at least an 80% chance of being above the position average. Names are alphabetical.")
    t = pool[pool["metric"] == m]
    cols = st.columns(3)
    for col, tier in zip(cols, ["above average", "around average", "below average"]):
        names = t[t["tier"] == tier].sort_values("player")
        col.markdown(f"**{tier.capitalize()}** ({len(names)})")
        col.markdown("\n".join(f"- {p} · {tm}" for p, tm in zip(names["player"], names["team"])) or "none")

if low:
    st.info("Not ranked because they are not reliable per player over 20 matches: " + ", ".join(low)
            + ". See them at team level on Team styles and as moments on the Clips page.")
