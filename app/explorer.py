"""Build-Up Explorer: team styles, player fingerprints, and every passing decision on a pitch.

Run (after scripts/option_value.py and scripts/fingerprint.py):
    streamlit run app/explorer.py -- --results outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEQ = [[0, "#d6e6f8"], [0.5, "#4b8fdd"], [1, "#123d73"]]
OPTION_COLS = ["decision_id", "match_id", "period", "minute_start", "second_start", "player_name",
               "player_in_possession_name", "team_shortname", "opt_x", "opt_y", "passer_x", "passer_y",
               "xpass_completion", "completion", "xthreat", "P", "cost", "V", "V_reward", "V_progression", "V_cost",
               "chosen", "received", "phase", "passer_pressure_id", "is_best", "pitch_length", "pitch_width"]
PRESSURE = {1: "none", 2: "low", 3: "medium", 4: "high", 5: "very high"}


def results_dir() -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs")
    args, _ = ap.parse_known_args(sys.argv[1:])
    return Path(args.results)


@st.cache_data
def load(src: str):
    src = Path(src)
    ts = pd.read_csv(src / "team_style.csv", index_col=0)
    fp = pd.read_csv(src / "player_fingerprint.csv", index_col=0)
    rel = pd.read_csv(src / "metric_reliability.csv")
    opts = pd.read_csv(src / "option_values.csv", usecols=lambda c: c in OPTION_COLS, low_memory=False)
    return ts, fp, rel, opts


def team_map(ts: pd.DataFrame) -> go.Figure:
    hover = [f"<b>{t}</b><br>Directness {r.directness:+.2f}<br>Risk {r.risk_axis:+.2f}"
             f"<br>Provider {r.provider * 1000:.2f} mG/pass<br>Progress in 10 s {r.progress_10s:.0%}"
             f"<br>Dangerous counters conceded {r.conceded_danger:.1%}" for t, r in ts.iterrows()]
    fig = go.Figure(go.Scatter(
        x=ts["directness"], y=ts["risk_axis"], mode="markers+text", text=ts.index, textposition="top center",
        hovertext=hover, hoverinfo="text",
        marker=dict(size=18, color=ts["provider"] * 1000, colorscale=SEQ, line=dict(color="white", width=2),
                    colorbar=dict(title="Provider<br>(milli-goals/pass)"))))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="← patient · Directness · direct →", yaxis_title="← safe · Risk accepted · risky →")
    return fig


def pitch(length: float = 105, width: float = 68) -> go.Figure:
    L, W = length / 2, width / 2
    fig = go.Figure()
    line = dict(color="#b9b8b2", width=1.5)
    shapes = [dict(type="rect", x0=-L, y0=-W, x1=L, y1=W, line=line),
              dict(type="line", x0=0, y0=-W, x1=0, y1=W, line=line),
              dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line)]
    for s in (-1, 1):
        shapes += [dict(type="rect", x0=s * L, y0=-20.16, x1=s * (L - 16.5), y1=20.16, line=line),
                   dict(type="rect", x0=s * L, y0=-9.16, x1=s * (L - 5.5), y1=9.16, line=line)]
    fig.update_layout(shapes=shapes, plot_bgcolor="#f3f6ef", height=470, showlegend=False,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(range=[-L - 2, L + 2], visible=False),
                      yaxis=dict(range=[-W - 2, W + 2], visible=False, scaleanchor="x"))
    return fig


def decision_figure(d: pd.DataFrame) -> go.Figure:
    r0 = d.iloc[0]
    fig = pitch(r0.get("pitch_length", 105) or 105, r0.get("pitch_width", 68) or 68)
    vmax = max(d["V"].abs().max(), 1e-6)
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r.passer_x, r.opt_x], y=[r.passer_y, r.opt_y], mode="lines",
                                 line=dict(color=ORANGE if r.chosen else "#c9c8c2", width=3 if r.chosen else 1.5,
                                           dash="solid" if r.chosen else "dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d["opt_x"], y=d["opt_y"], mode="markers+text", text=d["player_name"], textposition="top center",
        marker=dict(size=16, color=d["V"] * 1000, colorscale=SEQ, cmin=-vmax * 1000, cmax=vmax * 1000,
                    line=dict(color=np.where(d["is_best"], AQUA, "white"), width=np.where(d["is_best"], 4, 2))),
        hovertext=[f"<b>{r.player_name}</b>{' (chosen)' if r.chosen else ''}{' · best option' if r.is_best else ''}"
                   f"<br>Completion {r.completion:.0%} (SkillCorner xPass {r.xpass_completion:.0%}) · Threat {r.xthreat * 1000:.1f} mG"
                   f"<br>Receiver's next options {r.P * 1000:.1f} mG · Turnover cost {r.cost * 1000:.1f} mG"
                   f"<br><b>Option value {r.V * 1000:+.2f} mG</b>" for r in d.itertuples()],
        hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[r0.passer_x], y=[r0.passer_y], mode="markers+text", text=[r0.player_in_possession_name],
                             textposition="bottom center", marker=dict(size=18, color=BLUE, symbol="diamond"),
                             hoverinfo="text", hovertext=["Passer"]))
    return fig


def bars(values: dict, color: str, tiers: dict | None = None) -> go.Figure:
    labels = list(values)[::-1]
    v = [values[k] * 100 for k in labels]
    text = [f"{x:.0f}" + (f" · {tiers[k]}" if tiers and isinstance(tiers.get(k), str) else "") for k, x in zip(labels, v)]
    fig = go.Figure(go.Bar(x=v, y=labels, orientation="h", marker_color=color, text=text, textposition="outside"))
    fig.update_layout(height=60 + 38 * len(labels), margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(range=[0, 125], title="percentile within position"))
    return fig


def main():
    st.set_page_config(page_title="Build-Up Explorer", layout="wide")
    src = results_dir()
    ts, fp, rel, opts = load(str(src))
    labels = rel.set_index("metric")["label"].to_dict()
    style_keys = rel.loc[rel["kind"] == "style", "metric"].tolist()
    quality_keys = rel.loc[rel["kind"] == "quality", "metric"].tolist()

    st.title("Build-Up Explorer")
    st.caption("Every pass valued as completion × (threat + the receiver's next options) − interception × "
               "turnover cost, in goals. Completion adds the space around each option from tracking. SkillCorner A-League 2024/25, 20 matches. mG = milli-goals.")
    tab_team, tab_player, tab_rank, tab_dec, tab_method = st.tabs(
        ["Team styles", "Player fingerprint", "Rankings", "Decision replay", "How reliable is this?"])

    with tab_team:
        st.plotly_chart(team_map(ts), width="stretch")
        show = ts[["directness", "risk_axis", "long_share", "forwardness", "tempo", "back_under_pressure",
                   "provider", "decision_quality", "picked_best", "progress_10s", "conceded_danger"]]
        st.dataframe(show.sort_values("provider", ascending=False).round(3), width="stretch")

    with tab_player:
        c1, c2, c3 = st.columns(3)
        pos = c1.selectbox("Position", ["CB", "FB", "MF", "AM", "FW"], index=2)
        pool = fp[fp["pos_group"] == pos]
        team = c2.selectbox("Team", ["All"] + sorted(pool["team"].unique()))
        pool = pool if team == "All" else pool[pool["team"] == team]
        names = pool.sort_values("n_decisions", ascending=False).index.tolist()
        player = c3.selectbox("Player", names)
        if player:
            r = fp.loc[player]
            st.subheader(f"{player} · {r['team']} · {r['pos_group']} · {int(r['n_decisions'])} passing decisions")
            a, b = st.columns([1.3, 1])
            a.markdown("**Style**: what he tends to do")
            a.plotly_chart(bars({labels[k]: r[k + "_pct"] for k in style_keys}, BLUE), width="stretch")
            b.markdown("**Quality**: how well, with the tier where the data supports one")
            tiers = {labels[k]: r.get(k + "_tier") for k in quality_keys}
            b.plotly_chart(bars({labels[k]: r[k + "_pct"] for k in quality_keys}, ORANGE, tiers), width="stretch")
            st.caption("Tiers: 'above/below average' means at least an 80% chance he is above/below the position "
                       "average after accounting for sample size. Creator has no tier: it is not reliable per player "
                       "over 20 matches.")

    with tab_rank:
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Rank by", ["provider", "decision_quality"], format_func=lambda k: labels[k])
        pos = c2.selectbox("Position ", ["CB", "FB", "MF", "AM", "FW"], index=2)
        t = fp[fp["pos_group"] == pos].sort_values(metric + "_p_above", ascending=False)
        st.dataframe(t[["team", "n_decisions", metric + "_tier", metric + "_p_above", metric + "_pct"]].rename(columns={
            metric + "_tier": "tier", metric + "_p_above": "P(above position average)", metric + "_pct": "percentile"}),
            width="stretch")

    with tab_dec:
        c1, c2, c3 = st.columns(3)
        team = c1.selectbox("Team ", sorted(opts["team_shortname"].unique()))
        o = opts[opts["team_shortname"] == team]
        passer = c2.selectbox("Passer", sorted(o["player_in_possession_name"].dropna().unique()))
        o = o[o["player_in_possession_name"] == passer]
        ch = o[o["chosen"] == True].copy()  # noqa: E712
        ch["label"] = (ch["minute_start"].astype(int).astype(str) + "' → " + ch["player_name"]
                       + np.where(ch["is_best"], "  ✓ best", ""))
        pick = c3.selectbox("Decision", ch.sort_values(["match_id", "period", "minute_start"])["decision_id"],
                            format_func=lambda i: ch.set_index("decision_id").at[i, "label"])
        d = o[o["decision_id"] == pick]
        if len(d):
            r = d[d["chosen"] == True].iloc[0]  # noqa: E712
            best = d.loc[d["V"].idxmax()]
            st.plotly_chart(decision_figure(d), width="stretch")
            st.markdown(f"Pressure on the passer: **{PRESSURE.get(int(r.passer_pressure_id), '?')}**. "
                        f"Chosen: **{r.player_name}** ({r.V * 1000:+.2f} mG). Best option: **{best.player_name}** "
                        f"({best.V * 1000:+.2f} mG). Attacking left to right.")
            tbl = d[["player_name", "chosen", "is_best", "completion", "xpass_completion", "xthreat", "P", "cost", "V"]].copy()
            for c in ["xthreat", "P", "cost", "V"]:
                tbl[c] = tbl[c] * 1000
            st.dataframe(tbl.sort_values("V", ascending=False).rename(columns={
                "completion": "completion (with tracking)", "xpass_completion": "SkillCorner xPass", "xthreat": "threat (mG)", "P": "next options (mG)",
                "cost": "turnover cost (mG)", "V": "value (mG)"}).round(2), width="stretch")

    with tab_method:
        st.markdown("Each number is computed on odd and even matches separately. **Reliability** is how well the two "
                    "halves agree after removing the position average, projected to all 20 matches. Above 0.5: "
                    "good enough to rank. 0.3 to 0.5: tiers. Below 0.3: team level or clips only.")
        st.dataframe(rel, width="stretch")


if __name__ == "__main__":
    main()
