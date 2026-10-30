"""Plotly figures for the Build-Up Explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, AQUA, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#e4e3de"
SEQ = [[0, "#d6e6f8"], [0.5, "#4b8fdd"], [1, "#123d73"]]
DIV = [[0, ORANGE], [0.5, "#f1f0ea"], [1, BLUE]]
LEVEL_COLOR = {"rank": BLUE, "tier": "#7fb0e8", "low": "#c9c8c2"}
FONT = dict(family="Inter, system-ui, sans-serif", size=13)


def _base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), font=FONT,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------- teams ----------

def team_map(ts: pd.DataFrame, color: str = "provider", highlight: str | None = None) -> go.Figure:
    hover = [f"<b>{t}</b><br>Directness {r.directness:+.2f}<br>Risk {r.risk_axis:+.2f}"
             f"<br>Provider {r.provider * 1000:.2f} mG/pass<br>Progress in 10 s {r.progress_10s:.0%}"
             + (f"<br>Dangerous counters conceded {r.conceded_danger:.1%}" if "conceded_danger" in ts else "")
             for t, r in ts.iterrows()]
    c = ts[color] * (1000 if color == "provider" else 1)
    size = [26 if t == highlight else 18 for t in ts.index]
    fig = go.Figure(go.Scatter(
        x=ts["directness"], y=ts["risk_axis"], mode="markers+text", text=ts.index, textposition="top center",
        hovertext=hover, hoverinfo="text",
        marker=dict(size=size, color=c, colorscale=SEQ, line=dict(color="white", width=2),
                    colorbar=dict(title="Provider<br>(mG/pass)" if color == "provider" else color))))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(xaxis_title="← patient · Directness · direct →", yaxis_title="← safe · Risk accepted · risky →")
    return _base(fig, 560)


# ---------- player profiles ----------

def profile_ranges(rows: pd.DataFrame, show_observed: bool = True) -> go.Figure:
    """One line per metric: an 80% range bar, a dot at the estimate and a tick at the observed percentile."""
    rows = rows.iloc[::-1]
    y = rows["label"].tolist()
    fig = go.Figure()
    for lvl, col in LEVEL_COLOR.items():
        r = rows[rows["level"] == lvl]
        if not len(r):
            continue
        fig.add_trace(go.Bar(y=r["label"], x=(r["hi"] - r["lo"]) * 100, base=r["lo"] * 100, orientation="h",
                             marker=dict(color=col, opacity=0.45), width=0.5, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            y=r["label"], x=r["estimate"] * 100, mode="markers", showlegend=False,
            marker=dict(size=13, color=col, line=dict(color="white", width=2)),
            hovertext=[f"<b>{k.label}</b><br>Estimate {k.estimate:.0%} (80% range {k.lo:.0%} to {k.hi:.0%})"
                       f"<br>Observed {k.observed:.0%} · chance above average {k.p_above:.0%}"
                       f"<br>Reliability for this player {k.player_reliability:.2f}" for k in r.itertuples()],
            hoverinfo="text"))
    if show_observed:
        fig.add_trace(go.Scatter(y=y, x=rows["observed"] * 100, mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(symbol="line-ns", size=14, line=dict(color=MUTED, width=2))))
    text = [f"{e:.0%}  ({lo:.0%}–{hi:.0%})" if lvl != "low" else f"{e:.0%}  ({lo:.0%}–{hi:.0%}) low confidence"
            for e, lo, hi, lvl in zip(rows["estimate"], rows["lo"], rows["hi"], rows["level"])]
    fig.add_trace(go.Scatter(y=y, x=[101] * len(y), mode="text", text=text, textposition="middle right",
                             hoverinfo="skip", showlegend=False, textfont=dict(size=12, color=MUTED)))
    fig.add_vline(x=50, line_color=MUTED, line_width=1, line_dash="dot")
    fig.update_layout(barmode="overlay", xaxis=dict(range=[0, 175], tickvals=[0, 25, 50, 75, 100],
                                                    title="percentile within position", gridcolor=GRID),
                      yaxis=dict(categoryorder="array", categoryarray=y))
    return _base(fig, 70 + 44 * len(y))


def risk_quadrant(peers: pd.DataFrame, player: str) -> go.Figure:
    """Forwardness (x) against risk accepted (y), peers in grey, the player with 80% ranges."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=peers["forwardness"] * 100, y=peers["risk"] * 100, mode="markers",
                             text=peers.index, hoverinfo="text", showlegend=False,
                             marker=dict(size=8, color="#c9c8c2")))
    if player in peers.index:
        p = peers.loc[player]
        fig.add_trace(go.Scatter(
            x=[p["forwardness"] * 100], y=[p["risk"] * 100], mode="markers+text", text=[player],
            textposition="top center", showlegend=False, hoverinfo="skip",
            marker=dict(size=16, color=ORANGE, line=dict(color="white", width=2)),
            error_x=dict(type="data", symmetric=False, array=[(p["forwardness_hi"] - p["forwardness"]) * 100],
                         arrayminus=[(p["forwardness"] - p["forwardness_lo"]) * 100], color=ORANGE),
            error_y=dict(type="data", symmetric=False, array=[(p["risk_hi"] - p["risk"]) * 100],
                         arrayminus=[(p["risk"] - p["risk_lo"]) * 100], color=ORANGE)))
    for x, y, t in [(75, 97, "Bold progressor"), (75, 3, "Efficient progressor"), (25, 97, "Loose recycler"),
                    (25, 3, "Safe recycler")]:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(color=MUTED, size=12))
    fig.add_hline(y=50, line_color=MUTED, line_width=1)
    fig.add_vline(x=50, line_color=MUTED, line_width=1)
    fig.update_layout(xaxis=dict(range=[0, 100], title="Forwardness percentile (vs average option)", gridcolor=GRID),
                      yaxis=dict(range=[0, 100], title="Risk accepted percentile", gridcolor=GRID))
    return _base(fig, 420)


# ---------- pitch and decisions ----------

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
                      margin=dict(l=0, r=0, t=0, b=0), font=FONT,
                      xaxis=dict(range=[-L - 2, L + 2], visible=False),
                      yaxis=dict(range=[-W - 2, W + 2], visible=False, scaleanchor="x"))
    return fig


def _num(v, default):
    return default if v is None or pd.isna(v) else float(v)


def decision_figure(d: pd.DataFrame) -> go.Figure:
    r0 = d.iloc[0]
    fig = pitch(_num(r0.get("pitch_length"), 105), _num(r0.get("pitch_width"), 68))
    vmax = max(d["V"].abs().max(), 1e-6)
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r.passer_x, r.opt_x], y=[r.passer_y, r.opt_y], mode="lines",
                                 line=dict(color=ORANGE if r.chosen else "#c9c8c2", width=3 if r.chosen else 1.5,
                                           dash="solid" if r.chosen else "dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d["opt_x"], y=d["opt_y"], mode="markers+text", text=d["player_name"], textposition="top center",
        marker=dict(size=16, color=d["V"] * 1000, colorscale=DIV, cmin=-vmax * 1000, cmax=vmax * 1000,
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


def value_terms(d: pd.DataFrame) -> go.Figure:
    """The three terms of each option's value, stacked, so the trade-off is visible."""
    d = d.sort_values("V")
    fig = go.Figure()
    for col, name, color, sign in [("V_reward", "Threat", "#7fb0e8", 1), ("V_progression", "Receiver's next options", BLUE, 1),
                                   ("V_cost", "Turnover cost", ORANGE, -1)]:
        fig.add_trace(go.Bar(y=d["player_name"], x=sign * d[col] * 1000, name=name, orientation="h", marker_color=color))
    fig.add_trace(go.Scatter(y=d["player_name"], x=d["V"] * 1000, mode="markers", name="Option value",
                             marker=dict(color="black", size=9, symbol="diamond")))
    fig.update_layout(barmode="relative", xaxis_title="milli-goals", legend=dict(orientation="h", y=-0.25))
    return _base(fig, 120 + 34 * len(d))


# ---------- method and situations ----------

def reliability_chart(claims: dict) -> go.Figure:
    c = sorted(claims.values(), key=lambda k: k.reliability)
    from .claims import RANK_MIN, TIER_MIN
    fig = go.Figure(go.Bar(y=[k.label for k in c], x=[k.reliability for k in c], orientation="h",
                           marker_color=[LEVEL_COLOR[k.level] for k in c],
                           text=[f"{k.reliability:.2f}" for k in c], textposition="outside"))
    for x, t in [(TIER_MIN, "tiers"), (RANK_MIN, "rank")]:
        fig.add_vline(x=x, line_dash="dot", line_color=MUTED, annotation_text=t, annotation_position="top")
    fig.update_layout(xaxis=dict(range=[0, 1], title="reliability over 20 matches (within position)", gridcolor=GRID))
    return _base(fig, 80 + 34 * len(c))


def direction_by_pressure(table: dict, title: str, pct: bool) -> go.Figure:
    df = pd.DataFrame(table)  # columns pressure level, rows direction
    fig = go.Figure()
    for direction, color in [("backward", ORANGE), ("sideways", MUTED), ("forward", BLUE)]:
        if direction in df.index:
            v = df.loc[direction] * (100 if pct else 1000)
            fig.add_trace(go.Bar(x=df.columns, y=v, name=direction, marker_color=color,
                                 text=[f"{x:.0f}%" if pct else f"{x:+.1f}" for x in v], textposition="outside"))
    fig.update_layout(barmode="group", xaxis_title="pressure on the passer", yaxis_title=title,
                      legend=dict(orientation="h", y=1.1))
    return _base(fig, 380)
