"""Figures for the write-up, from the build-up and fingerprint outputs.

Usage:
    python scripts/figures.py --results outputs --out figures [--player "K. Bos"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#d6e6f8", "#a9cbf1", "#79ade8", "#4b8fdd", "#2a78d6", "#1d5aa5", "#123d73"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def _labels(fig, ax, xs, ys, names, fontsize=8.5):
    """Direct labels that avoid each other and the points: try a few offsets per point, keep the
    first whose box doesn't overlap a label already placed."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    placed = []
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    offsets = [(10, 6), (10, -8), (-10, 6), (-10, -8), (18, 16), (-18, 16), (18, -18), (-18, -18),
               (26, 0), (-26, 0), (30, 26), (-30, 26), (30, -28), (-30, -28)]
    for (x, y), name, (px, py) in zip(zip(xs, ys), names, pts):
        for dx, dy in offsets:
            ha = "left" if dx >= 0 else "right"
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            color=INK2, ha=ha, va="center", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=6))
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            hits_point = any(bb.contains(qx, qy) for qx, qy in pts)
            inside = ax.get_window_extent(r).contains(bb.x0, bb.y0) and ax.get_window_extent(r).contains(bb.x1, bb.y1)
            if not any(bb.overlaps(o) for o in placed) and not hits_point and inside:
                placed.append(bb)
                break
            t.remove()
        else:
            t = ax.annotate(name, (x, y), xytext=(8, 5), textcoords="offset points", fontsize=fontsize, color=INK2)
            placed.append(t.get_window_extent(r))


DIV = ["#eb6834", "#f3b293", "#e4e3df", "#a9cbf1", "#2a78d6"]   # orange (bad) -> grey -> blue (good)

# Colour options for the team map: column, colourbar label, title clause, colour map kind.
TEAM_COLOURS = {
    "provider": ("provider", "Provider: value handed to the receiver per pass (goals)", "and how well", "seq"),
    "progress_10s": ("progress_10s", "Build-up passes followed by progress within 10 s (share)", "and how often it gets them forward", "seq"),
    "goals_against": ("goals_against", "Goals allowed per match", "and what they concede", "seq_bad"),
    "shots_against": ("shots_against", "Shots allowed per match", "and the shots they allow", "seq_bad"),
    "counters_conceded": ("counters_conceded", "Dangerous counters conceded per match (box or shot within 10 s)",
                          "and the counters they allow", "seq_bad"),
    "goal_diff": ("goal_diff", "Goal difference per match", "and their results", "div"),
    "on_ball_gain": ("on_ball_gain", "Value gained on the ball per possession (SkillCorner EPV, goals)",
                     "and how much they add on the ball", "seq"),
}


def _cmap(kind: str, values: pd.Series):
    if kind == "div":
        m = max(abs(values.min()), abs(values.max()))
        return matplotlib.colors.LinearSegmentedColormap.from_list("div", DIV), -m, m
    colors = SEQ if kind == "seq" else ["#fde3d6", "#f7b99c", "#f08d62", "#eb6834", "#b84a1e", "#7a2e10"]
    return matplotlib.colors.LinearSegmentedColormap.from_list(kind, colors), values.min(), values.max()


def team_style_map(ts: pd.DataFrame, path: Path, colour: str = "provider", ax=None, compact: bool = False):
    """How each team builds up (two style axes), coloured by a quality or outcome measure."""
    col, label, clause, kind = TEAM_COLOURS[colour]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
    fig = ax.figure
    cmap, vmin, vmax = _cmap(kind, ts[col])
    sc = ax.scatter(ts["directness"], ts["risk_axis"], c=ts[col], cmap=cmap, vmin=vmin, vmax=vmax,
                    s=110 if compact else 170, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    lim = max(ts["directness"].abs().max(), ts["risk_axis"].abs().max()) * 1.45
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("← patient    Directness    direct →" if compact else "← patient          Directness          direct →")
    ax.set_ylabel("← safe    Risk    risky →" if compact else "← safe          Risk accepted          risky →")
    if not compact:
        for x, y, t in [(-lim, lim, "Patient, risky"), (lim, lim, "Direct, risky"),
                        (-lim, -lim, "Patient, safe"), (lim, -lim, "Direct, safe")]:
            ax.text(x * 0.95, y * 0.93, t, ha="left" if x < 0 else "right", va="top" if y > 0 else "bottom",
                    fontsize=8, color=MUTED, style="italic")
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(label, color=INK2, fontsize=7.5 if compact else 10)
    cb.outline.set_visible(False)
    names = ts.index if colour in ("provider", "progress_10s") else [f"{t} ({int(n)})" for t, n in zip(ts.index, ts["matches"])]
    if compact:
        ax.set_title(label, loc="left", fontsize=9.5, color=INK, pad=6)
        return sc, names
    ax.set_title(f"How A-League teams build up, {clause}", loc="left", fontsize=12, color=INK, pad=12)
    note = "" if colour in ("provider", "progress_10s") else " (n) = matches in the data; per-match outcomes from so few games are noisy."
    fig.text(0.01, 0.01, "Directness: long-pass share and forwardness vs the average option. Risk: lane difficulty "
             "(SkillCorner xPass) vs the average option, and fewer back passes. "
             "SkillCorner A-League 2024/25, 20 matches." + note, fontsize=7, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _labels(fig, ax, ts["directness"], ts["risk_axis"], names)   # after layout, so positions are final
    fig.savefig(path, dpi=200)
    plt.close(fig)


def team_map_panels(ts: pd.DataFrame, path: Path, colours=("provider", "progress_10s", "goals_against", "goal_diff")):
    """The same style map four times, each coloured by a different outcome, to compare at a glance."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    placed = []
    for ax, c in zip(axes.flat, colours):
        placed.append((ax, team_style_map(ts, path, colour=c, ax=ax, compact=True)[1]))
    fig.suptitle("Same build-up styles, different outcomes", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.005, "Each point is a team; (n) = matches in the 20-match sample. Outcomes per match from 1 to 7 games "
             "are noisy: read colours as tendencies, not verdicts. SkillCorner A-League 2024/25.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    for ax, names in placed:
        _labels(fig, ax, ts["directness"], ts["risk_axis"], names, fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def reliability_chart(rel: pd.DataFrame, path: Path):
    """Which numbers can rank players: full-sample reliability within position."""
    r = rel.sort_values("player_reliability")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [BLUE if k == "style" else ORANGE for k in r["kind"]]
    ax.barh(r["label"], r["player_reliability"], color=colors, height=0.6, zorder=3)
    for y, (v, n) in enumerate(zip(r["player_reliability"], r["players"])):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0.3, color=MUTED, lw=0.8, ls="--", zorder=2)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--", zorder=2)
    ax.text(0.305, len(r) - 0.4, "tiers", fontsize=7.5, color=MUTED)
    ax.text(0.505, len(r) - 0.4, "rankings", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Reliability within position over 20 matches (split-half, Spearman-Brown)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["Style: what he does", "Quality: how well"], loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Style is reliable from 20 matches; quality needs tiers", loc="left", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def back_pass_chart(res: dict, path: Path):
    """Back passes: worse than the alternatives on average, but more often the best option under pressure."""
    sb = pd.DataFrame(res["direction_by_pressure"]["share_best"]).T
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(sb.index))
    for i, (col, c) in enumerate([("backward", ORANGE), ("sideways", AQUA), ("forward", BLUE)]):
        ax.bar(x + (i - 1) * 0.26, sb[col], width=0.24, color=c, label=col, zorder=3)
    ax.set_xticks(x, [s.replace("none/low", "none / low") for s in sb.index])
    ax.set_xlabel("Pressure on the passer")
    ax.set_ylabel("Share of passes that were the best option")
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_title("Under high pressure, the back pass is more often the right call", loc="left", fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def player_card(fp: pd.DataFrame, rel: pd.DataFrame, player: str, path: Path):
    """One player's fingerprint: style percentiles within position, quality with tiers."""
    r = fp.loc[player]
    labels = rel.set_index("metric")["label"]
    style = rel[rel["kind"] == "style"]["metric"].tolist()
    quality = ["provider", "decision_quality", "creator"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, keys, color, title in [(axes[0], style, BLUE, "Style (percentile within position)"),
                                   (axes[1], quality, ORANGE, "Quality (percentile within position)")]:
        vals = [r[k + "_pct"] for k in keys]
        y = np.arange(len(keys))[::-1]
        ax.barh(y, [1] * len(keys), color=GRID, height=0.55, zorder=2)
        ax.barh(y, vals, color=color, height=0.55, zorder=3)
        ax.set_yticks(y, [labels[k] for k in keys], fontsize=8.5)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
        ax.grid(False)
        for yy, k, v in zip(y, keys, vals):
            tag = r.get(k + "_tier") if isinstance(r.get(k + "_tier"), str) else ""
            ax.text(1.02, yy, f"{v * 100:.0f}" + (f"  ({tag})" if tag else ""), va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2)
    fig.suptitle(f"{player} · {r['team']} · {r['pos_group']} · {int(r['n_decisions'])} passing decisions",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def safe_pass_chart(stories: dict, path: Path):
    """Build-up passes by direction and pressure: how often lost, and how often a loss becomes a chance."""
    a = pd.DataFrame(stories["A_build_up_passes_by_direction_pressure"]).T
    a[["dir", "press"]] = a.index.to_series().str.split("|", expand=True).values
    dirs, press = ["back", "sideways", "forward"], ["not high pressure", "high pressure"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    x = np.arange(len(dirs))
    for ax, col, title in [(axes[0], "loss_rate", "How often the pass is lost"),
                           (axes[1], "chance_after_loss", "When lost: how often it becomes a chance")]:
        for i, (pr, c) in enumerate(zip(press, [AQUA, ORANGE])):
            sub = a[a["press"] == pr].set_index("dir").loc[dirs]
            v = sub[col].astype(float).values
            ax.bar(x + (i - 0.5) * 0.36, v, width=0.34, color=c, label=pr.replace("not high", "low / medium"), zorder=3)
            for xx, vv, n in zip(x + (i - 0.5) * 0.36, v, sub["lost"].astype(int)):
                ax.text(xx, vv + 0.004, f"{vv:.0%}" + (f"\n({n} lost)" if col == "chance_after_loss" else ""),
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(x, ["Back", "Sideways", "Forward"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 0.3)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=10, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("The safe pass is rarely lost, but when it is, it hurts", x=0.01, ha="left", fontsize=12.5)
    fig.text(0.01, 0.01, "Open-play passes in SkillCorner's build-up and create phases. A chance = box entry or shot "
             "within 10 s of the loss. SkillCorner A-League 2024/25, 20 matches.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def buildup_turnover_chart(t: pd.DataFrame, path: Path):
    """How often a lost ball becomes a dangerous counter: by phase, and inside build-up by zone and how."""
    BASE = t["dangerous_10s"].mean()
    ph = t.groupby("loss_phase_type")["dangerous_10s"].agg(["mean", "size"])
    ph = ph.loc[[c for c in ["build_up", "create", "chaotic", "direct", "finish"] if c in ph.index]]
    names = {"build_up": "Build-up", "create": "Create", "chaotic": "Chaotic", "direct": "Direct", "finish": "Finish"}
    b = t[t["loss_phase_type"].isin(["build_up", "create"])]
    b = b[b["loss_third"] != "attacking_third"]
    z = b.groupby(["loss_third", "lost_by"])["dangerous_10s"].agg(["mean", "size"]).reset_index()
    z["label"] = (z["loss_third"].map({"defensive_third": "Own third", "middle_third": "Middle third"}) + " · "
                  + z["lost_by"].map({"dispossessed": "dispossessed", "failed_pass": "pass cut out",
                                      "lost_at_reception": "lost on the first touch"}))
    z = z.sort_values("mean")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    for ax, labels, vals, ns, color in [
            (axes[0], [names[i] for i in ph.index][::-1], ph["mean"][::-1], ph["size"][::-1], BLUE),
            (axes[1], z["label"], z["mean"], z["size"], ORANGE)]:
        y = np.arange(len(vals))
        ax.barh(y, vals, color=color, height=0.6, zorder=3)
        ax.set_yticks(y, labels, fontsize=8.5)
        for yy, v, n in zip(y, vals, ns):
            ax.text(v + 0.004, yy, f"{v:.0%}  (n={n})", va="center", fontsize=8, color=INK2)
        ax.axvline(BASE, color=MUTED, lw=0.8, ls="--", zorder=2)
        ax.set_xlim(0, 0.24)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
    axes[0].text(BASE + 0.003, -0.55, f"all turnovers {BASE:.1%}", fontsize=7.5, color=MUTED, va="center")
    axes[0].set_title("Losing the ball, by phase of play", loc="left", fontsize=10, color=INK2)
    axes[1].set_title("Inside build-up: where and how it was lost", loc="left", fontsize=10, color=INK2)
    fig.suptitle("Turnovers in build-up are the ones that get punished", x=0.01, ha="left", fontsize=12.5)
    fig.text(0.01, 0.01, "Share of open-play turnovers followed by a box entry or shot within 10 s. "
             "SkillCorner A-League 2024/25, 20 matches.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--player", default=None)
    args = ap.parse_args()
    src, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ts = pd.read_csv(src / "team_style.csv", index_col=0)
    rel = pd.read_csv(src / "metric_reliability.csv")
    fp = pd.read_csv(src / "player_fingerprint.csv", index_col=0)
    res = json.load(open(src / "buildup_results.json"))
    team_style_map(ts, out / "fig1_team_style_map.png")
    if "goals_against" in ts:
        team_style_map(ts, out / "fig1b_team_style_goals_allowed.png", colour="goals_against")
        team_map_panels(ts, out / "fig1c_team_style_outcomes.png")
    reliability_chart(rel, out / "fig2_reliability.png")
    back_pass_chart(res, out / "fig3_back_passes.png")
    big = fp[(fp["n_decisions"] >= 150) & fp["pos_group"].isin(["MF", "CB", "FB"])]
    if (src / "turnovers.parquet").exists():
        buildup_turnover_chart(pd.read_parquet(src / "turnovers.parquet"), out / "fig5_buildup_turnovers.png")
    if "on_ball_gain" in ts:
        team_style_map(ts, out / "fig1d_team_style_on_ball_gain.png", colour="on_ball_gain")
    if (src / "stories.json").exists():
        safe_pass_chart(json.load(open(src / "stories.json")), out / "fig6_safe_pass.png")
    player = args.player or big.sort_values("provider", ascending=False).index[0]
    player_card(fp, rel, player, out / "fig4_player_card.png")


if __name__ == "__main__":
    main()
