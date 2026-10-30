"""Streamlit glue shared by the pages: loading once, the data-status panel, and claim badges."""

from __future__ import annotations

import argparse
import sys

import streamlit as st

from . import data, state
from .claims import Claim

BADGE = {"rank": ":blue-badge[Ranked]", "tier": ":violet-badge[Tiers]", "low": ":gray-badge[Low confidence]"}


def results_dir() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs")
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.results


@st.cache_resource(show_spinner="Loading results…")
def _load(path: str) -> state.AppState:
    return state.build(data.load_results(path))


def app() -> state.AppState:
    return _load(st.session_state.get("results_dir", results_dir()))


def need(*keys: str) -> None:
    """Stop the page with a clear message when a file it needs was not produced."""
    s = app()
    missing = [k for k in keys if not s.res.has(k)]
    if missing:
        names = {src.key: src.names[-1] for src in data.SOURCES}
        st.info("This page needs " + ", ".join(f"`{names[k]}`" for k in missing) + " in the results folder. "
                "Run the pipeline (see README) or point the app at another folder in the sidebar.")
        st.stop()


def badge(c: Claim | None) -> str:
    if not c:
        return ""
    return BADGE[c.level] + (" :orange-badge[Indicative]" if c.note.startswith("Indicative") else "")


def mg(x: float, signed: bool = False) -> str:
    return f"{x * 1000:+.2f} mG" if signed else f"{x * 1000:.2f} mG"


def sidebar() -> None:
    s = app()
    with st.sidebar:
        st.caption("SkillCorner open data · A-League 2024/25 · 20 matches · mG = milli-goals")
        with st.expander("Data status", expanded=bool(s.res.issues)):
            st.text_input("Results folder", key="results_dir", value=results_dir(),
                          help="Folder with the pipeline outputs (searched one level deep).")
            for k, p in s.res.paths.items():
                st.markdown(f"✓ `{p.name}`")
            for issue in s.res.issues:
                st.markdown(f"✗ {issue}")
