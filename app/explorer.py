"""Build-Up Explorer: player profiles, team styles, and every passing decision on a pitch.

Run (after scripts/option_value.py, scripts/fingerprint.py and scripts/clips.py):
    streamlit run app/explorer.py -- --results outputs

Layout:
    lib/data.py      finds and checks the pipeline outputs, derives tables (option revaluation, execution, clips)
    lib/claims.py    what each metric may claim (rank / tiers / low confidence), from its reliability
    lib/profiles.py  percentile estimates with 80% ranges, risk quadrants
    lib/state.py     everything the pages share, built once
    lib/charts.py    figures;  lib/ui.py  Streamlit glue
    views/           one file per page
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from lib import ui  # noqa: E402

st.set_page_config(page_title="Build-Up Explorer", page_icon="⚽", layout="wide")

pages = {
    "": [st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True)],
    "Players": [st.Page("views/players.py", title="Player profile", icon=":material/person:"),
                st.Page("views/squad.py", title="Squad overview", icon=":material/groups:"),
                st.Page("views/rankings.py", title="Rankings", icon=":material/leaderboard:")],
    "Teams and passes": [st.Page("views/teams.py", title="Team styles", icon=":material/scatter_plot:"),
                         st.Page("views/replay.py", title="Pass replay", icon=":material/sports_soccer:"),
                         st.Page("views/situations.py", title="Situations", icon=":material/compress:"),
                         st.Page("views/clips.py", title="Clips", icon=":material/movie:")],
    "Method": [st.Page("views/method.py", title="How reliable is this?", icon=":material/verified:")],
}
nav = st.navigation(pages)
ui.sidebar()
nav.run()
