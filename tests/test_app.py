import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(Path(__file__).parent))

from app_fixtures import write  # noqa: E402
from lib import claims, data, profiles, state  # noqa: E402


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return write(tmp_path_factory.mktemp("results"))


def test_missing_files_are_issues_not_crashes(tmp_path):
    write(tmp_path, options=False)
    (tmp_path / "clips.csv").unlink()
    res = data.load_results(tmp_path)
    assert res.has("players") and not res.has("options") and not res.has("clips")
    assert any("option_values" in i for i in res.issues) and any("clips.csv" in i for i in res.issues)
    s = state.build(res)
    assert "execution" not in s.claims and len(s.profiles)


def test_results_found_one_level_down(tmp_path, results):
    sub = tmp_path / "fingerprint"
    sub.mkdir()
    (sub / "player_fingerprint.csv").write_bytes((results / "player_fingerprint.csv").read_bytes())
    assert data.find(tmp_path, "player_fingerprint.csv") == sub / "player_fingerprint.csv"


def test_revalue_matches_pipeline_at_default_weights(results):
    o = data.load_results(results)["options"]
    r = data.revalue(o, 1.0, 1.0)
    assert np.allclose(r["V"], o["V"]) and (r["is_best"] == o["is_best"]).all()
    no_fear = data.revalue(o, 1.0, 0.0)
    assert (no_fear["V"] >= o["V"] - 1e-12).all()


def test_claim_levels_follow_thresholds():
    assert [claims.level(x) for x in (0.55, 0.5, 0.32, 0.29, np.nan)] == ["rank", "rank", "tier", "low", "low"]


def test_estimates_respect_reliability_and_sample_size():
    sure = profiles.estimate(0.9, 0.95, 100, 100)
    noisy = profiles.estimate(0.9, 0.2, 100, 100)
    few = profiles.estimate(0.9, 0.55, 40, 100)
    many = profiles.estimate(0.9, 0.55, 250, 100)
    assert abs(sure.estimate - 0.9) < abs(noisy.estimate - 0.9)
    assert 0.5 < noisy.estimate < sure.estimate
    assert (noisy.hi - noisy.lo) > (sure.hi - sure.lo)
    assert (few.hi - few.lo) > (many.hi - many.lo)
    assert sure.lo <= sure.estimate <= sure.hi


def test_state_builds_profiles_and_execution(results):
    s = state.build(data.load_results(results))
    assert "execution" in s.claims
    assert set(s.profiles["metric"]) >= {"provider", "decision_quality", "execution", "forwardness", "risk"}
    assert s.profiles[["estimate", "lo", "hi"]].dropna().pipe(lambda d: ((d.lo <= d.estimate) & (d.estimate <= d.hi)).all())


def test_clips_link_to_decisions(results):
    res = data.load_results(results)
    c = data.link_clips(res["clips"], res["options"])
    missed = c[c["kind"] == "missed_option"]
    assert missed["decision_id"].notna().all()
    assert c.loc[c["match_id"] == 99, "decision_id"].isna().all()


@pytest.mark.skipif(not os.environ.get("BUILDUP_RESULTS"), reason="set BUILDUP_RESULTS to the real results folder")
def test_real_results_keep_the_agreed_claims():
    """Provider ranks, decision quality is tiers, creator is team-level. A model change that moves these should be a decision."""
    c = claims.claims(data.load_results(os.environ["BUILDUP_RESULTS"])["reliability"])
    assert (c["provider"].level, c["decision_quality"].level, c["creator"].level) == ("rank", "tier", "low")


PAGES = ["overview", "players", "squad", "rankings", "teams", "replay", "situations", "clips", "method"]


@pytest.mark.parametrize("page", PAGES)
def test_pages_render(results, page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP / "views" / f"{page}.py"), default_timeout=60)
    at.session_state["results_dir"] = str(results)
    sys.path.insert(0, str(APP))
    at.run()
    assert not at.exception, [e.message for e in at.exception]
