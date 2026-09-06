"""Non-interactive smoke test for calibrate_thresholds.py's core logic.

Tests:
- compute_rates returns sensible values for trivial score arrays
- build_table produces correct shape and expected current-marker
- patch_config round-trips correctly (uses a temp copy of config.py)
"""
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.calibrate_thresholds import compute_rates, build_table, patch_config, CONFIG_PATH


# ---------------------------------------------------------------------------
# compute_rates
# ---------------------------------------------------------------------------

def _mock_data():
    """10 real (y=0) and 10 synthetic (y=1) with controlled scores."""
    # real: all score 0.1 (far below any low_max ≥ 0.2)
    # synthetic: half score 0.5, half score 0.9
    real_scores  = np.full(10, 0.1)
    synth_scores = np.array([0.5]*5 + [0.9]*5)
    scores = np.concatenate([real_scores, synth_scores])
    y_true = np.array([0]*10 + [1]*10, dtype=np.int8)
    return scores, y_true


def test_compute_rates_no_false_alarms():
    """Real clips at 0.1 with low_max=0.2 → all real stay in 'low' → zero false alarms."""
    scores, y_true = _mock_data()
    r = compute_rates(scores, y_true, low_max=0.2, medium_max=0.8)
    assert r["fpr_medium_high"] == pytest.approx(0.0)
    assert r["fpr_high"] == pytest.approx(0.0)


def test_compute_rates_no_misses_at_tight_boundary():
    """All synthetic > 0.4 with medium_max=0.4 → no synthetic left in 'low' or 'medium'."""
    scores = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    y_true = np.array([0]*10 + [1]*10, dtype=np.int8)
    r = compute_rates(scores, y_true, low_max=0.2, medium_max=0.4)
    assert r["fnr_low"] == pytest.approx(0.0)
    assert r["tpr_high"] == pytest.approx(1.0)


def test_compute_rates_boundary_inclusive_on_higher_band():
    """Score exactly at low_max → 'medium', not 'low' (convention from bands.py)."""
    scores = np.array([0.3, 0.3])  # exactly at low_max=0.3
    y_true = np.array([1, 1], dtype=np.int8)  # both synthetic
    r = compute_rates(scores, y_true, low_max=0.3, medium_max=0.7)
    # should NOT be classified as 'low' — boundary belongs to 'medium'
    assert r["fnr_low"] == pytest.approx(0.0)


def test_compute_rates_all_miss():
    """synthetic clips all score 0.0 → all land in 'low' → FNR_low = 1.0."""
    scores = np.array([0.0, 0.0, 0.0])
    y_true = np.array([1, 1, 1], dtype=np.int8)
    r = compute_rates(scores, y_true, low_max=0.3, medium_max=0.7)
    assert r["fnr_low"] == pytest.approx(1.0)
    assert r["tpr_high"] == pytest.approx(0.0)


def test_compute_rates_split_synthetic():
    """5 synth at 0.5 (medium), 5 synth at 0.9 (high) with thresholds 0.3/0.7."""
    scores, y_true = _mock_data()
    r = compute_rates(scores, y_true, low_max=0.3, medium_max=0.7)
    # 0/10 synthetic missed as low
    assert r["fnr_low"] == pytest.approx(0.0)
    # 5/10 synthetic reach high
    assert r["tpr_high"] == pytest.approx(0.5)
    # 0/10 real above 0.3
    assert r["fpr_medium_high"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_table
# ---------------------------------------------------------------------------

def test_build_table_shape():
    scores, y_true = _mock_data()
    df = build_table(scores, y_true, current_low_max=0.30, current_medium_max=0.70)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8  # one row per CANDIDATE_PAIRS entry
    assert "FNR_low(miss%)" in df.columns
    assert "FPR_flag(FA%)" in df.columns


def test_build_table_current_marker():
    scores, y_true = _mock_data()
    df = build_table(scores, y_true, current_low_max=0.30, current_medium_max=0.70)
    marked = df[df["current"] != ""]
    assert len(marked) == 1
    assert float(marked.iloc[0]["low_max"]) == pytest.approx(0.30)
    assert float(marked.iloc[0]["medium_max"]) == pytest.approx(0.70)


def test_build_table_no_spurious_markers():
    scores, y_true = _mock_data()
    # Use a pair NOT in CANDIDATE_PAIRS as current
    df = build_table(scores, y_true, current_low_max=0.99, current_medium_max=0.999)
    assert (df["current"] == "").all()


# ---------------------------------------------------------------------------
# patch_config (uses a temp copy — never touches the real config.py)
# ---------------------------------------------------------------------------

def test_patch_config_round_trips():
    """patch_config should update the dict values and leave the rest unchanged."""
    original_text = CONFIG_PATH.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_config = Path(tmpdir) / "config.py"
        shutil.copy(CONFIG_PATH, tmp_config)

        # Temporarily redirect CONFIG_PATH
        import scripts.calibrate_thresholds as ct
        original_config_path = ct.CONFIG_PATH
        ct.CONFIG_PATH = tmp_config

        try:
            ct.patch_config(0.25, 0.65)
            patched = tmp_config.read_text(encoding="utf-8")

            assert '"low_max": 0.25' in patched
            assert '"medium_max": 0.65' in patched
            # Ensure nothing else was clobbered
            assert "RISK_THRESHOLDS" in patched
            assert "SAMPLE_RATE" in patched
        finally:
            ct.CONFIG_PATH = original_config_path

    # Confirm the real config.py is untouched
    assert CONFIG_PATH.read_text(encoding="utf-8") == original_text


def test_patch_config_idempotent():
    """Patching to current values should produce a file with those same values."""
    from voxguard import config

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_config = Path(tmpdir) / "config.py"
        shutil.copy(CONFIG_PATH, tmp_config)

        import scripts.calibrate_thresholds as ct
        original_config_path = ct.CONFIG_PATH
        ct.CONFIG_PATH = tmp_config

        try:
            current_low = float(config.RISK_THRESHOLDS["low_max"])
            current_med = float(config.RISK_THRESHOLDS["medium_max"])
            ct.patch_config(current_low, current_med)
            patched = tmp_config.read_text(encoding="utf-8")
            assert f'"low_max": {current_low}' in patched
            assert f'"medium_max": {current_med}' in patched
        finally:
            ct.CONFIG_PATH = original_config_path
