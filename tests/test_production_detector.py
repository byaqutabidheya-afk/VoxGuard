"""Tests for build_production_detector / verify_production_classifiers (Phase F4 prep).

No model weights are loaded: ``WeightedAverageDetector`` is replaced by a recorder, and the production
classifier files are tiny placeholders under ``tmp_path`` (``config.BASE_DIR`` and the two
``PRODUCTION_*_CLASSIFIERS`` dicts are monkeypatched), so the tests are independent of this machine's
``models/`` directory (which is not under version control).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxguard import config
from voxguard.classifier import ensemble
from voxguard.classifier.ensemble import build_production_detector, verify_production_classifiers

WHOLECLIP = {
    "wav2vec2": "models/classifiers/w2v_whole.joblib",
    "wavlm": "models/classifiers/wavlm_whole_v2.joblib",
}
STREAMING = {
    "wav2vec2": "models/classifiers/w2v_chunk.joblib",
    "wavlm": "models/classifiers/wavlm_chunk_v2.joblib",
}


def _make_head(base: Path, rel: str, *, sidecar: bool = True, scaler: bool = True) -> None:
    model = base / rel
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"x")
    scaler_path = model.with_name(f"{model.stem}_scaler.joblib")
    if sidecar:
        model.with_suffix(".json").write_text(json.dumps({"type": "logreg", "input_dim": 768, "scaler_path": scaler_path.name}))
    if scaler:
        scaler_path.write_bytes(b"x")


@pytest.fixture
def prod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """All four heads present under tmp_path; config points at them; the detector class is a recorder."""
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config, "PRODUCTION_WHOLECLIP_CLASSIFIERS", dict(WHOLECLIP))
    monkeypatch.setattr(config, "PRODUCTION_STREAMING_CLASSIFIERS", dict(STREAMING))
    monkeypatch.setattr(config, "PRODUCTION_ENSEMBLE_WEIGHT_A", 0.37)
    for rel in (*WHOLECLIP.values(), *STREAMING.values()):
        _make_head(tmp_path, rel)

    built: list[dict] = []

    class _Recorder:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    monkeypatch.setattr(ensemble, "WeightedAverageDetector", _Recorder)
    return tmp_path, built


# --------------------------------------------------------------------------- wiring


def test_wholeclip_is_the_default_mode_and_uses_wholeclip_heads(prod) -> None:
    base, built = prod
    detector = build_production_detector()
    assert isinstance(detector, ensemble.WeightedAverageDetector)
    assert built[0]["wav2vec2_classifier_path"] == base / WHOLECLIP["wav2vec2"]
    assert built[0]["wavlm_classifier_path"] == base / WHOLECLIP["wavlm"]


def test_streaming_mode_uses_streaming_heads(prod) -> None:
    base, built = prod
    build_production_detector("streaming")
    assert built[0]["wav2vec2_classifier_path"] == base / STREAMING["wav2vec2"]
    assert built[0]["wavlm_classifier_path"] == base / STREAMING["wavlm"]


@pytest.mark.parametrize("mode", ["wholeclip", "streaming"])
def test_weight_comes_from_config_not_a_literal(prod, mode: str) -> None:
    _, built = prod
    build_production_detector(mode)
    assert built[0]["weight_a"] == 0.37


def test_config_is_read_at_call_time(prod, monkeypatch: pytest.MonkeyPatch) -> None:
    base, built = prod
    _make_head(base, "models/classifiers/other.joblib")
    monkeypatch.setattr(config, "PRODUCTION_WHOLECLIP_CLASSIFIERS", {"wav2vec2": "models/classifiers/other.joblib", "wavlm": WHOLECLIP["wavlm"]})
    build_production_detector("wholeclip")
    assert built[0]["wav2vec2_classifier_path"] == base / "models/classifiers/other.joblib"


# --------------------------------------------------------------------------- unknown mode


@pytest.mark.parametrize("bad", ["whole_clip", "WHOLECLIP", "", "stream", "both"])
def test_unknown_mode_raises_clear_value_error(prod, bad: str) -> None:
    _, built = prod
    with pytest.raises(ValueError, match="Unknown production detector mode"):
        build_production_detector(bad)
    assert built == []                      # nothing was constructed


def test_unknown_mode_in_verify_raises_value_error(prod) -> None:
    with pytest.raises(ValueError, match="Unknown production detector mode"):
        verify_production_classifiers(["wholeclip", "nope"])


def test_malformed_config_entry_raises_value_error(prod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PRODUCTION_STREAMING_CLASSIFIERS", {"wav2vec2": STREAMING["wav2vec2"]})
    with pytest.raises(ValueError, match="wavlm"):
        build_production_detector("streaming")


# --------------------------------------------------------------------------- existence check


def test_verify_passes_and_returns_paths_when_everything_exists(prod) -> None:
    base, _ = prod
    resolved = verify_production_classifiers()
    assert set(resolved) == {"wholeclip", "streaming"}
    assert resolved["streaming"]["wavlm"] == base / STREAMING["wavlm"]


def test_missing_model_file_is_named(prod) -> None:
    base, built = prod
    (base / WHOLECLIP["wavlm"]).unlink()
    with pytest.raises(FileNotFoundError) as exc:
        build_production_detector("wholeclip")
    msg = str(exc.value)
    assert str(base / WHOLECLIP["wavlm"]) in msg
    assert "PRODUCTION_WHOLECLIP_CLASSIFIERS" in msg and "'wavlm'" in msg
    assert built == []                      # failed BEFORE any detector (and so any backbone) was built


def test_missing_sidecar_and_scaler_are_detected_too(prod) -> None:
    base, _ = prod
    (base / STREAMING["wav2vec2"]).with_suffix(".json").unlink()
    (base / "models/classifiers/wavlm_chunk_v2_scaler.joblib").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        verify_production_classifiers(["streaming"])
    msg = str(exc.value)
    assert "w2v_chunk.json" in msg and "wavlm_chunk_v2_scaler.joblib" in msg


def test_every_missing_path_is_listed_not_just_the_first(prod) -> None:
    base, _ = prod
    (base / WHOLECLIP["wav2vec2"]).unlink()
    (base / STREAMING["wavlm"]).unlink()
    with pytest.raises(FileNotFoundError) as exc:
        verify_production_classifiers()
    msg = str(exc.value)
    assert WHOLECLIP["wav2vec2"].split("/")[-1] in msg and STREAMING["wavlm"].split("/")[-1] in msg


def test_verify_defaults_to_checking_both_families(prod) -> None:
    base, _ = prod
    (base / STREAMING["wav2vec2"]).unlink()
    with pytest.raises(FileNotFoundError, match="PRODUCTION_STREAMING_CLASSIFIERS"):
        verify_production_classifiers()


def test_build_checks_only_its_own_mode(prod) -> None:
    """A missing streaming head must not block building the whole-clip detector, and vice versa."""
    base, built = prod
    (base / STREAMING["wavlm"]).unlink()
    build_production_detector("wholeclip")                    # fine
    assert len(built) == 1
    with pytest.raises(FileNotFoundError):
        build_production_detector("streaming")
    assert len(built) == 1


def test_importing_config_does_no_file_check() -> None:
    """The check must not run at import time: config is imported by scripts/tests on machines with no models/."""
    import subprocess
    import sys

    code = (
        "import sys, types\n"
        "from voxguard import config\n"
        "assert 'voxguard.classifier.ensemble' not in sys.modules, 'config must not import the detector module'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --------------------------------------------------------------------------- shared accessor


def test_get_production_detector_builds_once_per_mode_and_shares(prod) -> None:
    _, built = prod
    ensemble.get_production_detector.cache_clear()
    try:
        a1 = ensemble.get_production_detector("streaming")
        a2 = ensemble.get_production_detector("streaming")
        b = ensemble.get_production_detector("wholeclip")
    finally:
        ensemble.get_production_detector.cache_clear()
    assert a1 is a2 and a1 is not b
    assert len(built) == 2                   # one build per mode


def test_get_production_detector_does_not_cache_a_failed_build(prod) -> None:
    base, built = prod
    model = base / STREAMING["wavlm"]
    saved = model.read_bytes()
    model.unlink()
    ensemble.get_production_detector.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            ensemble.get_production_detector("streaming")
        model.write_bytes(saved)             # fixed on disk -> next call succeeds (the failure was not cached)
        assert ensemble.get_production_detector("streaming") is not None
    finally:
        ensemble.get_production_detector.cache_clear()
