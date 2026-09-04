"""
prosody.py — handcrafted prosodic/behavioral feature extraction (Phase 2).

Computes a fixed 10-dimensional feature vector per audio clip using only
librosa (CPU-only, no model download / GPU needed). These features are cheap
enough to run per-chunk later in Phase 5's real-time streaming path, which is
why F0 estimation uses librosa.yin rather than the more accurate but
substantially slower librosa.pyin (see comment on the F0 call below).
"""

from __future__ import annotations

import sys

import librosa
import numpy as np

from voxguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProsodyFeatureExtractor:
    """Extracts a fixed 10-dim handcrafted prosody/energy feature vector per clip.

    All features are computed from short-time frames at a single, shared
    frame/hop configuration (``FRAME_LENGTH`` / ``HOP_LENGTH``) so that the
    per-frame F0, RMS, and voicing arrays line up index-for-index.
    """

    # Frame configuration shared by F0, RMS, and ZCR extraction so their
    # per-frame arrays are aligned.
    FRAME_LENGTH: int = 2048
    HOP_LENGTH: int = 512

    # Plausible human speech F0 range (Hz). librosa.yin has no built-in
    # voicing decision — unvoiced/silent frames tend to clamp to this
    # search range's boundary, which we use below as one voicing signal.
    F0_MIN: float = 50.0
    F0_MAX: float = 500.0

    # A frame counts as "silence/pause" if its RMS is below this fraction of
    # the clip's own peak RMS — loudness-invariant, unlike a fixed dB cutoff.
    SILENCE_RMS_RATIO: float = 0.05

    FEATURE_NAMES: list[str] = [
        "f0_mean_hz",
        "f0_std_hz",
        "f0_range_hz",
        "voiced_fraction",
        "f0_jitter_hz",
        "pause_ratio",
        "speaking_rate_onsets_per_sec",
        "rms_mean",
        "rms_std",
        "zcr_mean",
    ]

    def extract(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Computes the 10-dim prosody feature vector for one clip.

        Parameters
        ----------
        waveform:
            1-D array of audio samples.
        sr:
            Sample rate of *waveform* in Hz.

        Returns
        -------
        np.ndarray, shape ``(10,)``, dtype ``float32``
            Feature values in the fixed order given by ``FEATURE_NAMES``.
            If no pitch can be detected at all (e.g. a near-silent clip),
            the F0-derived features (indices 0, 1, 2, 4) are returned as
            ``0.0`` and a warning is logged, rather than raising.
        """
        waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.size == 0:
            logger.warning("Empty waveform passed to ProsodyFeatureExtractor; returning zero vector.")
            return np.zeros(len(self.FEATURE_NAMES), dtype=np.float32)

        duration_s = len(waveform) / sr if sr > 0 else 0.0

        # --- F0 estimation ---------------------------------------------------
        # librosa.yin (time-domain autocorrelation-style estimator) instead of
        # librosa.pyin: yin is markedly faster and non-probabilistic (no HMM
        # smoothing over pitch candidates), which is the right speed/accuracy
        # tradeoff here since this same extractor will run per-chunk in
        # Phase 5's real-time streaming path, where pyin's extra latency
        # would eat into the streaming budget for a prosody signal that is
        # only ever used as a coarse behavioral cue, not a precise pitch track.
        f0 = librosa.yin(
            waveform,
            fmin=self.F0_MIN,
            fmax=self.F0_MAX,
            sr=sr,
            frame_length=self.FRAME_LENGTH,
            hop_length=self.HOP_LENGTH,
        )

        rms = librosa.feature.rms(
            y=waveform, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
        )[0]

        # yin/rms frame counts can differ by one near the clip boundary; align.
        n_frames = min(len(f0), len(rms))
        f0 = f0[:n_frames]
        rms = rms[:n_frames]

        # --- Voicing decision --------------------------------------------------
        # yin has no built-in voiced/unvoiced flag (unlike pyin's voiced_flag):
        # it always emits a value in [F0_MIN, F0_MAX]. We treat a frame as
        # voiced only if it also carries real energy (not silence/pause) and
        # its estimate isn't pinned to the search-range boundary, which is
        # yin's typical behavior on unvoiced/silent input.
        peak_rms = float(rms.max()) if rms.size else 0.0
        silence_threshold = self.SILENCE_RMS_RATIO * peak_rms
        energetic = rms > silence_threshold
        not_boundary = (f0 > self.F0_MIN * 1.01) & (f0 < self.F0_MAX * 0.99)
        voiced_mask = energetic & not_boundary & ~np.isnan(f0)

        voiced_f0 = f0[voiced_mask]

        if voiced_f0.size == 0:
            logger.warning("No pitch detected in clip; returning 0.0 for F0-derived features.")
            f0_mean = f0_std = f0_range = jitter = 0.0
        else:
            f0_mean = float(voiced_f0.mean())
            f0_std = float(voiced_f0.std())
            f0_range = float(voiced_f0.max() - voiced_f0.min())
            # Frame-to-frame diff over the voiced-only sequence — a simple
            # microvariation proxy, not a strict consecutive-frame jitter
            # measure (voiced frames separated by an unvoiced gap are still
            # adjacent in this sequence).
            jitter = float(np.mean(np.abs(np.diff(voiced_f0)))) if voiced_f0.size > 1 else 0.0

        voiced_fraction = float(voiced_mask.mean()) if voiced_mask.size else 0.0
        pause_ratio = float((~energetic).mean()) if energetic.size else 0.0

        onset_frames = librosa.onset.onset_detect(y=waveform, sr=sr, hop_length=self.HOP_LENGTH)
        speaking_rate = float(len(onset_frames) / duration_s) if duration_s > 0 else 0.0

        rms_mean = float(rms.mean()) if rms.size else 0.0
        rms_std = float(rms.std()) if rms.size else 0.0

        zcr = librosa.feature.zero_crossing_rate(
            waveform, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
        )[0]
        zcr_mean = float(zcr.mean()) if zcr.size else 0.0

        return np.array(
            [
                f0_mean,
                f0_std,
                f0_range,
                voiced_fraction,
                jitter,
                pause_ratio,
                speaking_rate,
                rms_mean,
                rms_std,
                zcr_mean,
            ],
            dtype=np.float32,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m voxguard.features.prosody <audio_file>")
        sys.exit(1)

    from voxguard.config import SAMPLE_RATE
    from voxguard.utils.audio_io import load_audio

    waveform, sr = load_audio(sys.argv[1], target_sr=SAMPLE_RATE)

    extractor = ProsodyFeatureExtractor()
    vector = extractor.extract(waveform, sr)

    print(f"Prosody features for: {sys.argv[1]}")
    for name, value in zip(ProsodyFeatureExtractor.FEATURE_NAMES, vector):
        print(f"  {name:<32} {value:.4f}")
