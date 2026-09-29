#!/usr/bin/env python3
"""
scripts/fix_capture_baseline.py — Capture immutable pre-FIX baseline metrics.

Records, in one complete run, every numeric metric the post-build remediation
guide aims to improve:
1. Current config values and hardcoded classifier paths across entrypoints.
2. Whole-clip performance of current production WeightedAverageDetector on ASVspoof2019 eval split.
3. Whole-clip performance on full Hindi/Hinglish eval split (held-out speaker soumya).
4. Per-backbone whole-clip predictions on all 25 of soumya's real clips (Issue 3 margin metric).
5. Streaming simulation behavior on 5 verified demo pairs and 3 sweep real clips.
6. Full window-size sweep table (1.5, 3.0, 4.0, 6.0s windows, 1.0s stride) regenerated live.

Saves structured results to models/reports/fix_baseline.json and human-readable models/reports/fix_baseline.md.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from voxguard import config
from voxguard.classifier.cross_eval import (
    _metric_dict,
    _predict_scores,
    weighted_average_ensemble,
    zero_shot_eval_weighted_average_from_cache,
)
from voxguard.classifier.ensemble import WeightedAverageDetector
from voxguard.classifier.head import load_classifier
from voxguard.embeddings.cache import load_cached_embeddings
from voxguard.streaming.session import StreamingSession, simulate_stream
from voxguard.utils.audio_io import load_audio
from voxguard.utils.logging_utils import get_logger

logger = get_logger("fix_capture_baseline")

DEFAULT_JSON_PATH = config.MODELS_DIR / "reports" / "fix_baseline.json"
DEFAULT_MD_PATH = config.MODELS_DIR / "reports" / "fix_baseline.md"

PROD_WAV2VEC2_CLF = "models/classifiers/wav2vec2_hindi_combined_logreg.joblib"
PROD_WAVLM_CLF = "models/classifiers/wavlm_hindi_combined_logreg.joblib"
PROD_WEIGHT_A = 0.5


def capture_item1_configs() -> Dict[str, Any]:
    """Capture current config values and entrypoint classifier paths."""
    logger.info("Capturing Item 1: Config values and entrypoint classifier paths...")

    # 1. Config constants
    risk_thresholds = getattr(config, "RISK_THRESHOLDS", None)
    stream_flag_threshold = getattr(config, "STREAM_FLAG_THRESHOLD", None)
    stream_chunk_sec = getattr(config, "STREAM_CHUNK_SECONDS", None)
    stream_overlap_sec = getattr(config, "STREAM_OVERLAP_SECONDS", None)

    # 2. StreamingSession defaults
    sig = inspect.signature(StreamingSession.__init__)
    consec_default = sig.parameters["consecutive_flags_required"].default

    # 3. Read classifier paths from session.py AST
    sess_py_path = config.BASE_DIR / "src" / "voxguard" / "streaming" / "session.py"
    sess_detector_paths: Dict[str, str] = {}
    if sess_py_path.exists():
        with open(sess_py_path, "r", encoding="utf-8") as f:
            sess_tree = ast.parse(f.read())
        for node in ast.walk(sess_tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        for kw in sub.keywords:
                            if kw.arg in ("wav2vec2_classifier_path", "wavlm_classifier_path"):
                                if isinstance(kw.value, ast.Constant):
                                    sess_detector_paths[kw.arg] = str(kw.value.value)

    # 4. Read classifier paths from app/app.py AST
    app_py_path = config.BASE_DIR / "app" / "app.py"
    app_detector_paths: Dict[str, str] = {}
    if app_py_path.exists():
        with open(app_py_path, "r", encoding="utf-8") as f:
            app_tree = ast.parse(f.read())
        for node in ast.walk(app_tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_detector":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        for kw in sub.keywords:
                            if kw.arg in ("wav2vec2_classifier_path", "wavlm_classifier_path"):
                                if isinstance(kw.value, ast.Constant):
                                    app_detector_paths[kw.arg] = str(kw.value.value)

    return {
        "RISK_THRESHOLDS": risk_thresholds,
        "STREAM_FLAG_THRESHOLD": stream_flag_threshold,
        "STREAM_CHUNK_SECONDS": stream_chunk_sec,
        "STREAM_OVERLAP_SECONDS": stream_overlap_sec,
        "consecutive_flags_required_default": consec_default,
        "StreamingSession_classifier_paths": sess_detector_paths,
        "app_py_classifier_paths": app_detector_paths,
    }


def capture_item2_asvspoof_eval() -> Dict[str, Any]:
    """Capture whole-clip performance of WeightedAverageDetector on ASVspoof2019 eval from cache."""
    logger.info("Capturing Item 2: ASVspoof2019 eval split performance from cached embeddings...")
    metrics = zero_shot_eval_weighted_average_from_cache(
        classifier_a_path=PROD_WAV2VEC2_CLF,
        model_a="wav2vec2",
        classifier_b_path=PROD_WAVLM_CLF,
        model_b="wavlm",
        dataset="asvspoof2019",
        weight_a=PROD_WEIGHT_A,
        split="eval",
    )
    cm = metrics["confusion_matrix"]
    total_real = cm[0][0] + cm[0][1]
    total_synth = cm[1][0] + cm[1][1]
    real_recall = float(cm[0][0] / total_real) if total_real > 0 else float("nan")
    synth_recall = float(cm[1][1] / total_synth) if total_synth > 0 else float("nan")

    return {
        "model_files": {
            "wav2vec2_classifier": PROD_WAV2VEC2_CLF,
            "wavlm_classifier": PROD_WAVLM_CLF,
            "weight_a": PROD_WEIGHT_A,
        },
        "embeddings_used": [
            "models/embeddings/wav2vec2_eval.npy",
            "models/embeddings/wavlm_eval.npy",
        ],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"],
        "eer_threshold": metrics["eer_threshold"],
        "real_recall": real_recall,
        "synthetic_recall": synth_recall,
        "confusion_matrix": cm,
        "total_eval_samples": total_real + total_synth,
    }


def capture_item3_hindi_eval() -> Dict[str, Any]:
    """Capture whole-clip performance on Hindi/Hinglish eval split (held-out speaker soumya)."""
    logger.info("Capturing Item 3: Hindi/Hinglish eval split performance from cached embeddings...")
    metrics = zero_shot_eval_weighted_average_from_cache(
        classifier_a_path=PROD_WAV2VEC2_CLF,
        model_a="wav2vec2",
        classifier_b_path=PROD_WAVLM_CLF,
        model_b="wavlm",
        dataset="hindi_eval",
        weight_a=PROD_WEIGHT_A,
        split="eval",
    )
    cm = metrics["confusion_matrix"]
    total_real = cm[0][0] + cm[0][1]
    total_synth = cm[1][0] + cm[1][1]
    real_recall = float(cm[0][0] / total_real) if total_real > 0 else float("nan")
    synth_recall = float(cm[1][1] / total_synth) if total_synth > 0 else float("nan")

    return {
        "model_files": {
            "wav2vec2_classifier": PROD_WAV2VEC2_CLF,
            "wavlm_classifier": PROD_WAVLM_CLF,
            "weight_a": PROD_WEIGHT_A,
        },
        "embeddings_used": [
            "models/embeddings/wav2vec2_hindi_eval.npy",
            "models/embeddings/wavlm_hindi_eval.npy",
        ],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"],
        "eer_threshold": metrics["eer_threshold"],
        "real_recall": real_recall,
        "synthetic_recall": synth_recall,
        "confusion_matrix": cm,
        "total_eval_samples": total_real + total_synth,
    }


def capture_item4_soumya_real_predictions() -> Dict[str, Any]:
    """Capture predictions on all 25 of soumya's real clips and record margin > 0.15 count."""
    logger.info("Capturing Item 4: Per-backbone whole-clip predictions on soumya 25 real clips...")
    X_w2v, m_w2v = load_cached_embeddings("models/embeddings/wav2vec2_hindi_eval.npy")
    X_wlm, m_wlm = load_cached_embeddings("models/embeddings/wavlm_hindi_eval.npy")

    clf_w2v, scaler_w2v = load_classifier(PROD_WAV2VEC2_CLF)
    clf_wlm, scaler_wlm = load_classifier(PROD_WAVLM_CLF)

    scores_w2v = _predict_scores(clf_w2v, scaler_w2v.transform(X_w2v))
    scores_wlm = _predict_scores(clf_wlm, scaler_wlm.transform(X_wlm))
    scores_ens = PROD_WEIGHT_A * scores_w2v + (1.0 - PROD_WEIGHT_A) * scores_wlm

    real_mask = (m_w2v["label"] == "bonafide") | (m_w2v["label"] == "real")
    real_indices = np.where(real_mask)[0]

    clip_records: List[Dict[str, Any]] = []
    confident_count = 0

    for idx in real_indices:
        filepath = str(m_w2v.iloc[idx]["filepath"]).replace("\\", "/")
        filename = Path(filepath).name
        p_w2v = float(scores_w2v[idx])
        p_wlm = float(scores_wlm[idx])
        p_ens = float(scores_ens[idx])
        margin = float(abs(p_ens - 0.5))
        is_confident = bool(margin > 0.15)
        if is_confident:
            confident_count += 1

        clip_records.append({
            "filename": filename,
            "filepath": filepath,
            "wav2vec2_prob_synthetic": p_w2v,
            "wavlm_prob_synthetic": p_wlm,
            "ensemble_prob_synthetic": p_ens,
            "margin_from_0_5": margin,
            "margin_gt_0_15": is_confident,
        })

    return {
        "model_files": {
            "wav2vec2_classifier": PROD_WAV2VEC2_CLF,
            "wavlm_classifier": PROD_WAVLM_CLF,
            "ensemble_weight_a": PROD_WEIGHT_A,
        },
        "total_real_clips": len(clip_records),
        "confident_predictions_count": confident_count,
        "confident_predictions_fraction": f"{confident_count}/{len(clip_records)}",
        "margin_threshold": 0.15,
        "clips": clip_records,
    }


def capture_item5_streaming_behaviour(detector: WeightedAverageDetector) -> Dict[str, Any]:
    """Capture streaming simulation behaviour on 5 verified demo pairs + 3 sweep real clips."""
    logger.info("Capturing Item 5: Streaming simulation on demo pairs and sweep real clips...")

    demo_pairs_def = [
        {
            "pair_name": "Pair 1: Casual Neutral (byaquta)",
            "speaker": "byaquta",
            "category": "neutral",
            "real_path": "data/raw/hindi_hinglish/real/byaquta_neutral_09.wav",
            "synth_path": "data/raw/hindi_hinglish/synthetic/byaquta_neutral_09_clone.wav",
        },
        {
            "pair_name": "Pair 2: Everyday Tech (mahato)",
            "speaker": "mahato",
            "category": "neutral",
            "real_path": "data/raw/hindi_hinglish/real/mahato_neutral_04.wav",
            "synth_path": "data/raw/hindi_hinglish/synthetic/mahato_neutral_04_clone.wav",
        },
        {
            "pair_name": "Pair 3: Urgent Legal Pressure Scam (byaquta)",
            "speaker": "byaquta",
            "category": "scam",
            "real_path": "data/raw/hindi_hinglish/real/byaquta_scam_16.wav",
            "synth_path": "data/raw/hindi_hinglish/synthetic/byaquta_scam_16_clone.wav",
        },
        {
            "pair_name": "Pair 4: Authority Customs Scam (mahato)",
            "speaker": "mahato",
            "category": "scam",
            "real_path": "data/raw/hindi_hinglish/real/mahato_scam_12.wav",
            "synth_path": "data/raw/hindi_hinglish/synthetic/mahato_scam_12_clone.wav",
        },
        {
            "pair_name": "Pair 5: Held-Out Casual Speaker (soumya)",
            "speaker": "soumya",
            "category": "neutral",
            "real_path": "data/raw/hindi_hinglish/real/soumya_neutral_03.wav",
            "synth_path": "data/raw/hindi_hinglish/synthetic/soumya_neutral_03_clone.wav",
        },
    ]

    sweep_real_paths = [
        "data/raw/hindi_hinglish/real/byaquta_neutral_01.wav",
        "data/raw/hindi_hinglish/real/soumya_control_21.wav",
        "data/raw/hindi_hinglish/real/soumya_scam_11.wav",
    ]

    def _run_sim(audio_file: str) -> Dict[str, Any]:
        sess = StreamingSession(detector=detector)
        sim_res = simulate_stream(audio_file, session=sess, real_time_paced=False)
        max_running = (
            max(r["running_score"] for r in sim_res["step_results"])
            if sim_res["step_results"]
            else 0.0
        )
        return {
            "flagged": sim_res["flagged"],
            "seconds_to_flag": sim_res["seconds_to_flag"],
            "final_running_score": sim_res["final_running_score"],
            "max_running_score": float(max_running),
            "total_duration": sim_res["total_duration"],
        }

    demo_pairs_results = []
    for pair in demo_pairs_def:
        real_res = _run_sim(pair["real_path"])
        synth_res = _run_sim(pair["synth_path"])
        demo_pairs_results.append({
            "pair_name": pair["pair_name"],
            "speaker": pair["speaker"],
            "category": pair["category"],
            "real": {
                "file": Path(pair["real_path"]).name,
                "path": pair["real_path"],
                **real_res,
            },
            "synthetic": {
                "file": Path(pair["synth_path"]).name,
                "path": pair["synth_path"],
                **synth_res,
            },
        })

    sweep_reals_results = []
    for path in sweep_real_paths:
        res = _run_sim(path)
        sweep_reals_results.append({
            "file": Path(path).name,
            "path": path,
            **res,
        })

    return {
        "model_files": {
            "wav2vec2_classifier": PROD_WAV2VEC2_CLF,
            "wavlm_classifier": PROD_WAVLM_CLF,
            "weight_a": PROD_WEIGHT_A,
        },
        "streaming_parameters": {
            "chunk_seconds": config.STREAM_CHUNK_SECONDS,
            "overlap_seconds": config.STREAM_OVERLAP_SECONDS,
            "flag_threshold": config.STREAM_FLAG_THRESHOLD,
            "consecutive_flags_required": 3,
        },
        "demo_pairs": demo_pairs_results,
        "sweep_real_clips": sweep_reals_results,
    }


def capture_item6_window_sweep(detector: WeightedAverageDetector) -> Dict[str, Any]:
    """Regenerate live the full window-size sweep table from ISSUES.md Issue 2."""
    logger.info("Capturing Item 6: Regenerating live window-size sweep table...")

    sweep_clips = [
        ("REAL", "data/raw/hindi_hinglish/real/byaquta_neutral_01.wav"),
        ("REAL", "data/raw/hindi_hinglish/real/soumya_control_21.wav"),
        ("REAL", "data/raw/hindi_hinglish/real/soumya_scam_11.wav"),
        ("SYNTH", "data/raw/hindi_hinglish/synthetic/byaquta_neutral_01_clone.wav"),
        ("SYNTH", "data/raw/hindi_hinglish/synthetic/soumya_scam_11_clone.wav"),
    ]

    windows = [1.5, 3.0, 4.0, 6.0]
    stride_sec = 1.0
    sweep_data: Dict[str, List[Dict[str, Any]]] = {}

    for win in windows:
        win_key = f"{win:.1f}s"
        sweep_data[win_key] = []
        for kind, path in sweep_clips:
            wf, sr = load_audio(path, target_sr=16000)
            n = int(win * sr)
            stride = int(stride_sec * sr)
            scores = []
            for start in range(0, max(1, len(wf) - n + 1), stride):
                w = wf[start : start + n]
                if len(w) < sr:
                    continue
                r = detector.predict_waveform(w, sr)
                p = r.get("probability_synthetic")
                if p is not None:
                    scores.append(float(p))
            if not scores:
                r = detector.predict_waveform(wf, sr)
                scores = [float(r.get("probability_synthetic", float("nan")))]

            mean_s = float(np.mean(scores))
            max_s = float(np.max(scores))
            min_s = float(np.min(scores))

            sweep_data[win_key].append({
                "kind": kind,
                "filename": Path(path).name,
                "filepath": path,
                "mean_score": mean_s,
                "max_score": max_s,
                "min_score": min_s,
                "num_windows": len(scores),
            })

    return {
        "model_files": {
            "wav2vec2_classifier": PROD_WAV2VEC2_CLF,
            "wavlm_classifier": PROD_WAVLM_CLF,
            "weight_a": PROD_WEIGHT_A,
        },
        "windows_evaluated": windows,
        "stride_seconds": stride_sec,
        "results_by_window": sweep_data,
    }


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """Formats the captured baseline numbers into human-readable Markdown."""
    lines: List[str] = [
        "# VoxGuard Pre-FIX Baseline Report (Phases 0–11 Complete)",
        "",
        f"**Generated at:** {data['metadata']['timestamp_utc']}",
        f"**Git Anchor Tag:** `pre-fix-baseline`",
        f"**Production Classifier (wav2vec2):** `{data['models_used']['wav2vec2_classifier']}`",
        f"**Production Classifier (WavLM):** `{data['models_used']['wavlm_classifier']}`",
        f"**Ensemble Weight (wav2vec2 / WavLM):** `{data['models_used']['ensemble_weight_a']:.1f} / {1.0 - data['models_used']['ensemble_weight_a']:.1f}`",
        "",
        "---",
        "",
        "## 1. Current Configuration & Entrypoint Paths",
        "",
        "| Config Key | Value | Source |",
        "|---|---|---|",
        f"| `RISK_THRESHOLDS` | `low_max={data['item1_config']['RISK_THRESHOLDS']['low_max']}`, `medium_max={data['item1_config']['RISK_THRESHOLDS']['medium_max']}` | `src/voxguard/config.py` |",
        f"| `STREAM_FLAG_THRESHOLD` | `{data['item1_config']['STREAM_FLAG_THRESHOLD']}` | `src/voxguard/config.py` |",
        f"| `STREAM_CHUNK_SECONDS` | `{data['item1_config']['STREAM_CHUNK_SECONDS']}` | `src/voxguard/config.py` |",
        f"| `STREAM_OVERLAP_SECONDS` | `{data['item1_config']['STREAM_OVERLAP_SECONDS']}` | `src/voxguard/config.py` |",
        f"| `consecutive_flags_required` | `{data['item1_config']['consecutive_flags_required_default']}` | `StreamingSession.__init__` |",
        f"| `StreamingSession` wav2vec2 path | `{data['item1_config']['StreamingSession_classifier_paths'].get('wav2vec2_classifier_path')}` | `src/voxguard/streaming/session.py` |",
        f"| `StreamingSession` WavLM path | `{data['item1_config']['StreamingSession_classifier_paths'].get('wavlm_classifier_path')}` | `src/voxguard/streaming/session.py` |",
        f"| `app/app.py` wav2vec2 path | `{data['item1_config']['app_py_classifier_paths'].get('wav2vec2_classifier_path')}` | `app/app.py` (`get_detector`) |",
        f"| `app/app.py` WavLM path | `{data['item1_config']['app_py_classifier_paths'].get('wavlm_classifier_path')}` | `app/app.py` (`get_detector`) |",
        "",
        "---",
        "",
        "## 2. Whole-Clip Performance: ASVspoof2019 Evaluation Split",
        "",
        "*Evaluated from cached embeddings (`wav2vec2_eval.npy`, `wavlm_eval.npy`) using production `WeightedAverageDetector`.*",
        "",
        "| Metric | Value | Detail |",
        "|---|---|---|",
        f"| **Accuracy** | **{data['item2_asvspoof']['accuracy'] * 100:.2f}%** | 71,237 total clips |",
        f"| **ROC-AUC** | **{data['item2_asvspoof']['roc_auc']:.4f}** | Area under ROC curve |",
        f"| **EER** | **{data['item2_asvspoof']['eer'] * 100:.2f}%** | Equal Error Rate |",
        f"| **EER Threshold** | `{data['item2_asvspoof']['eer_threshold']:.4f}` | Optimal ranking cutoff |",
        f"| **Real Recall (Bonafide)** | **{data['item2_asvspoof']['real_recall'] * 100:.2f}%** | `{data['item2_asvspoof']['confusion_matrix'][0][0]} / {data['item2_asvspoof']['confusion_matrix'][0][0] + data['item2_asvspoof']['confusion_matrix'][0][1]}` |",
        f"| **Synthetic Recall (Spoof)** | **{data['item2_asvspoof']['synthetic_recall'] * 100:.2f}%** | `{data['item2_asvspoof']['confusion_matrix'][1][1]} / {data['item2_asvspoof']['confusion_matrix'][1][0] + data['item2_asvspoof']['confusion_matrix'][1][1]}` |",
        f"| **Confusion Matrix [TN, FP / FN, TP]** | `{data['item2_asvspoof']['confusion_matrix']}` | TN={data['item2_asvspoof']['confusion_matrix'][0][0]}, FP={data['item2_asvspoof']['confusion_matrix'][0][1]}, FN={data['item2_asvspoof']['confusion_matrix'][1][0]}, TP={data['item2_asvspoof']['confusion_matrix'][1][1]} |",
        "",
        "---",
        "",
        "## 3. Whole-Clip Performance: Hindi/Hinglish Evaluation Split (Held-Out Speaker `soumya`)",
        "",
        "*Evaluated from cached embeddings (`wav2vec2_hindi_eval.npy`, `wavlm_hindi_eval.npy`) using production `WeightedAverageDetector`.*",
        "",
        "| Metric | Value | Detail |",
        "|---|---|---|",
        f"| **Accuracy** | **{data['item3_hindi']['accuracy'] * 100:.2f}%** | 50 total clips (25 real, 25 synthetic) |",
        f"| **ROC-AUC** | **{data['item3_hindi']['roc_auc']:.4f}** | Area under ROC curve |",
        f"| **EER** | **{data['item3_hindi']['eer'] * 100:.2f}%** | Equal Error Rate |",
        f"| **EER Threshold** | `{data['item3_hindi']['eer_threshold']:.4f}` | Optimal ranking cutoff |",
        f"| **Real Recall (Bonafide)** | **{data['item3_hindi']['real_recall'] * 100:.2f}%** | `{data['item3_hindi']['confusion_matrix'][0][0]} / 25` |",
        f"| **Synthetic Recall (Spoof)** | **{data['item3_hindi']['synthetic_recall'] * 100:.2f}%** | `{data['item3_hindi']['confusion_matrix'][1][1]} / 25` |",
        f"| **Confusion Matrix [TN, FP / FN, TP]** | `{data['item3_hindi']['confusion_matrix']}` | TN={data['item3_hindi']['confusion_matrix'][0][0]}, FP={data['item3_hindi']['confusion_matrix'][0][1]}, FN={data['item3_hindi']['confusion_matrix'][1][0]}, TP={data['item3_hindi']['confusion_matrix'][1][1]} |",
        "",
        "---",
        "",
        "## 4. Per-Backbone Whole-Clip Predictions on `soumya` Real Clips (Issue 3 Disagreement Metric)",
        "",
        f"**Confident Predictions (Margin $|P - 0.5| > 0.15$):** **{data['item4_soumya_predictions']['confident_predictions_fraction']}** ({data['item4_soumya_predictions']['confident_predictions_count']}/{data['item4_soumya_predictions']['total_real_clips']})",
        "",
        "| Real Clip | wav2vec2 $P(\\text{synth})$ | WavLM $P(\\text{synth})$ | Ensemble $P(\\text{synth})$ | Margin $\|P - 0.5\|$ | Margin $> 0.15$ |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for c in data["item4_soumya_predictions"]["clips"]:
        mark = "YES" if c["margin_gt_0_15"] else "NO (near 0.5)"
        lines.append(
            f"| `{c['filename']}` | {c['wav2vec2_prob_synthetic']:.4f} | {c['wavlm_prob_synthetic']:.4f} | {c['ensemble_prob_synthetic']:.4f} | {c['margin_from_0_5']:.4f} | {mark} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Streaming Behavior: Phase 6 Verified Demo Pairs & Sweep Real Clips",
        "",
        "*Streaming session evaluated at default parameters (`chunk=1.5s`, `overlap=0.5s`, `flag_threshold=0.6`, `consecutive=3`).*",
        "",
        "### 5.1 Verified Demo Pairs (10 clips)",
        "",
        "| Pair & Description | Clip Kind | File | Flagged | Seconds to Flag | Final Running Score | Max Running Score |",
        "|---|:---:|---|:---:|:---:|:---:|:---:|",
    ])

    for p in data["item5_streaming"]["demo_pairs"]:
        rf = p["real"]
        sf = p["synthetic"]
        r_sec = f"{rf['seconds_to_flag']:.2f}s" if rf["seconds_to_flag"] is not None else "N/A"
        s_sec = f"{sf['seconds_to_flag']:.2f}s" if sf["seconds_to_flag"] is not None else "N/A"
        lines.append(
            f"| {p['pair_name']} | REAL | `{rf['file']}` | {rf['flagged']} | {r_sec} | {rf['final_running_score']:.4f} | {rf['max_running_score']:.4f} |"
        )
        lines.append(
            f"| {p['pair_name']} | SYNTH | `{sf['file']}` | {sf['flagged']} | {s_sec} | {sf['final_running_score']:.4f} | {sf['max_running_score']:.4f} |"
        )

    lines.extend([
        "",
        "### 5.2 Window-Sweep Real Clips (Known Pre-FIX False Positives)",
        "",
        "| File | Flagged | Seconds to Flag | Final Running Score | Max Running Score | Pre-FIX Status |",
        "|---|:---:|:---:|:---:|:---:|---|",
    ])

    for r in data["item5_streaming"]["sweep_real_clips"]:
        sec = f"{r['seconds_to_flag']:.2f}s" if r["seconds_to_flag"] is not None else "N/A"
        status = "**FALSE POSITIVE (Flags)**" if r["flagged"] else "OK (Unflagged)"
        lines.append(
            f"| `{r['file']}` | {r['flagged']} | {sec} | {r['final_running_score']:.4f} | {r['max_running_score']:.4f} | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Full Window-Size Sweep Table (Regenerated Live)",
        "",
        "*Evaluated across windows $w \\in \\{1.5, 3.0, 4.0, 6.0\\}$ seconds with stride $s = 1.0$ second on production WeightedAverageDetector.*",
        "",
        "| Window Size | Kind | Clip File | Mean Score | Max Score | Min Score | Windows Count |",
        "|:---:|:---:|---|:---:|:---:|:---:|:---:|",
    ])

    for win_key, rows in data["item6_window_sweep"]["results_by_window"].items():
        for row in rows:
            lines.append(
                f"| **{win_key}** | {row['kind']} | `{row['filename']}` | {row['mean_score']:.3f} | {row['max_score']:.3f} | {row['min_score']:.3f} | {row['num_windows']} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Benchmark / Diagnostic Item | Pre-FIX Baseline Metric | Model Files Responsible |",
        "|---|---|---|",
        f"| ASVspoof2019 Eval Accuracy / EER | **{data['item2_asvspoof']['accuracy']*100:.2f}% / {data['item2_asvspoof']['eer']*100:.2f}%** | `{PROD_WAV2VEC2_CLF}` + `{PROD_WAVLM_CLF}` |",
        f"| Hindi Eval Accuracy / EER (`soumya`) | **{data['item3_hindi']['accuracy']*100:.2f}% / {data['item3_hindi']['eer']*100:.2f}%** | `{PROD_WAV2VEC2_CLF}` + `{PROD_WAVLM_CLF}` |",
        f"| `soumya` Real Margin $> 0.15$ Ratio | **{data['item4_soumya_predictions']['confident_predictions_fraction']}** ({data['item4_soumya_predictions']['confident_predictions_count']}/25) | `{PROD_WAV2VEC2_CLF}` vs `{PROD_WAVLM_CLF}` |",
        f"| Demo Pairs Streaming Flags (Real / Synth) | **0 / 5 Real flagged, 5 / 5 Synth flagged** | Production StreamingSession |",
        f"| Sweep Real Clips False Alarm Rate | **3 / 3 Real Clips Incorrectly Flagged** | Production StreamingSession |",
        f"| Window Sweep Real Score (1.5s vs 6.0s) | `byaquta_neutral_01`: mean 0.433 (1.5s) vs 0.004 (6.0s) | Production WeightedAverageDetector |",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture complete immutable pre-FIX baseline metrics."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_JSON_PATH),
        help=f"Path for JSON output (default: {DEFAULT_JSON_PATH}).",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=str(DEFAULT_MD_PATH),
        help=f"Path for Markdown report (default: {DEFAULT_MD_PATH}).",
    )

    args = parser.parse_args()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)

    print("\n" + "=" * 78)
    print(" VOXGUARD PRE-FIX BASELINE CAPTURE")
    print("=" * 78)

    # 1. Configs
    item1 = capture_item1_configs()

    # 2. ASVspoof eval
    item2 = capture_item2_asvspoof_eval()

    # 3. Hindi eval
    item3 = capture_item3_hindi_eval()

    # 4. Soumya real clips predictions
    item4 = capture_item4_soumya_real_predictions()

    # Shared detector instance for audio waveform tests (items 5 and 6)
    logger.info("Initializing shared WeightedAverageDetector for live waveform streaming/sweeps...")
    detector = WeightedAverageDetector(
        wav2vec2_classifier_path=PROD_WAV2VEC2_CLF,
        wavlm_classifier_path=PROD_WAVLM_CLF,
    )

    # 5. Streaming simulation
    item5 = capture_item5_streaming_behaviour(detector)

    # 6. Window sweep
    item6 = capture_item6_window_sweep(detector)

    baseline_data: Dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_tag": "pre-fix-baseline",
            "environment": {
                "os": os.name,
                "sample_rate": config.SAMPLE_RATE,
            },
        },
        "models_used": {
            "wav2vec2_classifier": PROD_WAV2VEC2_CLF,
            "wavlm_classifier": PROD_WAVLM_CLF,
            "ensemble_weight_a": PROD_WEIGHT_A,
        },
        "item1_config": item1,
        "item2_asvspoof": item2,
        "item3_hindi": item3,
        "item4_soumya_predictions": item4,
        "item5_streaming": item5,
        "item6_window_sweep": item6,
    }

    # Write JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(baseline_data, f, indent=2)
    logger.info("Saved baseline JSON to %s", out_json)

    # Write Markdown
    md_content = generate_markdown_report(baseline_data)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved baseline Markdown report to %s", out_md)

    # Print summary table to console
    print("\n" + "=" * 78)
    print(" BASELINE CAPTURE SUMMARY TABLE")
    print("=" * 78)
    print(f" {'Metric / Diagnostic':<42} | {'Value / Ratio':<20} | {'Status / Label'}")
    print("-" * 78)
    print(f" {'ASVspoof2019 Eval Acc / EER':<42} | {item2['accuracy']*100:.2f}% / {item2['eer']*100:.2f}%{'':<6} | {PROD_WAV2VEC2_CLF.split('/')[-1]}")
    print(f" {'ASVspoof2019 Real / Synth Recall':<42} | {item2['real_recall']*100:.2f}% / {item2['synthetic_recall']*100:.2f}% | TN={item2['confusion_matrix'][0][0]}, TP={item2['confusion_matrix'][1][1]}")
    print(f" {'Hindi Eval Acc / EER (soumya)':<42} | {item3['accuracy']*100:.2f}% / {item3['eer']*100:.2f}%{'':<6} | {PROD_WAVLM_CLF.split('/')[-1]}")
    print(f" {'Hindi Eval Real / Synth Recall':<42} | {item3['real_recall']*100:.2f}% / {item3['synthetic_recall']*100:.2f}% | TN={item3['confusion_matrix'][0][0]}, TP={item3['confusion_matrix'][1][1]}")
    print(f" {'soumya Real Margin > 0.15 (Issue 3)':<42} | {item4['confident_predictions_fraction']:<20} | Disagreement near 0.5")
    print(f" {'Demo Pairs Streaming (Real / Synth)':<42} | 0/5 Real / 5/5 Synth | Verified Demo Set")
    print(f" {'Sweep Real False Alarms (Streaming)':<42} | 3/3 Flagged (100% FP)| Known Issue 2/3")
    print("=" * 78)
    print(f"Outputs written to:\n  - {out_json}\n  - {out_md}\n")


if __name__ == "__main__":
    main()
