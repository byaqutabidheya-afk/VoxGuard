"""
xtts_clone.py — XTTS-v2 voice cloning synthesizer.

Wraps the Coqui TTS library's XTTS-v2 multilingual model for one-shot voice cloning
using a natural reference audio clip. Configured with an opt-in `use_gpu` flag,
defaulting to CPU execution (`use_gpu=False`) for local machines and gracefully
falling back to CPU if CUDA is unavailable.
"""

from __future__ import annotations

import os
import sys

# ── FFmpeg Shared Build DLL Configuration (Windows / torchcodec compatibility) ──
FFMPEG_SHARED_BIN = (
    r"C:\Users\ultra\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build-shared\bin"
)

try:
    if os.path.exists(FFMPEG_SHARED_BIN):
        current_path = os.environ.get("PATH", "")
        if FFMPEG_SHARED_BIN not in current_path:
            os.environ["PATH"] = f"{FFMPEG_SHARED_BIN}{os.pathsep}{current_path}"
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(FFMPEG_SHARED_BIN)
            except Exception:
                pass
except Exception:
    pass

# Now import standard and project libraries
import tempfile
from pathlib import Path
from typing import Optional, Union

from voxguard import config
from voxguard.utils.logging_utils import get_logger

logger = get_logger("xtts_clone")

# Log warning if FFmpeg shared bin is missing on Windows
if sys.platform == "win32" and not os.path.exists(FFMPEG_SHARED_BIN):
    logger.warning(
        f"Custom FFmpeg shared build bin not found at: {FFMPEG_SHARED_BIN}. "
        "Relying on system PATH for FFmpeg / torchcodec DLLs."
    )

# Global singleton to avoid reloading model weights on every synthesis call
_XTTS_MODEL = None
_CACHED_GPU_MODE: Optional[bool] = None
DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def resolve_gpu_flag(use_gpu: bool) -> bool:
    """
    Validates requested GPU mode and gracefully falls back to CPU if CUDA is unavailable.
    """
    if not use_gpu:
        return False

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "GPU requested (use_gpu=True) but CUDA is not available. Falling back to CPU."
            )
            return False
        return True
    except Exception as exc:
        logger.warning(f"Failed to check CUDA availability ({exc}). Falling back to CPU.")
        return False


def get_xtts_model(
    model_name: str = DEFAULT_MODEL_NAME,
    use_gpu: bool = False,
    gpu: Optional[bool] = None,
):
    """
    Returns the cached XTTS-v2 model instance, initializing it on first use.

    Parameters
    ----------
    model_name:
        Hugging Face / Coqui model identifier (defaults to xtts_v2).
    use_gpu:
        Whether to use GPU. Defaults to False (CPU mode).
    gpu:
        Legacy alias for use_gpu.
    """
    global _XTTS_MODEL, _CACHED_GPU_MODE

    if gpu is not None:
        use_gpu = gpu

    effective_gpu = resolve_gpu_flag(use_gpu)

    if _XTTS_MODEL is None or _CACHED_GPU_MODE != effective_gpu:
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "Coqui TTS package not found. Please install it with `pip install TTS`."
            ) from exc

        logger.info(f"Initializing XTTS-v2 model ({model_name}, gpu={effective_gpu})...")
        _XTTS_MODEL = TTS(model_name, gpu=effective_gpu)
        _CACHED_GPU_MODE = effective_gpu
        logger.info(f"XTTS-v2 model initialized successfully (gpu={effective_gpu}).")

    return _XTTS_MODEL


def clone_voice(
    reference_audio_path: Union[str, Path],
    text: str,
    language: str = "hi",
    output_path: Optional[Union[str, Path]] = None,
    use_gpu: bool = False,
    gpu: Optional[bool] = None,
) -> str:
    """
    Synthesizes speech in the target speaker's voice using XTTS-v2 one-shot voice cloning.

    Parameters
    ----------
    reference_audio_path:
        Path to the reference audio clip (typically 6-10s clean natural speech).
    text:
        Sentence text to synthesize.
    language:
        Language code (default: "hi" for Hindi/Hinglish).
    output_path:
        Destination path for the generated WAV file. If None, writes to a temporary file.
    use_gpu:
        Whether to run synthesis on GPU (default: False, CPU mode).
    gpu:
        Legacy alias for use_gpu.

    Returns
    -------
    str:
        The resolved output file path where the synthesized audio was saved.
    """
    if gpu is not None:
        use_gpu = gpu

    ref_path = Path(reference_audio_path).resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference audio clip not found: {ref_path}")

    if output_path is None:
        temp_fd, temp_file = tempfile.mkstemp(suffix=".wav", prefix="xtts_clone_")
        os.close(temp_fd)
        out_path = Path(temp_file).resolve()
    else:
        out_path = Path(output_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tts = get_xtts_model(use_gpu=use_gpu)

    logger.debug(
        f"Cloning voice from '{ref_path.name}' for text ({len(text)} chars) -> '{out_path.name}'"
    )

    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=str(ref_path),
            language=language,
            file_path=str(out_path),
        )
    except Exception as exc:
        logger.error(f"XTTS-v2 synthesis failed for '{ref_path.name}': {exc}")
        raise RuntimeError(f"XTTS-v2 synthesis error: {exc}") from exc

    return str(out_path)


def clone_voice_indic_tts(
    reference_audio_path: Union[str, Path],
    text: str,
    language: str = "hi",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Fallback stub for AI4Bharat Indic TTS voice cloning.

    Note:
        AI4Bharat Indic TTS is maintained at https://github.com/AI4Bharat/indic-tts
        If XTTS-v2 proves unworkable or if native Indic acoustic models are preferred,
        this stub can be implemented as an alternative backend.
    """
    raise NotImplementedError(
        "AI4Bharat Indic TTS fallback is not implemented. "
        "XTTS-v2 is the primary synthesis backend for VoxGuard. "
        "Refer to https://github.com/AI4Bharat/indic-tts for Indic TTS integration."
    )
