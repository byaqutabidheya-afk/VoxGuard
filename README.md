# VoxGuard

VoxGuard is a real-time voice cloning detection and prevention system built for SIH26104. It classifies audio as real or synthetic using pretrained self-supervised speech embeddings (wav2vec2 / HuBERT / WavLM) combined with handcrafted prosody and behavioural features, then adds a prevention layer that surfaces actionable guidance and contextual risk enrichment to the end user. The system further extends to speaker voiceprint verification (confirming that a caller is who they claim to be), a multimodal call-context risk fusion module, an explainability overlay that attributes detection decisions to specific acoustic cues, a FastAPI REST interface for platform integration, and a privacy-preserving session logging layer with configurable retention policies — delivering a full-stack, production-ready safeguard against AI-generated voice fraud.

## Project Structure

```
voxguard/
├── data/               # raw and processed audio (gitignored; kept via .gitkeep)
├── models/             # saved classifier heads / checkpoints (gitignored)
├── src/voxguard/       # core library
│   ├── config.py       # central paths, constants, thresholds
│   ├── embeddings/     # SSL embedding extraction (Phase 1)
│   ├── features/       # prosody / behavioural feature extraction (Phase 2)
│   ├── classifier/     # classifier head training and inference (Phase 3)
│   ├── streaming/      # real-time chunked inference (Phase 4)
│   ├── speaker/        # voiceprint enrolment and verification (Phase 5)
│   ├── fusion/         # multimodal risk fusion (Phase 7)
│   ├── explain/        # explainability overlay (Phase 9)
│   ├── privacy/        # session logging and retention policy (Phases 6, 8, 11)
│   └── utils/          # shared audio I/O and logging helpers
├── app/app.py          # Gradio UI entrypoint
├── api/main.py         # FastAPI REST entrypoint
├── tests/              # project test suite
├── scripts/            # one-off data download / preparation scripts
└── notebooks/          # Kaggle (primary GPU) and Colab (backup) notebooks
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

```bash
# Gradio UI
python app/app.py

# FastAPI REST server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Development Status

This repository is currently scaffolding only. Logic will be added phase by phase starting with Phase 1 (embedding extraction).
