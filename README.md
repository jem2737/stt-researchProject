# STT Research Project

Speech-based vocal-state classification using **wav2vec2 embeddings** and **scikit-learn classifiers**, with a workflow designed for both offline experiments and live microphone inference.

This repository explores whether a lightweight pipeline can distinguish vocal states such as **normal**, **stressed**, and **distressed** from speech audio, while also separating **speaker-presence detection** from **emotion classification** during live use.

## Overview

The project follows a two-stage pipeline:

1. **Speaker detection** checks whether a meaningful speaker is present.
2. **Vocal-state classification** predicts the label only when speech is detected.

This design helps avoid forcing silence, background noise, or crowd-like audio into an emotion label when there is no clear speaker to classify.

The core embedding backbone is **`facebook/wav2vec2-base`**, and the extracted embeddings are used to train several classical machine learning models for comparison.

## Why This Project

This project was built as part of a speech / embedded AI research workflow focused on:

- stress-related vocal-state classification
- lightweight model comparison
- live audio inference from a microphone
- future deployment and benchmarking on platforms such as the **NVIDIA Jetson Orin Nano**

Rather than training a large end-to-end speech classifier from scratch, this repository uses pretrained speech embeddings and classical downstream models to keep the workflow easier to inspect, benchmark, and iterate on.

## Features

- Audio conversion into a consistent WAV format
- File-path manifest generation from class-organized folders
- Embedding extraction with **wav2vec2**
- Training for multiple scikit-learn classifiers
- Separate training for:
  - speaker-presence detection
  - vocal-state classification
- Live microphone inference using **PyAudio**
- Saved classifier artifacts with **joblib**
- Training logs written to `src/logs/`

## Model Stack

### Embedding model
- `facebook/wav2vec2-base`

### Classifiers
- Logistic Regression
- SVM (linear)
- SVM (RBF)
- k-NN
- Random Forest
- MLP

These models are trained on fixed-length wav2vec2 embeddings rather than raw waveforms directly.

## Repository Structure

```text
stt-researchProject/
├── data/
├── documentation/
│   ├── Speaker_class_scripts/
│   └── file_structure.md
├── src/
│   ├── logs/
│   ├── cli.py
│   ├── convert_audio.py
│   ├── embed.py
│   ├── listen.py
│   ├── make_path_json.py
│   └── train.py
├── README.md
└── requirements.txt
```

## How It Works

### 1. Convert raw audio
`convert_audio.py` standardizes audio files into a format that can be used consistently across training and inference.

### 2. Build a manifest
`make_path_json.py` scans the cleaned audio folders and creates a JSON mapping of file paths grouped by class.

### 3. Extract embeddings
`embed.py` loads audio, runs it through wav2vec2, and converts the hidden states into a fixed-size embedding vector.

### 4. Train classifiers
`train.py` trains multiple scikit-learn models on the embeddings and saves the trained classifiers.

### 5. Run live inference
`listen.py` captures microphone audio, builds a rolling window, embeds the current audio chunk, and applies the two-stage decision process:
- first check for speaker presence
- then classify vocal state if speech is present

## Installation

Clone the repository:

```bash
git clone https://github.com/jem2737/stt-researchProject.git
cd stt-researchProject
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

The project currently depends on packages including:

- numpy
- scipy
- soundfile
- librosa
- pyaudio
- torch
- transformers
- scikit-learn
- joblib
- datasets
- huggingface_hub

## Data Expectations

The preprocessing pipeline expects audio to be organized by class in folders.

Example:

```text
data/raw/
├── normal/
├── stressed/
└── distressed/
```

For speaker-presence training, a separate dataset structure can also be used.

### Preferred audio format
During preprocessing, files are converted toward a standard format such as:

- WAV
- mono
- 16 kHz
- float32

## Usage

### Train the models

Basic command:

```bash
python src/cli.py train
```

Optional arguments:

```bash
python src/cli.py train \
  --raw-data-path path/to/raw \
  --clean-data-path path/to/clean_data \
  --classifier-path path/to/classifier \
  --clean-data-json path/to/audio_paths.json \
  --prepare-data
```

Skip preprocessing if your data is already cleaned and your JSON manifest already exists:

```bash
python src/cli.py train --no-prepare-data
```

### Run live listening

Basic command:

```bash
python src/cli.py listen
```

Optional classifier paths:

```bash
python src/cli.py listen \
  --type_classifier-path path/to/emotion_classifier.joblib \
  --spk_classifier-path path/to/speaker_classifier.joblib
```

## Outputs

After training, the project writes:

- trained classifier files under `classifier/`
- logs under `src/logs/`

Example classifier structure:

```text
classifier/
├── raw/
└── raw_speaker/
```

## Live Inference Behavior

The live pipeline is designed around a rolling audio window and is meant to avoid unnecessary classification when there is no meaningful speaker in the input.

This is especially useful when testing with:
- silence
- background noise
- multiple voices
- inconsistent microphone conditions

By separating speaker detection from emotion classification, the system can behave more predictably during real-time use.

## Jetson / Edge Deployment Notes

This project is a good fit for benchmarking on edge hardware because it separates the pipeline into clear stages:

- audio capture
- preprocessing
- embedding generation
- lightweight classifier inference

That makes it easier to evaluate tradeoffs such as:

- latency
- CPU vs GPU benefit
- memory usage
- deployability
- overall live responsiveness

For future Jetson-specific deployment, likely areas of focus include:
- ONNX / TensorRT acceleration for embeddings
- memory usage during embedding generation
- whether GPU acceleration materially improves end-to-end live performance

## Current Status

This repository is best described as a **research / prototype workflow** rather than a polished package.

It is useful for:
- experimenting with speech classification pipelines
- comparing classical classifiers on speech embeddings
- testing live audio inference
- building toward an embedded deployment benchmark

## Possible Future Improvements

- Add example audio samples
- Add benchmark tables and plots
- Add confusion matrices to the README
- Add confidence scores during live inference
- Add Jetson-specific setup instructions
- Add reproducible experiment configs
- Add a dataset preparation walkthrough with examples
- Add screenshots or diagrams of the live pipeline

## Author

**James McDonald**
