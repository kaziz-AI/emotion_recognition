# Emotion Recognition in Audio Conversations with RAG and ZenML

This project implements an advanced emotion recognition system for audio conversations using a RAG (Retrieval Augmented Generation) approach and ZenML for pipeline orchestration.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Service API](#service-api)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project aims to detect emotions (sad, happy, angry, neutral, non-neutral) in audio conversations by combining:
- Audio transcription and segmentation
- Multimodal analysis (audio + textual features)
- Contextualizing emotions within the conversation flow
- RAG approach to enhance analysis with relevant examples

The entire process is orchestrated via ZenML to ensure reproducibility and traceability of executions.

## Key Features

1. **Audio Transcription and Segmentation**
   - Uses WhisperX for accurate transcription with temporal alignment
   - Speaker diarization (who speaks when)
   - Segmentation into coherent utterances by speaker

2. **Multimodal Feature Extraction**
   - **Audio**: MFCC, pitch, energy, speech rate, spectral features
   - **Text**: Contextual embeddings via language models
   - Fusion of audio and textual features

3. **RAG Architecture**
   - Generation of combined embeddings for each utterance
   - Vector storage via ChromaDB
   - Contextual search to enhance analysis

4. **Emotion Classification**
   - Base multimodal model
   - Refinement with LLM and RAG approach
   - Detected emotions: sad, happy, angry, neutral, non-neutral

5. **Contextualization**
   - Analysis of previous context in the conversation
   - Consideration of speaker history
   - Adaptation based on position in conversation

## Architecture

The project follows a modular pipeline architecture:

1. **Transcription Pipeline**
   - Input: Raw audio files
   - Output: Transcription segmented into utterances with metadata

2. **Feature Extraction Pipeline**
   - Input: Segmented utterances
   - Output: Audio and textual features

3. **RAG Emotion Analysis Pipeline**
   - Input: Utterances and features
   - Steps:
     - Combined embedding generation
     - Vector storage
     - Conversational context processing
     - Base emotion classification
     - Refined analysis with LLM and RAG
   - Output: Detected emotions with explanations

4. **Evaluation Pipeline** (optional)
   - Input: Predictions and ground truth
   - Output: Evaluation metrics and visualizations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd emotion_recognition_rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the project
pip install -e .
```

### Prerequisites

- Python 3.8+
- GPU recommended for inference (but CPU possible)
- API keys for OpenAI (if using GPT)

## Configuration

```bash
# Initialize ZenML
zenml init

# Configure the ZenML stack
zenml stack register emotion_recognition_stack \
    -a default \
    -o default \
    -c default \
    -m default

# Configure secrets for API keys
zenml secret create openai_api_key --schema=api_key --value=<YOUR_API_KEY>
```

### Environment Variables

Create a `.env` file at the root of the project:

```
OPENAI_API_KEY=<your-openai-api-key>
HF_TOKEN=<your-huggingface-token>
CUDA_VISIBLE_DEVICES=0  # To specify the GPU (optional)
```

## Usage

### Data Preparation

Place your audio files in the `data/raw/` directory.

### Running the Complete Pipeline

```bash
# Basic execution
python run.py --audio_dir data/raw/ --output_dir results/

# With evaluation (if you have annotated data)
python run.py --audio_dir data/raw/ --output_dir results/ \
    --ground_truth data/ground_truth.json \
    --evaluation_output eval/results.json
```

### Model Training (optional)

If you have annotated data, you can train your own model:

```bash
python -m pipelines.training_pipeline \
    --train_data_path data/annotated/train/ \
    --val_data_path data/annotated/val/ \
    --model_output_path models/emotion_classifier.pt
```

## Service API

The project includes an API for on-demand emotion analysis:

```bash
# Start the API
python utils/api_service.py
```

### API Usage Example

```python
import requests

# Analyze an audio file
files = {'audio_file': open('conversation.wav', 'rb')}
data = {'context': '{"conversation_history": []}'}  # Optional

response = requests.post('http://localhost:8000/analyze_emotion', files=files, data=data)
results = response.json()

print(results['emotions'])
```

## Evaluation

The system's performance is evaluated on several metrics:

- **Overall Accuracy**: Percentage of correctly identified emotions
- **Recall by Class**: Ability to detect each type of emotion
- **F1-score**: Harmonic mean between precision and recall
- **Confusion Matrix**: Visualization of errors between classes

## Contributing

Contributions are welcome! Please see the CONTRIBUTING.md file for guidelines.

## License

This project is licensed under the MIT - see the LICENSE file for details.

---

Developed by [Kheireddine AZIZ/Technofutur] © [2025]
