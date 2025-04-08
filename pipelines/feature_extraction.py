# pipelines/feature_extraction.py
from zenml.pipelines import pipeline
from steps.feature_extraction.audio_features import extract_audio_features
from steps.feature_extraction.text_features import extract_text_features

@pipeline
def feature_extraction_pipeline(utterances, audio_files_path: str):
    """Pipeline pour extraire les features audio et textuelles."""
    audio_features = extract_audio_features(utterances, audio_files_path)
    text_features = extract_text_features(utterances)
    return audio_features, text_features