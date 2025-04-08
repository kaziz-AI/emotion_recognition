# pipelines/audio_transcription.py
from zenml.pipelines import pipeline
from steps.transcription.whisper_transcriber import transcribe_audio
from steps.transcription.utterance_segmenter import segment_utterances

@pipeline
def audio_transcription_pipeline(audio_files_path: str):
    """Pipeline pour transcrire l'audio et segmenter en utterances."""
    transcriptions = transcribe_audio(audio_files_path)
    utterances = segment_utterances(transcriptions)
    return utterances