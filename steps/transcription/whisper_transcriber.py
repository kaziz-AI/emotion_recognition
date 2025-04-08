# steps/transcription/whisper_transcriber.py
import os
import whisperx
from typing import List, Dict
from zenml.steps import step

@step
def transcribe_audio(audio_files_path: str) -> Dict[str, Dict]:
    """
    Transcrit les fichiers audio en texte avec horodatage.
    
    Args:
        audio_files_path: Chemin vers les fichiers audio
        
    Returns:
        Dict contenant les transcriptions et métadonnées pour chaque fichier
    """
    transcriptions = {}
    
    # Charge les modèles WhisperX
    model = whisperx.load_model("large-v2", device="cuda")
    
    # Traite chaque fichier audio
    for audio_file in os.listdir(audio_files_path):
        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
            file_path = os.path.join(audio_files_path, audio_file)
            
            # Transcription avec alignement audio
            audio = whisperx.load_audio(file_path)
            result = model.transcribe(audio, language="fr")
            
            # Alignement mot par mot
            model_a, metadata = whisperx.load_align_model(language_code="fr", device="cuda")
            result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda")
            
            # Diarisation des locuteurs
            diarize_model = whisperx.DiarizationPipeline(use_auth_token="your_hf_token")
            diarize_segments = diarize_model(audio)
            
            # Assigner les locuteurs aux segments
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            transcriptions[audio_file] = result
            
    return transcriptions