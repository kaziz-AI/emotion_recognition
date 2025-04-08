# steps/feature_extraction/audio_features.py
import numpy as np
import librosa
import torch
from typing import Dict, List
from zenml.steps import step
from pydub import AudioSegment

@step
def extract_audio_features(utterances: Dict[str, List[Dict]], audio_files_path: str) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extrait les features audio pour chaque utterance.
    
    Args:
        utterances: Utterances segmentées par fichier
        audio_files_path: Chemin vers les fichiers audio
        
    Returns:
        Dict contenant les features audio pour chaque utterance
    """
    audio_features = {}
    
    for file_name, file_utterances in utterances.items():
        file_path = f"{audio_files_path}/{file_name}"
        audio_file = AudioSegment.from_file(file_path)
        
        file_features = {}
        for i, utterance in enumerate(file_utterances):
            start_ms = int(utterance["start_time"] * 1000)
            end_ms = int(utterance["end_time"] * 1000)
            
            # Extraire le segment audio correspondant à l'utterance
            utterance_audio = audio_file[start_ms:end_ms]
            
            # Exporter temporairement pour traitement avec librosa
            temp_path = f"temp_utterance_{i}.wav"
            utterance_audio.export(temp_path, format="wav")
            
            # Extraire diverses features audio
            y, sr = librosa.load(temp_path, sr=None)
            
            # 1. MFCC (coefficients cepstraux)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 2. Caractéristiques prosodiques
            # Pitch (F0)
            pitch, _ = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
            
            # Énergie
            energy = np.mean(librosa.feature.rms(y=y))
            
            # Débit de parole (approximation par zero-crossing rate)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # 3. Caractéristiques spectrales
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            
            # Combiner toutes les features
            features = {
                "mfcc": mfcc.mean(axis=1),  # Moyenne sur l'axe temporel
                "pitch_mean": pitch_mean,
                "energy": energy,
                "speech_rate": zcr,
                "spectral_centroid": spec_centroid,
                "spectral_bandwidth": spec_bandwidth,
                "spectral_contrast": np.mean(spec_contrast)
            }
            
            file_features[i] = features
            
            # Supprimer le fichier temporaire
            import os
            os.remove(temp_path)
        
        audio_features[file_name] = file_features
    
    return audio_features