# steps/emotion/emotion_classifier.py
from typing import Dict, List
import numpy as np
import torch
from torch import nn
from zenml.steps import step
from models.multimodal_model import EmotionClassifier

@step
def classify_emotions(
    utterances: Dict[str, List[Dict]],
    audio_features: Dict[str, Dict[int, np.ndarray]],
    text_features: Dict[str, Dict[int, np.ndarray]]
) -> Dict[str, List[Dict]]:
    """
    Classifie les émotions à partir des features audio et textuelles.
    
    Args:
        utterances: Utterances segmentées
        audio_features: Features audio
        text_features: Features textuelles
        
    Returns:
        Prédictions d'émotions pour chaque utterance
    """
    # Classes d'émotions
    emotions = ["sad", "happy", "angry", "neutral", "non-neutral"]
    
    # Charger un modèle pré-entraîné (ou utiliser un modèle par défaut)
    try:
        model = EmotionClassifier.load("models/emotion_classifier.pt")
    except:
        # Créer un modèle simple si aucun n'est disponible
        audio_dim = sum([len(audio_features[list(audio_features.keys())[0]][0][feat]) 
                        if isinstance(audio_features[list(audio_features.keys())[0]][0][feat], np.ndarray) 
                        else 1 
                        for feat in audio_features[list(audio_features.keys())[0]][0].keys()])
        text_dim = text_features[list(text_features.keys())[0]][0].shape[0]
        model = EmotionClassifier(audio_dim, text_dim, len(emotions))
    
    emotion_predictions = {}
    
    for file_name, file_utterances in utterances.items():
        file_predictions = []
        
        for i, utterance in enumerate(file_utterances):
            # Extraire et préparer les features
            audio_feature = audio_features[file_name][i]
            text_feature = text_features[file_name][i]
            
            # Transformer en format attendu par le modèle
            audio_vector = np.concatenate([
                audio_feature["mfcc"],
                np.array([
                    audio_feature["pitch_mean"],
                    audio_feature["energy"],
                    audio_feature["speech_rate"],
                    audio_feature["spectral_centroid"],
                    audio_feature["spectral_bandwidth"],
                    audio_feature["spectral_contrast"]
                ])
            ])
            
            # Convertir en tenseurs
            audio_tensor = torch.tensor(audio_vector, dtype=torch.float32).unsqueeze(0)
            text_tensor = torch.tensor(text_feature, dtype=torch.float32).unsqueeze(0)
            
            # Prédiction
            with torch.no_grad():
                emotion_logits = model(audio_tensor, text_tensor)
                emotion_probs = torch.softmax(emotion_logits, dim=1).squeeze().numpy()
            
            # Émotions prédites avec probabilités
            emotion_dict = dict(zip(emotions, emotion_probs.tolist()))
            
            # Émotion principale
            main_emotion = emotions[np.argmax(emotion_probs)]
            
            # Sauvegarder les prédictions
            prediction = {
                "utterance_index": i,
                "main_emotion": main_emotion,
                "emotion_probabilities": emotion_dict,
                "confidence": float(np.max(emotion_probs))
            }
            
            file_predictions.append(prediction)
        
        emotion_predictions[file_name] = file_predictions
    
    return emotion_predictions