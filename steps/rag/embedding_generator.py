# steps/rag/embedding_generator.py
from typing import Dict, List
import numpy as np
from zenml.steps import step

@step
def generate_embeddings(
    utterances: Dict[str, List[Dict]],
    audio_features: Dict[str, Dict[int, np.ndarray]],
    text_features: Dict[str, Dict[int, np.ndarray]]
) -> Dict[str, List[Dict]]:
    """
    Génère des embeddings combinant features textuelles et audio.
    
    Args:
        utterances: Utterances segmentées
        audio_features: Features audio par utterance
        text_features: Features textuelles par utterance
        
    Returns:
        Utterances enrichies avec embeddings combinés
    """
    enriched_utterances = {}
    
    for file_name, file_utterances in utterances.items():
        enriched_file_utterances = []
        
        for i, utterance in enumerate(file_utterances):
            # Récupérer les features
            audio_feature = audio_features[file_name][i]
            text_feature = text_features[file_name][i]
            
            # Créer un embedding combiné (simple concaténation pour l'exemple)
            # Dans un cas réel, on pourrait utiliser une méthode plus sophistiquée
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
            
            # Normaliser les vecteurs
            audio_vector = audio_vector / np.linalg.norm(audio_vector)
            text_vector = text_feature / np.linalg.norm(text_feature)
            
            # Concaténer avec des poids (70% texte, 30% audio)
            combined_embedding = np.concatenate([
                text_vector * 0.7,
                audio_vector * 0.3
            ])
            
            # Enrichir l'utterance
            enriched_utterance = utterance.copy()
            enriched_utterance["embedding"] = combined_embedding
            enriched_file_utterances.append(enriched_utterance)
        
        enriched_utterances[file_name] = enriched_file_utterances
    
    return enriched_utterances