# steps/feature_extraction/text_features.py
from typing import Dict, List
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from zenml.steps import step

@step
def extract_text_features(utterances: Dict[str, List[Dict]]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Extrait les features textuelles pour chaque utterance.
    
    Args:
        utterances: Utterances segmentées par fichier
        
    Returns:
        Dict contenant les embeddings textuels pour chaque utterance
    """
    # Charger le modèle et le tokenizer pour les embeddings
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    model = AutoModel.from_pretrained("camembert-base")
    
    text_features = {}
    
    for file_name, file_utterances in utterances.items():
        file_features = {}
        
        for i, utterance in enumerate(file_utterances):
            text = utterance["text"]
            
            # Tokénisation et génération d'embeddings
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Utiliser la moyenne des embeddings de la dernière couche cachée
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            file_features[i] = embeddings
        
        text_features[file_name] = file_features
    
    return text_features