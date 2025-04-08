# models/multimodal_model.py
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class EmotionClassifier(nn.Module):
    def __init__(self, audio_dim, text_dim, num_emotions):
        super(EmotionClassifier, self).__init__()
        
        # Couches pour les features audio
        self.audio_layers = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Couches pour les features textuelles
        self.text_layers = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Couche de fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(128, 64),  # 64+64=128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions)
        )
    
    def forward(self, audio_features, text_features):
        # Traitement des features audio
        audio_encoding = self.audio_layers(audio_features)
        
        # Traitement des features textuelles
        text_encoding = self.text_layers(text_features)
        
        # Concaténation des features
        combined = torch.cat([audio_encoding, text_encoding], dim=1)
        
        # Classification finale
        emotion_logits = self.fusion_layer(combined)
        
        return emotion_logits
    
    @classmethod
    def load(cls, path):
        """Charge un modèle pré-entraîné à partir d'un fichier."""
        checkpoint = torch.load(path)
        model = cls(
            audio_dim=checkpoint['audio_dim'],
            text_dim=checkpoint['text_dim'],
            num_emotions=checkpoint['num_emotions']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def save(self, path, audio_dim, text_dim, num_emotions):
        """Sauvegarde le modèle dans un fichier."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'audio_dim': audio_dim,
            'text_dim': text_dim,
            'num_emotions': num_emotions
        }, path)