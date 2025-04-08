# utils/evaluation.py
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_emotions(
    predictions: Dict[str, List[Dict]],
    ground_truth: Dict[str, List[Dict]]
) -> Dict:
    """
    Évalue les prédictions d'émotions par rapport à la vérité terrain.
    
    Args:
        predictions: Prédictions d'émotions
        ground_truth: Vérités terrain
        
    Returns:
        Métriques d'évaluation
    """
    # Extraire les émotions prédites et les vraies émotions
    y_pred = []
    y_true = []
    
    for file_name, file_predictions in predictions.items():
        file_ground_truth = ground_truth.get(file_name, [])
        
        if not file_ground_truth:
            continue
        
        for pred in file_predictions:
            idx = pred["utterance_index"]
            
            # Vérifier que l'index existe dans la vérité terrain
            if idx < len(file_ground_truth):
                y_pred.append(pred["final_emotion"])
                y_true.append(file_ground_truth[idx]["emotion"])
    
    # Classes d'émotions
    emotions = ["sad", "happy", "angry", "neutral", "non-neutral"]
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=emotions)
    
    # Rapport de classification
    report = classification_report(y_true, y_pred, labels=emotions, output_dict=True)
    
    # Créer une visualisation de la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité terrain")
    plt.title("Matrice de confusion - Reconnaissance d'émotions")
    
    # Sauvegarder la figure
    plt.savefig("eval/confusion_matrix.png")
    
    # Précision par émotion
    precision_by_emotion = {emotion: report[emotion]["precision"] for emotion in emotions}
    
    # Rappel par émotion
    recall_by_emotion = {emotion: report[emotion]["recall"] for emotion in emotions}
    
    # Précision moyenne
    average_precision = report["macro avg"]["precision"]
    
    # Rappel moyen
    average_recall = report["macro avg"]["recall"]
    
    # F1-score moyen
    average_f1 = report["macro avg"]["f1-score"]
    
    # Exactitude globale
    accuracy = report["accuracy"] if "accuracy" in report else np.sum(np.diag(cm)) / np.sum(cm)
    
    # Résultats
    results = {
        "accuracy": accuracy,
        "precision": average_precision,
        "recall": average_recall,
        "f1_score": average_f1,
        "precision_by_emotion": precision_by_emotion,
        "recall_by_emotion": recall_by_emotion,
        "confusion_matrix": cm.tolist(),
        "report": report
    }
    
    return results