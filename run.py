# run.py
import os
import json
import argparse
from zenml.client import Client
from pipelines.audio_transcription import audio_transcription_pipeline
from pipelines.feature_extraction import feature_extraction_pipeline
from pipelines.rag_pipeline import rag_emotion_pipeline
from utils.evaluation import evaluate_emotions

def run_pipelines(audio_dir, output_dir=None, ground_truth_file=None, evaluation_output=None):
    """
    Exécute les pipelines de traitement pour la reconnaissance d'émotions.
    
    Args:
        audio_dir: Répertoire contenant les fichiers audio
        output_dir: Répertoire de sortie (optionnel)
        ground_truth_file: Fichier de vérité terrain (optionnel)
        evaluation_output: Fichier de sortie pour l'évaluation (optionnel)
    """
    # Initialiser le client ZenML
    client = Client()
    stack = client.active_stack
    print(f"Utilisation du stack ZenML: {stack.name}")
    
    # Créer le répertoire de sortie si nécessaire
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Pipeline de transcription
    print("Exécution du pipeline de transcription...")
    transcription_pipeline = audio_transcription_pipeline()
    transcription_pipeline.run(audio_files_path=audio_dir)
    
    # Récupérer les résultats
    latest_transcription_run = transcription_pipeline.get_runs()[-1]
    utterances = latest_transcription_run.get_step_output("segment_utterances")
    
    # Sauvegarder les transcriptions si un répertoire de sortie est fourni
    if output_dir:
        with open(os.path.join(output_dir, "utterances.json"), "w") as f:
            json.dump(utterances, f, indent=2)
    
    print(f"Transcription terminée. {sum(len(u) for u in utterances.values())} utterances identifiées.")
    
    # Pipeline d'extraction de features
    print("Exécution du pipeline d'extraction de features...")
    feature_pipeline = feature_extraction_pipeline()
    feature_pipeline.run(utterances=utterances, audio_files_path=audio_dir)
    
    # Récupérer les résultats
    latest_feature_run = feature_pipeline.get_runs()[-1]
    audio_features = latest_feature_run.get_step_output("extract_audio_features")
    text_features = latest_feature_run.get_step_output("extract_text_features")
    
    print("Extraction de features terminée.")
    
    # Pipeline RAG principal
    print("Exécution du pipeline RAG pour la reconnaissance d'émotions...")
    rag_pipeline = rag_emotion_pipeline()
    rag_pipeline.run(
        utterances=utterances,
        audio_features=audio_features,
        text_features=text_features
    )
    
    # Récupérer les résultats
    latest_rag_run = rag_pipeline.get_runs()[-1]
    emotion_results = latest_rag_run.get_step_output("analyze_emotions_with_llm")
    
    # Sauvegarder les résultats si un répertoire de sortie est fourni
    if output_dir:
        with open(os.path.join(output_dir, "emotion_results.json"), "w") as f:
            json.dump(emotion_results, f, indent=2)
    
    print("Reconnaissance d'émotions terminée.")
    
    # Évaluation des résultats si une vérité terrain est fournie
    if ground_truth_file:
        print("Évaluation des résultats...")
        
        # Charger la vérité terrain
        with open(ground_truth_file, "r") as f:
            ground_truth = json.load(f)
        
        # Évaluer les résultats
        evaluation = evaluate_emotions(emotion_results, ground_truth)
        
        # Afficher les résultats
        print(f"Exactitude: {evaluation['accuracy']:.4f}")
        print(f"F1-score: {evaluation['f1_score']:.4f}")
        
        # Sauvegarder l'évaluation si un fichier de sortie est fourni
        if evaluation_output:
            with open(evaluation_output, "w") as f:
                json.dump(evaluation, f, indent=2)
    
    print("Traitement terminé!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconnaissance d'émotions dans les conversations audio")
    parser.add_argument("--audio_dir", required=True, help="Répertoire contenant les fichiers audio")
    parser.add_argument("--output_dir", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--ground_truth", help="Fichier JSON contenant la vérité terrain")
    parser.add_argument("--evaluation_output", help="Fichier de sortie pour l'évaluation")
    
    args = parser.parse_args()
    
    run_pipelines(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        ground_truth_file=args.ground_truth,
        evaluation_output=args.evaluation_output
    )