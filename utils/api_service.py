# utils/api_service.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json
from zenml.client import Client
from pipelines.audio_transcription import audio_transcription_pipeline
from pipelines.feature_extraction import feature_extraction_pipeline
from pipelines.rag_pipeline import rag_emotion_pipeline

app = FastAPI(title="API de reconnaissance d'émotions")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_emotion")
async def analyze_emotion(
    audio_file: UploadFile = File(...),
    context: str = Form(None)
):
    """
    Analyse les émotions dans un fichier audio.
    
    Args:
        audio_file: Fichier audio à analyser
        context: Contexte de la conversation (optionnel, au format JSON)
        
    Returns:
        Résultats de l'analyse d'émotions
    """
    # Créer un répertoire temporaire pour le fichier audio
    with tempfile.TemporaryDirectory() as temp_dir:
        # Sauvegarder le fichier audio
        file_path = os.path.join(temp_dir, audio_file.filename)
        with open(file_path, "wb") as f:
            f.write(await audio_file.read())
        
        # Initialiser le client ZenML
        client = Client()
        
        # Pipeline de transcription
        transcription_pipeline = audio_transcription_pipeline()
        transcription_pipeline.run(audio_files_path=temp_dir)
        
        # Récupérer les résultats
        latest_transcription_run = transcription_pipeline.get_runs()[-1]
        utterances = latest_transcription_run.get_step_output("segment_utterances")
        
        # Pipeline d'extraction de features
        feature_pipeline = feature_extraction_pipeline()
        feature_pipeline.run(utterances=utterances, audio_files_path=temp_dir)
        
        # Récupérer les résultats
        latest_feature_run = feature_pipeline.get_runs()[-1]
        audio_features = latest_feature_run.get_step_output("extract_audio_features")
        text_features = latest_feature_run.get_step_output("extract_text_features")
        
        # Contexte de conversation si fourni
        conversation_histories = None
        if context:
            try:
                conversation_histories = json.loads(context)
            except:
                pass
        
        # Pipeline RAG principal
        rag_pipeline = rag_emotion_pipeline()
        rag_pipeline.run(
            utterances=utterances,
            audio_features=audio_features,
            text_features=text_features,
            conversation_histories=conversation_histories
        )
        
        # Récupérer les résultats
        latest_rag_run = rag_pipeline.get_runs()[-1]
        emotion_results = latest_rag_run.get_step_output("analyze_emotions_with_llm")
        
        return {
            "utterances": utterances,
            "emotions": emotion_results
        }

@app.get("/health")
async def health_check():
    """Vérifie la santé de l'API."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)