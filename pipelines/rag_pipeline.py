# pipelines/rag_pipeline.py
from zenml.pipelines import pipeline
from steps.rag.embedding_generator import generate_embeddings
from steps.rag.vector_store import store_embeddings
from steps.context.context_processor import process_context
from steps.emotion.emotion_classifier import classify_emotions
from steps.emotion.llm_emotion_analyzer import analyze_emotions_with_llm

@pipeline
def rag_emotion_pipeline(
    utterances, 
    audio_features, 
    text_features,
    conversation_histories=None
):
    """Pipeline RAG principal pour l'analyse d'émotions."""
    # Générer et stocker les embeddings
    embeddings = generate_embeddings(utterances, audio_features, text_features)
    vector_store = store_embeddings(embeddings)
    
    # Traiter le contexte conversationnel
    context_data = process_context(utterances, conversation_histories)
    
    # Classification des émotions
    emotion_predictions = classify_emotions(utterances, audio_features, text_features)
    
    # Analyse d'émotions avec LLM et RAG
    final_emotions = analyze_emotions_with_llm(
        utterances, 
        emotion_predictions, 
        context_data, 
        vector_store
    )
    
    return final_emotions