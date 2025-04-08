# steps/emotion/llm_emotion_analyzer.py
from typing import Dict, List
import numpy as np
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from zenml.steps import step
import chromadb

@step
def analyze_emotions_with_llm(
    utterances: Dict[str, List[Dict]],
    emotion_predictions: Dict[str, List[Dict]],
    context_data: Dict[str, List[Dict]],
    vector_store: chromadb.Collection
) -> Dict[str, List[Dict]]:
    """
    Analyse les émotions avec un LLM en utilisant le RAG pour intégrer le contexte.
    
    Args:
        utterances: Utterances segmentées
        emotion_predictions: Prédictions d'émotions du modèle de base
        context_data: Contexte pour chaque utterance
        vector_store: Base de données vectorielle des utterances
        
    Returns:
        Émotions finales pour chaque utterance
    """
    # Configuration de l'environnement LLM
    llm = OpenAI(temperature=0.3)
    
    # Template de prompt pour l'analyse d'émotions
    prompt_template = """
    Analyse l'émotion dans cette utterance en tenant compte du contexte conversationnel.
    
    Utterance à analyser: "{utterance_text}"
    Locuteur: {speaker}
    
    Contexte précédent:
    {previous_context}
    
    Prédiction du modèle de base:
    - Émotion principale: {predicted_emotion}
    - Confiance: {confidence}
    
    Utterances similaires:
    {similar_utterances}
    
    Détermine l'émotion la plus appropriée parmi : sad, happy, angry, neutral, non-neutral.
    Explique ton raisonnement en considérant:
    1. Le contenu lexical de l'utterance
    2. Le contexte de la conversation
    3. Le locuteur et son historique
    4. Les caractéristiques prosodiques suggérées par le modèle de base
    
    Émotion:
    """
    
    prompt = PromptTemplate(
        input_variables=["utterance_text", "speaker", "previous_context", 
                        "predicted_emotion", "confidence", "similar_utterances"],
        template=prompt_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    final_emotions = {}
    
    for file_name, file_utterances in utterances.items():
        file_final_emotions = []
        
        for i, utterance in enumerate(file_utterances):
            # Récupérer la prédiction du modèle de base
            prediction = emotion_predictions[file_name][i]
            predicted_emotion = prediction["main_emotion"]
            confidence = prediction["confidence"]
            
            # Récupérer le contexte
            context = context_data[file_name][i]
            
            # Formater le contexte précédent
            previous_context_str = ""
            for prev in context["previous_context"]:
                previous_context_str += f"- {prev['speaker']}: {prev['text']}\n"
            
            # Rechercher des utterances similaires dans la base vectorielle
            # Convertir l'embedding de l'utterance actuelle en liste
            query_embedding = utterance.get("embedding", np.zeros(768)).tolist()  # Valeur par défaut en cas d'absence
            
            # Recherche dans le vector store
            similar_results = vector_store.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"file_name": file_name}  # Limiter aux utterances du même fichier
            )
            
            # Formater les utterances similaires
            similar_utterances_str = ""
            for idx, doc in enumerate(similar_results.get("documents", [[]])[0]):
                metadata = similar_results.get("metadatas", [[]])[0][idx]
                similar_utterances_str += f"- {metadata.get('speaker', 'unknown')}: {doc}\n"
            
            # Appeler le LLM pour l'analyse
            result = chain.run(
                utterance_text=utterance["text"],
                speaker=utterance["speaker"],
                previous_context=previous_context_str if previous_context_str else "Aucun contexte précédent disponible.",
                predicted_emotion=predicted_emotion,
                confidence=confidence,
                similar_utterances=similar_utterances_str if similar_utterances_str else "Aucune utterance similaire trouvée."
            )
            
            # Extraire l'émotion finale du résultat
            final_emotion = result.strip().split("\n")[-1].strip()
            
            # Si le LLM n'a pas répondu avec une émotion valide, utiliser la prédiction du modèle
            valid_emotions = ["sad", "happy", "angry", "neutral", "non-neutral"]
            if final_emotion not in valid_emotions:
                final_emotion = predicted_emotion
            
            # Sauvegarder le résultat final
            final_emotion_data = {
                "utterance_index": i,
                "text": utterance["text"],
                "speaker": utterance["speaker"],
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "final_emotion": final_emotion,
                "reasoning": result
            }
            
            file_final_emotions.append(final_emotion_data)
        
        final_emotions[file_name] = file_final_emotions
    
    return final_emotions