# steps/context/context_processor.py
from typing import Dict, List, Optional
from zenml.steps import step

@step
def process_context(
    utterances: Dict[str, List[Dict]],
    conversation_histories: Optional[Dict[str, List[Dict]]] = None
) -> Dict[str, List[Dict]]:
    """
    Traite le contexte conversationnel pour chaque utterance.
    
    Args:
        utterances: Utterances segmentées
        conversation_histories: Historiques de conversation (optionnel)
        
    Returns:
        Contexte pour chaque utterance
    """
    context_data = {}
    
    for file_name, file_utterances in utterances.items():
        file_context = []
        
        # Récupérer l'historique de la conversation si disponible
        conversation_history = conversation_histories.get(file_name, []) if conversation_histories else []
        
        for i, utterance in enumerate(file_utterances):
            # Contexte précédent (3 dernières utterances)
            previous_context = []
            for j in range(max(0, i-3), i):
                prev_utterance = file_utterances[j]
                previous_context.append({
                    "speaker": prev_utterance["speaker"],
                    "text": prev_utterance["text"]
                })
            
            # Ajouter des informations de l'historique si disponible
            history_context = []
            if conversation_history:
                # Trouver les tours de parole pertinents dans l'historique
                # Pour simplifier, on prend les 2 dernières interactions avec le même locuteur
                same_speaker_interactions = [
                    h for h in conversation_history 
                    if h["speaker"] == utterance["speaker"]
                ][-2:]
                
                for interaction in same_speaker_interactions:
                    history_context.append({
                        "speaker": interaction["speaker"],
                        "text": interaction["text"],
                        "emotion": interaction.get("emotion", "unknown")
                    })
            
            # Construire le contexte complet
            context = {
                "utterance_index": i,
                "previous_context": previous_context,
                "history_context": history_context,
                "speaker": utterance["speaker"],
                "turn_position": {
                    "is_first": i == 0,
                    "is_last": i == len(file_utterances) - 1,
                    "total_turns": len(file_utterances)
                }
            }
            
            file_context.append(context)
        
        context_data[file_name] = file_context
    
    return context_data