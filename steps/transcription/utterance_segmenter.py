# steps/transcription/utterance_segmenter.py
from typing import Dict, List
from zenml.steps import step

@step
def segment_utterances(transcriptions: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """
    Segmente les transcriptions en utterances par locuteur.
    
    Args:
        transcriptions: Transcriptions avec métadonnées
        
    Returns:
        Dict contenant les utterances par fichier
    """
    utterances_by_file = {}
    
    for file_name, transcription in transcriptions.items():
        utterances = []
        current_speaker = None
        current_text = []
        current_start = None
        current_end = None
        
        # Parcourir tous les segments pour créer des utterances par locuteur
        for segment in transcription["segments"]:
            speaker = segment.get("speaker", "unknown")
            
            if current_speaker is None:
                # Première utterance
                current_speaker = speaker
                current_start = segment["start"]
                current_text.append(segment["text"])
                current_end = segment["end"]
            elif speaker == current_speaker:
                # Même locuteur, continuation de l'utterance
                current_text.append(segment["text"])
                current_end = segment["end"]
            else:
                # Changement de locuteur, création d'une nouvelle utterance
                utterances.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                    "start_time": current_start,
                    "end_time": current_end,
                    "audio_file": file_name
                })
                
                # Réinitialisation pour la nouvelle utterance
                current_speaker = speaker
                current_text = [segment["text"]]
                current_start = segment["start"]
                current_end = segment["end"]
        
        # Ajouter la dernière utterance
        if current_text:
            utterances.append({
                "speaker": current_speaker,
                "text": " ".join(current_text),
                "start_time": current_start,
                "end_time": current_end,
                "audio_file": file_name
            })
            
        utterances_by_file[file_name] = utterances
    
    return utterances_by_file