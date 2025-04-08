# steps/rag/vector_store.py
from typing import Dict, List
import chromadb
from zenml.steps import step

@step
def store_embeddings(enriched_utterances: Dict[str, List[Dict]]) -> chromadb.Collection:
    """
    Stocke les embeddings dans une base de données vectorielle.
    
    Args:
        enriched_utterances: Utterances avec embeddings
        
    Returns:
        Collection ChromaDB contenant les embeddings
    """
    # Initialiser ChromaDB
    client = chromadb.Client()
    collection = client.create_collection("utterance_embeddings")
    
    # Préparer les données à insérer
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    for file_name, file_utterances in enriched_utterances.items():
        for i, utterance in enumerate(file_utterances):
            # Identifiant unique
            utterance_id = f"{file_name}_{i}"
            ids.append(utterance_id)
            
            # Embedding
            embeddings.append(utterance["embedding"].tolist())
            
            # Métadonnées utiles pour les recherches
            metadata = {
                "file_name": file_name,
                "speaker": utterance["speaker"],
                "start_time": utterance["start_time"],
                "end_time": utterance["end_time"],
                "utterance_index": i
            }
            metadatas.append(metadata)
            
            # Texte de l'utterance
            documents.append(utterance["text"])
    
    # Ajouter à la collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    
    return collection