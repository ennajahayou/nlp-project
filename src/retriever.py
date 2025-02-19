from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
import os

class DocumentRetriever:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model_name = config["embedding_model_name"]
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.persist_directory = config["persist_directory"]
        
        # Charger la base vectorielle existante
        if os.path.exists(self.persist_directory):
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            raise ValueError(f"Le dossier {self.persist_directory} n'existe pas. Veuillez lancer l'indexation avant.")
    
    def get_relevant_documents(self, query: str, k: int = 4):
        """
        Retourne une liste de tuples (document, score) pour les k documents les plus pertinents
        """
        # similarity_search_by_vector peut également retourner les scores (selon version de langchain)
        # On peut contourner en faisant un vectordb.similarity_search_with_score(...)
        
        docs_with_scores = self.vectordb.similarity_search_with_score(query, k=k)
        
        # docs_with_scores est une liste de tuples (Document, float)
        return docs_with_scores
