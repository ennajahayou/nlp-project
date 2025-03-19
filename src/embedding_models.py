# src/embedding_models.py

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from typing import Dict, List, Any

class BGEM3Embeddings:
    """
    Wrapper class for BGE-M3 embeddings that handles the specifics of this model.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize the BGE-M3 embeddings model.
        
        Args:
            model_name: Name of the model (default: "BAAI/bge-m3")
        """
        # BGE-M3 specific encode kwargs
        encode_kwargs = {'normalize_embeddings': True}
        
        # For BGE-M3, we need to specify the model_kwargs differently
        model_kwargs = {'device': 'cpu'}
        
        # Initialize the base embeddings model
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Query embedding
        """
        return self.embeddings.embed_query(text)