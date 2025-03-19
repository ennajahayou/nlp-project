# src/retriever.py

from langchain.vectorstores import Chroma
from typing import List, Dict, Any, Optional, Tuple
import logging
from src.indexer import DocumentIndexer
import os

class DocumentRetriever:
    """
    Class for retrieving relevant documents from the vector database based on user queries.
    """
    
    def __init__(self, 
                 indexer: Optional[DocumentIndexer] = None,
                 vector_db_path: str = "./chroma_db",
                 top_k: int = 2):
        """
        Initialize the DocumentRetriever.
        
        Args:
            indexer: DocumentIndexer instance (optional)
            vector_db_path: Path to the vector database
            top_k: Number of top results to return
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.vector_db_path = vector_db_path
        self.top_k = top_k
        
        # If an indexer is provided, use its embedding model and vector store
        if indexer:
            self.embeddings = indexer.embeddings
            self.vector_store = indexer.vector_store
        else:
            self.embeddings = None
            self.vector_store = None
            
        # Try to load existing vector store if not initialized
        if not self.vector_store:
            self._load_vector_store()
            
    def _load_vector_store(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            Boolean indicating whether vector store was successfully loaded
        """
        if not os.path.exists(self.vector_db_path):
            self.logger.error(f"Vector store not found at {self.vector_db_path}")
            return False
            
        try:
            # We need to create a temporary indexer to get the embeddings model
            if not self.embeddings:
                temp_indexer = DocumentIndexer(vector_db_path=self.vector_db_path)
                self.embeddings = temp_indexer.embeddings
            
            self.logger.info(f"Loading vector store from {self.vector_db_path}")
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            return False
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query the vector database for documents relevant to the query.
        
        Args:
            query_text: The query text
            top_k: Number of results to return (overrides default if provided)
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        if not self.vector_store:
            self.logger.error("Vector store not initialized")
            return []
            
        if top_k is None:
            top_k = self.top_k
            
        self.logger.info(f"Querying vector store with: '{query_text}'")
        
        try:
            # Query the vector store with metadata and scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query_text,
                k=top_k
            )
            
            # Format the results
            results = []
            for doc, score in docs_with_scores:
                # Convert similarity score to a more intuitive format (0-100%)
                # Higher is better, and the score is a distance, so we need to normalize it
                similarity_score = 1.0 - min(1.0, float(score))  # Ensure it's between 0 and 1
                similarity_percentage = round(similarity_score * 100, 2)
                
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": similarity_percentage,
                    "raw_score": float(score)
                }
                results.append(result)
            
            self.logger.info(f"Found {len(results)} relevant documents")
            return results
        except Exception as e:
            self.logger.error(f"Error querying vector store: {e}")
            return []
    
    def get_document_sources(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract unique document sources and their highest scores from results.
        
        Args:
            results: List of result dictionaries from query()
            
        Returns:
            Dictionary mapping document sources to their scores
        """
        source_scores = {}
        
        for result in results:
            source = result["metadata"].get("source", "unknown")
            score = result["score"]
            
            # Keep the highest score for each source
            if source in source_scores:
                source_scores[source] = max(source_scores[source], score)
            else:
                source_scores[source] = score
                
        # Sort by score in descending order
        sorted_sources = {k: v for k, v in sorted(
            source_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
        
        return sorted_sources
    
    def advanced_query(self, query_text: str, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Perform an advanced query that returns both detailed results and summarized document sources.
        
        Args:
            query_text: The query text
            top_k: Number of results to return (overrides default if provided)
            
        Returns:
            Tuple containing (detailed_results, source_scores)
        """
        # Get detailed results
        results = self.query(query_text, top_k)
        
        # Extract sources and their scores
        source_scores = self.get_document_sources(results)
        
        return results, source_scores