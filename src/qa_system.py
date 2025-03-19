# src/qa_system.py

from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
import logging
from typing import List, Dict, Any, Optional, Tuple

class QASystem:
    """
    Complete Question-Answering system that combines document retrieval with LLM generation.
    """
    
    def __init__(self, 
                 retriever: DocumentRetriever, 
                 llm_handler: LLMHandler,
                 max_context_length: int = 3000,
                 top_k: int = 3):
        """
        Initialize the QA System.
        
        Args:
            retriever: DocumentRetriever instance
            llm_handler: LLMHandler instance
            max_context_length: Maximum context length to send to the LLM
            top_k: Number of documents to retrieve
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.retriever = retriever
        self.llm_handler = llm_handler
        self.max_context_length = max_context_length
        self.top_k = top_k
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
      """
      Format the retrieved results into a context string for the LLM.
    
      Args:
        results: List of retrieval results
        
      Returns:
        Formatted context string
      """
      context_parts = []
    
      for i, result in enumerate(results):
        # Extract document source and content
        source = result["metadata"].get("source", "Unknown")
        page = result["metadata"].get("page", "")
        page_info = f"(page {page})" if page else ""
        content = result["content"].strip()
        
        # Ne garder que 150 caractères au maximum par extrait
        if len(content) > 300:
            content = content[:300] + "..."
        
        # Format as a segment with source information
        segment = f"[Extrait {i+1} - Source: {source} {page_info}]\n{content}"
        context_parts.append(segment)
    
    # Join all parts with separators
      context = "\n\n".join(context_parts)
    
      return context
    
    def answer(self, question: str) -> Dict[str, Any]:
      """
      Answer a question using the retrieval-augmented generation process.
    
      Args:
        question: The user's question
        
      Returns:
        Dictionary containing the answer, sources, and other metadata
      """
      self.logger.info(f"Processing question: {question}")
    
      try:
        # Retrieve relevant documents
        results, source_scores = self.retriever.advanced_query(question, self.top_k)
        
        if not results:
            return {
                "answer": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
                "sources": [],
                "context_used": "",
                "model_used": self.llm_handler.model_config["name"]
            }
        
        # Format context from retrieved documents
        context = self._format_context(results)
        
        # Vérifier si llm_handler.llm existe avant de l'utiliser
        if not hasattr(self.llm_handler, 'llm') or self.llm_handler.llm is None:
            return {
                "answer": "Le modèle de langage n'est pas correctement initialisé. Veuillez essayer avec un autre modèle.",
                "sources": self._format_sources(source_scores),
                "context_used": context,
                "model_used": self.llm_handler.model_config["name"],
                "error": "LLM not initialized"
            }
        
        # Generate answer using LLM
        try:
            answer = self.llm_handler.answer_question(question, context)
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Une erreur s'est produite lors de la génération de la réponse: {str(e)}. Essayez un autre modèle.",
                "sources": self._format_sources(source_scores),
                "context_used": context,
                "model_used": self.llm_handler.model_config["name"],
                "error": str(e)
            }
        
        return {
            "answer": answer,
            "sources": self._format_sources(source_scores),
            "context_used": context,
            "model_used": self.llm_handler.model_config["name"]
        }
      except Exception as e:
        self.logger.error(f"Error in QA system: {e}")
        return {
            "answer": f"Une erreur s'est produite dans le système: {str(e)}",
            "sources": [],
            "context_used": "",
            "model_used": self.llm_handler.model_config["name"] if hasattr(self.llm_handler, 'model_config') else "unknown",
            "error": str(e)
        }
    
    def _format_sources(self, source_scores):
      """Format sources into a standard structure."""
      sources = []
      for source, score in source_scores.items():
        sources.append({
            "name": source,
            "relevance": score
        })
      return sources
    
    def switch_llm(self, model_name: str) -> bool:
        """
        Switch to a different language model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            Boolean indicating success
        """
        return self.llm_handler.switch_model(model_name)
    
    # Ajoutez cette méthode à la classe QASystem dans src/qa_system.py

    def _try_fallback_models(self, question, context):
      """
      Tente d'utiliser des modèles de secours si le modèle principal échoue.
    
      Args:
        question: La question de l'utilisateur
        context: Le contexte extrait
        
      Returns:
        Réponse générée ou message d'erreur
      """
    # Liste des modèles à essayer dans l'ordre
      fallback_models = ["mistral", "deepseek"]
    
    # Retirer le modèle actuel s'il est dans la liste
      if self.llm_handler.model_name in fallback_models:
        fallback_models.remove(self.llm_handler.model_name)
    
    # Ajouter le modèle actuel au début pour réessayer une fois
      fallback_models = [self.llm_handler.model_name] + fallback_models
    
      last_error = None
    
    # Essayer chaque modèle jusqu'à ce qu'un fonctionne
      for model_name in fallback_models:
        try:
            # Si ce n'est pas le modèle actuel, essayer de changer
            if model_name != self.llm_handler.model_name:
                self.logger.info(f"Trying fallback model: {model_name}")
                success = self.switch_llm(model_name)
                if not success:
                    continue
            
            # Essayer de générer une réponse
            return self.llm_handler.answer_question(question, context)
            
        except Exception as e:
            self.logger.warning(f"Error with model {model_name}: {str(e)}")
            last_error = str(e)
            continue
    
    # Si tous les modèles ont échoué
      return f"Je n'ai pas pu générer une réponse avec les modèles disponibles. Erreur: {last_error}"


# Puis modifiez la méthode answer comme ceci:

def answer(self, question: str) -> Dict[str, Any]:
    """
    Answer a question using the retrieval-augmented generation process.
    
    Args:
        question: The user's question
        
    Returns:
        Dictionary containing the answer, sources, and other metadata
    """
    self.logger.info(f"Processing question: {question}")
    
    # Retrieve relevant documents
    results, source_scores = self.retriever.advanced_query(question, self.top_k)
    
    if not results:
        return {
            "answer": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
            "sources": [],
            "context_used": "",
            "model_used": self.llm_handler.model_config["name"]
        }
    
    # Format context from retrieved documents
    context = self._format_context(results)
    
    try:
        # Generate answer using LLM
        answer = self.llm_handler.answer_question(question, context)
    except Exception as e:
        self.logger.error(f"Error generating answer with primary model: {e}")
        # Try fallback models
        answer = self._try_fallback_models(question, context)
    
    # Extract source information
    sources = []
    for source, score in source_scores.items():
        sources.append({
            "name": source,
            "relevance": score
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "context_used": context,
        "model_used": self.llm_handler.model_config["name"]
    }