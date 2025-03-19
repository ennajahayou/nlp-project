# src/chatbot.py

from src.qa_system import QASystem
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime

class Conversation:
    """
    Class to represent a conversation with history.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        """
        Initialize a conversation.
        
        Args:
            conversation_id: Optional identifier for the conversation
        """
        self.conversation_id = conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.messages = []
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender ('user' or 'assistant')
            content: Content of the message
            metadata: Optional metadata for the message (e.g., timestamps, sources)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
            
        self.messages.append(message)
    
    def get_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Args:
            max_messages: Maximum number of most recent messages to return
            
        Returns:
            List of message dictionaries
        """
        if max_messages is None:
            return self.messages
        else:
            return self.messages[-max_messages:]
    
    def format_for_prompt(self, max_messages: Optional[int] = None, include_metadata: bool = False) -> str:
        """
        Format the conversation history for inclusion in a prompt.
        
        Args:
            max_messages: Maximum number of most recent messages to include
            include_metadata: Whether to include metadata in the formatted history
            
        Returns:
            Formatted conversation history string
        """
        history = self.get_history(max_messages)
        formatted = []
        
        for message in history:
            role = message["role"].capitalize()
            content = message["content"]
            
            if role == "User":
                formatted.append(f"User: {content}")
            elif role == "Assistant":
                formatted.append(f"Assistant: {content}")
                
            # Add metadata if requested and available
            if include_metadata and "metadata" in message:
                if "sources" in message["metadata"]:
                    sources = message["metadata"]["sources"]
                    formatted.append(f"Sources: {', '.join(s.get('name', 'Unknown') for s in sources)}")
                    
        return "\n\n".join(formatted)
    
    def save(self, directory: str = "./conversations"):
        """
        Save the conversation to a file.
        
        Args:
            directory: Directory to save the conversation in
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.conversation_id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation_id": self.conversation_id,
                "messages": self.messages
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Conversation':
        """
        Load a conversation from a file.
        
        Args:
            filepath: Path to the conversation file
            
        Returns:
            Conversation instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        conversation = cls(conversation_id=data.get("conversation_id"))
        conversation.messages = data.get("messages", [])
        
        return conversation


class RAGChatbot:
    """
    Chatbot that uses a RAG system to answer questions while maintaining conversation history.
    """
    
    def __init__(self, 
                qa_system: QASystem,
                conversation: Optional[Conversation] = None,
                history_context_size: int = 5,
                max_history_tokens: int = 1000):
        """
        Initialize the RAG Chatbot.
        
        Args:
            qa_system: QASystem to use for answering questions
            conversation: Optional existing conversation
            history_context_size: Number of recent messages to include in context
            max_history_tokens: Maximum number of tokens to include from history
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.qa_system = qa_system
        self.conversation = conversation or Conversation()
        self.history_context_size = history_context_size
        self.max_history_tokens = max_history_tokens
    
    def _create_prompt_with_history(self, query: str, retrieved_context: str) -> str:
        """
        Create a prompt that includes conversation history and retrieved context.
        
        Args:
            query: Current user query
            retrieved_context: Context retrieved from documents
            
        Returns:
            Complete prompt string
        """
        # Format conversation history
        history = self.conversation.format_for_prompt(max_messages=self.history_context_size)
        
        # Build the prompt with history and retrieved context
        prompt = f"""Vous êtes un assistant IA expert qui répond aux questions en utilisant les informations fournies dans le contexte documentaire et qui maintient une conversation cohérente en tenant compte de l'historique du dialogue.

HISTORIQUE DE LA CONVERSATION:
{history}

CONTEXTE DOCUMENTAIRE:
{retrieved_context}

QUESTION ACTUELLE:
{query}

Utilisez le contexte documentaire pour répondre à la question actuelle, tout en tenant compte de l'historique de la conversation. Si la réponse ne se trouve pas dans le contexte documentaire, dites-le clairement mais essayez de fournir une réponse utile basée sur l'historique si possible.

RÉPONSE:"""
        
        return prompt
    
    def process_query(self, query: str) -> Dict[str, Any]:
      """
      Process a user query and generate a response.
    
      Args:
        query: User's query or message
        
      Returns:
        Dictionary with response and metadata
      """
    # Add user message to conversation history
      self.conversation.add_message("user", query)
    
      self.logger.info(f"Processing query: {query}")
    
    # First, retrieve documents normally
      retrieved_documents = self.qa_system.retriever.query(query, self.qa_system.top_k)
    
    # Format the retrieved context
      retrieved_context = self.qa_system._format_context(retrieved_documents)
    
    # Create prompt with conversation history
      prompt_with_history = self._create_prompt_with_history(query, retrieved_context)
    
    # Generate response using the LLM
      try:
        # Au lieu d'utiliser directement le LLM, utiliser answer_question du llm_handler
        # qui nettoie correctement la réponse
        cleaned_response = self.qa_system.llm_handler.answer_question(query, retrieved_context)
        
        # Extract sources for the response
        sources = []
        for doc in retrieved_documents:
            source_name = doc["metadata"].get("source", "Unknown")
            source_score = doc["score"]
            
            # Check if this source is already in the list
            if not any(s["name"] == source_name for s in sources):
                sources.append({
                    "name": source_name,
                    "relevance": source_score
                })
        
        # Prepare result with metadata
        result = {
            "answer": cleaned_response,
            "sources": sources,
            "model_used": self.qa_system.llm_handler.model_config["name"]
        }
        
        # Add assistant message to conversation history
        self.conversation.add_message("assistant", cleaned_response, {"sources": sources})
        
        return result
        
      except Exception as e:
        self.logger.error(f"Error generating response: {e}")
        error_message = f"Désolé, j'ai rencontré une erreur en générant une réponse: {str(e)}"
        
        # Add error message to conversation history
        self.conversation.add_message("assistant", error_message)
        
        return {
            "answer": error_message,
            "sources": [],
            "model_used": self.qa_system.llm_handler.model_config["name"]
        }
    
    def save_conversation(self, directory: str = "./conversations"):
        """
        Save the current conversation to a file.
        
        Args:
            directory: Directory to save the conversation in
        """
        self.conversation.save(directory)
        self.logger.info(f"Conversation saved to {directory}/{self.conversation.conversation_id}.json")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], conversation_id: Optional[str] = None):
        """
        Create a RAGChatbot instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            conversation_id: Optional existing conversation ID to load
            
        Returns:
            RAGChatbot instance
        """
        # Create retriever
        retriever = DocumentRetriever(
            vector_db_path=config['indexing']['vector_db']['path'],
            top_k=config['chatbot']['top_k']
        )
        
        # Create LLM handler
        llm_handler = LLMHandler(model_name=config['chatbot'].get('model', 'deepseek'))
        
        # Create QA system
        qa_system = QASystem(
            retriever=retriever,
            llm_handler=llm_handler,
            max_context_length=config['chatbot']['max_context_length'],
            top_k=config['chatbot']['top_k']
        )
        
        # Create or load conversation
        conversation = None
        if conversation_id:
            conversation_path = os.path.join(
                config['chatbot'].get('conversations_dir', './conversations'),
                f"{conversation_id}.json"
            )
            if os.path.exists(conversation_path):
                conversation = Conversation.load(conversation_path)
        
        if conversation is None:
            conversation = Conversation()
        
        # Create chatbot
        return cls(
            qa_system=qa_system,
            conversation=conversation,
            history_context_size=config['chatbot'].get('history_size', 5),
            max_history_tokens=config['chatbot'].get('max_history_tokens', 1000)
        )