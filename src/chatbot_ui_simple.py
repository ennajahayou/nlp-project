# src/chatbot_ui_simple.py

import streamlit as st
import yaml
import os
import sys
import time

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot import RAGChatbot, Conversation
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.qa_system import QASystem

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_chatbot():
    """Initialize the chatbot and required components."""
    # Load configuration
    config = load_config()
    
    # Create retriever
    retriever = DocumentRetriever(
        vector_db_path=config['indexing']['vector_db']['path'],
        top_k=5
    )
    
    # Create LLM handler
    llm_handler = LLMHandler(model_name="deepseek")
    
    # Create QA system
    qa_system = QASystem(
        retriever=retriever,
        llm_handler=llm_handler,
        max_context_length=3000,
        top_k=5
    )
    
    # Create chatbot
    return RAGChatbot(qa_system=qa_system)

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
    
    st.title("💬 Chatbot RAG")
    st.write("Posez des questions sur vos documents")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = setup_chatbot()
    
    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    prompt = st.chat_input("Posez votre question...")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Génération de la réponse..."):
                # Process the query
                try:
                    result = st.session_state.chatbot.process_query(prompt)
                    response = result["answer"]
                    
                    # Display response
                    st.write(response)
                    
                    # Add to message history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Erreur: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()