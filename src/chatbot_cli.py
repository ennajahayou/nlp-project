# src/chatbot_cli.py

import os
import sys
import yaml
import argparse
from colorama import Fore, Style, init
from datetime import datetime

# Add the parent directory to the path to import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot import RAGChatbot, Conversation
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.qa_system import QASystem

# Initialize colorama for colored terminal output
init()

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def list_conversations(directory="./conversations"):
    """List all saved conversations."""
    if not os.path.exists(directory):
        print(f"No conversations directory found at {directory}")
        return []
    
    conversations = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            conv_id = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            mod_time = os.path.getmtime(file_path)
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            conversations.append((conv_id, mod_date, file_path))
    
    # Sort by modification time (newest first)
    conversations.sort(key=lambda x: x[1], reverse=True)
    return conversations

def print_welcome():
    """Print welcome message."""
    print("\n" + "="*70)
    print(f"{Fore.CYAN}  CHATBOT RAG INTERACTIF{Style.RESET_ALL}")
    print("  Posez des questions sur votre base documentaire et maintenez une conversation")
    print("="*70)
    print(f"\n{Fore.YELLOW}Commandes spéciales:{Style.RESET_ALL}")
    print("  !exit        - Quitter le chatbot")
    print("  !save        - Sauvegarder la conversation")
    print("  !clear       - Effacer l'historique de la conversation")
    print("  !switch nom  - Changer de modèle LLM (deepseek, mistral, etc.)")
    print("  !help        - Afficher cette aide")
    print("="*70 + "\n")

def format_conversation_history(conversation):
    """Format the conversation history for display."""
    history = []
    
    for message in conversation.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            history.append(f"{Fore.GREEN}Vous: {Style.RESET_ALL}{content}")
        elif role == "assistant":
            history.append(f"{Fore.BLUE}Assistant: {Style.RESET_ALL}{content}")
            
            # Add sources if available
            if "metadata" in message and "sources" in message["metadata"]:
                sources = message["metadata"]["sources"]
                if sources:
                    source_text = ", ".join([f"{s['name']} ({s['relevance']}%)" for s in sources[:3]])
                    history.append(f"{Fore.YELLOW}Sources: {Style.RESET_ALL}{source_text}")
    
    return "\n\n".join(history)

def run_interactive_chat(config, conversation_id=None):
    """Run the interactive chatbot session."""
    
    # Initialize chatbot configuration if not already in config
    if 'chatbot' not in config:
        config['chatbot'] = {
            'model': 'deepseek',
            'top_k': config['qa_system']['top_k'],
            'max_context_length': config['qa_system']['max_context_length'],
            'history_size': 5,
            'max_history_tokens': 1000,
            'conversations_dir': './conversations'
        }
    
    # Create chatbot
    chatbot = RAGChatbot.from_config(config, conversation_id)
    
    # Print welcome message
    print_welcome()
    
    # Print initial state
    model_name = chatbot.qa_system.llm_handler.model_config["name"]
    print(f"{Fore.CYAN}Modèle actuel: {Style.RESET_ALL}{model_name}")
    print(f"{Fore.CYAN}ID Conversation: {Style.RESET_ALL}{chatbot.conversation.conversation_id}")
    
    # If continuing conversation, show history
    if chatbot.conversation.messages:
        print(f"\n{Fore.YELLOW}Historique de la conversation:{Style.RESET_ALL}")
        print(format_conversation_history(chatbot.conversation))
    
    print("\n" + "-"*70)
    
    # Main interaction loop
    try:
        while True:
            # Get user input
            user_input = input(f"\n{Fore.GREEN}Vous: {Style.RESET_ALL}")
            
            # Handle special commands
            if user_input.lower() == "!exit":
                # Save conversation before exiting
                chatbot.save_conversation(config['chatbot'].get('conversations_dir', './conversations'))
                print(f"\n{Fore.YELLOW}Conversation sauvegardée. Au revoir!{Style.RESET_ALL}")
                break
                
            elif user_input.lower() == "!save":
                chatbot.save_conversation(config['chatbot'].get('conversations_dir', './conversations'))
                print(f"\n{Fore.YELLOW}Conversation sauvegardée avec ID: {chatbot.conversation.conversation_id}{Style.RESET_ALL}")
                continue
                
            elif user_input.lower() == "!clear":
                # Create new conversation with same ID
                chatbot.conversation = Conversation(chatbot.conversation.conversation_id)
                print(f"\n{Fore.YELLOW}Historique de la conversation effacé.{Style.RESET_ALL}")
                continue
                
            elif user_input.lower() == "!help":
                print_welcome()
                continue
                
            elif user_input.lower().startswith("!switch "):
                model_name = user_input[8:].strip().lower()
                if model_name in chatbot.qa_system.llm_handler.AVAILABLE_MODELS:
                    success = chatbot.qa_system.llm_handler.switch_model(model_name)
                    if success:
                        new_model = chatbot.qa_system.llm_handler.model_config["name"]
                        print(f"\n{Fore.YELLOW}Modèle changé pour: {new_model}{Style.RESET_ALL}")
                    else:
                        print(f"\n{Fore.RED}Échec du changement de modèle.{Style.RESET_ALL}")
                else:
                    available_models = list(chatbot.qa_system.llm_handler.AVAILABLE_MODELS.keys())
                    print(f"\n{Fore.RED}Modèle non disponible. Options: {', '.join(available_models)}{Style.RESET_ALL}")
                continue
            
            # Process regular user input
            print(f"\n{Fore.YELLOW}Recherche de documents pertinents...{Style.RESET_ALL}")
            result = chatbot.process_query(user_input)
            
            # Display response
            print(f"\n{Fore.BLUE}Assistant: {Style.RESET_ALL}{result['answer']}")
            
            # Display sources
            if result['sources']:
                sources_text = ", ".join([f"{s['name']} ({s['relevance']}%)" for s in result['sources'][:3]])
                print(f"\n{Fore.YELLOW}Sources: {Style.RESET_ALL}{sources_text}")
            
    except KeyboardInterrupt:
        # Save conversation on Ctrl+C
        print(f"\n\n{Fore.YELLOW}Interruption détectée. Sauvegarde de la conversation...{Style.RESET_ALL}")
        chatbot.save_conversation(config['chatbot'].get('conversations_dir', './conversations'))
        print(f"{Fore.YELLOW}Conversation sauvegardée. Au revoir!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Une erreur est survenue: {str(e)}{Style.RESET_ALL}")
        # Try to save conversation on error
        try:
            chatbot.save_conversation(config['chatbot'].get('conversations_dir', './conversations'))
            print(f"{Fore.YELLOW}Conversation sauvegardée malgré l'erreur.{Style.RESET_ALL}")
        except:
            print(f"{Fore.RED}Impossible de sauvegarder la conversation.{Style.RESET_ALL}")

def main():
    """Main entry point for the chatbot CLI."""
    parser = argparse.ArgumentParser(description='RAG Chatbot CLI')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--conversation', help='Conversation ID to continue')
    parser.add_argument('--list', action='store_true', help='List saved conversations')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure chatbot config exists
    if 'chatbot' not in config:
        config['chatbot'] = {
            'model': 'deepseek',
            'top_k': 5,
            'max_context_length': 3000,
            'history_size': 5,
            'max_history_tokens': 1000,
            'conversations_dir': './conversations'
        }
    
    # List conversations if requested
    if args.list:
        conversations_dir = config['chatbot'].get('conversations_dir', './conversations')
        conversations = list_conversations(conversations_dir)
        
        if not conversations:
            print("No saved conversations found.")
        else:
            print("\nSAVED CONVERSATIONS:")
            print("-"*70)
            for i, (conv_id, mod_date, _) in enumerate(conversations):
                print(f"{i+1}. {conv_id} (Last modified: {mod_date})")
        return
    
    # Run interactive session
    run_interactive_chat(config, args.conversation)

if __name__ == "__main__":
    main()