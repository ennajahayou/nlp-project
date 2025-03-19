# src/interactive_qa.py

import yaml
import os
import sys
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.qa_system import QASystem

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header."""
    clear_screen()
    print("="*80)
    print("                   RAG QUESTION-ANSWERING SYSTEM")
    print("="*80)
    print()

def print_model_info(llm_handler):
    """Print information about the current model."""
    print(f"Current model: {llm_handler.model_config['name']}")
    print(f"Running on: {llm_handler.device}")
    print()

def list_available_models(llm_handler):
    """List all available models."""
    models = llm_handler.list_available_models()
    
    print("\nAVAILABLE MODELS:")
    print("-"*50)
    for model in models:
        gpu_req = "Requires GPU" if model["requires_gpu"] else "Works on CPU"
        current = " (CURRENT)" if model["id"] == llm_handler.model_name else ""
        print(f"{model['id']} - {model['name']}: {model['description']} ({gpu_req}){current}")
    print()

def switch_model(qa_system):
    """Allow the user to switch to a different model."""
    print("\nSelect a model:")
    print("1. DeepSeek 67B (best quality, requires GPU)")
    print("2. Mistral 7B (good balance)")
    print("0. Cancel")
    
    choice = input("\nEnter your choice (0-2): ")
    
    model_map = {
        "1": "deepseek",
        "2": "mistral",
        
    }
    
    if choice in model_map:
        model_name = model_map[choice]
        print(f"\nSwitching to {model_name}...")
        success = qa_system.switch_llm(model_name)
        
        if success:
            print(f"Successfully switched to {model_name}!")
        else:
            print(f"Failed to switch to {model_name}. Please check logs for details.")
    
    input("\nPress Enter to continue...")

def main():
    """Run the interactive QA system."""
    try:
        # Load configuration
        config = load_config()
        
        print_header()
        print("Initializing QA system...")
        
        # Set up the retriever
        retriever = DocumentRetriever(
            vector_db_path=config['indexing']['vector_db']['path'],
            top_k=config['qa_system']['top_k']
        )
        
        # Set up the LLM handler with default model
        default_model = config['qa_system'].get('default_model', 'mistral')
        llm_handler = LLMHandler(model_name=default_model)
        
        # Set up the QA system
        qa_system = QASystem(
            retriever=retriever,
            llm_handler=llm_handler,
            max_context_length=config['qa_system']['max_context_length'],
            top_k=config['qa_system']['top_k']
        )
        
        # Main interaction loop
        while True:
            print_header()
            print_model_info(llm_handler)
            
            print("OPTIONS:")
            print("1. Ask a question")
            print("2. Change model")
            print("3. List available models")
            print("0. Exit")
            
            choice = input("\nEnter your choice (0-3): ")
            
            if choice == '0':
                print("\nExiting...")
                break
                
            elif choice == '1':
                print_header()
                print("Ask a question about the documents in your knowledge base.")
                print("Enter 'back' to return to the main menu.\n")
                
                question = input("Your question: ")
                
                if question.lower() == 'back':
                    continue
                
                print("\nSearching for relevant information...")
                result = qa_system.answer(question)
                
                print("\n" + "="*80)
                print("ANSWER:")
                print("-"*80)
                print(result["answer"])
                print("="*80)
                
                print("\nSOURCES:")
                for source in result["sources"]:
                    print(f"- {source['name']} (relevance: {source['relevance']}%)")
                
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                switch_model(qa_system)
                
            
            else:
                print("\nInvalid choice. Please try again.")
                input("Press Enter to continue...")
                
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()