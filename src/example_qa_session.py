# src/example_qa_session.py

import yaml
import os
import time
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.qa_system import QASystem

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_example_session():
    """Run an example Q&A session with predefined questions."""
    # Load configuration
    config = load_config()
    
    print("="*80)
    print("              EXAMPLE QA SESSION")
    print("="*80)
    print("\nThis script will run through a series of example questions using all available models.")
    print("It demonstrates how the QA system works with different LLMs.\n")
    
    # Set up the retriever
    print("Initializing document retriever...")
    retriever = DocumentRetriever(
        vector_db_path=config['indexing']['vector_db']['path'],
        top_k=config['qa_system']['top_k']
    )
    
    # Example questions
    example_questions = [
        "What is the main focus of these documents?",
        "What are the key challenges mentioned in the documents?",
        "Can you summarize the main points about machine learning?",
        "What are the ethical considerations discussed?",
        "How do these documents relate to real-world applications?"
    ]
    
    # Available models
    models = ["mistral", "deepseek"]
    
    # Check if GPU is available to possibly include deepseek
    import torch
    if torch.cuda.is_available():
        models.append("deepseek")
    
    # Run each model with each question
    for model_name in models:
        print("\n" + "="*80)
        print(f"TESTING MODEL: {model_name.upper()}")
        print("="*80)
        
        try:
            # Initialize LLM handler with current model
            print(f"\nInitializing {model_name} model...")
            llm_handler = LLMHandler(model_name=model_name)
            
            # Create QA system
            qa_system = QASystem(
                retriever=retriever,
                llm_handler=llm_handler,
                max_context_length=config['qa_system']['max_context_length'],
                top_k=config['qa_system']['top_k']
            )
            
            # Process each question
            for i, question in enumerate(example_questions):
                print(f"\n{'-'*80}")
                print(f"QUESTION {i+1}: {question}")
                print(f"{'-'*80}")
                
                start_time = time.time()
                
                # Get the answer
                result = qa_system.answer(question)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Print the answer
                print("\nANSWER:")
                print(result["answer"])
                
                # Print sources
                print("\nSOURCES:")
                for source in result["sources"]:
                    print(f"- {source['name']} (relevance: {source['relevance']}%)")
                
                print(f"\nProcessing time: {processing_time:.2f} seconds")
                
                # Pause between questions
                time.sleep(1)
                
        except Exception as e:
            print(f"\nError with model {model_name}: {str(e)}")
            continue
    
    print("\n" + "="*80)
    print("               EXAMPLE SESSION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_example_session()