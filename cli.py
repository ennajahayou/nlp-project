#!/usr/bin/env python
# cli.py

import argparse
import yaml
import os
import glob
import json
import sys
# Ensure src directory is in the path
module_path = os.path.abspath(os.path.dirname(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.qa_system import QASystem
from src.evaluator import RAGEvaluator

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def index_documents(config):
    """Index documents using the DocumentIndexer."""
    # Create indexer with configuration
    indexer = DocumentIndexer(
        embedding_model_name=config['indexing']['embeddings']['model'],
        vector_db_path=config['indexing']['vector_db']['path'],
        chunk_size=config['indexing']['document_processing']['chunk_size'],
        chunk_overlap=config['indexing']['document_processing']['chunk_overlap']
    )
    
    # Check if an index already exists
    index_exists = indexer.load_existing_index()
    
    # Find all documents in the specified directory
    documents_dir = config['data']['documents_dir']
    supported_extensions = ['.pdf', '.txt', '.md', '.markdown']
    documents = []
    
    for ext in supported_extensions:
        documents.extend(glob.glob(os.path.join(documents_dir, f'*{ext}')))
    
    if not documents:
        print(f"No supported documents found in {documents_dir}")
        return
    
    print(f"Found {len(documents)} documents to index")
    
    # Index the documents
    indexer.create_or_update_index(documents)
    print("Indexing complete")

def query_documents(config, query_text, top_k=5):
    """Query the document vector store."""
    # Create retriever
    retriever = DocumentRetriever(
        vector_db_path=config['indexing']['vector_db']['path'],
        top_k=top_k
    )
    
    # Perform the query
    results, source_scores = retriever.advanced_query(query_text, top_k)
    
    if not results:
        print("No relevant documents found.")
        return
    
    # Print source documents and their scores
    print("\n=== Document Sources ===")
    for source, score in source_scores.items():
        print(f"{source}: {score}% relevance")
    
    # Print detailed results
    print("\n=== Detailed Results ===")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} ({result['score']}% relevance) ---")
        print(f"Source: {result['metadata'].get('source', 'unknown')}")
        
        # If the source is a PDF, include page number if available
        if 'page' in result['metadata']:
            print(f"Page: {result['metadata']['page']}")
            
        # Print a snippet of the content (first 200 chars)
        content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        print(f"Content: {content_preview}")

def ask_question(config, question, model_name="deepseek", top_k=5):
    """Use the QA system to answer a question."""
    # Create retriever
    retriever = DocumentRetriever(
        vector_db_path=config['indexing']['vector_db']['path'],
        top_k=top_k
    )
    
    try:
        # Create LLM handler with the specified model
        print(f"Initializing {model_name} model...")
        llm_handler = LLMHandler(model_name=model_name)
        
        # Create QA system
        qa_system = QASystem(
            retriever=retriever,
            llm_handler=llm_handler,
            top_k=top_k
        )
        
        # Get answer
        print(f"Generating answer to: '{question}'")
        result = qa_system.answer(question)
        
        # Print the answer without the conversation history
        print("\n" + "="*80)
        print("RÉPONSE:")
        print("-"*80)
        
        answer = result["answer"]
        # Si la réponse contient encore le contexte ou la question, essayer de nettoyer
        if "Contexte:" in answer and "Question:" in answer:
            parts = answer.split("Question:")
            if len(parts) > 1:
                parts = parts[1].split("Réponse:")
                if len(parts) > 1:
                    answer = parts[1].strip()
        
        print(answer)
        print("="*80)
        
        # Print sources
        print("\nSOURCES:")
        for source in result["sources"]:
            print(f"- {source['name']} (relevance: {source['relevance']}%)")
            
        # Print model used
        print(f"\nModèle utilisé: {result['model_used']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def evaluate_system(config, model_name="deepseek", eval_data=None, output_dir=None):
    """Evaluate the RAG system performance."""
    if output_dir is None:
        output_dir = config['evaluation']['output_dir']
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup QA system
    print(f"Setting up QA system with {model_name} model for evaluation...")
    retriever = DocumentRetriever(
        vector_db_path=config['indexing']['vector_db']['path'],
        top_k=config['qa_system']['top_k']
    )
    
    llm_handler = LLMHandler(model_name=model_name)
    
    qa_system = QASystem(
        retriever=retriever,
        llm_handler=llm_handler,
        max_context_length=config['qa_system']['max_context_length'],
        top_k=config['qa_system']['top_k']
    )
    
    # Setup evaluator
    if eval_data is None:
        eval_data = config['evaluation'].get('evaluation_data_path')
    
    evaluator = RAGEvaluator(evaluation_data_path=eval_data, output_dir=output_dir)
    
    # Get embeddings for semantic similarity calculations
    embeddings = None
    try:
        indexer = DocumentIndexer(
            embedding_model_name=config['indexing']['embeddings']['model'],
            vector_db_path=config['indexing']['vector_db']['path']
        )
        embeddings = indexer.embeddings
    except Exception as e:
        print(f"Warning: Could not initialize embeddings model. Some metrics will not be available: {e}")
    
    # Run evaluation
    print(f"Starting evaluation with {model_name} model...")
    results = evaluator.evaluate_batch(
        qa_system=qa_system,
        embeddings_model=embeddings
    )
    
    # Create visualizations
    if config['evaluation']['visualizations']['enabled']:
        print("Generating visualizations...")
        viz_dir = evaluator.visualize_results(results)
        print(f"Visualizations saved to: {viz_dir}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-"*50)
    
    metrics = results.get("aggregated_metrics", {})
    for metric_name in config['evaluation']['metrics']:
        if metric_name in metrics:
            print(f"{metric_name.ljust(25)}: {metrics[metric_name]['mean']:.4f} (mean)")
    
    # Print path to full results
    print(f"\nDetailed results saved to: {output_dir}")
    
    return results

def list_models():
    """List available LLM models."""
    # Temporarily create a handler just to list models
    try:
        handler = LLMHandler()
        models = handler.list_available_models()
        
        print("\nAVAILABLE MODELS:")
        print("-"*50)
        for model in models:
            gpu_req = "Requires GPU" if model["requires_gpu"] else "Works on CPU"
            print(f"{model['id']} - {model['name']}: {model['description']} ({gpu_req})")
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='RAG System CLI')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    chat_parser = subparsers.add_parser('chat', help='Start an interactive chatbot session')
    chat_parser.add_argument('--conversation', help='Conversation ID to continue')
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the vector database')
    query_parser.add_argument('query_text', help='Query text')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    
    # Ask command (using QA system)
    ask_parser = subparsers.add_parser('ask', help='Ask a question using the QA system')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--model', default='deepseek', help='LLM model to use')
    ask_parser.add_argument('--top-k', type=int, default=5, help='Number of documents to retrieve')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the RAG system')
    eval_parser.add_argument('--model', default='deepseek', help='Model to use for evaluation')
    eval_parser.add_argument('--eval-data', help='Path to evaluation dataset')
    eval_parser.add_argument('--output-dir', help='Directory for evaluation results')
    
    # List models command
    models_parser = subparsers.add_parser('models', help='List available LLM models')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.command == 'index':
        index_documents(config)
    elif args.command == 'query':
        query_documents(config, args.query_text, args.top_k)
    elif args.command == 'ask':
        ask_question(config, args.question, args.model, args.top_k)
    elif args.command == 'evaluate':
        evaluate_system(config, args.model, args.eval_data, args.output_dir)
    elif args.command == 'models':
        list_models()
    elif args.command == 'chat':
       from src.chatbot_cli import run_interactive_chat
       run_interactive_chat(config, args.conversation)
    else:
        parser.print_help()
    

# Et dans le bloc de conditions pour exécuter les commandes, ajouter:

if __name__ == '__main__':
    main()