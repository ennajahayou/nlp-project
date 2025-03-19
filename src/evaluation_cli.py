# src/evaluation_cli.py

import argparse
import os
import sys
import json
import yaml
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluator import RAGEvaluator
from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.qa_system import QASystem
from src.indexer import DocumentIndexer

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_qa_system(config, model_name="mistral"):
    """Set up QA system based on configuration."""
    print(f"Setting up QA system with {model_name} model...")
    
    # Create retriever
    retriever = DocumentRetriever(
        vector_db_path=config['indexing']['vector_db']['path'],
        top_k=config['qa_system']['top_k']
    )
    
    # Create LLM handler
    llm_handler = LLMHandler(model_name=model_name)
    
    # Create QA system
    qa_system = QASystem(
        retriever=retriever,
        llm_handler=llm_handler,
        max_context_length=config['qa_system']['max_context_length'],
        top_k=config['qa_system']['top_k']
    )
    
    return qa_system, retriever, llm_handler

def create_evaluation_data(args, config):
    """Create evaluation dataset from input file or examples."""
    print("Creating evaluation dataset...")
    
    evaluator = RAGEvaluator(output_dir=args.output_dir)
    
    if args.input_file:
        # Load data from input file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_file.endswith('.json'):
                data = json.load(f)
            elif args.input_file.endswith('.csv'):
                df = pd.read_csv(args.input_file)
                data = df.to_dict('records')
            else:
                print(f"Unsupported file format: {args.input_file}")
                return None
    else:
        # Create sample evaluation data
        data = [
            {
                "question": "Qu'est-ce que le deep learning?",
                "reference_answer": "Le deep learning est une sous-catégorie du machine learning qui utilise des réseaux de neurones à plusieurs couches (d'où le terme 'deep') pour apprendre à partir des données. Il est particulièrement efficace pour traiter des données non structurées comme les images, le texte ou le son."
            },
            {
                "question": "Comment fonctionne l'attention dans les réseaux de neurones?",
                "reference_answer": "Le mécanisme d'attention dans les réseaux de neurones permet au modèle de se concentrer sur différentes parties de l'entrée avec différents degrés d'importance lors de la génération de la sortie. Il calcule des scores d'attention qui déterminent l'importance relative de chaque élément d'entrée."
            },
            {
                "question": "Quels sont les principaux avantages de l'apprentissage par renforcement?",
                "reference_answer": "L'apprentissage par renforcement permet aux agents d'apprendre par interaction avec leur environnement, d'optimiser des comportements à long terme plutôt que des récompenses immédiates, et de résoudre des problèmes où les solutions optimales ne sont pas connues à l'avance."
            }
        ]
    
    # Create and save evaluation dataset
    output_path = os.path.join(args.output_dir, f"evaluation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    evaluator.create_evaluation_dataset(data, output_path)
    
    print(f"Evaluation dataset created with {len(data)} examples at {output_path}")
    return output_path

def run_evaluation(args, config):
    """Run evaluation on the RAG system."""
    print("Starting evaluation of RAG system...")
    
    # Setup
    qa_system, retriever, llm_handler = setup_qa_system(config, args.model)
    
    # Create evaluator
    if args.eval_data:
        evaluator = RAGEvaluator(evaluation_data_path=args.eval_data, output_dir=args.output_dir)
    else:
        # Create evaluation dataset if not provided
        eval_data_path = create_evaluation_data(args, config)
        evaluator = RAGEvaluator(evaluation_data_path=eval_data_path, output_dir=args.output_dir)
    
    # Get embeddings model for similarity calculations
    embeddings = None
    try:
        indexer = DocumentIndexer(
            embedding_model_name=config['indexing']['embeddings']['model'],
            vector_db_path=config['indexing']['vector_db']['path']
        )
        embeddings = indexer.embeddings
    except Exception as e:
        print(f"Warning: Could not initialize embeddings model: {e}")
        print("Some semantic similarity metrics will not be available.")
    
    # Run batch evaluation
    results = evaluator.evaluate_batch(
        qa_system=qa_system,
        embeddings_model=embeddings
    )
    
    # Create visualizations
    viz_dir = evaluator.visualize_results(results)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    
    # Print summary
    print("\nSummary of key metrics:")
    metrics = results.get("aggregated_metrics", {})
    for metric in ["rouge1_f1", "rougeL_f1", "context_utilization", "semantic_similarity"]:
        if metric in metrics:
            print(f"  {metric}: {metrics[metric]['mean']:.4f} (mean)")
    
    return results

def main():
    """Main entry point for the evaluation CLI."""
    parser = argparse.ArgumentParser(description='RAG System Evaluation')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--model', default='deepseek', choices=['deepseek', 'mistral', 'phi'], 
                       help='LLM model to evaluate')
    parser.add_argument('--eval-data', help='Path to evaluation dataset (optional)')
    parser.add_argument('--input-file', help='Path to input file with QA pairs (optional)')
    parser.add_argument('--output-dir', default='./evaluation_results', 
                       help='Directory to store evaluation results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    try:
        run_evaluation(args, config)
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()