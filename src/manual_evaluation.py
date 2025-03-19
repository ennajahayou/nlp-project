# src/manual_evaluation.py

import os
import sys
import json
import yaml
from datetime import datetime
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import DocumentRetriever
from src.llm_handler import LLMHandler
from src.qa_system import QASystem

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_qa_system(config, model_name="deepseek"):
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
    
    return qa_system

def load_questions(questions_file):
    """Load questions from a file."""
    with open(questions_file, 'r', encoding='utf-8') as f:
        if questions_file.endswith('.json'):
            return json.load(f)
        elif questions_file.endswith('.csv'):
            df = pd.read_csv(f)
            return df.to_dict('records')
        else:
            lines = f.readlines()
            return [{"question": line.strip()} for line in lines if line.strip()]

def manual_evaluate(qa_system, questions):
    """Run manual evaluation on the QA system."""
    results = []
    
    for i, q_item in enumerate(questions):
        question = q_item["question"]
        reference = q_item.get("reference_answer", "")
        
        print(f"\nQuestion {i+1}/{len(questions)}:")
        print(f"{question}")
        
        if reference:
            print(f"\nReference Answer:")
            print(f"{reference}")
        
        # Get system response
        result = qa_system.answer(question)
        answer = result["answer"]
        
        print(f"\nGenerated Answer:")
        print(f"{answer}")
        
        # Display sources
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['name']} (relevance: {source['relevance']}%)")
        
        # Manual scoring
        while True:
            score = input("\nRate the answer (1-5, where 5 is best): ")
            try:
                score = int(score)
                if 1 <= score <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")
        
        feedback = input("Additional feedback (optional): ")
        
        # Save results
        eval_result = {
            "question": question,
            "reference_answer": reference,
            "generated_answer": answer,
            "sources": result["sources"],
            "manual_score": score,
            "feedback": feedback,
            "model_used": result.get("model_used", "")
        }
        
        results.append(eval_result)
        print("\n" + "-" * 50)
    
    return results

def save_results(results, output_file=None):
    """Save evaluation results to a file."""
    if output_file is None:
        output_dir = "./evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"manual_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Calculate metrics
    scores = [r["manual_score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "average_score": avg_score,
        "num_questions": len(results),
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Average Score: {avg_score:.2f}/5.0")
    
    # Create CSV summary
    csv_file = output_file.replace('.json', '.csv')
    df = pd.DataFrame([{
        "question": r["question"],
        "score": r["manual_score"],
        "feedback": r["feedback"],
        "model": r["model_used"]
    } for r in results])
    
    df.to_csv(csv_file, index=False)
    print(f"Summary saved to {csv_file}")
    
    return output_file

def main():
    """Main entry point for manual evaluation."""
    parser = argparse.ArgumentParser(description='Manual RAG System Evaluation')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--model', default='deepseek', help='LLM model to use')
    parser.add_argument('--questions', required=True, help='Path to file with questions')
    parser.add_argument('--output', help='Path for output file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load questions
    questions = load_questions(args.questions)
    if not questions:
        print("No questions found in the provided file.")
        return
    
    print(f"Loaded {len(questions)} questions for evaluation.")
    
    # Setup QA system
    qa_system = setup_qa_system(config, args.model)
    
    # Run manual evaluation
    print("\nStarting manual evaluation...\n")
    results = manual_evaluate(qa_system, questions)
    
    # Save results
    save_results(results, args.output)

if __name__ == "__main__":
    main()