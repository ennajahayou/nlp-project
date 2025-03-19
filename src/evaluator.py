# src/evaluator.py

# Suppression de l'importation problématique
# from langchain.evaluation import load_evaluator
# from langchain.evaluation.schema import StringEvaluation 
# from langchain.smith import RunEvalConfig, run_on_dataset

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class RAGEvaluator:
    """
    Class for evaluating the performance of a RAG (Retrieval Augmented Generation) system.
    """
    
    def __init__(self, evaluation_data_path: Optional[str] = None, output_dir: str = "./evaluation_results"):
        """
        Initialize the RAG Evaluator.
        
        Args:
            evaluation_data_path: Path to evaluation dataset (optional)
            output_dir: Directory to store evaluation results
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.evaluation_data_path = evaluation_data_path
        self.evaluation_data = None
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load evaluation data if provided
        if evaluation_data_path and os.path.exists(evaluation_data_path):
            self._load_evaluation_data()
    
    # Modifiez la méthode _load_evaluation_data dans src/evaluator.py

    def _load_evaluation_data(self):
      """Load evaluation data from the specified path."""
      try:
        with open(self.evaluation_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Vérifier si les données sont dans un format complet ou sous une clé particulière
        if isinstance(data, list):
            # Format attendu: liste directe d'exemples
            self.evaluation_data = data
        elif isinstance(data, dict):
            # Vérifier les clés possibles contenant les exemples
            if "evaluation" in data:
                self.evaluation_data = data["evaluation"]
            elif "examples" in data:
                self.evaluation_data = data["examples"]
            elif "data" in data:
                self.evaluation_data = data["data"]
            elif "questions" in data:
                self.evaluation_data = data["questions"]
            elif len(data) == 1:
                # S'il n'y a qu'une seule clé, utiliser sa valeur
                self.evaluation_data = list(data.values())[0]
            else:
                # Si c'est un dictionnaire avec plusieurs questions directement
                for key in data:
                    if isinstance(data[key], dict) and "question" in data[key]:
                        # Convertir le dictionnaire en liste
                        self.evaluation_data = [data[key] for key in data]
                        break
                else:
                    # Si aucun format reconnu n'est trouvé
                    self.logger.warning("Format d'évaluation non reconnu, tentative de conversion...")
                    try:
                        # Essayer de convertir un dict de questions en format attendu
                        self.evaluation_data = [{"question": k, "reference_answer": v} 
                                              for k, v in data.items() if isinstance(v, str)]
                    except:
                        raise ValueError(f"Format de données d'évaluation non reconnu: {data.keys()}")
        else:
            raise ValueError(f"Format de données d'évaluation non reconnu")
            
        # Renommer les clés si nécessaire (answer -> reference_answer)
        for item in self.evaluation_data:
            if "answer" in item and "reference_answer" not in item:
                item["reference_answer"] = item.pop("answer")
        
        self.logger.info(f"Loaded evaluation data with {len(self.evaluation_data)} examples")
      except Exception as e:
        self.logger.error(f"Error loading evaluation data: {e}")
        self.evaluation_data = None

    def create_evaluation_dataset(self, qa_pairs: List[Dict[str, str]], 
                                 output_path: Optional[str] = None) -> str:
        """
        Create an evaluation dataset from a list of question-answer pairs.
        
        Args:
            qa_pairs: List of dictionaries with 'question' and 'reference_answer' keys
            output_path: Path to save the evaluation dataset (optional)
            
        Returns:
            Path to the saved evaluation dataset
        """
        # Format the evaluation data
        evaluation_data = []
        for i, qa_pair in enumerate(qa_pairs):
            if 'question' not in qa_pair or 'reference_answer' not in qa_pair:
                self.logger.warning(f"Skipping QA pair {i} due to missing keys")
                continue
                
            evaluation_data.append({
                "id": f"example_{i}",
                "question": qa_pair['question'],
                "reference_answer": qa_pair['reference_answer'],
                "metadata": qa_pair.get('metadata', {})
            })
        
        # Save the evaluation data
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"evaluation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Created evaluation dataset with {len(evaluation_data)} examples at {output_path}")
        
        # Update the current evaluation data path
        self.evaluation_data_path = output_path
        self.evaluation_data = evaluation_data
        
        return output_path
    
    def evaluate_single_response(self, 
                              question: str, 
                              generated_answer: str, 
                              reference_answer: Optional[str] = None,
                              retrieved_context: Optional[str] = None,
                              embeddings_model = None) -> Dict[str, Any]:
        """
        Evaluate a single response from the RAG system.
        
        Args:
            question: The user's question
            generated_answer: The answer generated by the system
            reference_answer: The reference (ground truth) answer (optional)
            retrieved_context: The context retrieved from the vector database (optional)
            embeddings_model: Model to use for semantic similarity (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Initialize results dictionary
        eval_results = {
            "question": question,
            "generated_answer": generated_answer,
            "metrics": {}
        }
        
        # Add optional inputs to results if provided
        if reference_answer:
            eval_results["reference_answer"] = reference_answer
        if retrieved_context:
            eval_results["retrieved_context"] = retrieved_context
        
        # 1. Answer Relevance (if reference answer is available)
        if reference_answer:
            # ROUGE scores (lexical overlap)
            try:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge_scores = scorer.score(reference_answer, generated_answer)
                
                eval_results["metrics"]["rouge1_f1"] = rouge_scores['rouge1'].fmeasure
                eval_results["metrics"]["rouge2_f1"] = rouge_scores['rouge2'].fmeasure
                eval_results["metrics"]["rougeL_f1"] = rouge_scores['rougeL'].fmeasure
                
                # BLEU score (n-gram precision)
                try:
                    smoothie = SmoothingFunction().method1
                    reference_tokens = [reference_answer.split()]
                    candidate_tokens = generated_answer.split()
                    
                    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, 
                                              smoothing_function=smoothie)
                    eval_results["metrics"]["bleu_score"] = bleu_score
                except Exception as e:
                    self.logger.warning(f"Error calculating BLEU score: {e}")
                    eval_results["metrics"]["bleu_score"] = None
                    
                # Semantic similarity (if embeddings model is provided)
                if embeddings_model:
                    try:
                        ref_embedding = embeddings_model.embed_documents([reference_answer])[0]
                        gen_embedding = embeddings_model.embed_documents([generated_answer])[0]
                        
                        similarity = cosine_similarity([ref_embedding], [gen_embedding])[0][0]
                        eval_results["metrics"]["semantic_similarity"] = float(similarity)
                    except Exception as e:
                        self.logger.warning(f"Error calculating semantic similarity: {e}")
                        eval_results["metrics"]["semantic_similarity"] = None
            except Exception as e:
                self.logger.error(f"Error calculating lexical metrics: {e}")
        
        # 2. Context Utilization (if context is available)
        if retrieved_context:
            # Check if the answer uses information from the context
            try:
                # Simple overlap calculation
                context_words = set(retrieved_context.lower().split())
                answer_words = set(generated_answer.lower().split())
                
                overlap_words = context_words.intersection(answer_words)
                context_utilization = len(overlap_words) / len(answer_words) if answer_words else 0
                
                eval_results["metrics"]["context_utilization"] = context_utilization
                
                # Check for hallucinations (information not in context)
                # This is a simple approximation, not perfect
                potential_hallucination_score = 1 - context_utilization
                eval_results["metrics"]["potential_hallucination_score"] = potential_hallucination_score
                
                # Semantic context relevance (if embeddings model is provided)
                if embeddings_model:
                    try:
                        context_embedding = embeddings_model.embed_documents([retrieved_context])[0]
                        question_embedding = embeddings_model.embed_documents([question])[0]
                        
                        context_relevance = cosine_similarity([question_embedding], [context_embedding])[0][0]
                        eval_results["metrics"]["context_relevance"] = float(context_relevance)
                    except Exception as e:
                        self.logger.warning(f"Error calculating context relevance: {e}")
                        eval_results["metrics"]["context_relevance"] = None
            except Exception as e:
                self.logger.error(f"Error evaluating context utilization: {e}")
        
        # 3. Answer Quality Metrics
        # Length and complexity
        eval_results["metrics"]["answer_length"] = len(generated_answer.split())
        
        return eval_results
    
    def evaluate_batch(self, 
                 qa_system,
                 evaluation_data: Optional[List[Dict[str, Any]]] = None,
                 embeddings_model = None,
                 output_path: Optional[str] = None) -> Dict[str, Any]:
      """
      Evaluate the RAG system on a batch of questions.
    
      Args:
        qa_system: The QA system to evaluate
        evaluation_data: List of evaluation examples (optional, will use self.evaluation_data if None)
        embeddings_model: Model to use for semantic similarity (optional)
        output_path: Path to save evaluation results (optional)
        
      Returns:
        Dictionary with evaluation results
      """
      if evaluation_data is None:
        if self.evaluation_data is None:
            raise ValueError("No evaluation data provided or loaded")
        evaluation_data = self.evaluation_data
    
      self.logger.info(f"Starting batch evaluation with {len(evaluation_data)} examples")
    
    # Initialize results
      all_results = []
      aggregated_metrics = {}
    
    # Process each evaluation example
      for i, example in enumerate(evaluation_data):
        self.logger.info(f"Evaluating example {i+1}/{len(evaluation_data)}")
        
        question = example["question"]
        reference_answer = example.get("reference_answer")
        
        # Get system response
        try:
            qa_result = qa_system.answer(question)
            generated_answer = qa_result["answer"]
            retrieved_context = qa_result.get("context_used", "")
            
            # Evaluate the response
            eval_result = self.evaluate_single_response(
                question=question,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                retrieved_context=retrieved_context,
                embeddings_model=embeddings_model
            )
            
            # Add example ID and metadata
            eval_result["id"] = example.get("id", f"example_{i}")
            eval_result["metadata"] = example.get("metadata", {})
            
            all_results.append(eval_result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating example {i}: {e}")
            # Add a failed evaluation to the results
            all_results.append({
                "id": example.get("id", f"example_{i}"),
                "question": question,
                "error": str(e),
                "metrics": {}
            })
    
    # Aggregate metrics
      metric_names = set()
      for result in all_results:
        metric_names.update(result.get("metrics", {}).keys())
    
      for metric in metric_names:
        values = [r.get("metrics", {}).get(metric) for r in all_results if metric in r.get("metrics", {})]
        values = [v for v in values if v is not None]  # Filter out None values
        
        if values:
            aggregated_metrics[metric] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values)
            }
    
    # Fonction locale pour convertir les types numpy en types Python standards
      def convert_numpy(obj):
        """Convertit les types numpy en types Python standards pour la sérialisation JSON."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    # Conversion des types numpy en types Python standards avant sérialisation
      for metric, values in aggregated_metrics.items():
        for stat, value in values.items():
            aggregated_metrics[metric][stat] = convert_numpy(value)

    # Préparation du rapport final
      evaluation_report = {
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(evaluation_data),
        "aggregated_metrics": aggregated_metrics,
        "individual_results": all_results
    }

    # Enregistrement des résultats
      if output_path is None:
        output_path = os.path.join(self.output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

      with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
    
      self.logger.info(f"Completed batch evaluation. Results saved to {output_path}")
    
      return evaluation_report

      
    def visualize_results(self, evaluation_results: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """
        Generate visualizations for evaluation results.
        
        Args:
            evaluation_results: Evaluation results from evaluate_batch
            output_dir: Directory to save visualizations (optional)
            
        Returns:
            Path to the directory with visualizations
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract metrics
        aggregated_metrics = evaluation_results.get("aggregated_metrics", {})
        individual_results = evaluation_results.get("individual_results", [])
        
        # 1. Create a summary table
        summary_data = []
        for metric, values in aggregated_metrics.items():
            summary_data.append({
                "Metric": metric,
                "Mean": values.get("mean"),
                "Median": values.get("median"),
                "Min": values.get("min"),
                "Max": values.get("max"),
                "Std Dev": values.get("std")
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # 2. Create bar charts for key metrics
        plt.figure(figsize=(10, 6))
        metrics_to_plot = [m for m in aggregated_metrics.keys() 
                          if any(word in m for word in ["rouge", "bleu", "similarity", "relevance"])]
        
        if metrics_to_plot:
            means = [aggregated_metrics[m]["mean"] for m in metrics_to_plot]
            plt.bar(metrics_to_plot, means)
            plt.title("Average Scores for Key Metrics")
            plt.ylabel("Score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            bar_chart_path = os.path.join(output_dir, "key_metrics_bar_chart.png")
            plt.savefig(bar_chart_path)
            plt.close()
        
        # 3. Create histograms for metrics
        for metric, values in aggregated_metrics.items():
            metric_values = [r.get("metrics", {}).get(metric) for r in individual_results]
            metric_values = [v for v in metric_values if v is not None]
            
            if metric_values:
                plt.figure(figsize=(8, 5))
                plt.hist(metric_values, bins=10, alpha=0.7)
                plt.title(f"Distribution of {metric}")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                hist_path = os.path.join(output_dir, f"{metric}_histogram.png")
                plt.savefig(hist_path)
                plt.close()
        
        # 4. Create scatter plot for context relevance vs answer quality (if available)
        if "context_relevance" in aggregated_metrics and "rougeL_f1" in aggregated_metrics:
            context_relevance = [r.get("metrics", {}).get("context_relevance") for r in individual_results]
            rouge_scores = [r.get("metrics", {}).get("rougeL_f1") for r in individual_results]
            
            # Filter out None values
            valid_points = [(cr, rs) for cr, rs in zip(context_relevance, rouge_scores) if cr is not None and rs is not None]
            
            if valid_points:
                x_vals, y_vals = zip(*valid_points)
                
                plt.figure(figsize=(8, 5))
                plt.scatter(x_vals, y_vals, alpha=0.7)
                plt.title("Context Relevance vs Answer Quality")
                plt.xlabel("Context Relevance")
                plt.ylabel("RougeL F1 Score")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                scatter_path = os.path.join(output_dir, "context_vs_quality_scatter.png")
                plt.savefig(scatter_path)
                plt.close()
        
        self.logger.info(f"Visualizations saved to {output_dir}")
        return output_dir
    
    def numpy_encoder(obj):
      """
      Helper function to handle numpy data types in JSON serialization.
      """
      if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
      elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
      elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
      elif isinstance(obj, (datetime,)):
        return obj.isoformat()
      else:
        return obj