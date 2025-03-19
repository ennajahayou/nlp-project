# src/example_queries.py

from src.retriever import DocumentRetriever
import yaml
import json

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_example_queries(config_path="config.yaml"):
    """Run a set of example queries and display the results."""
    # Load configuration
    config = load_config(config_path)
    
    # Create retriever
    retriever = DocumentRetriever(
        vector_db_path=config['indexing']['vector_db']['path']
    )
    
    # List of example queries
    example_queries = [
        "Qu'est-ce que l'intelligence artificielle?",
        "Comment fonctionne l'apprentissage par renforcement?",
        "Quels sont les enjeux éthiques de l'IA?",
        "Expliquez les réseaux de neurones profonds",
        "Quelles sont les applications de l'IA dans la santé?"
    ]
    
    # Run each query and display results
    for i, query in enumerate(example_queries):
        print(f"\n\n{'='*50}")
        print(f"EXAMPLE QUERY {i+1}: {query}")
        print(f"{'='*50}")
        
        # Get results
        results, source_scores = retriever.advanced_query(query)
        
        # Display document sources
        print("\nDOCUMENT SOURCES:")
        print("-"*30)
        for source, score in source_scores.items():
            print(f"{source}: {score}% relevance")
        
        # Display detailed results
        print("\nDETAILED RESULTS:")
        print("-"*30)
        for j, result in enumerate(results):
            print(f"\nResult {j+1} ({result['score']}% relevance):")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
            
            # If the source is a PDF, include page number if available
            if 'page' in result['metadata']:
                print(f"Page: {result['metadata']['page']}")
                
            # Print a snippet of the content (first 150 chars)
            content_preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
            print(f"Content: {content_preview}")

if __name__ == "__main__":
    run_example_queries()