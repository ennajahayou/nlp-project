import os
import yaml
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class DocumentRetriever:
    def __init__(self, config_path: str):
        

        ## Initializes the DocumentRetriever

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.vector_store_path = config['vector_store_path']
        self.embedding_model = config['embedding_model']
        self.embedder = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.client = Chroma(persist_directory=self.vector_store_path, embedding_function=self.embedder)

    def search_documents(self, query: str, top_k: int = 5):


        query_embedding = self.embedder.embed_query(query)[0]
        results = self.client.similarity_search(query_embedding, k=top_k)
        
        return [(result.metadata, result.page_content) for result in results]

    def display_results(self, query: str, top_k: int = 5):
        """
        Displays search results for a given query.
        :param query: The search query.
        :param top_k: Number of top results to return.
        """
        results = self.search_documents(query, top_k)
        print(f"Search results for query: '{query}'")
        for i, (metadata, content) in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Metadata: {metadata}")
            print(f"Content: {content[:500]}...")

# Example usage
if __name__ == "__main__":
    retriever = DocumentRetriever(config_path="../config/config.yaml")
    retriever.display_results("What is self-driving technology?", top_k=3)
