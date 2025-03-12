import os
import yaml
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class DocumentIndexer:
    def __init__(self, config_path: str):
        

        ## Initializes the DocumentIndexer

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.data_path = config['data_path']
        self.embedding_model = config['embedding_model']
        self.vector_store_path = config['vector_store_path']

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(config['chunk_size']), chunk_overlap=int(config['chunk_overlap'])
        )

        self.embedder = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.client = Chroma(persist_directory=self.vector_store_path, embedding_function=self.embedder)

    def load_documents(self, domain_folder: str):

        domain_path = os.path.join(self.data_path, domain_folder)
        documents = []
        
        if not os.path.exists(domain_path):
            raise FileNotFoundError(f"Domain folder '{domain_folder}' not found in data/ directory.")
        
        for file in os.listdir(domain_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(domain_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        
        return documents

    def split_documents(self, documents):
        """
        Splits documents into smaller chunks for indexing
        """
        return self.text_splitter.split_documents(documents)

    def embed_documents(self, chunks):
        """
        Generates embeddings for document chunks
        """
        return self.embedder.embed_documents([chunk.page_content for chunk in chunks])

    def store_embeddings(self, chunks):


        for i, chunk in enumerate(chunks):
            self.client.add_texts(texts=[chunk.page_content], metadatas=[{"id": i}])
        
        print("Embeddings stored successfully!")

    def index_domain(self, domain_folder: str):

        print(f"Indexing documents from {domain_folder}...")
        documents = self.load_documents(domain_folder)
        chunks = self.split_documents(documents)
        self.store_embeddings(chunks)
        print(f"Indexing completed for {domain_folder}!")

# Example usage (not to be included in module if used as a library)
if __name__ == "__main__":
    indexer = DocumentIndexer(config_path="../config/config.yaml")
    for domain in ["data1", "data2", "data3"]:
        indexer.index_domain(domain)
