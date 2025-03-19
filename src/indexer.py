# src/indexer.py

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
# Import our custom BGE-M3 wrapper
from src.embedding_models import BGEM3Embeddings
import os
from typing import List, Dict, Any, Optional, Union
import logging

class DocumentIndexer:
    """
    Class for indexing documents into a vector database.
    Supports different document types, embedding models, and vector stores.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-m3",
                 vector_db_path: str = "./chroma_db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the DocumentIndexer.
        
        Args:
            embedding_model_name: Name of the HuggingFace embedding model to use
            vector_db_path: Path to store the vector database
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between consecutive chunks
        """
        self.embedding_model_name = embedding_model_name
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        self.embeddings = self._initialize_embeddings(embedding_model_name)
        
        # Initialize vector store
        self.vector_store = None
    
    def _initialize_embeddings(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name or path of the embedding model
            
        Returns:
            Embedding model instance
        """
        self.logger.info(f"Initializing embedding model: {model_name}")
        
        # Handle special model types
        if model_name.startswith("openai:"):
            # Format: "openai:text-embedding-ada-002"
            openai_model = model_name.split(":", 1)[1]
            self.logger.info(f"Using OpenAI embedding model: {openai_model}")
            return OpenAIEmbeddings(model=openai_model)
        
        # Specific handling for BGE-M3
        if model_name.lower() == "baai/bge-m3":
            self.logger.info(f"Using BGE-M3 embedding model with custom wrapper")
            try:
                return BGEM3Embeddings(model_name=model_name)
            except Exception as e:
                self.logger.error(f"Error initializing BGE-M3 model: {e}")
                self.logger.info("Falling back to default embedding model")
                return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Other BGE models use a different class
        elif "bge" in model_name.lower():
            self.logger.info(f"Using BGE embedding model: {model_name}")
            try:
                return HuggingFaceBgeEmbeddings(model_name=model_name)
            except Exception as e:
                self.logger.error(f"Error initializing BGE model: {e}")
                self.logger.info("Falling back to default embedding model")
                return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Standard HuggingFace models
        try:
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            self.logger.error(f"Error initializing embedding model {model_name}: {e}")
            self.logger.info("Trying alternative embedding model that doesn't require TensorFlow")
            
            # Try using a model that doesn't depend on TensorFlow
            fallback_models = [
                "BAAI/bge-small-en-v1.5",  # Good performance, smaller model
                "all-mpnet-base-v2",       # Good performance, no TF dependency
                "all-MiniLM-L6-v2"         # Smaller, faster model
            ]
            
            for fallback_model in fallback_models:
                try:
                    self.logger.info(f"Trying fallback model: {fallback_model}")
                    if "bge" in fallback_model.lower():
                        return HuggingFaceBgeEmbeddings(model_name=fallback_model)
                    else:
                        return HuggingFaceEmbeddings(model_name=fallback_model)
                except Exception as fallback_error:
                    self.logger.error(f"Error with fallback model {fallback_model}: {fallback_error}")
            
            raise ValueError(f"Failed to initialize any embedding model. Please install required dependencies or specify a different model.")
    
    def _get_loader_for_file(self, file_path: str):
        """
        Get the appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document loader instance
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path)
        elif file_extension in ['.md', '.markdown']:
            return UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _get_text_splitter(self, file_path: str = None):
        """
        Get the appropriate text splitter based on file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Text splitter instance
        """
        if file_path and os.path.splitext(file_path)[1].lower() in ['.md', '.markdown']:
            return MarkdownTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
    
    def load_and_split_document(self, file_path: str) -> List[Any]:
        """
        Load a document and split it into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        self.logger.info(f"Loading document: {file_path}")
        
        # Get appropriate loader
        loader = self._get_loader_for_file(file_path)
        
        # Load the document
        documents = loader.load()
        
        # Add source filename to metadata
        for doc in documents:
            doc.metadata["source"] = os.path.basename(file_path)
        
        # Split the document
        splitter = self._get_text_splitter(file_path)
        chunks = splitter.split_documents(documents)
        
        self.logger.info(f"Split {file_path} into {len(chunks)} chunks")
        return chunks
    
    def load_and_split_documents(self, file_paths: List[str]) -> List[Any]:
        """
        Load multiple documents and split them into chunks.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of document chunks with metadata
        """
        all_chunks = []
        for file_path in file_paths:
            chunks = self.load_and_split_document(file_path)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def create_or_update_index(self, file_paths: Union[str, List[str]]) -> None:
        """
        Create or update the vector index with documents.
        
        Args:
            file_paths: Path or list of paths to document files
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Load and split documents
        chunks = self.load_and_split_documents(file_paths)
        
        # If vector store already exists, add to it, otherwise create new
        if self.vector_store:
            self.logger.info("Adding documents to existing vector store")
            self.vector_store.add_documents(chunks)
        else:
            self.logger.info("Creating new vector store")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vector_db_path
            )
        
        # Persist the vector store
        self.vector_store.persist()
        self.logger.info(f"Vector store persisted to {self.vector_db_path}")
    
    def load_existing_index(self) -> bool:
        """
        Load an existing vector index if it exists.
        
        Returns:
            Boolean indicating whether an index was successfully loaded
        """
        if os.path.exists(self.vector_db_path):
            self.logger.info(f"Loading existing vector store from {self.vector_db_path}")
            self.vector_store = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
            return True
        else:
            self.logger.info("No existing vector store found")
            return False
    
    def change_embedding_model(self, model_name: str) -> None:
        """
        Change the embedding model and rebuild the index if needed.
        
        Args:
            model_name: Name of the new HuggingFace embedding model
        """
        self.logger.info(f"Changing embedding model to: {model_name}")
        self.embedding_model_name = model_name
        self.embeddings = self._initialize_embeddings(model_name)
        
        # If vector store exists, it needs to be rebuilt with the new embeddings
        if self.vector_store:
            self.logger.warning("Embedding model changed. Vector store must be rebuilt.")
            self.vector_store = None