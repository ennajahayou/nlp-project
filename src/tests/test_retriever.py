# tests/test_retriever.py

import unittest
import os
import sys
import yaml
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever

class TestDocumentRetriever(unittest.TestCase):
    """
    Unit tests for DocumentRetriever functionality.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for the vector store
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db_path = os.path.join(self.temp_dir, "test_chroma_db")
        
        # Create test documents
        self.test_docs_dir = os.path.join(self.temp_dir, "test_docs")
        os.makedirs(self.test_docs_dir, exist_ok=True)
        
        # Create a sample text file for testing
        self.test_file_path = os.path.join(self.test_docs_dir, "test_doc.txt")
        with open(self.test_file_path, "w") as f:
            f.write("This is a test document about artificial intelligence. " 
                   "AI is transforming many industries including healthcare and finance. "
                   "Machine learning is a subset of AI that allows systems to learn from data.")
        
        # Create a test config
        self.config = {
            'indexing': {
                'embeddings': {'model': 'sentence-transformers/all-MiniLM-L6-v2'},  # Use a small, fast model for tests
                'vector_db': {'path': self.vector_db_path},
                'document_processing': {'chunk_size': 500, 'chunk_overlap': 50}
            },
            'data': {'documents_dir': self.test_docs_dir}
        }
        
        # Create and initialize the indexer with test document
        self.indexer = DocumentIndexer(
            embedding_model_name=self.config['indexing']['embeddings']['model'],
            vector_db_path=self.vector_db_path,
            chunk_size=self.config['indexing']['document_processing']['chunk_size'],
            chunk_overlap=self.config['indexing']['document_processing']['chunk_overlap']
        )
        self.indexer.create_or_update_index(self.test_file_path)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temp directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_retriever_initialization(self):
        """Test initialization of retriever with existing indexer."""
        retriever = DocumentRetriever(
            indexer=self.indexer,
            top_k=3
        )
        self.assertIsNotNone(retriever.vector_store)
        self.assertIsNotNone(retriever.embeddings)
        self.assertEqual(retriever.top_k, 3)
    
    def test_standalone_retriever_initialization(self):
        """Test initialization of retriever without indexer."""
        retriever = DocumentRetriever(
            vector_db_path=self.vector_db_path,
            top_k=3
        )
        self.assertIsNotNone(retriever.vector_store)
        self.assertIsNotNone(retriever.embeddings)
    
    def test_basic_query(self):
        """Test basic document retrieval with a simple query."""
        retriever = DocumentRetriever(
            indexer=self.indexer,
            top_k=2
        )
        
        results = retriever.query("What is artificial intelligence?")
        
        # Verify we got results
        self.assertTrue(len(results) > 0)
        
        # Check result structure
        first_result = results[0]
        self.assertIn("content", first_result)
        self.assertIn("metadata", first_result)
        self.assertIn("score", first_result)
        
        # Verify metadata
        self.assertIn("source", first_result["metadata"])
        
        # Check content is relevant (contains "artificial intelligence")
        self.assertIn("artificial intelligence", first_result["content"].lower())
    
    def test_advanced_query(self):
        """Test advanced query returning both results and source scores."""
        retriever = DocumentRetriever(
            indexer=self.indexer,
            top_k=2
        )
        
        results, source_scores = retriever.advanced_query("machine learning")
        
        # Verify we got results
        self.assertTrue(len(results) > 0)
        
        # Verify source scores
        self.assertTrue(len(source_scores) > 0)
        
        # Should contain our test document
        filename = os.path.basename(self.test_file_path)
        self.assertIn(filename, source_scores)
    
    def test_empty_query(self):
        """Test handling of empty query string."""
        retriever = DocumentRetriever(
            indexer=self.indexer
        )
        
        results = retriever.query("")
        
        # Should return results even with empty query
        # (typically this returns all documents, but behavior can vary)
        self.assertIsInstance(results, list)
    
    def test_irrelevant_query(self):
        """Test query with terms not in the documents."""
        retriever = DocumentRetriever(
            indexer=self.indexer
        )
        
        results = retriever.query("quantum physics theory string theory")
        
        # Should still return results (albeit with lower relevance scores)
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()