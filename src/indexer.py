import os
import yaml
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document  # Importer Document ici

class DocumentIndexer:
    def __init__(self, config: dict):
        self.config = config
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
        self.embedding_model_name = config["embedding_model_name"]
        self.persist_directory = config["persist_directory"]
        
        # Initialiser le modèle d'embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Assurez-vous que le dossier de persistence existe
        os.makedirs(self.persist_directory, exist_ok=True)

    def create_index(self, documents: List):
        """
        documents : liste de documents LangChain à indexer
        """
        # 1) Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        splitted_docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                meta = doc.metadata.copy()
                # Créer un objet Document à partir du chunk et des métadonnées
                splitted_docs.append(Document(page_content=chunk, metadata=meta))
        
        # 2) Stockage via Chroma
        vectordb = Chroma.from_documents(
            documents=splitted_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Sauvegarder l’index sur disque (persistence)
        vectordb.persist()
        return vectordb
