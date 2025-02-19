import os
from langchain.document_loaders import PyPDFLoader
from typing import List
import yaml

class DataLoader:
    def __init__(self, config: dict):
        self.data_directory = config["data_directory"]
    
    def load_documents(self) -> List:
        """
        Parcourt le répertoire data et charge tous les PDF en tant que Documents.
        Retourne une liste de Documents (format LangChain).
        """
        documents = []
        for file_name in os.listdir(self.data_directory):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(self.data_directory, file_name)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
        return documents
