from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class RAGQA:
    def __init__(self, retriever, config: dict):
        """
        retriever : instance de DocumentRetriever ou vectorstore.as_retriever()
        config : dictionnaire de configuration
        """
        self.config = config
        self.retriever_top_k = config.get("retriever_top_k", 4)
        self.chain_type = config.get("chain_type", "stuff")
        
        # Initialiser le modèle LLM (Mistral)
        model_name = config["llm_model_name"]
        temperature = config.get("model_temperature", 0.1)
        max_length = config.get("model_max_length", 2048)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configuration pour optimiser l'utilisation mémoire
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True  # Utiliser une quantification 8-bit pour réduire l'empreinte mémoire
        )
        
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        self.retriever = retriever.vectordb.as_retriever(search_kwargs={"k": self.retriever_top_k})
        
        # Template de prompt optimisé pour Mistral
        prompt_template = """<s>[INST] Vous êtes un assistant IA expert. Utilisez le contexte ci-dessous pour répondre à la question de façon précise et concise.
        
        Contexte:
        {context}
        
        Question: {question} [/INST]
        """
        
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Création de la chaîne avec le prompt personnalisé
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=self.chain_type,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def answer_question(self, question: str) -> dict:
        """
        Retourne un dictionnaire contenant la réponse et éventuellement les documents sources.
        """
        result = self.qa_chain({"query": question})
        return result  # {'result': '...', 'source_documents': [...]}