from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from typing import Optional

class RAGQA:
    def __init__(self, retriever, config: dict):
        """
        retriever : instance de DocumentRetriever ou vectorstore.as_retriever()
        config : dictionnaire de configuration
        """
        self.config = config
        self.retriever_top_k = config.get("retriever_top_k", 4)
        self.chain_type = config.get("chain_type", "stuff")
        
        # Initialiser le modèle LLM (ex: Flan-T5)
        # Vous pouvez utiliser un pipeline transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
        
        model_name = config["llm_model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer
        )
        
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        # Option 1: Utiliser directly un retrieveur depuis la classe DocumentRetriever
        # Option 2: Faire un vectordb.as_retriever() pour le passer à RetrievalQA
        # Ici, on va transformer le vectordb en 'retriever' (LangChain) si besoin
        
        self.retriever = retriever.vectordb.as_retriever(search_kwargs={"k": self.retriever_top_k})
        
        # Template de prompt si besoin
        prompt_template = """
        Vous êtes un assistant IA. Utilisez le contexte ci-dessous pour répondre à la question.
        Si la réponse n'est pas dans le contexte, dites que vous ne savez pas.

        Contexte:
        {context}

        Question:
        {question}

        Réponse:
        """
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Création de la chaîne
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=self.chain_type,  # "stuff", "map_reduce", ...
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def answer_question(self, question: str) -> dict:
        """
        Retourne un dictionnaire contenant la réponse et éventuellement les documents sources.
        """
        result = self.qa_chain({"query": question})
        return result  # {'result': '...', 'source_documents': [...]}
