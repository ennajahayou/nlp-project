from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class ChatBot:
    def __init__(self, retriever, config: dict):
        self.config = config
        # Récupérer le retriever sans arguments
        self.retriever = retriever.vectordb.as_retriever()
        # Mettre à jour le nombre de résultats dans search_kwargs
        self.retriever.search_kwargs["k"] = config["retriever_top_k"]
        
        model_name = config["llm_model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Construire la chaîne conversationnelle
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory
        )

    def chat(self, user_input: str):
        """
        Gère une conversation continue avec l'utilisateur.
        """
        result = self.chat_chain({"question": user_input})
        return result["answer"]
