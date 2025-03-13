from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.prompts import PromptTemplate

class ChatBot:
    def __init__(self, retriever, config: dict):
        self.config = config
        # Récupérer le retriever sans arguments
        self.retriever = retriever.vectordb.as_retriever()
        # Mettre à jour le nombre de résultats dans search_kwargs
        self.retriever.search_kwargs["k"] = config["retriever_top_k"]
        
        model_name = config["llm_model_name"]
        temperature = config.get("model_temperature", 0.1)
        max_length = config.get("model_max_length", 2048)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configuration pour optimiser l'utilisation mémoire
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            load_in_8bit=True  # Pour économiser la mémoire
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
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Personnalisation du prompt pour Mistral
        condense_question_prompt = PromptTemplate.from_template(
            "<s>[INST] En fonction de l'historique de conversation suivant et d'une nouvelle question, "
            "reformule la nouvelle question pour qu'elle inclue tout le contexte pertinent. "
            "Historique:\n{chat_history}\nQuestion: {question} [/INST]"
        )
        
        qa_prompt = PromptTemplate.from_template(
            "<s>[INST] Vous êtes un assistant IA spécialisé dans les documents techniques. "
            "En utilisant seulement les informations ci-dessous, répondez à la question de manière claire et concise. "
            "Si vous ne trouvez pas la réponse dans les informations fournies, dites que vous ne savez pas.\n\n"
            "Contexte: {context}\n\n"
            "Question: {question} [/INST]"
        )
        
        # Construire la chaîne conversationnelle avec prompts personnalisés
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

    def chat(self, user_input: str):
        """
        Gère une conversation continue avec l'utilisateur.
        """
        result = self.chat_chain({"question": user_input})
        return result["answer"]