import argparse
import yaml
import os

from src.data_loader import DataLoader
from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever
from src.rag_qa import RAGQA
from src.evaluation import Evaluation
from src.chatbot import ChatBot


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="CLI pour le projet RAG")
    parser.add_argument("command", type=str, help="Commande à exécuter: index, query, qa, eval, chat")
    parser.add_argument("--query", type=str, help="Requête utilisateur pour la recherche ou QA")
    parser.add_argument("--ref", type=str, help="Réponse de référence pour l'évaluation")
    args = parser.parse_args()
    
    config = load_config()

    if args.command == "index":
        # 1) Charger les documents
        loader = DataLoader(config)
        docs = loader.load_documents()
        
        # 2) Indexer
        indexer = DocumentIndexer(config)
        vectordb = indexer.create_index(docs)
        print("Indexation terminée !")
    
    elif args.command == "query":
        if not args.query:
            print("Veuillez spécifier --query")
            return
        
        # Charger la base vectorielle puis effectuer une recherche
        retriever = DocumentRetriever(config)
        docs_with_scores = retriever.get_relevant_documents(args.query, k=config["retriever_top_k"])
        
        print(f"Top {config['retriever_top_k']} documents pertinents :")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"{i+1}. Score = {score:.4f}, Extrait = {doc.page_content[:200]}...")
    
    elif args.command == "qa":
        if not args.query:
            print("Veuillez spécifier --query")
            return
        
        # QA
        retriever = DocumentRetriever(config)
        rag_qa = RAGQA(retriever, config)
        result = rag_qa.answer_question(args.query)
        
        print("Réponse générée :")
        print(result["result"])
        print("\nDocuments sources utilisés :")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata}")
    
    elif args.command == "eval":
        # Évaluation simple
        if not args.query or not args.ref:
            print("Veuillez spécifier --query et --ref")
            return
        
        retriever = DocumentRetriever(config)
        rag_qa = RAGQA(retriever, config)
        result = rag_qa.answer_question(args.query)
        generated_answer = result["result"]
        
        evaluator = Evaluation()
        score = evaluator.evaluate(generated_answer, args.ref)
        
        print(f"Réponse générée : {generated_answer}")
        print(f"Réponse de référence : {args.ref}")
        print(f"Score : {score}")
    
    elif args.command == "chat":
        """
        Exemple de boucle de chat simplifiée.
        Pour chaque input utilisateur, on appelle chat().
        L'historique est géré par la mémoire interne de la classe ChatBot.
        """
        retriever = DocumentRetriever(config)
        chatbot = ChatBot(retriever, config)
        
        print("Démarrage du ChatBot (Tapez 'exit' pour quitter)")
        while True:
            user_input = input("Vous: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Fermeture du chatbot.")
                break
            response = chatbot.chat(user_input)
            print(f"Bot: {response}")

    else:
        print("Commande inconnue. Options : index, query, qa, eval, chat.")

if __name__ == "__main__":
    main()
