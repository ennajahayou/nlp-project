import gradio as gr
import yaml

# Importer vos classes
from src.retriever import DocumentRetriever
from src.chatbot import ChatBot

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 1) Charger la configuration et initialiser le chatbot
config = load_config()
retriever = DocumentRetriever(config)
chatbot = ChatBot(retriever, config)

# 2) Définir la fonction de réponse pour Gradio
def respond(message, history):
    """
    Gradio gère l'historique comme une liste de [message_utilisateur, réponse_bot].
    On ajoute le message de l'utilisateur, puis on appelle chatbot.chat(message).
    """
    bot_reply = chatbot.chat(message)
    history.append((message, bot_reply))
    return history, history

# 3) Construire l'interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot RAG avec LangChain")
    
    # Composant pour afficher la conversation (liste de [user, bot])
    chatbot_interface = gr.Chatbot(label="Conversation")
    # Zone de saisie pour le message utilisateur
    message_box = gr.Textbox(label="Tapez votre message ici…")
    
    # Lors de la soumission dans message_box, on appelle la fonction respond
    message_box.submit(fn=respond,
                       inputs=[message_box, chatbot_interface],
                       outputs=[chatbot_interface, chatbot_interface])
    
    # Bouton pour réinitialiser la conversation
    def clear_conversation():
        return []
    clear_btn = gr.Button("Effacer la conversation")
    clear_btn.click(fn=clear_conversation,
                    inputs=[],
                    outputs=[chatbot_interface])

# 4) Lancer l'application Gradio avec un lien public
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
