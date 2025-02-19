import gradio as gr
import yaml

# Importe vos classes
from src.retriever import DocumentRetriever
from src.chatbot import ChatBot

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# 1) Charger la config et initialiser le chatbot
config = load_config()
retriever = DocumentRetriever(config)
chatbot = ChatBot(retriever, config)

# 2) Définir une fonction de réponse pour Gradio
def respond(message, history):
    """
    Gradio gère l'historique comme une liste de [user_msg, bot_msg].
    On ajoute le message de l'utilisateur, puis on appelle chatbot.chat(message).
    """
    bot_reply = chatbot.chat(message)
    history.append((message, bot_reply))
    return history, history

# 3) Construire l'interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot RAG avec LangChain")

    # Chatbot() de Gradio attend une liste d'échanges [ [user, bot], [user, bot], ... ]
    chatbot_interface = gr.Chatbot(label="Conversation")
    message_box = gr.Textbox(label="Tapez votre message ici…")

    # Lorsqu’on soumet le message_box, on appelle respond()
    # - La fonction prend en entrée (message, history)
    # - Elle renvoie (nouvelle_history, nouvelle_history) qui met à jour l’affichage
    message_box.submit(fn=respond,
                       inputs=[message_box, chatbot_interface],
                       outputs=[chatbot_interface, chatbot_interface])

    # Bouton de reset
    clear_btn = gr.Button("Effacer la conversation")
    def clear_conversation():
        return []
    clear_btn.click(fn=clear_conversation,
                    inputs=[],
                    outputs=[chatbot_interface])

# 4) Lancer l'appli Gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
