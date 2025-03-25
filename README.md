# Guide d'utilisation du projet

## Structure du Projet

```
├── cli.py
├── config.yaml
├── requirements.txt
├── data/
│   └── (vos fichiers PDF)
└── src/
    ├── __init__.py
    ├── indexer.py
    ├── retriever.py
    ├── llm_handler.py
    ├── qa_system.py
    ├── prompt_templates.py
    ├── evaluator.py
    ├── chatbot.py
    ├── chatbot_cli.py
    └── chatbot_ui.py
```

## Mode Conversation Dynamique
Pour lancer une session interactive où vous pouvez changer de LLM et poser des questions en direct, utilisez :

```bash
python -m src.interactive_qa
```

Cela ouvrira un environnement interactif où vous pourrez interagir avec différents modèles de langage.

## Évaluation du Modèle
Pour évaluer un modèle spécifique avec un jeu de données d'évaluation, utilisez la commande suivante :

```bash
python cli.py evaluate --model deepseek --eval-data ./data/evaluation_data.json
```

Remplacez `deepseek` par le nom du modèle q et assurez-vous que le fichier `evaluation_data.json` contient les bonnes données.

## Lancer le Chatbot avec Streamlit
Pour démarrer l'interface utilisateur du chatbot avec Streamlit, exécutez la commande suivante :

```bash
streamlit run src/chatbot_ui_simple.py --server.port 8501 --server.address 127.0.0.1
```

Cela lancera une interface web accessible sur `http://127.0.0.1:8501/`.

## Prérequis
Avant de commencer, assurez-vous d'avoir installé toutes les dépendances nécessaires. Si ce n'est pas encore fait, exécutez :

```bash
pip install -r requirements.txt
```

## Contact
Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur le dépôt GitHub ou à contacter l'équipe de développement.

