# src/prompt_templates.py

"""
Collection of prompt templates optimized for different LLM models.
These templates are designed to get the best performance from each model
for the retrieval-augmented generation task.
"""

# Template simple pour la plupart des modèles
STANDARD_TEMPLATE = """
En tant qu'assistant IA, réponds à la question ci-dessous en utilisant uniquement les informations fournies dans les extraits de documents.
Si l'information nécessaire n'est pas présente dans les extraits, indique simplement que tu ne disposes pas de suffisamment d'informations.

Extraits de documents:
{context}

Question: {question}

Réponse:
"""

# Template optimisé pour DeepSeek
DEEPSEEK_TEMPLATE = """
Tu es un assistant IA qui répond uniquement à partir des informations contenues dans les extraits ci-dessous.
Ne génère pas d'informations qui ne figurent pas dans ces extraits.

Extraits de documents:
{context}

Question: {question}

Réponse (basée uniquement sur les extraits ci-dessus):
"""

# Template optimisé pour Mistral
MISTRAL_TEMPLATE = """
<s>[INST] Réponds à la question suivante en te basant uniquement sur les informations présentes dans les extraits fournis.
N'invente aucune information qui ne serait pas mentionnée explicitement dans ces extraits.

Extraits de documents:
{context}

Question: {question} [/INST]

"""

# Mapping des modèles vers leurs templates optimaux
MODEL_TEMPLATES = {
    "deepseek": DEEPSEEK_TEMPLATE,
    "mistral": MISTRAL_TEMPLATE
}

# Mapping des types de tâches vers des templates spécialisés
TASK_TEMPLATES = {
    "standard": STANDARD_TEMPLATE,
    "concise": """
Fournis une réponse brève et directe à la question, en te basant uniquement sur les extraits de documents ci-dessous.
Limite ta réponse à 2-3 phrases maximum.

Extraits:
{context}

Question: {question}

Réponse concise:
""",
    "explanatory": """
Explique en détail la réponse à la question ci-dessous, en te basant uniquement sur les extraits fournis.
Organise ta réponse de manière structurée et pédagogique.

Extraits de documents:
{context}

Question: {question}

Explication détaillée:
"""
}

def get_template_for_model(model_name, task_type="standard"):
    """
    Get the optimal prompt template for a specific model and task type.
    
    Args:
        model_name: Name of the LLM model
        task_type: Type of task (standard, concise, explanatory)
        
    Returns:
        String template with {context} and {question} placeholders
    """
    # First try to get a model-specific template
    if model_name.lower() in MODEL_TEMPLATES:
        return MODEL_TEMPLATES[model_name.lower()]
    
    # If no model-specific template, use task-specific template
    if task_type.lower() in TASK_TEMPLATES:
        return TASK_TEMPLATES[task_type.lower()]
    
    # Default to standard template
    return STANDARD_TEMPLATE