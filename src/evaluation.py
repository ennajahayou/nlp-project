class Evaluation:
    def __init__(self):
        pass

    def evaluate(self, generated_answer: str, reference_answer: str) -> float:
        """
        Compare la réponse générée avec une référence (ex: via une similarité sémantique).
        Retourne un score (par ex. 0 à 1).
        """
        # Cette partie est à implémenter selon vos besoins.
        # On peut utiliser par exemple une similarité cosinus sur embeddings, ou un BLEU, etc.
        
        if not reference_answer:
            return 0.0
        
        # Exemple simpliste : note 1.0 si égalité parfaite, 0.0 sinon
        return 1.0 if generated_answer.strip() == reference_answer.strip() else 0.0
