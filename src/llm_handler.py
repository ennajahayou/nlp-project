# src/llm_handler.py

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
from typing import List, Dict, Any, Optional, Union
import os
from src.prompt_templates import get_template_for_model

class LLMHandler:
    """
    Class for handling different language models and generating responses based on retrieved context.
    """
    
    # Available models configuration
    AVAILABLE_MODELS = {
        "deepseek": {
            "name": "DeepSeek Coder",
            "model_id": "deepseek-ai/deepseek-coder-1.3b-base",  # Version plus petite et non restreinte
            "description": "Bon modèle pour le code et le contenu technique",
            "max_length": 1024,
            "temperature": 0.7,
            "requires_gpu": False  # Version 1.3B peut fonctionner sur CPU
        },
        "mistral": {
            "name": "Mistral 7B Instruct",
            "model_id": "mistralai/Mistral-7B-v0.1",  # Alternative non restreinte basée sur Mistral
            "description": "Bon équilibre entre qualité et efficacité",
            "max_length": 1024,
            "temperature": 0.7,
            "requires_gpu": False
        }
    }
    
    def __init__(self, model_name: str = "mistral", device: str = None):
        """
        Initialize the LLM Handler.
        
        Args:
            model_name: Name of the model to use (one of: "deepseek", "mistral")
            device: Device to run the model on ("cpu", "cuda", "mps", etc.)
                   If None, will automatically detect the best available device
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Validate model name
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Available models: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = self.AVAILABLE_MODELS[model_name]
        
        # Determine device
        if device is None:
            device = self._get_optimal_device(self.model_config["requires_gpu"])
        
        self.device = device
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.logger.info(f"Initializing {self.model_config['name']} model...")
        self.tokenizer = None
        self.model = None
        self.llm = None
        
        try:
            self._initialize_model()
            # Vérifier si llm a été correctement initialisé
            if self.llm is None:
                raise RuntimeError(f"Model initialization completed but llm attribute is None")
        except Exception as e:
            self.logger.error(f"Error initializing {model_name} model: {e}")
            # Si le modèle demandé échoue, essayer l'autre modèle comme secours
            fallback_model = "deepseek" if model_name == "mistral" else "mistral"
            try:
                self.logger.info(f"Attempting to initialize fallback model ({fallback_model})")
                self.model_name = fallback_model
                self.model_config = self.AVAILABLE_MODELS[fallback_model]
                self._initialize_model()
                if self.llm is None:
                    raise RuntimeError("Fallback initialization completed but llm attribute is still None")
            except Exception as fallback_e:
                self.logger.error(f"Fallback initialization also failed: {fallback_e}")
                raise RuntimeError(f"Failed to initialize model {model_name} and fallback: {str(e)} / {str(fallback_e)}")
        
        # Initialize prompt templates
        self.qa_prompt_template = self._create_qa_prompt_template()
    
    def _get_optimal_device(self, requires_gpu: bool) -> str:
        """
        Determine the optimal device based on availability and model requirements.
        
        Args:
            requires_gpu: Whether the model requires GPU to run effectively
            
        Returns:
            Device string ("cuda", "mps", or "cpu")
        """
        if requires_gpu:
            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                self.logger.info("CUDA is available, using GPU")
                return "cuda"
            # Check for MPS (Apple Silicon)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.logger.info("MPS is available, using Apple Silicon acceleration")
                return "mps"
            else:
                self.logger.warning(f"Model {self.model_name} recommended with GPU, but no GPU available. Using CPU (will be slow)")
                return "cpu"
        else:
            # For models that don't necessarily need GPU
            if torch.cuda.is_available():
                self.logger.info("CUDA is available, using GPU for faster inference")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.logger.info("MPS is available, using Apple Silicon acceleration")
                return "mps"
            else:
                return "cpu"
    
    def _initialize_model(self):
      """
      Initialize the model and tokenizer with simpler parameters.
      """
      try:
        model_id = self.model_config["model_id"]
        self.logger.info(f"Loading model: {model_id}")
        
        # Paramètres simplifiés pour le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Assurer que le tokenizer a un pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Paramètres simplifiés pour le modèle - moins d'options pour éviter les problèmes
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }
        
        # Ajouter des paramètres GPU uniquement si on n'est pas sur CPU
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = self.device
        
        self.logger.info(f"Loading model with simplified params: {model_kwargs}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        # Paramètres de génération simplifiés
        generation_config = {
            "max_new_tokens": 512,  # Valeur plus conservatrice
            "temperature": 0.5,     # Plus déterministe
            "do_sample": True
        }
        
        # Créer la pipeline de génération
        self.logger.info(f"Creating generation pipeline with config: {generation_config}")
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            **generation_config
        )
        
        # Créer le wrapper LangChain
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.logger.info(f"Successfully initialized {self.model_config['name']}")
        
      except Exception as e:
        self.logger.error(f"Error initializing model: {e}")
        raise RuntimeError(f"Failed to initialize model {self.model_name}: {str(e)}")
    
    def _create_qa_prompt_template(self, task_type="standard") -> PromptTemplate:
        """
        Create a prompt template for question answering.
        
        Args:
            task_type: Type of QA task (standard, academic, concise, explanatory)
        
        Returns:
            PromptTemplate instance
        """
        template = get_template_for_model(self.model_name, task_type)
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template.strip()
        )
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Generate an answer to a question based on the provided context.
        
        Args:
            question: The user's question
            context: The context information retrieved from the vector database
            
        Returns:
            Generated answer
        """
        if not self.llm:
            raise RuntimeError("Model not initialized")
        
        try:
            # Créer un prompt très direct sans balises ni formatage spécial
            prompt = f"""Voici des extraits de documents:

{context}

Question: {question}

Réponse:"""
            
            # Générer la réponse en appelant directement le modèle
            self.logger.info(f"Generating answer for question: {question}")
            
            # Appel direct pour éviter les problèmes de LLMChain
            response = self.llm(prompt, stop=["Question:", "\nQuestion", "\n\nQuestion"])
            
            # Nettoyer la réponse
            cleaned_response = self._clean_response(response)
            
            return cleaned_response
                
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return f"Désolé, une erreur s'est produite lors de la génération de la réponse: {str(e)}"

    def _clean_response(self, response: str) -> str:
        """
        Clean up the LLM response to extract only the direct answer.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Cleaned response
        """
        if not response:
            return "Je n'ai pas pu générer de réponse basée sur le contexte fourni."
            
        # Si la réponse contient des balises spécifiques au modèle DeepSeek
        if "<|im_start|>" in response and "<|im_end|>" in response:
            try:
                # Essayer d'extraire la partie assistant uniquement
                assistant_parts = response.split("<|im_start|>assistant")
                if len(assistant_parts) > 1:
                    # Prendre la dernière partie après "assistant"
                    response = assistant_parts[-1]
                    # Supprimer la balise de fin
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0]
            except Exception as e:
                self.logger.warning(f"Erreur lors du nettoyage de la réponse DeepSeek: {e}")
        
        # Pour le modèle Mistral
        elif "[/INST]" in response:
            try:
                # Extraire ce qui vient après le dernier [/INST]
                response = response.split("[/INST]")[-1]
            except Exception as e:
                self.logger.warning(f"Erreur lors du nettoyage de la réponse Mistral: {e}")
        
        # Supprimer tout le texte de prompt qui pourrait avoir été répété
        if "Voici des extraits de documents:" in response:
            try:
                response = response.split("Réponse:", 1)[-1]
            except:
                pass
        
        # Split by common separators that might indicate a new question
        separators = ["Question:", "Human:", "User:", "Q:", "Voici le contexte:"]
        
        # Get only the first part (before any new question)
        for separator in separators:
            if separator in response:
                response = response.split(separator)[0]
        
        # Remove any common prefixes
        prefixes = ["Answer:", "Réponse:", "assistant"]
        for prefix in prefixes:
            if response.strip().startswith(prefix):
                response = response.strip()[len(prefix):].strip()
        
        return response.strip()
    
    def list_available_models(self) -> List[Dict[str, str]]:
        """
        List all available models with their descriptions.
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        for model_id, config in self.AVAILABLE_MODELS.items():
            models.append({
                "id": model_id,
                "name": config["name"],
                "description": config["description"],
                "requires_gpu": config["requires_gpu"]
            })
        return models
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different language model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            Boolean indicating success
        """
        if model_name not in self.AVAILABLE_MODELS:
            self.logger.error(f"Model '{model_name}' not supported")
            return False
        
        if model_name == self.model_name:
            self.logger.info(f"Already using {self.model_config['name']}")
            return True
        
        try:
            # Clean up current model to free memory
            if self.model:
                del self.model
                del self.tokenizer
                del self.llm
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Update model name and config
            self.model_name = model_name
            self.model_config = self.AVAILABLE_MODELS[model_name]
            
            # Re-initialize with new model
            device = self._get_optimal_device(self.model_config["requires_gpu"])
            self.device = device
            self._initialize_model()
            
            # Update prompt template for the new model
            self.qa_prompt_template = self._create_qa_prompt_template()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error switching model: {e}")
            return False