from .LiteLLMModel import LiteLLMModel
from .BedrockModel import BedrockModel

try:
    from backend.logger import model_logger as logger
except:
    from test_env.logger import model_logger as logger

SUPPORTED_MODELS = {
    "litellm": [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4-turbo-2024-04-09",
        # "gpt-4-0125-preview",  Not available

        "gemini/gemini-1.5-flash",
        'gemini/gemini-1.5-pro', # 'gemini/gemini-1.5-pro-latest', 'gemini-1.5-pro-exp-0827', 'gemini-1.5-pro-002'
        'gemini/gemini-exp-1121',
        'gemini/gemini-2.0-flash',
        'gemini/gemini-2.0-flash-thinking-exp-01-21',
        'gemini/gemini-2.0-pro-exp-02-05',
        
        'openai/qwen2.5-vl-72b-instruct',
        # "claude-3-5-sonnet-20241022",
        # 'claude-3-haiku-20240307'
        # 'claude-3-opus-20240229',
    ],
    "bedrock":{
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "llama3-2-90b-instruct",
        # "claude-3-5-haiku-20241022", Text only not available
    }
} # Map from model name to model ID

class BackboneModel:
    """Backbone model class for all models"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_type = self._get_model_type(model_name)
        self.model = self._get_model_instance(model_name)
        self.completion = self.model.completion
        
    def _get_model_instance(self, model_name: str):
        """Get the model instance.

        Args:
            model_name: The name of the model
        """
        if self.model_type == "litellm":
            return LiteLLMModel(model_name)
        elif self.model_type == "bedrock":
            return BedrockModel(model_name)
        else:
            logger.error(f"Model type {self.model_type} is not supported")
            raise ValueError(f"Model type {self.model_type} is not supported")
    
    def _get_model_type(self, model_name: str):
        """Get the model type.

        Args:
            model_name: The name of the model
        """
        for model_type, models in SUPPORTED_MODELS.items():
            if model_name in models:
                return model_type
        
        logger.error(f"Model {model_name} is not supported")
        raise ValueError(f"Model {model_name} is not supported")
    
