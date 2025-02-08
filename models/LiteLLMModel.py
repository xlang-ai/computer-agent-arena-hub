import os
import json
import time
from litellm import completion
from .BaseModel import BaseModel

try:
    # for deploy environment
    from backend.agents.utils.utils import Timer
    from backend.logger import model_logger as logger
except:
    # for local environment
    from utils import Timer
    from test_env.logger import model_logger as logger

def setup_api_keys():
    """Setup the API keys.

    Args:
        None
    """
    keys_str = os.environ['API_KEYS']
    keys = json.loads(keys_str)
    
    for api_env_name in keys:
        os.environ[api_env_name] = keys[api_env_name]
setup_api_keys()

class LiteLLMModel(BaseModel):
    """LiteLLM model class for all models"""
    @BaseModel.retry(max_retries=3, retry_delay=2, backoff_factor=2, exceptions=(Exception,))
    def _completion(self, messages: list, max_tokens: int, top_p: float, temperature: float):
        """Completion method for the LiteLLM model.

        Args:
            messages: The messages to complete
            max_tokens: The maximum number of tokens
            top_p: The top-p sampling parameter
            temperature: The sampling temperature
        """
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "verbose": True
        }
        
        # Add special configuration for Qwen models
        if "qwen" in self.model_name.lower():
            kwargs["api_key"] = os.getenv("DASHSCOPE_API_KEY")
            kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
        response = completion(**kwargs)
        
        return response
    
    def completion(self, messages: list, max_tokens=2000, top_p=0.9, temperature=0.5):
        """Completion method for the LiteLLM model.

        Args:
            messages: The messages to complete
            max_tokens: The maximum number of tokens
            top_p: The top-p sampling parameter
            temperature: The sampling temperature
        """
        try:
            with Timer() as timer:
                response = self._completion(
                    messages=messages, 
                    max_tokens=max_tokens,
                    top_p=top_p,
                    temperature=temperature
                    )

            if isinstance(response, str) or not response:
                logger.error(f"Unexpected string response: {response}")
                return {
                    "response_text": "",
                    "model_usage": {},
                    "response": response,
                    "error": "Unexpected string response"
                }

            response_text = response.choices[0].message.content.strip()
            model_usage = response.usage

            # logger.warning(f"{self.model_name} response: \n{model_usage} \n{response_text}")
            return {
                "response_text": response_text,
                "model_usage": {
                    'completion_tokens': model_usage.get('completion_tokens'),
                    'prompt_tokens': model_usage.get('prompt_tokens'),
                    'model_time': timer.duration,
                },
                "response": response,
                "error": None
            }
            
        except Exception as e:
            logger.exception(f"{self.model_name} completion error: {str(e)}")
            return {
                "response_text": "",
                "model_usage": {},
                "response": None,
                "error": str(e)
            }