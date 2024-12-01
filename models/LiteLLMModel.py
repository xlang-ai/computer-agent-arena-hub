import os
import json
import time
from litellm import completion
from .BaseModel import BaseModel
from backend.logger import model_logger as logger
from backend.agents.utils.utils import Timer

def setup_api_keys():
    keys_str = os.environ['API_KEYS']
    keys = json.loads(keys_str)
    
    for api_env_name in keys:
        os.environ[api_env_name] = keys[api_env_name]

setup_api_keys()

class LiteLLMModel(BaseModel):
    
    @BaseModel.retry(max_retries=3, retry_delay=2, backoff_factor=2, exceptions=(Exception,))
    def _completion(self, messages: list, max_tokens: int, top_p: float, temperature: float):
        response = completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                verbose=True
            )
        
        return response
    
    def completion(self, messages: list, max_tokens=2000, top_p=0.9, temperature=0.5):
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