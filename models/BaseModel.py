"""
Base model class for all models
"""
import time
import random
from typing import List, Dict, Any
from functools import wraps
from abc import ABC, abstractmethod
    
class BaseModel(ABC):
    """Base model class for all models"""
    def __init__(self, model_name: str):
        """Initialize the base model.

        Args:
            model_name: The name of the model
        """
        self.model_name = model_name
    
    @abstractmethod
    def completion(
        self, 
        messages : List, 
        max_tokens : int = 2000, 
        top_p : float = 0.9, 
        temperature: float = 0.5) -> Dict:
        """
            Abstract method for model completion, output response and usage info.
            
            Args:
                messages (list): Input message list
                max_tokens (int, optional): Maximum token number. Defaults to 2000.
                top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
                temperature (float, optional): Sample temperature. Defaults to 0.5.
                
            Returns:
                dict: {
                    "response_text": str,  
                    "model_usage": dict,   
                    "response": object,    
                    "error": str (if any)
                }
            """
        pass
    
    def retry(max_retries=3, retry_delay=2, backoff_factor=2, exceptions=(Exception,)):
        """
        A decorator to add retry logic to a function.
        
        Args:
            max_retries (int): Maximum number of retries.
            retry_delay (float): Initial delay between retries in seconds.
            backoff_factor (float): Exponential backoff factor.
            exceptions (tuple): Tuple of exception classes to catch and retry.
        
        Returns:
            A decorated function with retry logic.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                retries = 0
                while retries < max_retries:
                    try:
                        # Attempt to execute the decorated function
                        return func(self, *args, **kwargs)
                    except exceptions as e:
                        retries += 1
                        if retries >= max_retries:
                            raise  # Exceeded retries, re-raise the exception
                        delay = retry_delay * (backoff_factor ** (retries - 1)) + random.uniform(0, 0.5)
                        print(f"Retrying {func.__name__} due to {str(e)} ({retries}/{max_retries}), waiting {delay:.2f}s...")
                        time.sleep(delay)
            return wrapper
        return decorator

    @abstractmethod
    @retry(max_retries=5, retry_delay=2, backoff_factor=2, exceptions=(Exception,))
    def _completion(self, messages: list, max_tokens: int, top_p: float, temperature: float) -> Any:
        """
            Call model completion API.

            This method is wrapped with a retry mechanism to handle transient failures or API rate limits.

            Args:
                messages (list): Input message list.
                max_tokens (int): Maximum number of tokens to include in the response.
                top_p (float): Top-p sampling parameter to control the diversity of the output.
                temperature (float): Sampling temperature to adjust the randomness of the output.

            Returns:
                Model response in any format.
        """
        raise NotImplementedError("Must be implemented in subclass")
    
    