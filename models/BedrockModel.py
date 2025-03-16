"""
Bedrock model class for all models
"""
import os
import boto3
from PIL import Image
import io
import base64
import json
from .BaseModel import BaseModel

try:
    # for deploy environment
    from backend.agents.utils.utils import Timer
    from backend.logger import model_logger as logger
except:
    # for local environment
    from utils import Timer
    from test_env.logger import model_logger as logger
    

ModelName2ModelID = {
    "claude-3-5-haiku-20241022": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-opus-20240229": "us.anthropic.claude-3-opus-20240229-v1:0",
    "llama3-2-90b-instruct": "us.meta.llama3-2-90b-instruct-v1:0"
} # Map from model name to model ID

class BedrockModel(BaseModel):
    """Bedrock model class for all models"""
    def __init__(self, model_name: str):
        """Initialize the Bedrock model.

        Args:
            model_name: The name of the model
        """
        super().__init__(model_name)
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name = "us-east-1"
        )

    def _format_content(self, content: list) -> list:
        """Format the content.

        Args:
            content: The content to format

        Returns:
            The formatted content
        """
        formatted_content = []
        if not self.model_name.startswith("llama"): # TEMP
            for item in content:
                if item["type"] == "text":
                    formatted_content.append(item)
                elif item["type"] == "image_url":
                    image_url = item["image_url"]["url"]
                    base64_data = image_url.split(",")[1]
                    formatted_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    })
                    
        else:
            for item in content:
                if item["type"] == "text":
                    formatted_content.append({"text": item["text"]})
                elif item["type"] == "image_url":
                    image_url = item["image_url"]["url"]
                    base64_data = image_url.split(",")[1]
                    
                    img_bytes = base64.b64decode(base64_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.resize((img.size[0]//2, img.size[1]//2))
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    
                    formatted_content.append({
                        "image":{
                            "format": "png",
                            "source": {"bytes": buffer.getvalue()}
                        }
                    })

        return formatted_content

    def _format_messages(self, messages: list) -> list:
        """Format the messages.

        Args:
            messages: The messages to format

        Returns:
            The formatted messages
        """
        formatted_messages = []
        for message in messages:
            if message["role"] == "system":
                formatted_messages.append({
                    "role": "user",
                    "content": self._format_content(message["content"]) if isinstance(message["content"], list) else [{"type": "text", "text": message["content"]}]
                })
            elif message["role"] in ["user", "assistant"]:
                formatted_messages.append({
                    "role": message["role"],
                    "content": self._format_content(message["content"]) if isinstance(message["content"], list) else [{"type": "text", "text": message["content"]}]
                })
            else:
                logger.warning(f"Unsupported message role: {message['role']}, skipping")
            
        return formatted_messages
    
    @BaseModel.retry(max_retries=3, retry_delay=2, backoff_factor=2, exceptions=(Exception,))
    def _completion(self, messages: list, max_tokens: int, top_p: float, temperature: float):
        """Completion method for the Bedrock model.

        Args:
            messages: The messages to complete
            max_tokens: The maximum number of tokens
            top_p: The top-p sampling parameter
            temperature: The sampling temperature
        """
        formatted_messages = self._format_messages(messages)
        
        if not self.model_name.startswith("llama"): # TEMP
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": formatted_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }

            response = self.client.invoke_model(
                modelId=ModelName2ModelID[self.model_name],
                body=json.dumps(request_body)
            )
            
            response = json.loads(response.get('body').read())
        else:
            if self.model_name.startswith("llama"): # TEMP
                for i in range(1, len(formatted_messages)):
                    formatted_messages[0]['content'].extend(formatted_messages[i]['content'])
                formatted_messages = [formatted_messages[0]]

            response = self.client.converse(
                modelId=ModelName2ModelID[self.model_name],
                messages=formatted_messages,
                )
            
        return response
    
    def completion(self, messages: list, max_tokens=2000, top_p=1, temperature=0.5):
        """Completion method for the Bedrock model.

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
            if not self.model_name.startswith("llama"): # TEMP
                response_text = response['content'][0]['text']
                
                return {
                    "response_text": response_text,
                    "model_usage": {
                        'completion_tokens': response.get('usage', {}).get('output_tokens'),
                        'prompt_tokens': response.get('usage', {}).get('input_tokens'),
                        'model_time': timer.duration
                    },
                    "response": response,
                    "error": None
                }
            else:
                response_text = response['output']['message']['content'][0]['text']
                
                return {
                    "response_text": response_text,
                    "model_usage": {
                        'completion_tokens': response.get('usage', {}).get('outputTokens'),
                        'prompt_tokens': response.get('usage', {}).get('inputTokens'),
                        'model_time': timer.duration
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