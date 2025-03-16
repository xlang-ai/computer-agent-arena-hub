from typing import List, Optional, Dict
import json
import os
import base64
import time
import boto3
import os
import requests
import time
import backoff
from requests.exceptions import SSLError

from .utils import encode_image, parse_aguvis_response

from .prompt import (
    AGUVIS_PLANNER_SYS_PROMPT,
    AGUVIS_SYS_PROMPT,
    AGUVIS_PLANNING_PROMPT,
    AGUVIS_INNER_MONOLOGUE_APPEND_PROMPT,
    AGUVIS_GROUNDING_PROMPT,
    AGUVIS_GROUNDING_APPEND_PROMPT
)

try:
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.models.BackboneModel import BackboneModel
    from backend.agents.utils.schemas import ObservationType
    from backend.desktop_env.desktop_env import DesktopEnv
    from backend.logger import aguvis_logger as logger
    from backend.agents.AgentManager import SessionConfig
except:
    from BaseAgent import BaseAgent
    from models.BackboneModel import BackboneModel
    from utils import ObservationType
    from temp.desktop_env import DesktopEnv
    from backend.agents.AgentManager import SessionConfig


s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

def upload_image_to_s3(base64_img):
    try:
        img_data = base64.b64decode(base64_img)
        filename = f"aguvis_images/{time.time()}.png"
        
        s3.put_object(
            Bucket="agent-arena-data",
            Key=filename,
            Body=img_data,
            ContentType='image/png'
        )
        
        url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': "agent-arena-data",
                'Key': filename
            },
            ExpiresIn=1800  # 30分钟
        )
        return url
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        return None
        
class AguvisAgent(BaseAgent):
    
    def __init__(
        self, 
        env: DesktopEnv,
        obs_options = ["screenshot"],
        platform: str = "Ubuntu",
        
        planner_model: str = "",
        executor_model: str = "",
        max_history_length: int = 5,
        max_tokens: int = 2000,
        top_p: float = 1,
        temperature: float = 0,
        action_space: str = "pyautogui",
        config: Optional[SessionConfig] = None,
    ):
        super().__init__(
            env=env,
            obs_options=obs_options,
            max_history_length=max_history_length,
            platform=platform,
            action_space=action_space,
            config=config,
        )

        self.aguvis_api_url = os.getenv("AGUVIS_API_URL")
        self.planner_model = os.getenv("AGUVIS_MODEL_NAME")
        self.executor_model = os.getenv("AGUVIS_MODEL_NAME")
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature

        self.thoughts = []
        self.actions = []
        self.observations = []

    @BaseAgent.run_decorator
    def run(self, task_instruction: str):
        while True:
            obs, obs_info = self.get_observation()
            actions, predict_info = self.predict(instruction=task_instruction, obs=obs)
            for i, action in enumerate(actions):
                terminated, step_info = self.step(action=action)
                if terminated:
                    self.terminated = terminated
                    return

    @BaseAgent.predict_decorator
    def predict(self, instruction: str, obs: Dict):
        previous_actions = "\n".join([f"Step {i+1}: {action}" for i, action in enumerate(self.actions)]) if self.actions else "None"

        aguvis_messages = []
        aguvis_messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": AGUVIS_SYS_PROMPT}]
            }
        )
        aguvis_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": AGUVIS_PLANNING_PROMPT.format(
                        instruction=instruction,
                        previous_actions=previous_actions
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{obs['screenshot']}"
                    }
                }
            ],
        })
        aguvis_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": AGUVIS_INNER_MONOLOGUE_APPEND_PROMPT}]
            }
        )
        img_url = upload_image_to_s3(obs['screenshot'])
        if img_url:
            aguvis_messages[1]['content'][1]['image_url']['url'] = img_url
            
        start_time = time.time()
        
        aguvis_response = self.llm_completion({
            "model": self.executor_model,
            "messages": aguvis_messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature
        }, self.executor_model)
        model_duration = time.time() - start_time
        
        aguvis_messages = aguvis_response['choices'][0]

        logger.info(f"aguvis_response: \n{aguvis_messages['message']['content']}")
        low_level_instruction, pyautogui_actions = parse_aguvis_response(aguvis_messages['message']['content'])
        
        logger.info(f"pyautogui_actions: \n{pyautogui_actions}")
        self.actions.append(low_level_instruction)

        self._obs, self._actions, self._thought = obs, pyautogui_actions, low_level_instruction
        self.history.append({
            "obs": self._obs,
            "actions": self._actions,
            "thought": self._thought
        })
        
        aguvis_model_usage = aguvis_response['usage']
        aguvis_model_usage['model_time'] = model_duration

        actions = [pyautogui_actions]
        predict_info = {
            "model_usage": aguvis_model_usage,
            "response": low_level_instruction,
            "messages": aguvis_messages['message']['content']
        }

        return actions, predict_info


    """@backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure
        # each example won't exceed the time limit
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                #openai.RateLimitError,
                #openai.BadRequestError,
                #openai.InternalServerError,

                # Google exceptions
                #InvalidArgument,
                #ResourceExhausted,
                #InternalServerError,
                #BadRequest,

                # Groq exceptions
                # todo: check
        ),
        interval=30,
        max_tries=10
    )
    def llm_completion(self, payload, model):
        headers = {"Content-Type": "application/json"}
        
        # 保存请求日志
        log_dir = "request_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        with open(f"{log_dir}/request_{timestamp}.json", "w") as f:
            json.dump({
                "url": self.aguvis_api_url,
                "headers": headers,
                "payload": payload
            }, f, indent=2)
        
        response = requests.post(self.aguvis_api_url, headers=headers, json=payload)
        
        # 保存响应日志
        with open(f"{log_dir}/response_{timestamp}.txt", "w") as f:
            f.write(f"Status: {response.status_code}\n")
            f.write(f"Content: {response.text}")
        
        if response.status_code != 200:
            logger.error("Failed to call LLM: " + response.text)
            time.sleep(5)
            return ""
        return response.json()"""
    
    
    

        
       
    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure
        # each example won't exceed the time limit
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                #openai.RateLimitError,
                #openai.BadRequestError,
                #openai.InternalServerError,

                # Google exceptions
                #InvalidArgument,
                #ResourceExhausted,
                #InternalServerError,
                #BadRequest,

                # Groq exceptions
                # todo: check
        ),
        interval=30,
        max_tries=10
    )
    def llm_completion(self, payload, model):
            headers = {
                "Content-Type": "application/json",
            }
            #logger.info("Generating content with Aguvis model: %s", model)

            response = requests.post(
                self.aguvis_api_url,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                #return response.json()['choices'][0]['message']['content']
                return response.json()