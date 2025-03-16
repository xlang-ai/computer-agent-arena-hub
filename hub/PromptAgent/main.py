"""
Implementation of the Prompt agent class for the agent hub.
"""
from typing import List, Dict, Optional, Any, Union

from .utils import (
    decode_image_from_base64,
    encode_image_to_base64,
    draw_grid_with_number_labels,
    parse_code_from_string,
    parse_code_from_som_string
)

from .prompt import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE_UBUNTU,
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE_WINDOWS,
    SYS_PROMPT_IN_A11Y_OUT_CODE, 
    SYS_PROMPT_IN_VISION_ACCESSIBILITY_OUT_CODE, 
    SYS_PROMPT_IN_SOM_OUT_TAG
)

import re

try:
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.models.BackboneModel import BackboneModel
    from backend.desktop_env.desktop_env import DesktopEnv
except:
    from BaseAgent import BaseAgent
    from models.BackboneModel import BackboneModel
    from test_env.desktop_env import DesktopEnv

class PromptAgent(BaseAgent):
    """Implementation of the Prompt agent class for the agent hub.
    Prompt agent is a basic ReAct-style agent that uses a prompt to generate actions after observing the environment.
    """
    
    def __init__(self,
            env: DesktopEnv,
            model_name: str,
            obs_options=["screenshot"],
            
            max_tokens=2000,
            top_p=1,
            temperature=0.5,
            platform="Ubuntu",
            action_space="pyautogui",
            
            max_trajectory_length=5,
            a11y_tree_max_tokens=10000,
            config=None,
            **kwargs
    ):
        """Initialize the Prompt agent.

        Args:
            env: The environment
            model_name: The name of the model
            obs_options: The observation options
            max_trajectory_length: The maximum trajectory length
            a11y_tree_max_tokens: The maximum tokens for the accessibility tree
        """
        super().__init__(
            env=env,
            obs_options=obs_options,
            max_history_length=max_trajectory_length,
            platform=platform,
            action_space=action_space,
            config=config
        )
        
        self.class_name = self.__class__.__name__
        self.model_name = model_name
        self.model = BackboneModel(model_name=model_name)
        
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        
        # Initialize messages list to store conversation history in proper format
        self.messages = []
        
        prompt_mapping = {

            ("screenshot", "pyautogui"): SYS_PROMPT_IN_SCREENSHOT_OUT_CODE_WINDOWS if self.platform == 'Windows' else SYS_PROMPT_IN_SCREENSHOT_OUT_CODE_UBUNTU,
            ("a11y_tree", "pyautogui"): SYS_PROMPT_IN_A11Y_OUT_CODE,
            ("a11y_tree,screenshot", "pyautogui"): SYS_PROMPT_IN_VISION_ACCESSIBILITY_OUT_CODE,
            ("som", "pyautogui"): SYS_PROMPT_IN_SOM_OUT_TAG
        }

        obs_key = self.obs_config.to_string()
        key = (obs_key, self.action_space)
        
        if key not in prompt_mapping:
            raise ValueError(f"Invalid combination: obs_options={self.obs_config.to_string()}, action_space={self.action_space}")

        self.system_message = prompt_mapping[key]
        
    @BaseAgent.predict_decorator
    def predict(self, task_instruction: str, obs: Dict):
        """Predict the next action.

        Args:
            task_instruction: The task instruction
            obs: The observation
        """
        system_message = self.system_message
        messages = []
        
        # system prompt
        system_msg = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message.format(
                        task_instruction=task_instruction,
                        resolution=self.env.resolution,
                        platform=self.platform,
                        )
                },
            ]
        }
        
        # Add system message to the conversation if it's the first message
        if not self.messages:
            self.messages.append(system_msg)
            
        # Start with system message for this prediction
        messages.append(system_msg)

        def create_message(obs_type, text_content, image_content=None, use_grid=True):
            """Create a message.

            Args:
                obs_type: The observation type
                text_content: The text content
                image_content: The image content
                use_grid: Whether to use a grid
            """
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the {obs_type} as below. What's the next step that you will do to help with the task?\n{text_content}"
                    }
                ]
            }

            if image_content:
                if use_grid:
                    # Draw grid on the image and overlay coordinates
                    image_content = decode_image_from_base64(image_content)
                    image_content = draw_grid_with_number_labels(image_content, 100)
                    image_content = encode_image_to_base64(image_content)
                
                data_uri = f"data:image/png;base64,{image_content}"
                message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "high"}
                })
                
            return message
        
        obs_handlers = {
            ("a11y_tree", "screenshot"): lambda item: create_message(
                obs_type="screenshot and info from accessibility tree",
                text_content=item['a11y_tree'],
                image_content=item['screenshot']
            ),
            ("som",): lambda item: create_message(
                obs_type="tagged screenshot",
                text_content="Please analyze the tagged screenshot.",
                image_content=item["som"]
            ),
            ("screenshot",): lambda item: create_message(
                obs_type="screenshot",
                text_content="Please analyze the screenshot.",
                image_content=item["screenshot"]
            ),
            ("a11y_tree",): lambda item: create_message(
                obs_type="info from accessibility tree",
                text_content=item["a11y_tree"]
            )
        }
        
        # Add conversation history to messages
        if len(self.messages) > 1:  # Skip system message
            messages.extend(self.messages[1:])
        
        # Create and add the current observation message
        obs_keys = tuple(sorted(self.obs_config.obs_options))
        if obs_keys in obs_handlers:
            current_obs_message = obs_handlers[obs_keys](obs)
            messages.append(current_obs_message)
            # Add to history only if we're not in continue_conversation mode
            # (in that case, we'll add it after getting the response)
            self.messages.append(current_obs_message)
        else:
            raise ValueError(f"Invalid observation type: {obs_keys} {obs_handlers.keys()}")
        
        response = self.model.completion(
            messages=messages,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            temperature=self.temperature
        )


        if not response['response_text']:
            self.logger.error(f"{self.class_name} Empty response: {response}")
            return None, None
        
        response_text, model_usage, response = response['response_text'], response['model_usage'], response['response']
        
        # Add assistant response to conversation history
        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response_text
                }
            ]
        }
        self.messages.append(assistant_message)
        
        actions = self.parse_actions(response_text)
        thoughts = self.parse_thoughts(response_text)
        self.logger.info(f"{self.class_name} model_usage: {model_usage}")
        
        self._obs, self._actions, self._thought = obs, actions, thoughts
        self.history.append({
            "obs": self._obs,
            "actions": self._actions,
            "thought": self._thought
        })
        
        return (
            actions,
            {
                "model_usage": model_usage,
                "response": thoughts, 
                "messages": messages
            }
        )

    
    def parse_actions(self, response: str, masks: Optional[List[Any]] = None) -> List[Any]:
        """Parse the actions from the response.

        Args:
            response: The response
            masks: The masks
        """
        action_parsers = {
            "pyautogui": {
                "screenshot": parse_code_from_string,
                "a11y_tree": parse_code_from_string,
                "a11y_tree,screenshot": parse_code_from_string,
                "som": lambda r, m: parse_code_from_som_string(r, m)
            }
        }

        obs_key = self.obs_config.to_string()
        
        try:
            parser = action_parsers[self.action_space][obs_key]
        except KeyError:
            raise ValueError(f"Invalid combination: action_space={self.action_space}, obs_options={self.obs_config.to_string()}")

        return parser(response, masks) if obs_key == "som" else parser(response)
    
    def parse_thoughts(self, response: str) -> str:
        """Parse the thoughts from the response.

        Args:
            response: The response
        """
        # 移除response中用```包裹的部分
        return re.sub(r"```.*?```", "", response, flags=re.DOTALL).strip()
    
    @BaseAgent.run_decorator
    def run(self, task_instruction: str):
        """Run the agent.

        Args:
            task_instruction: The task instruction
        """
        # Reset messages for a new run
        self.messages = []
        
        while True:
            obs, obs_info = self.get_observation()
            actions, predict_info = self.predict(task_instruction=task_instruction, obs=obs)
            self.logger.info(f"PromptAgent actions: {len(actions)} {actions}")
            for i, action in enumerate(actions):
                terminated, step_info = self.step(action=action)
                if terminated:
                    self.terminated = terminated
                    return
    
    @BaseAgent.continue_conversation_decorator
    def continue_conversation(self, user_message: str):
        """继续与用户的对话，将新消息追加到历史中。
        
        Args:
            user_message: 用户的新消息
        """
        self.logger.info(f"Continuing conversation with message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        
        # 添加用户消息到历史
        user_msg = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        }
        self.messages.append(user_msg)
        
        # 重置步数计数器，确保每次继续对话时都有足够的步数限制
        if hasattr(self, 'agent_manager') and self.agent_manager is not None:
            self.agent_manager.step_idx = 0
            self.logger.info("Reset agent_manager.step_idx to 0 for continuing conversation")
        
        # 继续对话直到终止
        self.terminated = False
        while not self.terminated:
            obs, obs_info = self.get_observation()
            # 注意这里传入空字符串作为task_instruction，表示这是继续对话而不是新任务
            actions, predict_info = self.predict(task_instruction="", obs=obs)
            
            if not actions:
                self.logger.warning("No actions returned from predict")
                break
                
            self.logger.info(f"PromptAgent continue_conversation actions: {len(actions)} {actions}")
            
            for i, action in enumerate(actions):
                terminated, step_info = self.step(action=action)
                if terminated:
                    self.terminated = terminated
                    return
