"""
Implementation of the TARS agent class for the agent hub.
TARS agent is an advanced agent that uses UI-TARS model to generate actions after observing the environment.
"""
import re
import logging
from typing import List, Dict, Optional, Any, Union, Tuple

try:
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.models.BackboneModel import BackboneModel 
    from backend.logger import tars_logger as logger
    from backend.desktop_env.desktop_env import DesktopEnv
except:
    from BaseAgent import BaseAgent
    from models.BackboneModel import BackboneModel
    from test_env.desktop_env import DesktopEnv

from .utils import (
    decode_image_from_base64,
    encode_image_to_base64,
    draw_grid_with_number_labels,
    parse_action_qwen2vl,
    parsing_response_to_pyautogui_code,
    FINISH_WORD,
    WAIT_WORD,
    ENV_FAIL_WORD,
    CALL_USER
)
from .prompt import (
    REFLECTION_ACTION_SPACE,
    NO_THOUGHT_PROMPT_0103,
    MULTI_STEP_PROMPT_1229,
)

class TARSAgent(BaseAgent):
    """Implementation of the TARS agent class.
    TARS agent uses UI-TARS model to generate actions by understanding UI elements and their relationships.
    """
    
    def __init__(self,
            env: DesktopEnv,
            model_name: str,
            obs_options: List[str] = ["screenshot"],
            max_tokens: int = 2000,
            top_p: float = 1,
            temperature: float = 0.5,
            platform: str = "Ubuntu",
            action_space: str = "pyautogui",
            max_trajectory_length: int = 5,
            prompt_template: str = "multi_step",
            language: str = "English",
            config: Optional[Dict] = None,
            **kwargs
    ):
        """Initialize the TARS agent.

        Args:
            env: The environment
            model_name: The name of the model
            obs_options: The observation options
            max_tokens: Maximum number of tokens for model response
            top_p: Top p sampling parameter
            temperature: Temperature for sampling
            platform: Operating system platform
            action_space: Action space type
            max_trajectory_length: Maximum trajectory length
            prompt_template: Prompt template type
            language: Language for the prompt
            config: Additional configuration
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
        self.language = language
        self.logger = logger
        self.history_images = []
        self.history_responses = []
        
        # Select prompt template
        if prompt_template == "no_thought":
            self.system_message = NO_THOUGHT_PROMPT_0103
            self.action_space_prompt = REFLECTION_ACTION_SPACE
        elif prompt_template == "multi_step":
            self.system_message = MULTI_STEP_PROMPT_1229
            self.action_space_prompt = REFLECTION_ACTION_SPACE
        else:
            raise ValueError(f"Unknown prompt template: {prompt_template}")


    def _process_screenshot(self, screenshot: str) -> Tuple[Optional[str], Optional[str]]:
        """Process screenshot and return base64 encoded image.
        
        Args:
            screenshot: Base64 encoded screenshot
            
        Returns:
            Tuple of (processed image base64, error message if any)
        """
        try:
            if not screenshot:
                return None, "Empty screenshot content"
                
            image = decode_image_from_base64(screenshot)
            if image is None:
                return None, "Failed to decode image"
                
            image_with_grid = draw_grid_with_number_labels(image, 100)
            processed_image = encode_image_to_base64(image_with_grid)
            if processed_image is None:
                return None, "Failed to encode processed image"
                
            return processed_image, None
            
        except Exception as e:
            return None, f"Error processing screenshot: {str(e)}"

    @BaseAgent.predict_decorator
    def predict(self, task_instruction: str, obs: Dict) -> Tuple[Optional[List], Optional[Dict]]:
        """Predict the next action based on the current observation."""
        # Add history image handling
        if not hasattr(self, 'history_images'):
            self.history_images = []
        if not hasattr(self, 'history_responses'):
            self.history_responses = []
        
        # Add image to history
        if "screenshot" in obs:
            self.history_images.append(obs["screenshot"])
            # Limit history length
            if len(self.history_images) > self.max_history_length:
                self.history_images = self.history_images[-self.max_history_length:]

        messages = []
        
        # Add system message with proper formatting
        messages.append({
            "role": "system", 
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        })

        # Add task instruction
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message.format(
                        instruction=task_instruction,
                        action_space=self.action_space_prompt,
                        language=self.language
                    )
                }
            ]
        })

        # Add history context and images
        if self.history_responses:
            for idx, (hist_response, hist_image) in enumerate(zip(self.history_responses[-self.max_history_length:], 
                                                                self.history_images[-self.max_history_length-1:-1])):
                # Add historical image
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{hist_image}",
                                "detail": "high"
                            }
                        }
                    ]
                })
                
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": hist_response}]
                })

        # Add current screenshot
        if "screenshot" in obs:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{obs['screenshot']}",
                            "detail": "high"
                        }
                    }
                ]
            })

        # Get model response
        try:
            response = self.model.completion(
                messages=messages,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature
            )
        except Exception as e:
            self.logger.error(f"{self.class_name} Model completion error: {str(e)}")
            return None, None

        if not response or 'response_text' not in response:
            self.logger.error(f"{self.class_name} Invalid response format: {response}")
            return None, None

        response_text = response['response_text']
        
        model_usage = response.get('model_usage', {})

        # Parse actions and thoughts
        try:
            self.logger.info(f"Response Text: {response_text}")
            print("Response Text", response_text)
            actions = parse_action_qwen2vl(response_text, 1000, 720, 1080)
            if not actions:
                self.logger.warning(f"{self.class_name} No valid actions parsed from response")
                return None, None
            print("Actions", actions)
            pyautogui_actions = parsing_response_to_pyautogui_code(actions, 720, 1080)
            if not pyautogui_actions:
                self.logger.error(f"{self.class_name} Failed to generate pyautogui code")
                return None, None
            self.logger.info(f"PyAutoGUI Actions:{pyautogui_actions}")
            thoughts = actions[0]['thought']
            reflection = actions[0]['reflection']
        except Exception as e:
            self.logger.error(f"{self.class_name} Error parsing response: {str(e)}")
            return None, None

        # Store response in history
        if response_text:
            self.history_responses.append(response_text)
            if len(self.history_responses) > self.max_history_length:
                self.history_responses = self.history_responses[-self.max_history_length:]

        return (
            [pyautogui_actions],
            {
                "model_usage": model_usage,
                "response": thoughts,
                "messages": messages
            }
        )

    @BaseAgent.run_decorator
    def run(self, task_instruction: str) -> None:
        """Run the agent with the given task instruction."""
        try:
            while True:
                obs, obs_info = self.get_observation()
                if obs is None:
                    self.logger.error(f"{self.class_name} Failed to get observation")
                    break
                print(obs)
                actions, predict_info = self.predict(task_instruction=task_instruction, obs=obs)
                print(actions)
                if actions is None:
                    self.logger.error(f"{self.class_name} Failed to predict actions")
                    break
                
                self.logger.info(f"{self.class_name} actions: {len(actions)} {actions}")
                
                for action in actions:
                    if action == FINISH_WORD:
                        self.terminated = True
                        return
                        
                    terminated, step_info = self.step(action=action)
                    if terminated:
                        self.terminated = terminated
                        return
                        
                    if step_info.get("status") == ENV_FAIL_WORD:
                        self.logger.error(f"{self.class_name} Environment failure")
                        return
                        
        except Exception as e:
            self.logger.error(f"{self.class_name} Error in run loop: {str(e)}")
            self.terminated = True

    @BaseAgent.continue_conversation_decorator
    def continue_conversation(self, user_message: str) -> None:
        """继续与用户的对话，将新消息追加到历史中。
        
        Args:
            user_message: 用户的新消息
        """
        try:
            # 添加用户消息到历史
            self.history.append({
                "role": "user",
                "content": user_message
            })
            # 重置步数计数器，确保每次继续对话时都有足够的步数限制
            if hasattr(self, 'agent_manager') and self.agent_manager is not None:
                self.agent_manager.step_idx = 0
                self.logger.info("Reset agent_manager.step_idx to 0 for continuing conversation")
            
            while True:
                obs, obs_info = self.get_observation()
                if obs is None:
                    self.logger.error(f"{self.class_name} Failed to get observation")
                    break
                    
                # 注意这里不传入task_instruction，而是使用user_message
                actions, predict_info = self.predict(task_instruction="", obs=obs, user_message=user_message)
                
                if actions is None:
                    self.logger.error(f"{self.class_name} Failed to predict actions")
                    break
                
                self.logger.info(f"{self.class_name} actions: {len(actions)} {actions}")
                
                for action in actions:
                    if action == FINISH_WORD:
                        self.terminated = True
                        return
                        
                    terminated, step_info = self.step(action=action)
                    if terminated:
                        self.terminated = terminated
                        return
                        
                    if step_info.get("status") == ENV_FAIL_WORD:
                        self.logger.error(f"{self.class_name} Environment failure")
                        return
                        
        except Exception as e:
            self.logger.error(f"{self.class_name} Error in continue_conversation loop: {str(e)}")
            self.terminated = True
