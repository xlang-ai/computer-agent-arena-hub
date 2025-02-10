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
            top_p: float = 0.9,
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
        
        # Configure logging
        self.logger = logging.getLogger(self.class_name)
        
        # Select prompt template
        if prompt_template == "no_thought":
            self.system_message = NO_THOUGHT_PROMPT_0103
            self.action_space_prompt = REFLECTION_ACTION_SPACE
        elif prompt_template == "multi_step":
            self.system_message = MULTI_STEP_PROMPT_1229
            self.action_space_prompt = REFLECTION_ACTION_SPACE
        else:
            raise ValueError(f"Unknown prompt template: {prompt_template}")

    def _format_history(self, history: List[Dict]) -> str:
        """Format history into a string.
        
        Args:
            history: List of historical actions and observations
            
        Returns:
            Formatted history string
        """
        formatted = ""
        for i, previous_item in enumerate(history):
            formatted += f"Step {i + 1}:\n"
            if 'thought' in previous_item:
                formatted += f"Thought: {previous_item['thought']}\n"
            if 'actions' in previous_item:
                for j, action in enumerate(previous_item['actions'], start=1):
                    formatted += f"Action {j}: {action}\n"
        return formatted

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
        messages = []
        
        # Add system message with proper formatting
        messages.append({
            "role": "system",
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

        # Add history context
        history = self._format_history(self.history[-self.max_history_length:]) if self.history else ""

        # Process screenshot
        if "screenshot" in obs:
            # TODO: No need to draw grid for UI-TARS
            # processed_image, error = self._process_screenshot(obs["screenshot"])
            # if error:
            #     self.logger.error(f"{self.class_name} {error}")
            #     return None, None
            processed_image = obs["screenshot"]
                
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{history}\nAnalyze the current screen and determine the next action:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{processed_image}",
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
            actions = parse_action_qwen2vl(response_text, 1000, 720, 1080)
            if not actions:
                self.logger.warning(f"{self.class_name} No valid actions parsed from response")
                return None, None
                
            pyautogui_actions = parsing_response_to_pyautogui_code(actions, 720, 1080)
            if not pyautogui_actions:
                self.logger.error(f"{self.class_name} Failed to generate pyautogui code")
                return None, None
                
            thoughts = actions[0]['thought']
            reflection = actions[0]['reflection']
        except Exception as e:
            self.logger.error(f"{self.class_name} Error parsing response: {str(e)}")
            return None, None

        # Update history with validation
        if obs and pyautogui_actions and thoughts is not None:
            self._obs, self._actions, self._thought = obs, pyautogui_actions, thoughts
            self.history.append({
                "obs": self._obs,
                "actions": self._actions,
                "thought": self._thought
            })
            
            # Trim history if it exceeds max length
            if len(self.history) > self.max_history_length:
                self.history = self.history[-self.max_history_length:]
        else:
            self.logger.warning(f"{self.class_name} Missing required components for history update")

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
