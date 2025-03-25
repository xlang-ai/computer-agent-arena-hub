"""
Implementation of OpenAI Computer Using Agent for Computer Agent Arena
"""
import os
import json
import time
import base64
import requests
from typing import Callable, Dict, List, Any, Tuple
from .utils import acknowledge_safety_check_callback
try:
    # for deploy environment
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.action.main import Action
    from backend.agents.utils.utils import Timer
    from backend.agents.utils.exceptions import StepError

    from backend.logger import agent_logger as logger
except ImportError:
    # for test environments
    from BaseAgent import BaseAgent
    from action.main import Action
    from utils.utils import Timer
    from utils.exceptions import StepError
    
    import logging
    logger = logging.getLogger("agent_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

class OpenAICUAAgent(BaseAgent):
    """Implementation of OpenAI Computer Using Agent for Computer Agent Arena"""
    
    def __init__(
        self,
        env,
        config,
        platform="Ubuntu",
        action_space="pyautogui",
        obs_options=["screenshot"],
        max_history_length=100,
        model="computer-use-preview-2025-02-04",
        acknowledge_safety_check_callback: Callable = acknowledge_safety_check_callback,
        **kwargs
    ):
        """Initialize the OpenAI CUA Agent.
        
        Args:
            env: The environment to run the agent in
            config: Configuration for the agent session
            platform: The platform the agent is running on (Ubuntu, Windows, etc.)
            action_space: The action space to use (pyautogui, etc.)
            obs_options: The observation options to use (screenshot, etc.)
            max_history_length: The maximum number of steps in the trajectory
            model: The OpenAI model to use
            **kwargs: Additional arguments
        """
        super().__init__(
            env=env,
            config=config,
            platform=platform, 
            action_space=action_space,
            obs_options=obs_options,
            max_history_length=max_history_length,
            **kwargs
        )
        
        self.model = model
        self.messages = []
        
        # Computer environment tools
        self.tools = [{
            "type": "computer-preview",
            "display_width": 1280,
            "display_height": 720,
            "environment": "linux" if platform == "Ubuntu" else "windows"
        }]
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback
        # Track API usage
        self.total_tokens = 0
        self.api_calls = 0
        
        logger.info(f"Initialized OpenAICUAAgent with model={model}, platform={platform}, action_space={action_space}")
    
    def _handle_item(self, item):
        """Parse a pyautogui action from the OpenAI API response"""
        if item["type"] == "message":
            if item.get("content", None) != None:
                response = item.get("content")[0] if isinstance(item.get("content"), list) else item.get("content")
                response_type = response.get("type", "")
                response_text = response.get("text", "")
                logger.info(f"Received response text: {response_type} - {response_text}")
                if response_type == "output_text":
                    return response_text
                else:
                    return None
            else:
                return None
        
        if item["type"] == "function_call":
            return None
        
        if item["type"] == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            logger.warning(f"Original Action: {action}")
            result_code = self._convert_cua_action_to_pyautogui_action(action_type, action_args)
            if result_code:
                result = {
                    "action_space": "pyautogui",
                    "action": result_code,
                    "pending_checks": item.get("pending_safety_checks", []),
                    "call_id": item.get("call_id", "")
                }
                return result
            return None
        
    
    def _convert_cua_action_to_pyautogui_action(self, action_type, args):
        """Convert a CUA action to a pyautogui action format
        
        This function converts OpenAI CUA actions to pyautogui commands
        for the Computer Agent Arena
        
        Args:
            action_type: Type of the CUA action
            args: Arguments for the action
            
        Returns:
            String with pyautogui command code or None if the action can't be converted
        """
        if not action_type:
            logger.warning("Empty CUA action received")
            return None
        
        # CUA key mapping to pyautogui keys
        key_mapping = {
            # Basic characters
            "/": "slash",
            "\\": "backslash",
            # Modifier keys
            "alt": "alt",
            "cmd": "command",
            "ctrl": "ctrl",
            "shift": "shift",
            "option": "option",
            "super": "win",
            "win": "win",
            # Arrow keys
            "arrowdown": "down",
            "arrowleft": "left",
            "arrowright": "right",
            "arrowup": "up",
            "DOWN": "down",
            "LEFT": "left",
            "RIGHT": "right",
            "UP": "up",
            # Special keys
            "backspace": "backspace",
            "BACKSPACE": "backspace",
            "capslock": "capslock",
            "delete": "delete",
            "end": "end",
            "enter": "enter",
            "ENTER": "enter",
            "esc": "esc",
            "ESC": "esc",
            "escape": "esc",
            "home": "home",
            "insert": "insert",
            "pagedown": "pagedown",
            "pageup": "pageup",
            "space": "space",
            "SPACE": "space",
            "tab": "tab",
            "TAB": "tab",
            # Function keys
            "f1": "f1",
            "f2": "f2",
            "f3": "f3",
            "f4": "f4",
            "f5": "f5",
            "f6": "f6",
            "f7": "f7",
            "f8": "f8",
            "f9": "f9",
            "f10": "f10",
            "f11": "f11",
            "f12": "f12",
            # Return is often used in Linux contexts for Enter
            "Return": "enter",
        }
        
        try:
            if action_type == "click":
                x = args.get("x")
                y = args.get("y")
                button = args.get("button", "left")
                
                # Validate coordinates
                if x is None or y is None:
                    logger.warning(f"Invalid click coordinates: x={x}, y={y}")
                    return None
                
                # Validate button
                if button not in ["left", "middle", "right"]:
                    logger.warning(f"Invalid click button: {button}, defaulting to 'left'")
                    button = "left"
                
                return f"import pyautogui\npyautogui.moveTo({x}, {y})\npyautogui.click(button='{button}')"
                
            elif action_type == "double_click":
                x = args.get("x")
                y = args.get("y")
                
                # Validate coordinates
                if x is None or y is None:
                    logger.warning(f"Invalid double_click coordinates: x={x}, y={y}")
                    return None
                
                return f"import pyautogui\npyautogui.moveTo({x}, {y})\npyautogui.doubleClick()"
                
            elif action_type == "type":
                text = args.get("text", "")
                
                if not text:
                    logger.warning("Empty text for type action")
                    return "import pyautogui\n# Empty text, no action taken"
                
                # Escape special characters for Python string
                escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                
                return f"import pyautogui\npyautogui.write('{escaped_text}')"
                
            elif action_type == "keypress":
                keys = args.get("keys", [])
                
                if not keys:
                    logger.warning("Empty keys for keypress action")
                    return None
                
                # Map to pyautogui keys and normalize
                mapped_keys = []
                for key in keys:
                    if isinstance(key, str):
                        # For Linux compatibility, handle the key mapping more thoroughly
                        mapped_key = key_mapping.get(key, key)
                        # Also try lowercase version if not found
                        if mapped_key == key and key.lower() != key:
                            mapped_key = key_mapping.get(key.lower(), key)
                        mapped_keys.append(mapped_key)
                
                if not mapped_keys:
                    return None
                
                # Format for pyautogui.hotkey
                keys_str = ", ".join([f"'{k}'" for k in mapped_keys])
                
                return f"import pyautogui\npyautogui.hotkey({keys_str})"
                
            elif action_type == "scroll":
                x = args.get("x", None)
                y = args.get("y", None)
                scroll_x = args.get("scroll_x", 0)
                scroll_y = args.get("scroll_y", 0)
                
                # Normalize scroll values (Linux might use different scaling)
                scroll_y = int(scroll_y) if scroll_y else 0
                scroll_x = int(scroll_x) if scroll_x else 0
                
                # Default to current mouse position if coordinates not provided
                position_str = ""
                if x is not None and y is not None:
                    position_str = f", {x}, {y}"
                
                # Handle scroll direction
                if scroll_y != 0:
                    # Convert to clicks - normalize the amount
                    clicks = min(max(abs(scroll_y) // 20, 1), 10)  # Cap between 1-10 clicks
                    return f"import pyautogui\npyautogui.scroll({clicks * (1 if scroll_y < 0 else -1)}{position_str})"
                elif scroll_x != 0:
                    # Convert to clicks - normalize the amount
                    clicks = min(max(abs(scroll_x) // 20, 1), 10)  # Cap between 1-10 clicks
                    return f"import pyautogui\npyautogui.hscroll({clicks * (1 if scroll_x < 0 else -1)}{position_str})"
                else:
                    logger.warning("Scroll action with zero scrolling amount")
                    return None
                
            elif action_type == "move":
                x = args.get("x")
                y = args.get("y")
                
                # Validate coordinates
                if x is None or y is None:
                    logger.warning(f"Invalid move coordinates: x={x}, y={y}")
                    return None
                
                return f"import pyautogui\npyautogui.moveTo({x}, {y})"
                
            elif action_type == "drag":
                path = args.get("path", [])
                
                if not path or len(path) < 2:
                    logger.warning("Drag path must have at least two points")
                    return None
                
                # Extract start and end points
                start = path[0]
                end = path[-1]
                
                # Validate path coordinates - handle both (x,y) tuples/lists and {'x':x, 'y':y} dictionaries
                valid_path = True
                for point in path:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        continue
                    elif isinstance(point, dict) and 'x' in point and 'y' in point:
                        continue
                    else:
                        valid_path = False
                        break
                
                if not valid_path:
                    logger.warning("Invalid path format for drag action")
                    return None
                
                if len(path) == 2:
                    # Extract coordinates, handling both formats
                    if isinstance(start, (list, tuple)):
                        start_x, start_y = start
                    else:  # dict format
                        start_x, start_y = start.get('x'), start.get('y')
                        
                    if isinstance(end, (list, tuple)):
                        end_x, end_y = end
                    else:  # dict format
                        end_x, end_y = end.get('x'), end.get('y')
                    
                    return (
                        f"import pyautogui\n"
                        f"pyautogui.moveTo({start_x}, {start_y})\n"
                        f"pyautogui.dragTo({end_x}, {end_y}, duration=0.5, button='left')"
                    )
                # For complex paths with multiple points
                else:
                    actions = []
                    # Handle first point
                    if isinstance(path[0], (list, tuple)):
                        first_x, first_y = path[0]
                    else:  # dict format
                        first_x, first_y = path[0].get('x'), path[0].get('y')
                        
                    actions.append(f"import pyautogui\npyautogui.moveTo({first_x}, {first_y})")
                    
                    for i in range(1, len(path)):
                        if isinstance(path[i], (list, tuple)):
                            x, y = path[i]
                        else:  # dict format
                            x, y = path[i].get('x'), path[i].get('y')
                            
                        actions.append(f"pyautogui.dragTo({x}, {y}, duration=0.2, button='left')")
                    
                    return "\n".join(actions)
                
            elif action_type == "wait":
                ms = args.get("ms", 1000)  # Default to 1000ms (1 second)
                seconds = max(0.1, ms / 1000)  # Ensure minimum wait time
                
                return f"import time\ntime.sleep({seconds})"
                
            elif action_type == "screenshot":
                # Just return a wait action, as screenshots are handled automatically
                return "import time\ntime.sleep(0.1)  # Screenshot requested, no direct action needed"
                
            else:
                logger.warning(f"Unknown action type: {action_type}")
                return None
                
        except Exception as e:
            logger.exception(f"Error converting CUA action to agent action: {e}")
            return None
    
    
    def _create_response(self, **kwargs):
        """Create a response from the OpenAI API"""
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY_CUA')}",
            "Content-Type": "application/json",
            "Openai-beta": "responses=v1",
        }

        openai_org = os.getenv("OPENAI_ORG")
        if openai_org:
            headers["Openai-Organization"] = openai_org
            
        logger.debug(f"Making OpenAI API call to {url}")

        response = requests.post(url, headers=headers, json=kwargs)

        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} {response.text}")
        else:
            logger.debug(f"Received successful response from OpenAI API")
        return response.json()
    
    @BaseAgent.predict_decorator
    def predict(self, task_instruction: str) -> Tuple[List[Dict], Dict]:
        """Predict the next action based on the current state.
        
        Args:
            task_instruction: The task instruction
            
        Returns:
            Tuple containing:
                - actions: List of actions to take
                - predict_info: Information about the prediction
        """
        logger.info(f"Predicting next actions for task: {task_instruction[:50]}{'...' if len(task_instruction) > 50 else ''}")
        
        with Timer() as model_timer:
            response = self._create_response(
                model=self.model,
                input=self.messages,
                tools=self.tools,
                truncation="auto",
            )
        self.messages += response["output"]
        print(response["output"])
        
        actions = []
        responses = []
        for item in response["output"]:
            parsed_item = self._handle_item(item)
            if isinstance(parsed_item, dict) and parsed_item.get("action_space", None) == "pyautogui":
                actions.append(parsed_item)
            else:
                responses.append(parsed_item)
        
        logger.info(f"Predicted {len(actions)} actions")
        predict_info = {
            "model_usage": {
                "model_time": model_timer.duration,
                "prompt_tokens": response.get("usage", {}).get("total_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
            },
            "messages": self.messages,
            "response": responses.join()
        }
        return actions, predict_info
    
    @BaseAgent.step_decorator
    def step(self, action: Dict) -> Tuple[bool, Dict]:
        """Execute an action in the environment.
        
        Args:
            action: The action to execute
            
        Returns:
            Tuple containing:
                - terminated: Whether the episode has terminated
                - info: Information about the step
        """
        try:
            if not action:
                logger.warning("Empty action received, terminating episode")
                return True, {}
                
            logger.info(f"Executing action: {action.get('action_space', 'unknown')} - {action.get('action', '')[:50]}...")
            
            with Timer() as step_timer:
                # Convert the action to an Action object
                step_action = Action(action.get("action", ""), self.action_space)
                # Execute the action in the environment
                terminated, info = self.env.step(step_action.get_action())
                
                obs, obs_info = self.get_observation()
                screenshot_base64 = obs["screenshot"]
                
                # Handle safety checks
                if "pending_checks" in action and action["pending_checks"]:
                    logger.info(f"Processing {len(action['pending_checks'])} safety checks")
                    for check in action["pending_checks"]:
                        message = check["message"]
                        logger.warning(f"Safety check: {message}")
                        if not self.acknowledge_safety_check_callback(message):
                            logger.error(f"Safety check failed: {message}")
                            raise ValueError(
                                f"Safety check failed: {message}. Cannot continue with unacknowledged safety checks."
                            )
                        logger.info(f"Safety check acknowledged: {message}")
                
                self.messages.append({
                    "type": "computer_call_output",
                    "call_id": action["call_id"],
                    "acknowledged_safety_checks": action["pending_checks"],
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_base64}",
                    },
                })
                
            logger.debug(f"Action completed in {step_timer.duration:.2f}s")
            if terminated:
                logger.info("Environment signaled termination")
                
            return terminated, {
                "step_time": step_timer.duration,
                "action": action
            }
                
        except Exception as e:
            logger.exception(f"Environment step failed: {str(e)}")
            raise StepError(f"Failed to execute step: {str(e)}")
    
    @BaseAgent.run_decorator
    def run(self, task_instruction: str):
        """Run the agent to complete the task.
        
        Args:
            task_instruction: The task instruction
        """
        logger.info(f"Starting agent run with task: {task_instruction[:100]}{'...' if len(task_instruction) > 100 else ''}")
        
        # Reset conversation history
        self.messages = [
            {
                "role": "user", 
                "content": task_instruction
            }
        ]
        self.terminated = False
        step_count = 0
        
        # Continue until terminated
        logger.info("Beginning agent loop")
        while not self.terminated:
            step_count += 1
            logger.info(f"Step {step_count}: Predicting next actions")
            
            # Get next actions
            actions, predict_info = self.predict(task_instruction)
            if actions is None or actions == []:
                #TODO: this means the agent outputs no action but pure message for user to interact with
                self.agent_manager.send_interact_message(text=predict_info['response'])
                self.terminated = True
                time.sleep(5)
                return
            logger.warning(f"Actions: {actions}")
            for action_idx, action in enumerate(actions):
                if not action:
                    logger.warning(f"Step {step_count}, Action {action_idx+1}: Received empty action, skipping")
                    continue
                    
                logger.info(f"Step {step_count}, Action {action_idx+1}: Executing action")
                self.terminated, step_info = self.step(action)
                
                if self.terminated:
                    logger.info(f"Step {step_count}: Agent run terminated")
                    break

    @BaseAgent.continue_conversation_decorator
    def continue_conversation(self, user_message: str):
        """继续与用户的对话，将新消息追加到历史中。
        
        Args:
            user_message: 用户的新消息
        """
        logger.info(f"Continuing conversation with message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        
        # 添加用户消息到历史
        self.messages.append({
            "role": "user", 
            "content": user_message
        })
        
        self.terminated = False
        step_count = 0
        
        # 继续对话直到终止
        logger.info("Continuing agent loop")
        while not self.terminated:
            step_count += 1
            logger.info(f"Step {step_count}: Predicting next actions")
            
            # 获取下一步动作，注意这里不传入task_instruction
            actions, predict_info = self.predict("")
            if actions is None or actions == []:
                #TODO: this means the agent outputs no action but pure message for user to interact with
                self.agent_manager.send_interact_message(text=predict_info['response'])
                self.terminated = True
                time.sleep(5)
                return
            logger.warning(f"Actions: {actions}")
            for action_idx, action in enumerate(actions):
                if not action:
                    logger.warning(f"Step {step_count}, Action {action_idx+1}: Received empty action, skipping")
                    continue
                    
                logger.info(f"Step {step_count}, Action {action_idx+1}: Executing action")
                self.terminated, step_info = self.step(action)
                
                if self.terminated:
                    logger.info(f"Step {step_count}: Agent conversation terminated")
                    break
        