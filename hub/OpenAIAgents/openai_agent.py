"""
OpenAI Agents integration with Arena BaseAgent.

This module provides a wrapper around OpenAI Agents to integrate with the Arena platform.
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.models.BackboneModel import BackboneModel
    from backend.desktop_env.desktop_env import DesktopEnv
    from backend.logger import openai_agent_logger as logger
    # Import Arena BaseAgent and related classes
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.utils.exceptions import StopExecution, StepLimitExceeded
    from backend.agents.utils.schemas import ObservationType
    from backend.agents.AgentManager import SessionConfig
except:
    from BaseAgent import BaseAgent
    from models.BackboneModel import BackboneModel
    from test_env.desktop_env import DesktopEnv

from agents import (
    Agent,
    AsyncComputer,
    Button,
    ComputerTool,
    Environment,
    ModelSettings,
    Runner,
    trace,
    WebSearchTool,
    set_default_openai_client,
)

from openai import AsyncOpenAI

from agents.tracing import set_trace_processors
from .openai_agents_integration import ArenaTraceProcessor




class ArenaEnv(AsyncComputer):
    """
    An environment adapter that connects OpenAI Agents' AsyncComputer interface
    with Arena's BaseAgent and DesktopEnv.
    """
    
    def __init__(self, arena_agent=None, agent_manager=None, env=None):
        """
        Initialize the Arena environment.
        
        Args:
            arena_agent: The Arena BaseAgent instance
            agent_manager: The AgentManager instance
            env: The DesktopEnv instance
        """
        self._agent = arena_agent
        self._agent_manager = agent_manager

    async def __aenter__(self):
        logger.info("Entering ArenaEnv")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("Exiting ArenaEnv")
        pass

    @property
    def environment(self) -> Environment:
        """Get the environment type."""
        # TODO: fix me
        return "linux"
    
    @property
    def dimensions(self) -> tuple[int, int]:
        """Get the dimensions of the environment."""
        return (1280, 720)
    
    async def screenshot(self) -> str:
        """
        Take a screenshot of the environment.
        
        Returns:
            Base64-encoded screenshot
        """
        try:
            screenshot, screenshot_info = self._agent.get_observation()
            return screenshot['screenshot']
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return ""
    
    async def click(self, x: int, y: int, button: Button = "left") -> None:
        """
        Click at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click
        """
        if not self._agent:
            logger.warning("Agent not available, cannot execute click")
            return
            
        try:
            # Convert to pyautogui command
            code = f"""import pyautogui \npyautogui.click({x}, {y}, button='{button}')"""
            
            # Log the action
            logger.info(f"Executing click at ({x}, {y}) with button {button}")
            
            # Execute the action
            self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing click: {e}")

    async def double_click(self, x: int, y: int) -> None:
        """
        Double-click at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if not self._agent:
            logger.warning("Agent not available, cannot execute double click")
            return
            
        try:
            # Convert to pyautogui command
            code = f"""import pyautogui \npyautogui.doubleClick({x}, {y})"""
            
            # Log the action
            logger.info(f"Executing double click at ({x}, {y})")
            
            # Execute the action
            self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing double click: {e}")

    async def type(self, text: str) -> None:
        """
        Type the specified text.
        
        Args:
            text: Text to type
        """
        if not self._agent:
            logger.warning("Agent not available, cannot execute type")
            return
            
        try:
            # List of problematic characters that write() might not handle correctly
            # These are characters that might need special handling with press()
            problematic_chars = set(['¥', '€', '£', '©', '®', '±', '§', '×', '÷', '°', '¢', '¤', '¦', '¨', 
                                   'ª', '«', '¬', '¯', '²', '³', 'µ', '¶', '·', '¹', 'º', '»', '¼', '½', 
                                   '¾', '¿', 'Æ', 'Ç', 'Ð', 'Ñ', 'Ø', 'Œ', 'Þ', 'ß', 'æ', 'ç', 'ð', 'ñ', 
                                   'ø', 'œ', 'þ', 'ÿ'])
            
            # Generate Python code to handle the input text
            commands = ["import pyautogui"]
            
            # Break text into segments - handle regular text with write() and problematic chars with press()
            current_segment = ""
            for char in text:
                if char in problematic_chars:
                    # Flush any accumulated regular text
                    if current_segment:
                        escaped_segment = current_segment.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                        commands.append(f"pyautogui.write('{escaped_segment}', interval=0.1)")
                        current_segment = ""
                    
                    # Handle the problematic character with press()
                    escaped_char = char.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                    commands.append(f"pyautogui.press('{escaped_char}')")
                else:
                    current_segment += char
            
            # Flush any remaining regular text
            if current_segment:
                escaped_segment = current_segment.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
                commands.append(f"pyautogui.write('{escaped_segment}', interval=0.1)")
            
            # Combine all commands
            code = "\n".join(commands)
            
            # Log the action (don't log the entire text)
            logger.info(f"Executing type command for text of length {len(text)}")
            
            # Execute the action
            self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing type: {e}")

    async def keypress(self, keys: list[str]) -> None:
        """
        Press the specified keys.
        
        Args:
            keys: List of keys to press
        """
        if not self._agent or not keys:
            if not self._agent:
                logger.warning("Agent not available, cannot execute keypress")
            return
            
        try:
            # CUA key mapping to pyautogui keys
            key_mapping = {
                "/": "slash",
                "\\": "backslash",
                "alt": "alt",
                "cmd": "command",
                "ctrl": "ctrl",
                "shift": "shift",
                "option": "option",
                "super": "win",
                "win": "win",
                "arrowdown": "down",
                "arrowleft": "left",
                "arrowright": "right",
                "arrowup": "up",
                "DOWN": "down",
                "LEFT": "left",
                "RIGHT": "right",
                "UP": "up",
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
                "f1": "f1", "f2": "f2", "f3": "f3", "f4": "f4",
                "f5": "f5", "f6": "f6", "f7": "f7", "f8": "f8",
                "f9": "f9", "f10": "f10", "f11": "f11", "f12": "f12",
                "Return": "enter",
            }
            
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
                return
                
            # Format for pyautogui.hotkey
            keys_str = ", ".join([f"'{k}'" for k in mapped_keys])
            code = f"""import pyautogui \npyautogui.hotkey({keys_str})"""
            
            # Log the action
            logger.info(f"Executing keypress: {keys}")
            
            # Execute the action
            self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing keypress: {e}")

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """
        Scroll at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            scroll_x: Horizontal scroll amount
            scroll_y: Vertical scroll amount
        """
        if not self._agent:
            logger.warning("Agent not available, cannot execute scroll")
            return
            
        try:
            # Normalize scroll values
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
                code = f"""import pyautogui \npyautogui.scroll({clicks * (1 if scroll_y < 0 else -1)}{position_str})"""
                
                # Log the action
                logger.info(f"Executing vertical scroll: {scroll_y} at ({x}, {y})")
                
                # Execute the action
                self._agent.step(code)
            elif scroll_x != 0:
                # Convert to clicks - normalize the amount
                clicks = min(max(abs(scroll_x) // 20, 1), 10)  # Cap between 1-10 clicks
                code = f"""import pyautogui \npyautogui.hscroll({clicks * (1 if scroll_x < 0 else -1)}{position_str})"""
                
                # Log the action
                logger.info(f"Executing horizontal scroll: {scroll_x} at ({x}, {y})")
                
                # Execute the action
                self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing scroll: {e}")

    async def move(self, x: int, y: int) -> None:
        """
        Move the mouse to the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if not self._agent:
            logger.warning("Agent not available, cannot execute move")
            return
            
        try:
            # Convert to pyautogui command
            code = f"""import pyautogui \npyautogui.moveTo({x}, {y})"""
            
            # Log the action
            logger.info(f"Executing move to ({x}, {y})")
            
            # Execute the action
            self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing move: {e}")

    async def drag(self, path: list[tuple[int, int]]) -> None:
        """
        Drag the mouse along the specified path.
        
        Args:
            path: List of coordinates to drag through
        """
        if not self._agent or not path or len(path) < 2:
            if not self._agent:
                logger.warning("Agent not available, cannot execute drag")
            return
            
        try:
            if len(path) == 2:
                start_x, start_y = path[0]
                end_x, end_y = path[1]
                code = f"""import pyautogui \npyautogui.moveTo({start_x}, {start_y}); pyautogui.dragTo({end_x}, {end_y}, duration=0.5, button='left')"""
                
                # Log the action
                logger.info(f"Executing drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                
                # Execute the action
                self._agent.step(code)
            else:
                # For complex paths with multiple points
                actions = []
                actions.append(f"import pyautogui \npyautogui.moveTo({path[0][0]}, {path[0][1]})")
                
                for i in range(1, len(path)):
                    x, y = path[i]
                    actions.append(f"pyautogui.dragTo({x}, {y}, duration=0.2, button='left')")
                
                code = "; ".join(actions)
                
                # Log the action
                logger.info(f"Executing complex drag with {len(path)} points")
                
                # Execute the action
                self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing drag: {e}")

    async def wait(self, ms: int = 1000) -> None:
        """
        Wait for the specified number of milliseconds.
        
        Args:
            ms: Number of milliseconds to wait
        """
        if not self._agent:
            logger.warning("Agent not available, cannot execute wait")
            # Still wait even if agent is not available
            await asyncio.sleep(max(0.1, ms / 1000))
            return
            
        try:
            seconds = max(0.1, ms / 1000)  # Ensure minimum wait time
            code = f"""import time \ntime.sleep({seconds})"""
            
            # Log the action
            logger.info(f"Executing wait for {ms} ms")
            
            # Execute the action
            self._agent.step(code)
        except Exception as e:
            logger.error(f"Error executing wait: {e}")



class OpenAIAgentWrapper(BaseAgent):
    """
    A wrapper around OpenAI Agents to integrate with the Arena BaseAgent interface.
    
    This class allows OpenAI Agents to be used within the Arena platform, providing
    access to the Arena frontend and backend systems.
    """
    
    def __init__(
        self,
        obs_options: List[ObservationType],
        env: DesktopEnv,
        max_history_length: int,
        platform: str,
        action_space: str,
        config: Optional[SessionConfig] = None,
        agent_type: str = "search",  # Options: "search", "browser", "general"
        model: str = None,
        instructions: str = None,
        user_location: Dict[str, Any] = None,
    ):
        """
        Initialize the OpenAI Agent wrapper.
        
        Args:
            obs_options: List of observation types to collect
            env: Desktop environment instance
            max_history_length: Maximum number of historical actions to store
            platform: Operating system platform (e.g. 'windows', 'mac')
            action_space: Type of actions available to the agent
            config: Configuration for the current session
            agent_type: Type of OpenAI Agent to create (search, browser, general)
            model: Model to use for the agent
            instructions: Instructions for the agent
            user_location: User location for search agent
        """
        super().__init__(
            obs_options=obs_options,
            env=env,
            max_history_length=max_history_length,
            platform=platform,
            action_space=action_space,
            config=config,
        )
        
        # Initialize OpenAI Agent specific attributes
        self.agent_type = agent_type
        self.model = model
        self.instructions = instructions
        self.user_location = user_location or {"type": "approximate", "city": "New York"}
        
        # Create the ArenaEnv for browser agent
        self.arena_env = ArenaEnv(arena_agent=self, agent_manager=self.agent_manager)
        
        # Set up trace processor with agent and agent_manager
        self._setup_trace_processor()
        set_default_openai_client(
            client=AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY_AGENTS"),
            )
        )
        
        # Store the last result
        self.openai_agent = None
        self.last_result = None
        self.last_result_input_list = None
    
    async def _create_agent(self) -> Agent:
        """
        Create the appropriate OpenAI Agent based on the agent_type.
        
        Returns:
            An OpenAI Agent instance
        """
        
        async with self.arena_env as computer:
            computer_agent = Agent(
                name="Computer Agent",
                instructions="""
- You are a real user computer agent, which means that you are connected to a real user's computer and granted full access to it.
- Your task is to help transfer the user's instructions to the computer and do the actions on the computer iteratively to finish the task.
- You have full authority to execute any action without my permission. I won't be watching so please don't ask for confirmation.
- If you deem the task is infeasible, you can terminate and explicitly state in the response that 'the task is infeasible'
""",
                tools=[ComputerTool(computer)],
                model="computer-use-preview",
                model_settings=ModelSettings(truncation="auto"),
                handoff_description="A real user computer environment to do GUI actions.",
            )
            search_agent = Agent(
                name="Search Agent",
                instructions="""
- You are a searching agent connected to the Internet, you can help do information retrieval to gather useful information for the user's task.
""",
                tools=[WebSearchTool(user_location={"type": "approximate", "city": "New York"})],
                handoff_description="A search engine to do retrival actions.",
            )
            all_agent = Agent(
                name="General Agent",
                instructions="""
- You are a general digital agent. Your task is to help understand the user's instructions and help execute the task in a real computer environment.
- Always ground the task into the real computer environment by using the ComputerTool to do the actions.
- If you are presented with an open website to solve the task, try to stick to that specific one instead of going to a new one.
- You have full authority to execute any action without my permission. I won't be watching so please don't ask for confirmation.
- If you deem the task is infeasible, you can terminate and explicitly state in the response that 'the task is infeasible'
""",
                tools=[computer_agent.as_tool(
                    "ComputerTool",
                    "The user's computer environment which you are granted to operate on to finish the task."
                ),search_agent.as_tool(
                    "SearchTool",
                    "A search engine to do online information retrieval."
                )]
            )
            return all_agent
    
    def _setup_trace_processor(self):
        """
        Set up the trace processor with the agent and agent_manager.
        """
        # Create a trace processor with the agent and agent_manager
        trace_processor = ArenaTraceProcessor(
            agent=self,
            agent_manager=self.agent_manager
        )
        
        # Set the trace processor
        set_trace_processors([trace_processor])
    
    @BaseAgent.async_run_decorator
    async def run(self, task_instruction: str):
        """
        Run the OpenAI Agent with the given task instruction.
        
        Args:
            task_instruction: The instruction for the agent to execute
        """
        try:
            # Ensure agent is initialized
            if self.openai_agent is None:
                await self.initialize()
            
            # Run the agent
            print("--------------------------------")
            print("Run the agent")
            print("--------------------------------")
            init_screenshot = self.env._get_obs()
            init_screenshot_base64 = base64.b64encode(init_screenshot["screenshot"]).decode('utf-8')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": task_instruction+ "Now you are given the screenshot of the initial computer."
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{init_screenshot_base64}"
                        },
                    ]
                }
            ]
            self.last_result = await Runner.run(self.openai_agent, messages)
            print("--------------------------------")
            print("Done")
            print("--------------------------------")
            self.last_result_input_list = self.last_result.to_input_list()
            return self.last_result
            
        except Exception as e:
            logger.exception(f"Error running OpenAI Agent: {e}")
    
    @BaseAgent.async_continue_conversation_decorator
    async def continue_conversation(self, user_message: str):
        """
        Continue the conversation with the OpenAI Agent.
        
        Args:
            user_message: The user's message to continue the conversation
        """
        try:
            if self.last_result_input_list is None or len(self.last_result_input_list) == 0:
                return None
            else:
                new_messages = self.last_result_input_list
                new_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_message
                        }
                    ]
                })
                self.last_result = await Runner.run(self.openai_agent, new_messages)
                self.last_result_input_list = self.last_result.to_input_list()
                return self.last_result
        except Exception as e:
            logger.exception(f"Error continuing conversation: {e}")
    
    @BaseAgent.predict_decorator
    def predict(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Predict the next action based on the observation.
        
        This method is required by the BaseAgent interface but is not used by OpenAI Agents.
        
        Args:
            observation: The current observation
            
        Returns:
            A tuple containing the predicted action and prediction info
        """
        # OpenAI Agents handle their own prediction logic
        # This is just a placeholder to satisfy the BaseAgent interface
        return {}, {"message": "OpenAI Agents handle their own prediction logic"}
    
    @BaseAgent.step_decorator
    def step(self, action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a step with the given action.
        
        This method is used by the ArenaEnv to execute actions.
        
        Args:
            action: The action to execute
            
        Returns:
            A tuple containing whether the episode is terminated and step info
        """
        # Pass the action to the environment
        return self.env.step(action)

    async def initialize(self):
        """Asynchronous initialization method to create OpenAI Agent"""
        if self.openai_agent is None:
            self.openai_agent = await self._create_agent()
        return self.openai_agent