"""
Agent Manager module for handling agent execution and communication.

This module provides classes for managing agent sessions, runtime status,
and communication between agents and the frontend/backend systems.
"""

import time
import threading
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

from flask_socketio import SocketIO

from observation.obs_utils import encode_image
from utils.exceptions import StopExecution
from utils.schemas import AgentStatus

# Determine environment type
ENV_TYPE = "local"
try:
    from backend.api.data_manager import Conversation, Session
    from backend.main import env_managers, user_manager
    ENV_TYPE = "deploy"
except ImportError:
    pass


@dataclass
class SessionConfig:
    """Configuration for an agent session.
    
    Attributes:
        user_id: Unique identifier for the user
        session_id: Unique identifier for the session
        region: Geographic region for the session
        agent_idx: Index of the agent in the session
        session: Optional Session object for deployment
        conversation: Optional Conversation object for deployment
        socketio: Optional SocketIO instance for real-time communication
        stop_event: Optional threading Event for stopping execution
    """
    user_id: str
    session_id: str
    region: str
    agent_idx: int
    session: Any
    conversation: Any
    socketio: Optional[SocketIO] = None
    stop_event: Optional[threading.Event] = None


class AgentManager:
    """Manages agent execution, communication, and status updates.
    
    This class handles the lifecycle of an agent, including initialization,
    runtime status updates, message handling, and cleanup.
    """

    def __init__(self, agent: Any, config: SessionConfig) -> None:
        """Initialize the AgentManager.

        Args:
            agent: The agent instance to manage
            config: SessionConfig containing session configuration
        """
        self.agent = agent
        self.config = config
        self.step_idx: int = 0
        self.predict_idx: int = 0
        self.done: bool = False
        self.total_time: float = 0
        self.total_time_start: float = 0

    def initialize(self, task_instruction: str) -> None:
        """Initialize the agent session with a task instruction.

        Args:
            task_instruction: The instruction for the agent to execute
        """
        self.step_idx = 0
        self.predict_idx = 0
        self.done = False
        self.total_time = 0
        self.total_time_start = time.time()
        
        self.send_start_message(task_instruction)
        
        if ENV_TYPE == "deploy":
            env_manager = env_managers[self.config.region]
            binding_key = (self.config.user_id, self.config.session_id)
            
            # Update environment bindings
            env_manager.env_bindings[binding_key][f"agent{self.config.agent_idx}"]["status"] = AgentStatus.RUNNING
            
            # Update session and conversation
            self.config.session.set_task_instruction(task_instruction)
            self.config.conversation.task_instruction = task_instruction
            self.update_session_item({
                "role": "user",
                "type": "user_prompt",
                "content": task_instruction,
            })

    def stop_checkpoint(self) -> None:
        """Check if execution should be stopped."""
        if ENV_TYPE == "deploy" and self.config.stop_event and self.config.stop_event.is_set():
            raise StopExecution("Execution stopped by user")

    def finalize(self) -> None:
        """Cleanup and finalize the agent session."""
        if ENV_TYPE == "deploy":
            self.config.conversation.upload_conversation_data()
            
            region_manager = env_managers[self.config.region]
            binding_key = (self.config.user_id, self.config.session_id)
            
            if binding_key in region_manager.env_bindings:
                region_manager.set_binded_agent_status(
                    binding_key=binding_key,
                    agent_idx=f"agent{self.config.agent_idx}",
                    status=AgentStatus.DONE
                )
            region_manager.log_status()

    def send_agent_runtime_status_message(
        self,
        agent_status: str,
        obs_time: Optional[float] = None,
        predict_time: Optional[float] = None,
        step_time: Optional[float] = None
    ) -> None:
        """Send runtime status updates to the frontend.

        Args:
            agent_status: Current status of the agent
            obs_time: Time taken for observation
            predict_time: Time taken for prediction
            step_time: Time taken for step execution
        """
        if ENV_TYPE == "deploy":
            event_name = 'message_agent_runtime_status_left' if self.config.agent_idx == 0 else 'message_agent_runtime_status_right'
            self.config.socketio.emit(
                event_name,
                {
                    "user_id": self.config.user_id,
                    'content': {
                        'status': agent_status,
                        'obs_time': obs_time,
                        'predict_time': predict_time,
                        'step_time': step_time,
                    },
                }
            )

    def send_start_message(self, task_instruction: str) -> None:
        """Send a start message to the frontend.

        Args:
            task_instruction: The instruction for the agent to execute
        """
        if ENV_TYPE == "deploy":
            self.config.socketio.emit('message_response_left' if self.config.agent_idx == 0 else 'message_response_right', {
                'type': 'user',
                'content': {
                    'title': task_instruction,
                },
                "user_id": self.config.user_id,
            })

    def send_message(
        self,
        title: str,
        image: str,
        description: str,
        obs_time: float,
        agent_time: float,
        env_time: float,
        token: int,
        action: str,
        visualization: str
    ) -> None:
        """Send a message to the frontend.

        Args:
            title: Title of the message
            image: Image associated with the message
            description: Description of the message
        """
        if ENV_TYPE == "deploy":
            self.config.socketio.emit('message_response_left' if self.config.agent_idx == 0 else 'message_response_right', {
                "type": "agent",
                "content": {
                    "title": title,
                    "time": time.time() - self.total_time_start,
                    "image": image,
                    "description": description,
                    "obs_time": obs_time,
                    "agent_time": agent_time,
                    "env_time": env_time,
                    "token": token,
                    "action": action,
                    "visualization": visualization,
                },
                "user_id": self.config.user_id,
            })

    def send_end_message(self, description: str) -> None:
        """Send an end message to the frontend.

        Args:
            description: Description of the message
        """
        if ENV_TYPE == "deploy":
            self.config.socketio.emit('message_response_left' if self.config.agent_idx == 0 else 'message_response_right', {
                "type": "end",
                "content": {
                    "title": "end",
                    "time": time.time() - self.total_time_start,
                    "image": "",
                    "description": description,
                },
                "user_id": self.config.user_id,
            })

    def update_session_item(self, item: Dict[str, Any]) -> None:
        """Update the session item in the conversation.

        Args:
            item: Dictionary containing the item to update
        """
        if ENV_TYPE == "deploy":
            self.config.conversation.append_conversation_item(item)

    @contextmanager
    def runtime_status(self, status_prefix: str) -> None:
        """Context manager for handling agent runtime status messages.

        Args:
            status_prefix: The prefix for the status (e.g. "predict", "observation")
        """
        try:
            if ENV_TYPE == "deploy":
                self.send_agent_runtime_status_message(
                    agent_status=f"{status_prefix}_start")
            start_time = time.time()
            yield
        finally:
            duration = time.time() - start_time
            if ENV_TYPE == "deploy":
                self.send_agent_runtime_status_message(
                    agent_status=f"{status_prefix}_end",
                    **{f"{status_prefix}_time": duration}
                )

    def get_screenshot(self) -> str:
        """Get a screenshot from the agent's environment.

        Returns:
            str: Encoded image string
        """
        try:
            screenshot = self.agent.env._get_screenshot()
            if screenshot is None:
                return ""

            screenshot = encode_image(screenshot)
            return screenshot

        except Exception as e:

            self.agent.logger.exception(e)
            raise e

    def start_video_recording(self) -> None:
        """Start video recording for the agent's environment."""
        try:
            self.agent.env._start_video_recording()
        except Exception as e:
            self.agent.logger.exception(e)
            raise e

    def stop_video_recording(self) -> None:
        """Stop video recording for the agent's environment."""
        try:
            return self.agent.env._stop_video_recording()
        except Exception as e:
            self.agent.logger.exception(e)
            raise e
