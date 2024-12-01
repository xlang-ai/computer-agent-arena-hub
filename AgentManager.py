import time
import threading
from dataclasses import dataclass
from contextlib import contextmanager
from flask_socketio import SocketIO
from typing import Any, Dict, Optional

from .observation.obs_utils import encode_image
from backend.api.data_manager import Conversation, Session
from .utils.exceptions import StopExecution
from backend.main import env_managers, user_manager
from backend.constants import AgentStatus   
@dataclass
class SessionConfig:
    user_id: str
    session_id: str
    region: str
    agent_idx: int
    session: Session
    conversation: Conversation
    socketio: SocketIO
    stop_event: threading.Event


class AgentManager:
    def __init__(self, agent: Any, config: SessionConfig):
        self.agent = agent
        self.config = config

    def initialize(self, task_instruction: str):
        self.step_idx = 0
        self.predict_idx = 0
        self.done = False
        self.total_time = 0
        self.send_start_message(task_instruction)
        env_manager = env_managers[self.config.region]
        env_manager.env_bindings[(
            self.config.user_id, self.config.session_id)]["agent"+str(self.config.agent_idx)]["status"] = AgentStatus.RUNNING
        
        self.config.session.set_task_instruction(task_instruction)
        self.config.conversation.task_instruction = task_instruction
        self.update_session_item({
            "role": "user",
            "type": "user_prompt",
            "content": task_instruction,
        })
        self.total_time_start = time.time()

    def stop_checkpoint(self):
        if self.config.stop_event.is_set():
            raise StopExecution("Execution stopped by user")

    def finalize(self):
        self.config.conversation.upload_conversation_data()
        if (self.config.user_id, self.config.session_id) in env_managers[self.config.region].env_bindings:
            env_managers[self.config.region].set_binded_agent_status(
                binding_key = (self.config.user_id, self.config.session_id),
                agent_idx = "agent"+str(self.config.agent_idx),
                status = AgentStatus.DONE
            )
        env_managers[self.config.region].log_status()

    def send_agent_runtime_status_message(self, agent_status: str, obs_time: Optional[float] = None, predict_time: Optional[float] = None, step_time: Optional[float] = None):
        self.config.socketio.emit(
            'message_agent_runtime_status_left' if self.config.agent_idx == 0 else 'message_agent_runtime_status_right',
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

    def send_start_message(self, task_instruction: str):
        self.config.socketio.emit('message_response_left' if self.config.agent_idx == 0 else 'message_response_right', {
            'type': 'user',
            'content': {
                'title': task_instruction,
            },
            "user_id": self.config.user_id,
        })

    def send_message(self, title: str, image: str, description: str, obs_time: float, agent_time: float, env_time: float, token: int, action: str, visualization: str):
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

    def send_end_message(self, description: str):
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

    def update_session_item(self, item: Dict[str, Any]):
        self.config.conversation.append_conversation_item(item)

    @contextmanager
    def runtime_status(self, status_prefix: str):
        """Context manager for handling agent runtime status messages.

        Args:
            status_prefix: The prefix for the status (e.g. "predict", "observation")
        """
        try:
            self.send_agent_runtime_status_message(
                agent_status=f"{status_prefix}_start")
            start_time = time.time()
            yield
        finally:
            duration = time.time() - start_time
            self.send_agent_runtime_status_message(
                agent_status=f"{status_prefix}_end",
                **{f"{status_prefix}_time": duration}
            )

    def get_screenshot(self):
        try:
            screenshot = self.agent.env._get_screenshot()
            if screenshot is None:
                return ""

            screenshot = encode_image(screenshot)
            return screenshot

        except Exception as e:
            self.agent.logger.exception(e)
            raise e

    def start_video_recording(self):
        try:
            self.agent.env._start_video_recording()
        except Exception as e:
            self.agent.logger.exception(e)
            raise e

    def stop_video_recording(self):
        try:
            return self.agent.env._stop_video_recording()
        except Exception as e:
            self.agent.logger.exception(e)
            raise e
