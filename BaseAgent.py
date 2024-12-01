import time
import threading
from functools import wraps
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypedDict

from backend.agents.action.main import Action
from backend.constants import AGENT_MAX_STEPS
from backend.utils.utils import get_temp_video_url, process_action_and_visualize_multiple_clicks, simplify_action
from backend.desktop_env.desktop_env import DesktopEnv
from backend.agents.AgentManager import SessionConfig, AgentManager
from backend.logger import agent_logger as logger
from backend.agents.utils.schemas import ObservationType, OBS_DICT
from backend.agents.utils.exceptions import EnvironmentError, ProcessingError, StepError, StepLimitExceeded, StopExecution, VLMPredictionError
from backend.agents.observation.main import Observation
from backend.agents.utils.utils import Timer, need_visualization

class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    def __init__(
        self,
        obs_options: List[ObservationType],
        env: DesktopEnv,
        max_history_length: int,
        platform: str,
        action_space: str,
        config: SessionConfig = None,
    ):
        self._obs: Dict = None
        self._thought: Dict = None
        self._action: Dict = None
        self._step_result: Dict = None
        
        self._obs_info: Dict = {}
        self._predict_info: Dict = {}
        self._step_info: Dict = {}
        self.logger = logger
        self.history: List[Dict] = []

        self.env = env
        self.obs_config = Observation(obs_options)
        self.env.set_obs_options(dict(self.obs_config))
        self.action_space = action_space
        self.env.action_space = action_space
        self.max_history_length = max_history_length
        self.platform = platform
        self.terminated = False

        self.agent_manager = AgentManager(agent=self, config=config)

        if self.env is None:
            raise ValueError("Env is not provided")
        if self.max_history_length < 0:
            raise ValueError("Max history length should be non-negative")
        if self.platform is None:
            raise ValueError("Platform is not provided")

    def run_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                self.agent_manager.stop_checkpoint()
                self._task_instruction = kwargs.get("task_instruction")
                self.agent_manager.initialize(
                    task_instruction=self._task_instruction)
                func(self, *args, **kwargs)
                self.agent_manager.send_end_message(description=self._step_result)
            except StopExecution:
                self.agent_manager.send_end_message(description="user_stop_execution")
            except StepLimitExceeded:
                self.agent_manager.send_end_message(description="reach_max_steps")
            except Exception as e:
                self.logger.exception(
                    f"{self.agent_manager.config.user_id} {self.agent_manager.config.session_id} {self.agent_manager.config.agent_idx}")
                raise e
            finally:
                self.agent_manager.finalize()
        return wrapper

    
    @run_decorator
    @abstractmethod
    def run(self, task_instruction: str):
        """
        Basic implementation of the run method.
        This method can be overridden by subclasses for custom behavior.
        """
        pass
     
    def obs_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                self.agent_manager.stop_checkpoint()
                with self.agent_manager.runtime_status("obs"):
                    return func(self, *args, **kwargs)
            except StopExecution:
                self.logger.warning(
                    f"User ID:{self.agent_manager.config.user_id} Session ID:{self.agent_manager.config.session_id} Agent ID:{self.agent_manager.config.agent_idx} Stage:observation Execution stopped by user.")
                raise  
            except Exception as e:
                raise e
        return wrapper

    @obs_decorator
    def get_observation(self) -> Tuple[Dict, Dict]:
        """Get and process observations from the environment.

        Returns:
            Tuple[Dict, Dict]: A tuple containing:
                - Dict: Processed observation dictionary with environment state
                - Dict: Timing information for performance monitoring
                    - env_time: Total time spent getting observation
                    - fetch_obs_duration: Time spent fetching raw observation
                    - process_obs_duration: Time spent processing observation

        Raises:
            EnvironmentError: If there are issues getting observations from environment
            ProcessingError: If there are issues processing the raw observation
            ValueError: If observation data is invalid or missing required fields
        """
        try:
            with Timer() as total_timer:
                with Timer() as obs_timer:
                    try:
                        raw_obs = self.env._get_obs()
                        if raw_obs is None:
                            raise ValueError(
                                "Environment returned None observation")
                    except Exception as e:
                        raise EnvironmentError(
                            f"Failed to get observation from environment: {str(e)}")

                with Timer() as process_timer:
                    try:
                        self._obs = self.obs_config.process_observation(
                            raw_obs, self.platform)
                        if not self._obs:
                            raise ValueError(
                                "Processing returned empty observation")
                    except Exception as e:
                        raise ProcessingError(
                            f"Failed to process observation: {str(e)}")
            self._obs_info = {
                "env_time": total_timer.duration,
                "fetch_obs_duration": obs_timer.duration,
                "process_obs_duration": process_timer.duration
            }
            return self._obs, self._obs_info

        except (EnvironmentError, ProcessingError, ValueError) as e:
            self.logger.exception("Observation error: %s", str(e))
            raise
        except Exception as e:
            self.logger.exception(
                "Unexpected error getting observation: %s", str(e))
            raise

    # @staticmethod
    def step_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                self.agent_manager.stop_checkpoint()
                action = args[0] if args else kwargs.get("action")
                if self.agent_manager.step_idx == AGENT_MAX_STEPS:
                    raise StepLimitExceeded
                
                process_result = {}  # Shared storage for the thread result
                def run_process_thread():
                    process_result["simplify_result"] = simplify_action(action)
                    process_result["visualized_screenshot"] = process_action_and_visualize_multiple_clicks(action, self.agent_manager.get_screenshot())
                process_thread = threading.Thread(target=run_process_thread)
                process_thread.start()  # Start the simplify_action thread
                
                with self.agent_manager.runtime_status("step"):
                    if need_visualization(action, self.action_space):
                        self.agent_manager.start_video_recording()
                        time.sleep(0.5)
                        
                    self._step_result, self._step_info = func(self, *args, **kwargs)
                    
                    if need_visualization(action, self.action_space):
                        time.sleep(0.5)
                        recording_buffer = self.agent_manager.stop_video_recording()
                        tmp_url = get_temp_video_url(
                            recording_buffer=recording_buffer,
                            conversation_id=self.agent_manager.config.conversation.conversation_id,
                            step_idx=self.agent_manager.step_idx,
                            action=action,
                            visualize=True
                        )
                    else:
                        tmp_url = None
                    
                    process_thread.join()
                    self.agent_manager.send_message(
                        title=process_result.get("simplify_result", action),
                        image=process_result.get("visualized_screenshot", None),
                        description = self._predict_info.get("response"),
                        obs_time = self._obs_info.get("env_time") if self._obs_info else None,
                        agent_time = self._predict_info.get("model_usage",{}).get("model_time"),
                        env_time = self._step_info.get("step_time"),
                        token = self._predict_info.get('model_usage', {}).get('prompt_tokens'),
                        action = action,
                        visualization = tmp_url
                    )
                    self.agent_manager.update_session_item(item={
                        "step_idx": self.agent_manager.step_idx,
                        "role": "agent",
                        "type": "agent_step",
                        "action": kwargs.get("action"),
                        "new_observation": self._obs,
                        "step_result": self._step_result,
                        "step_time": self._step_info.get("step_time")
                    })
                    self.agent_manager.step_idx += 1
                    return self._step_result, self._step_info
            except StopExecution:
                self.logger.warning(
                    f"User ID:{self.agent_manager.config.user_id} \n Session ID:{self.agent_manager.config.session_id} \n Agent ID:{self.agent_manager.config.agent_idx} \n Stage:step \n Execution stopped by user.")
                raise StopExecution
            except StepLimitExceeded:
                self.logger.warning(
                    f"User ID:{self.agent_manager.config.user_id} \n Session ID:{self.agent_manager.config.session_id} \n Agent ID:{self.agent_manager.config.agent_idx} \n Stage:step \n Execution stopped by step limit.")
                raise StepLimitExceeded
            except Exception as e:
                raise e
        
        return wrapper

    @step_decorator
    def step(self, action: Optional[Dict]):  # -> Tuple[bool, Dict]
        """Execute one step in the environment with the given action.

        Args:
            action (Dict): Action to execute in the environment

        Returns:
            Tuple[bool, Dict]: A tuple containing:
                - bool: Done flag indicating if episode has ended
                - Dict: Performance metrics including step execution time

        Raises:
            StepError: If environment step execution fails
        """
        try:
            step_action = Action(action, self.action_space)
            with Timer() as step_timer:
                terminated, info = self.env.step(step_action.get_action())
                self.logger.warning(
                    f"action_space: {self.action_space} {step_action.get_action()}")
            
            info.update({"step_time": step_timer.duration})
            
            return terminated, info
        
        except Exception as e:
            self.logger.exception(f"Environment step failed: {str(e)}")
            raise StepError(f"Failed to execute step: {str(e)}")

    def get_history(self):
        return self.history


    # @staticmethod
    def predict_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            max_retries = 3
            max_total_time = 120  # seconds
            start_time = time.time()

            for attempt in range(max_retries):
                try:
                    self.agent_manager.stop_checkpoint()
                    with self.agent_manager.runtime_status("predict"):
                        self._actions, self._predict_info = func(self, *args, **kwargs)
                    self.agent_manager.update_session_item(item={
                        "predict_idx": self.agent_manager.predict_idx,
                        "role": "agent",
                        "type": "agent_predict",
                        "task_instruction": self._task_instruction,
                        "observation": self._obs,
                        "actions": self._actions,
                        "model_input": self._predict_info.get("messages"),
                        "model_response": self._predict_info.get("response"),
                        "model_cost": self._predict_info.get("model_usage"),
                        "env_cost": self._obs_info
                    })
                    self.agent_manager.predict_idx += 1
                    return self._actions, self._predict_info
                except StopExecution:
                    self.logger.warning(
                        f"User ID:{self.agent_manager.config.user_id} \n Session ID:{self.agent_manager.config.session_id} \n Agent ID:{self.agent_manager.config.agent_idx} \n Stage:predict \n Execution stopped by user.")
                    raise StopExecution
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= max_total_time or attempt == max_retries - 1:
                        raise VLMPredictionError(
                            f"Prediction failed after {attempt + 1} attempts: {str(e)}")
                    self.logger.warning(
                        f"Prediction attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                    time.sleep(1)  # 添加短暂延迟避免立即重试

            raise VLMPredictionError(
                f"Prediction failed after {max_retries} attempts")
        return wrapper

    
    @predict_decorator
    @abstractmethod
    def predict(self, kwargs):
        """
        Predict the next action based on the given inputs.

        Args:
            task_instruction (str): The current task instruction.
            obs (Dict): The current observation.
            history (List[Dict]): The history of previous observations, actions, and thoughts.

        Returns:
            Tuple[str, str, List[Dict]]: A tuple containing:
                - Input: (str) the input string used for prediction.
                - Output: (str) the predicted response or thought.
                - Others: (List[Dict]) other information to be recorded
            e.g. return Input, Output, Others = predict(task_instruction, obs, history)
        """
        pass
