"""
Base class for all agents in the system.
"""
import time
import threading
import traceback
from functools import wraps
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any



ENV_TYPE = ""
try:    
    # for deploy environment
    from backend.agents.AgentManager import SessionConfig, AgentManager
    from backend.agents.observation.main import Observation
    from backend.agents.action.main import Action
    from backend.agents.utils.schemas import ObservationType, OBS_DICT
    from backend.agents.utils.exceptions import ProcessingError, StepError, StepLimitExceeded, StopExecution, VLMPredictionError
    from backend.agents.utils.utils import Timer, need_visualization
    from backend.agents.utils.constants import AGENT_MAX_STEPS

    from backend.logger import agent_logger as logger
    from backend.utils.utils import get_temp_video_url, process_action_and_visualize_multiple_clicks, simplify_action
    from backend.desktop_env.desktop_env import DesktopEnv
    ENV_TYPE = "deploy"
except ImportError:
    # for test environments
    from AgentManager import SessionConfig, AgentManager
    from observation import Observation
    from action import Action
    from utils import ObservationType, OBS_DICT
    from utils import ProcessingError, StepError, StepLimitExceeded, StopExecution, VLMPredictionError
    from utils import Timer, need_visualization
    from utils.constants import AGENT_MAX_STEPS

    from test_env.logger import agent_logger as logger
    from test_env.utils import get_temp_video_url, process_action_and_visualize_multiple_clicks, simplify_action
    from test_env.desktop_env import DesktopEnv

class BaseAgent(ABC):
    """Base class for all agents in the system.
    
    This class provides core functionality for agent operations including:
    - Environment interaction (observation and action execution)
    - Action prediction
    - Session management
    - Error handling and logging
    
    Attributes:
        obs_options (List[ObservationType]): List of observation types to collect
        env (DesktopEnv): Desktop environment instance
        max_history_length (int): Maximum number of historical actions to store
        platform (str): Operating system platform (e.g. 'windows', 'mac')
        action_space (str): Type of actions available to the agent
        config (SessionConfig, optional): Configuration for the current session
    """

    def __init__(
        self,
        obs_options: List[ObservationType],
        env: DesktopEnv,
        max_history_length: int,
        platform: str,
        action_space: str,
        config: Optional[SessionConfig] = None,
    ):
        # Add type hints and validation for all instance variables
        self._obs: Optional[Dict] = None
        self._thought: Optional[Dict] = None
        self._action: Optional[Dict] = None
        self._step_result: Optional[Dict] = None
        self._task_instruction: Optional[str] = None
        
        # Use TypedDict for structured dictionaries
        self._obs_info: Dict[str, float] = {}
        self._predict_info: Dict[str, Any] = {}
        self._step_info: Dict[str, Any] = {}
        
        # Initialize core components
        self._initialize_components(
            env=env,
            obs_options=obs_options,
            action_space=action_space,
            platform=platform,
            max_history_length=max_history_length,
            config=config
        )
        
    def _initialize_components(self, **kwargs) -> None:
        """Initialize core agent components with validation."""
        # Validate required parameters
        if kwargs['env'] is None:
            raise ValueError("Environment must be provided")
        if kwargs['max_history_length'] < 0:
            raise ValueError("Max history length must be non-negative")
        if kwargs['platform'] is None:
            raise ValueError("Platform must be specified")

        # Set up environment
        self.env = kwargs['env']
        self.obs_config = Observation(kwargs['obs_options'])
        self.env.set_obs_options(dict(self.obs_config))
        
        # Configure action space
        self.action_space = kwargs['action_space']
        self.env.action_space = self.action_space
        
        # Set other attributes
        self.max_history_length = kwargs['max_history_length']
        self.platform = kwargs['platform']
        self.terminated = False
        self.history: List[Dict] = []
        self.logger = logger
        
        # Initialize agent manager
        self.agent_manager = AgentManager(agent=self, config=kwargs.get('config'))

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

    def async_run_decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                self.agent_manager.stop_checkpoint()
                self._task_instruction = kwargs.get("task_instruction")
                self.agent_manager.initialize(
                    task_instruction=self._task_instruction)
                result = await func(self, *args, **kwargs)
                self.agent_manager.send_end_message(description=self._step_result)
                return result
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
        return async_wrapper
    
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
                    if isinstance(action, dict):
                        # TODO: here for Operator, optimize me
                        process_result["simplify_result"] = simplify_action(action.get("action", ""))
                        process_result["visualized_screenshot"] = process_action_and_visualize_multiple_clicks(action.get("action", ""), self.agent_manager.get_screenshot())
                    else:
                        process_result["simplify_result"] = simplify_action(action)
                        process_result["visualized_screenshot"] = process_action_and_visualize_multiple_clicks(action, self.agent_manager.get_screenshot())
                process_thread = threading.Thread(target=run_process_thread)
                process_thread.start()  # Start the simplify_action thread
                
                with self.agent_manager.runtime_status("step"):
                    if need_visualization(action, self.action_space):
                        self.agent_manager.start_video_recording()
                        time.sleep(0.5)
                        
                    self._step_result, self._step_info = func(self, *args, **kwargs)
                    
                    if need_visualization(action, self.action_space) and self.agent_manager.config.conversation:
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
    def step(self, action: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """Execute one step in the environment with the given action.
        
        Args:
            action: Dictionary containing action parameters
                   Format depends on action_space configuration
        
        Returns:
            terminated: Whether the episode has ended
            info: Dictionary containing step execution metrics
        
        Raises:
            StepError: If environment step execution fails
        """
        try:
            step_action = Action(action, self.action_space)
            with Timer() as step_timer:
                terminated, info = self.env.step(step_action.get_action())
                
                # Log at debug level instead of warning for action execution
                self.logger.debug(
                    f"Executed action in {self.action_space} space: {step_action.get_action()}"
                )
            
            info["step_time"] = step_timer.duration
            return terminated, info
        
        except Exception as e:
            self.logger.exception("Step execution failed")
            raise StepError(f"Failed to execute step: {str(e)}") from e

    def get_history(self):
        return self.history


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
                    self.logger.warning(
                        f"Prediction attempt {attempt + 1} failed: {str(e)}\n{traceback.format_exc()}")
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= max_total_time or attempt == max_retries - 1:
                        raise VLMPredictionError(
                            f"Prediction failed after {attempt + 1} attempts: {str(e)}")
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

    def continue_conversation_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                self.agent_manager.stop_checkpoint()
                user_message = kwargs.get("user_message")
                
                # 不重置agent_manager的状态，只是发送新消息
                self.agent_manager.send_user_message(user_message)
                
                # 更新会话项
                self.agent_manager.update_session_item({
                    "role": "user",
                    "type": "user_message",
                    "content": user_message,
                })
                
                # 执行继续对话的方法
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
    
    @continue_conversation_decorator
    @abstractmethod
    def continue_conversation(self, user_message: str):
        """
        继续与用户的对话，将新消息追加到历史中。
        这个方法应该被子类重写以提供具体实现。
        
        Args:
            user_message: 用户的新消息
        """
        pass