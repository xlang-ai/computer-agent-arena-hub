import os
import json
import time
from typing import Any, cast, Optional, Dict, List

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)
from .utils import COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG,SYSTEM_PROMPT, SYSTEM_PROMPT_WINDOWS, APIProvider, PROVIDER_TO_DEFAULT_MODEL_NAME
from .utils import _make_api_tool_result, _response_to_params, _inject_prompt_caching, _maybe_filter_to_n_most_recent_images

try:
    # for deploy environment
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.AgentManager import SessionConfig
    from backend.agents.action.main import Action
    from backend.agents.utils.utils import Timer, need_visualization
    from backend.agents.utils.exceptions import EnvironmentError, ProcessingError, StepError, StepLimitExceeded, StopExecution, VLMPredictionError

    from backend.logger import computer_use_logger as logger
    from backend.utils.utils import pretty_print
    from backend.desktop_env.desktop_env import DesktopEnv
except:
    # for local environment
    from BaseAgent import BaseAgent
    from AgentManager import SessionConfig
    from action import Action
    from utils import Timer, need_visualization
    from utils import EnvironmentError, ProcessingError, StepError, StepLimitExceeded, StopExecution, VLMPredictionError

    from test_env.logger import computer_use_logger as logger
    from test_env.utils import pretty_print
    from test_env.desktop_env import DesktopEnv

class AnthropicComputerDemoAgent(BaseAgent):
    def __init__(self,
                 env: DesktopEnv,
                 obs_options=["screenshot"],
                 platform: str = "Ubuntu",
                 model_name: str = "claude-3-5-sonnet-20241022",
                 provider: APIProvider = APIProvider.BEDROCK,
                 max_tokens: int = 4096,
                 api_key: str = os.environ.get("ANTHROPIC_API_KEY", None),
                 system_prompt_suffix: str = "",
                 only_n_most_recent_images: Optional[int] = 10,
                 action_space: str = "claude_computer_use",
                 config: SessionConfig = None,
                 ):
        super().__init__(
            env=env,
            obs_options=obs_options,
            max_history_length=only_n_most_recent_images,
            platform=platform,
            action_space=action_space,
            config=config,
        )
        self.logger = logger
        self.class_name = self.__class__.__name__
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.system_prompt_suffix = system_prompt_suffix
        self.only_n_most_recent_images = only_n_most_recent_images
        self.messages: list[BetaMessageParam] = []

    @BaseAgent.predict_decorator
    def predict(self, task_instruction: str, obs: Dict = None, system: Any = None):
        system = BetaTextBlockParam(
            type="text",
            text=f"{SYSTEM_PROMPT_WINDOWS if self.platform == 'Windows' else SYSTEM_PROMPT}{' ' + self.system_prompt_suffix if self.system_prompt_suffix else ''}"
        )

        if not self.messages:
            self.messages.append({
                "role": "user",
                "content": [{"type": "text", "text": task_instruction}]
            })
            
        enable_prompt_caching = False
        betas = [COMPUTER_USE_BETA_FLAG]
        image_truncation_threshold = 10
        if self.provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=self.api_key)
            enable_prompt_caching = True
        elif self.provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif self.provider == APIProvider.BEDROCK:
            client = AnthropicBedrock(
                # Authenticate by either providing the keys below or use the default AWS credential providers, such as
                # using ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
                aws_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                # aws_region changes the aws region to which the request is made. By default, we read AWS_REGION,
                # and if that's not present, we default to us-east-1. Note that we do not read ~/.aws/config for the region.
                aws_region=os.getenv('AWS_DEFAULT_REGION'),
            )

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(self.messages)
            image_truncation_threshold = 50
            system["cache_control"] = {"type": "ephemeral"}

        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                self.messages,
                self.only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        try:
            self.logger.warning(f"predict message:\n{pretty_print(self.messages)}")
            start_time = time.time()  
            response = client.beta.messages.create(
                max_tokens=self.max_tokens,
                messages=self.messages,
                model=PROVIDER_TO_DEFAULT_MODEL_NAME[self.provider, self.model_name],
                system=[system],
                tools=[{'name': 'computer', 'type': 'computer_20241022', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': None}, {'type': 'bash_20241022', 'name': 'bash'}, {
                    'name': 'str_replace_editor', 'type': 'text_editor_20241022'}] if self.platform == 'Ubuntu' else [{'name': 'computer', 'type': 'computer_20241022', 'display_width_px': 1280, 'display_height_px': 720, 'display_number': None}],
                betas=betas,
            )
            model_time = time.time() - start_time 

        except (APIError, APIStatusError, APIResponseValidationError) as e:
            self.logger.exception(f"Anthropic API error: {str(e)}")
            return None, None

        except Exception as e:
            self.logger.exception(f"Error in Anthropic API: {str(e)}")
            return None, None

        response_params = _response_to_params(response)

        # Store response in message history
        self.messages.append({
            "role": "assistant",
            "content": response_params
        })

        actions: list[Any] = []
        reasonings: list[str] = []
        for content_block in response_params:
            if content_block["type"] == "tool_use":
                actions.append({
                    "name": content_block["name"],
                    "input": cast(dict[str, Any], content_block["input"]),
                    "id": content_block["id"]
                })
            elif content_block["type"] == "text":
                reasonings.append(content_block["text"])
        if isinstance(reasonings, list) and len(reasonings) > 0:
            reasonings = reasonings[0]
        else:
            reasonings = ""
        return actions, {
            "messages":  self.messages,
            "response": reasonings,
            "model_usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "model_time": model_time
            }
        }

    @BaseAgent.obs_decorator
    def get_observation(self):
        pass
    
    @BaseAgent.step_decorator
    def step(self, action: List):  # -> Tuple[bool, Dict]
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
            tool_result_content = []
            with Timer() as step_timer:
                for action_item in action:
                    step_action = Action(action_item, self.action_space)
                
                    terminated, info = self.env.step(step_action.get_action())
                    result_block = _make_api_tool_result(info["tool_result"], action_item["id"])
                    tool_result_content.append(result_block)
                    self.logger.warning(
                        f"action_space: {self.action_space} {step_action.get_action()}")
            
            info = {
                "step_time": step_timer.duration,
                "tool_result_content": tool_result_content
            }
            
            return terminated, info
        
        except Exception as e:
            self.logger.exception(f"Environment step failed: {str(e)}")
            raise StepError(f"Failed to execute step: {str(e)}")
    
    @BaseAgent.run_decorator
    def run(self, task_instruction: str):
        while True:
            self.get_observation()
            actions, predict_info = self.predict(task_instruction)
            if isinstance(actions, list) and len(actions) == 0:
                #TODO: this means the agent outputs no action but pure message for user to interact with
                self.agent_manager.send_interact_message(text=predict_info['response'])
                self.terminated = True
                return
            
            # self.logger.warning(f"Computer use actions: {len(actions)} {actions}")
            terminated, step_info = self.step(actions)

            self.messages.append({
                "role": 'user', 
                "content": step_info['tool_result_content']
            })

    @BaseAgent.continue_conversation_decorator
    def continue_conversation(self, task_instruction: str):
        self.messages.append({
            "role": 'user', 
            "content": task_instruction
        })
        while True:
            self.get_observation()
            actions, predict_info = self.predict(self.messages)
            if isinstance(actions, list) and len(actions) == 0:
                self.agent_manager.send_interact_message(text=predict_info['response'])
                self.terminated = True
                return

