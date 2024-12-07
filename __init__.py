try:
    from backend.agents.BaseAgent import BaseAgent
    from backend.agents.AgentManager import AgentManager, SessionConfig
    from backend.agents.action.main import Action
    from backend.agents.observation.main import Observation
    from backend.agents.utils.schemas import ObservationType, OBS_DICT
    from backend.agents.utils.exceptions import (
        EnvironmentError, 
        ProcessingError, 
        StepError, 
        StepLimitExceeded, 
        StopExecution,
        VLMPredictionError  
    )
    from backend.agents.utils.utils import Timer, need_visualization
    from backend.agents.hub.PromptAgent.main import PromptAgent
    from backend.agents.hub.Anthropic.main import AnthropicComputerDemoAgent
except ImportError:
    from BaseAgent import BaseAgent
    from utils.schemas import ObservationType, OBS_DICT
    from utils.exceptions import (
        EnvironmentError, 
        ProcessingError, 
        StepError, 
        StepLimitExceeded, 
        StopExecution,
        VLMPredictionError
    )
    from utils import Timer, need_visualization
    from hub.PromptAgent.main import PromptAgent
    from hub.Anthropic.main import AnthropicComputerDemoAgent
    from test import test_anthropic_agent, test_prompt_agent
    from test_env import *

# Make these available when importing from agents
__all__ = [
    'BaseAgent',
    'AgentManager',
    'SessionConfig',
    'Action',
    'Observation',
    'ObservationType',
    'OBS_DICT',
    'EnvironmentError',
    'ProcessingError',
    'StepError',
    'StepLimitExceeded',
    'StopExecution',
    'VLMPredictionError',
    'Timer',
    'need_visualization',
    'PromptAgent',
    'AnthropicComputerDemoAgent',
    'DesktopEnv',
]