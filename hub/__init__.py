"""
Hub for different agent implementations
"""

from .PromptAgent.main import PromptAgent
from .Anthropic.main import AnthropicComputerDemoAgent
from .UI_TARS.main import TARSAgent
from .OpenAICUA.main import OpenAICUAAgent
from .OpenAIAgents.openai_agent import OpenAIAgentWrapper

__all__ = [
    'PromptAgent',
    'AnthropicComputerDemoAgent',
    'TARSAgent',
    'OpenAICUAAgent',
    'OpenAIAgentWrapper'
]
