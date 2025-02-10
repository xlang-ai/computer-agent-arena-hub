"""
Hub for different agent implementations
"""

from .PromptAgent.main import PromptAgent
from .Anthropic.main import AnthropicComputerDemoAgent
from .UI_TARS.main import TARSAgent

__all__ = [
    'PromptAgent',
    'AnthropicComputerDemoAgent',
    'TARSAgent'
]
