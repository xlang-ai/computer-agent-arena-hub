"""
Hub for different agent implementations
"""

from .PromptAgent.main import PromptAgent
from .Anthropic.main import AnthropicComputerDemoAgent

__all__ = [
    'PromptAgent',
    'AnthropicComputerDemoAgent'
]