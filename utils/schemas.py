"""
Schemas for the agents
"""

from enum import Enum
from typing import TypedDict

class ObservationType(Enum):
    """
    Enum for the type of observation
    """
    SCREENSHOT = "screenshot"
    A11Y_TREE = "a11y_tree"
    TERMINAL = "terminal"
    SOM = "som"
    HTML = "html"  # Not supported yet


class OBS_DICT(TypedDict):
    """
    Type definition for observations.
        screenshot: base64 encoded screenshot
        a11y_tree: ally_tree
        som: base64 encoded som
        html: html
    """
    screenshot: str
    a11y_tree: str
    terminal: str
    som: str
    html: str  # Not supported yet

class ActionType(Enum):
    """
    Enum for the type of action
    """
    PYAUTOGUI = "pyautogui"
    CLAUDE = "claude_computer_use"

class AgentStatus(Enum):
    """
    Enum for the status of an agent
    """
    IDLE = "agent_idle"
    RUNNING = "agent_running"
    STOP = "agent_stop"
    DONE = "agent_done"