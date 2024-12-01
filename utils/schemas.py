from enum import Enum
from typing import TypedDict

class ObservationType(Enum):
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
    PYAUTOGUI = "pyautogui"
    CLAUDE = "claude_computer_use"
