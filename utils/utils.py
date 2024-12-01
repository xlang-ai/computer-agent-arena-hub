import time
from typing import Any
from .exceptions import StepError

class Timer:
    """Context manager for timing code blocks."""
    
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.duration = time.time() - self.start


def need_visualization(action: Any, action_space: str) -> bool:
    """Check if the action needs visualization.

    Args:
        action: The action to check
        action_space: The type of action space

    Returns:
        bool: Whether the action needs visualization
    """
    if not action: # action is None
        return False
    
    if action_space == "pyautogui":
        if action in ["WAIT","FAIL","DONE"]:
            # No visualization for WAIT, FAIL, DONE
            return False
        return True
    
    elif action_space == "claude_computer_use":
        if isinstance(action, dict):
            if action["name"] == "computer" and action["input"]["action"] == "screenshot":
                # No visualization for screenshot
                return False
            return True
        elif isinstance(action, list):
            if action[0]["name"] == "computer" and action[0]["input"]["action"] == "screenshot":
                # No visualization for screenshot
                return False
            return True
        else:
            raise StepError()
    else:
        raise StepError()