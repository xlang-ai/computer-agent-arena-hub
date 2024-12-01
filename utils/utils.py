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
    if not action:
        return False
    
    if action_space == "pyautogui":
        if action in ["WAIT","FAIL","DONE"]:
            return False
        return True
    
    elif action_space == "claude_computer_use":
        if isinstance(action, dict):
            if action["name"] == "computer" and action["input"]["action"] == "screenshot":
                return False
            return True
        elif isinstance(action, list):
            if action[0]["name"] == "computer" and action[0]["input"]["action"] == "screenshot":
                return False
            return True
        else:
            raise StepError()
    else:
        raise StepError()