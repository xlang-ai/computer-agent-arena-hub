"""
Action class for the agent.
"""
import ast
from typing import Dict, Any, Optional, Union

class Action:
    """Action class for the agent."""
    def __init__(self, raw_action: Union[Dict, str], action_space: str):
        """Initialize the Action class.

        Args:
            raw_action: The raw action
            action_space: The action space
        """
        self._action_space = None
        self._action = None
        self.action_space = action_space
        self.action = raw_action

    @property
    def action(self) -> str:
        return self._action

    @property
    def action_space(self) -> str:
        return self._action_space

    @action_space.setter
    def action_space(self, value: str):
        """
        Set the action space for the agent.
        Currently only supports 'pyautogui' as a valid action space.

        Args:
            value (str): The action space to set

        Raises:
            ValueError: If action_space is empty or invalid
        """
        if not value:
            raise ValueError("action_space is required")
        if value not in ["pyautogui", "claude_computer_use"]:
            raise ValueError(
                "Invalid action space. Allowed spaces are: pyautogui")
        self._action_space = value

    

    @action.setter
    def action(self, value: Optional[str]):
        """
        Set the action for the agent.
        For pyautogui action space, accepts special commands (WAIT, FAIL, DONE) or valid Python code.
        For claude_computer_use action space, accepts a dict with keys "name", "input" and "id".

        Args:
            value (str | dict): The action to set

        Raises:
            ValueError: If action is empty or invalid
        """
        if not value:
            raise ValueError("action cannot be empty")

        if self._action_space == "pyautogui":
            self._action = value
            # if value in ["WAIT", "FAIL", "DONE"]:
            #     self._action = value
            # elif self._is_valid_python_code(value):
            #     self._action = value
            # else:
            #     raise ValueError("Invalid action format for pyautogui")
        elif self._action_space == "claude_computer_use":
            self._action = value
            # if self._is_valid_claude_computer_use_action(value):
            #     self._action = value
        else:
            raise ValueError(
                f"Invalid action space: {self._action_space}, allowed spaces are: pyautogui, claude_computer_use")

    def __str__(self) -> str:
        """Return a string representation of the Action instance.

        Returns:
            str: A string showing the action space and action value
        """
        return f"Action(action_space='{self._action_space}', action='{self._action}')"

    def get_action(self) -> Optional[str]:
        """Get the action.

        Returns:
            str: The action
        """
        return self._action

    def to_dict(self) -> Dict[str, Any]:
        """Convert the action to a dictionary.

        Returns:
            dict: The action as a dictionary
        """
        return {"action_space": self._action_space, "action": self._action}

    def _is_valid_python_code(self, code: str) -> bool:
        """
        Validate if the given string is valid Python code syntax.

        Args:
            code (str): The code string to validate

        Returns:
            bool: True if code is valid Python syntax, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            raise ValueError("Invalid Python code syntax")

    def _is_valid_claude_computer_use_action(self, action: Dict[str, Any]) -> bool:
        """Validate if the given action is valid for the claude_computer_use action space.

        Args:
            action: The action to validate

        Returns:
            bool: True if action is valid, False otherwise
        """
        if not isinstance(action, dict):
            raise ValueError("Invalid action format for claude_computer_use")
        if not (action.get("name") and action.get("input") and action.get("id")):
            raise ValueError(
                "Invalid action format for claude_computer_use, 'name', 'input' and 'id' are required")
        return True
