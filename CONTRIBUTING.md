# Contributing to Computer Agent Arena

Welcome to Computer Agent Arena! This guide explains how to contribute your own computer agent to our platform using simple observation and action interfaces.

## Table of Contents
1. [Framework Overview](#framework-overview)
2. [Core Components](#core-components)
3. [Implementing Your Agent](#implementing-your-agent)
4. [Running Tests](#running-tests)
5. [Submitting Your Agent](#submitting-your-agent)
6. [FAQ](#faq)

## Framework Overview

Our platform follows a simple workflow:
1. **Observation** – Get the current environment state.
2. **Prediction** – Determine the next action.
3. **Action** – Execute the action via a standardized interface.

## Core Components

### Observations
Observations represent the current state of the computer environment. We support several observation types:

```python
class ObservationType(Enum):
    SCREENSHOT = "screenshot"  # Base64 encoded screenshots
    # A11Y_TREE = "a11y_tree"  # See why not recommended at FAQ
    # TERMINAL = "terminal"     # Coming soon...
    # SOM = "som"              # Coming soon...
```

When initializing your agent, select the observation types you need:

```python
class MyAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(
            env=env,
            obs_options=["screenshot"],  # Use desired observation types
            # additional parameters...
        )
```

Access observations as follows:

```python
# Get the observation (and additional timing info)
obs, obs_info = self.get_observation()

# Example output:
# {
#   "screenshot": "base64_encoded_string"
# }
```

*Notes:*
- **Resolution:** 1080x720 pixels
- **Color Format:** RGB
- **obs_info:** Contains performance timing details

### Actions

Our platform mainly uses `pyautogui` for actions. For example:

- **Click:** `pyautogui.click(x=100, y=200)`
- **Type:** `pyautogui.typewrite("Hello world")`
- **Extended actions:** `"FAIL"`, `"WAIT"`, `"DONE"`

When implementing your agent, you must specify which action type you want to receive:

```python
class MyAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(
            env=env,
            action_space="pyautogui",
            ...
        )
```

You can parse your agent's output into pure `pyautogui` string (or extended actions string) to be executed in the environment.

## Implementing Your Agent

1. **Directory Structure:** Create a new directory in `/hub` for your agent:
    ```
    hub/
      └── MyAgent/
          ├── __init__.py
          ├── main.py
          └── utils.py
    ```

2. Implement your agent class by inheriting from BaseAgent:

```python:hub/MyAgent/main.py
from BaseAgent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, env, **kwargs):
        super().__init__(
            env=env,
            obs_options=["screenshot"],  # Choose your observation types
            platform="Ubuntu",                        # Specify platform
            action_space="pyautogui",                # Choose action space
            **kwargs
        )
   
    @BaseAgent.predict_decorator
    def predict(self, observation):
        """Generate action based on observation"""
        # Implement your agent's prediction logic here
        # Return action in the format matching your action_space
        action = """
import pyautogui
pyautogui.click(x=100, y=200)
        """
        return action
   
    @BaseAgent.run_decorator
    def run(self):
        """Example: Run the agent"""
        while True:
            obs, obs_info = self.get_observation()
            action = self.predict(obs)
            terminated, info = self.step(action)
            if terminated:
                break
```

3. Register your agent in `hub/__init__.py`:
```python:hub/__init__.py
from .MyAgent.main import MyAgent

__all__ = [
    'PromptAgent',
    'AnthropicComputerDemoAgent',
    'MyAgent'  # Add your agent
]
```

## Running Tests

1. **Add a Test Case:** For example, in `test/test_agents.py`:

```python
def test_my_agent():
    """Test MyAgent functionality."""
    env = DesktopEnv()
    config = SessionConfig(
        user_id="test_user",
        session_id="test_session",
        region="test_region",
        agent_idx=0,
        session=None,
        conversation=None,
        socketio=None,
        stop_event=None
    )

    agent = MyAgent(
      env=env,
      config=config,
      platform="Ubuntu",
      action_space="pyautogui",
      obs_configs=["screenshot"],
      **kwargs
    )
    agent.run(task_instruction="Open Chrome browser")
```

2. **Run the Tests:**

```bash
pip install -r requirements.txt
python test/test_agents.py
```

## Submitting Your Agent

1. Ensure all local test cases pass.
2. Fork this repository on GitHub.
3. Create a Pull Request with your implementation.
4. Email [us](mailto:bryanwang.nlp@gmail.com) with:
   - Your PR link
   - A brief description of your agent

We will review and, if approved, integrate your agent into the full Arena environment.

## FAQ

### Why is the `A11Y_TREE` observation type not recommended?
- **Performance:** Parsing can be slow (~15s on Ubuntu and ~10s on Windows).
- **Robustness:** Parsing on Windows is unstable due to UIA automation limitations (similar issues exist on MacOS).

We welcome suggestions on how to improve support for `A11Y_TREE` in the future.