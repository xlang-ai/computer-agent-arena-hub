import os
import pytest
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_env import DesktopEnv
from hub.Anthropic import AnthropicComputerDemoAgent
from hub.PromptAgent import PromptAgent
from hub.UI_TARS import TARSAgent
from AgentManager import SessionConfig
from hub.Anthropic.utils import APIProvider

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

def get_api_key(key_name):
    """Get API key from environment variable API_KEYS"""
    api_keys = os.getenv("API_KEYS")
    if not api_keys:
        return None
    try:
        return json.loads(api_keys).get(key_name)
    except json.JSONDecodeError:
        return None

@pytest.mark.skipif(not get_api_key("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
def test_anthropic_agent():
    """Test agent prediction functionality"""
    return 
    agent = AnthropicComputerDemoAgent(
        env=env,
        obs_options=["screenshot"],
        platform="Ubuntu",
        config=config,
        provider=APIProvider.ANTHROPIC,
    )

    agent.run(task_instruction="Open Chrome browser")

@pytest.mark.skipif(not get_api_key("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_prompt_agent():
    """Test prompt agent"""
    return
    agent = PromptAgent(env=env,
                        model_name="gpt-4o-mini-2024-07-18",
                        obs_options=["screenshot"],
                        platform="Ubuntu",
                        config=config,
                        )
    agent.run(task_instruction="Open Chrome browser")

@pytest.mark.skipif(not get_api_key("UI_TARS_API_KEY"), reason="UI_TARS_API_KEY not set")
def test_tars_agent():
    """Test prompt agent"""
    agent = TARSAgent(env=env,
                        model_name="ui-tars-7b-dpo",
                        obs_options=["screenshot"],
                        platform="Ubuntu",
                        config=config,
                        prompt_template="multi_step",
                        language="English",
                        )
    agent.run(task_instruction="Open Chrome browser")

# TODO: Add tests for customized agents
"""
@pytest.mark.skipif(not get_api_key("MY_KEY"), reason="MY_KEY not set")
def test_my_agent():
    # Test your agent here
    pass
"""


if __name__ == "__main__":
    # pytest.main([__file__, "-v"])
    agent = TARSAgent(env=env,
                        model_name="openai/ui-tars-7b-dpo",
                        obs_options=["screenshot"],
                        platform="Ubuntu",
                        action_space="pyautogui",
                        config=config,
                        prompt_template="multi_step",
                        language="English",
                        )
    agent.run(task_instruction="Open Chrome browser")

