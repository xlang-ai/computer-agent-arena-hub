import os
import pytest

from temp.desktop_env import DesktopEnv
from hub.Anthropic import AnthropicComputerDemoAgent
from hub.PromptAgent import PromptAgent
from AgentManager import SessionConfig


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
def test_anthropic_agent():
    return True
    """Test agent prediction functionality"""
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

    agent = AnthropicComputerDemoAgent(
        env=env,
        obs_options=["screenshot"],
        platform="Ubuntu",
        config=config
    )

    agent.run(task_instruction="Open Chrome browser")

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_prompt_agent():
    """Test prompt agent"""
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
    agent = PromptAgent(env=env,
                        model_name="gpt-4o-mini-2024-07-18",
                        obs_options=["screenshot"],
                        platform="Ubuntu",
                        config=config,
                        )
    agent.run(task_instruction="Open Chrome browser")

# TODO: Add tests for customized agents
"""
def test_customized_agent():
    pass
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
