import os
import sys
import pytest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from temp.desktop_env import DesktopEnv
from hub.Anthropic import AnthropicComputerDemoAgent
from AgentManager import SessionConfig

def test_anthropic_agent_initialization():
    """Test basic initialization of the Anthropic agent"""
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
    
    assert agent is not None
    assert agent.platform == "Ubuntu"
    assert agent.obs_options == ["screenshot"]
    assert agent.class_name == "AnthropicComputerDemoAgent"

def test_anthropic_agent_with_windows():
    """Test agent initialization with Windows platform"""
    env = DesktopEnv()
    config = SessionConfig(
        session_id="test_session",
        task_id="test_task",
        platform="Windows"
    )
    
    agent = AnthropicComputerDemoAgent(
        env=env,
        obs_options=["screenshot"],
        platform="Windows",
        config=config
    )
    
    assert agent.platform == "Windows"

def test_anthropic_agent_invalid_obs_options():
    """Test agent initialization with invalid observation options"""
    env = DesktopEnv()
    config = SessionConfig(
        session_id="test_session",
        task_id="test_task",
        platform="Ubuntu"
    )
    
    with pytest.raises(ValueError):
        AnthropicComputerDemoAgent(
            env=env,
            obs_options=["invalid_option"],
            platform="Ubuntu",
            config=config
        )

def test_anthropic_agent_multiple_obs_options():
    """Test agent initialization with multiple observation options"""
    env = DesktopEnv()
    config = SessionConfig(
        session_id="test_session",
        task_id="test_task",
        platform="Ubuntu"
    )
    
    agent = AnthropicComputerDemoAgent(
        env=env,
        obs_options=["screenshot", "a11y_tree"],
        platform="Ubuntu",
        config=config
    )
    
    assert "screenshot" in agent.obs_options
    assert "a11y_tree" in agent.obs_options

@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
def test_anthropic_agent_predict():
    """Test agent prediction functionality"""
    env = DesktopEnv()
    config = SessionConfig(
        session_id="test_session",
        task_id="test_task",
        platform="Ubuntu"
    )
    
    agent = AnthropicComputerDemoAgent(
        env=env,
        obs_options=["screenshot"],
        platform="Ubuntu",
        config=config
    )
    
    actions, info = agent.predict("Open Chrome browser")
    assert actions is not None
    assert info is not None
    assert "messages" in info
    assert "response" in info
    assert "model_usage" in info

def test_anthropic_agent_step():
    """Test agent step functionality"""
    env = DesktopEnv()
    config = SessionConfig(
        session_id="test_session",
        task_id="test_task",
        platform="Ubuntu"
    )
    
    agent = AnthropicComputerDemoAgent(
        env=env,
        obs_options=["screenshot"],
        platform="Ubuntu",
        config=config
    )
    
    action = [{
        "name": "computer",
        "input": {
            "action": "screenshot"
        },
        "id": "test_id"
    }]
    
    terminated, info = agent.step(action)
    assert isinstance(terminated, bool)
    assert "step_time" in info
    assert "tool_result_content" in info

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
