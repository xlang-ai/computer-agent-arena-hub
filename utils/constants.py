AGENT_MAX_STEPS = 20
class AgentStatus(Enum):
    IDLE = "agent_idle"
    RUNNING = "agent_running"
    STOP = "agent_stop"
    DONE = "agent_done"