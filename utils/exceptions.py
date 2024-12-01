"""
Exceptions for the agents
"""

class AgentError(Exception):
    """Base exception class for all agent-related errors"""
    pass

class EnvironmentError(AgentError):
    """Raised when there are issues getting observations from the environment"""
    pass

class ProcessingError(AgentError):
    """Raised when there are issues processing the raw observation"""
    pass

class StepError(AgentError):
    """Raised when there are issues executing an action"""
    pass

class PredictionError(AgentError):
    """Raised when there are issues making predictions"""
    pass

class ValidationError(AgentError):
    """Raised when there are data validation issues"""
    pass 

class VLMPredictionError(Exception):
    """Raised when there are issues making predictions with the VLM"""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class StopExecution(Exception):
    """Raised when the execution should be stopped"""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class StepLimitExceeded(Exception):
    """Raised when the step limit is exceeded"""
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
