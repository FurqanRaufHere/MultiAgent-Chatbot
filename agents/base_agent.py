from utils.logger import log
from memory.memory_models import MemoryRecord
from typing import Any, Dict

class BaseAgent:
    """Base class for all worker and coordinator agents, providing basic structure and logging."""
    
    def __init__(self, name: str):
        self.name = name
        # Bind the logger to the specific agent name for clear tracing
        self.log = log.bind(name=name)

    def execute(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """The main method for the agent to perform its specialized task."""
        raise NotImplementedError("Subclasses must implement the execute method.")