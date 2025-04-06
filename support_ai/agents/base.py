from abc import ABC, abstractmethod
import subprocess
from typing import Dict, Any

class BaseAgent(ABC):
    def query_mistral(self, prompt: str) -> str:
        result = subprocess.run(
            ['ollama', 'run', 'mistral'],
            input=prompt.encode(),
            stdout=subprocess.PIPE
        )
        return result.stdout.decode()

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass