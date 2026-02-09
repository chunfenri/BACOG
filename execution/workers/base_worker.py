from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseWorker(ABC):
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
    
    @abstractmethod
    def execute(self, 
               prompt: str, 
               schema: Dict[str, Any],
               max_tokens: int,
               **kwargs) -> Dict[str, Any]:
        pass
    
    def validate_output(self, output: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        try:
            import jsonschema
            jsonschema.validate(output, schema)
            return True
        except jsonschema.ValidationError:
            return False
        except Exception as e:
            return False
    
    def get_worker_info(self) -> Dict[str, Any]:
        return {
            'worker_id': self.worker_id,
            'worker_type': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.worker_id}')"


class DummyWorker(BaseWorker):
    def __init__(self, worker_id: str = "dummy", response: Optional[Dict[str, Any]] = None):
        super().__init__(worker_id)
        self.response = response
    
    def execute(self, 
               prompt: str, 
               schema: Dict[str, Any],
               max_tokens: int,
               **kwargs) -> Dict[str, Any]:
        if self.response:
            return self.response
        else:
            return {
                "result": f"Dummy worker received prompt of {len(prompt)} chars",
                "prompt_preview": prompt[:100] if prompt else "",
                "max_tokens": max_tokens
            }

