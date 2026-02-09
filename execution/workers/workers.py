from typing import Dict, Any

from .base_worker import BaseWorker
from ...msir.llm_client import LLMClient


class QAWorker(BaseWorker):
    
    def __init__(self, worker_id: str, llm_client: LLMClient):
        super().__init__(worker_id)
        self.llm_client = llm_client
    
    def execute(self, 
               prompt: str, 
               schema: Dict[str, Any],
               max_tokens: int,
               **kwargs) -> Dict[str, Any]:
        try:
            content, usage = self.llm_client.call_with_usage(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
                                                                               
            try:
                result = self.llm_client._parse_json_response(content)
                if not isinstance(result, dict):
                    result = {"answer": content, "parse_error": True}
            except Exception:
                result = {"answer": content, "parse_error": True}
                                            
            result['_meta_usage'] = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "answer": "",
                "_meta_usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }


class SearchWorker(BaseWorker):
    def __init__(self, worker_id: str, llm_client: LLMClient):
        super().__init__(worker_id)
        self.llm_client = llm_client
    
    def execute(self, 
               prompt: str, 
               schema: Dict[str, Any],
               max_tokens: int,
               **kwargs) -> Dict[str, Any]:
        try:
            content, usage = self.llm_client.call_with_usage(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
                                                                               
            try:
                result = self.llm_client._parse_json_response(content)
                if not isinstance(result, dict):
                    result = {"rationalized_query": content, "parse_error": True}
            except Exception:
                result = {"rationalized_query": content, "parse_error": True}
            
                                            
            result['_meta_usage'] = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "rationalized_query": "",
                "_meta_usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }


class SummaryWorker(BaseWorker):
    def __init__(self, worker_id: str, llm_client: LLMClient):
        super().__init__(worker_id)
        self.llm_client = llm_client
    
    def execute(self, 
               prompt: str, 
               schema: Dict[str, Any],
               max_tokens: int,
               **kwargs) -> Dict[str, Any]:
        try:
            content, usage = self.llm_client.call_with_usage(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
                                                                               
            try:
                result = self.llm_client._parse_json_response(content)
                if not isinstance(result, dict):
                    result = {"summary": content, "parse_error": True}
            except Exception:
                result = {"summary": content, "parse_error": True}
            
            result['_meta_usage'] = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "summary": "",
                "_meta_usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }
