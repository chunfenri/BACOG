import re
from typing import Dict, Any, List, Optional
from ..configs.system_config import config

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    pass


class StateManager:
    def __init__(self, 
                 completion_threshold: float = None,
                 output_length_threshold: float = None):
        self.completion_threshold = completion_threshold if completion_threshold is not None else 0.9
        self.output_length_threshold = output_length_threshold if output_length_threshold is not None else 0.05

    def _assess_field_quality(self, 
                             response: Dict[str, Any],
                             must_have_fields: List[str]) -> float:
        if not must_have_fields:
            return 1.0
        
        for field in must_have_fields:
            if field not in response:
                return 0.0
            value = response[field]
            if not value or (isinstance(value, str) and not value.strip()):
                return 0.0
                
        return 1.0

    def _estimate_token_count(self, text: str) -> int:
        _CHARS_PER_TOKEN = 4
        return len(text) // _CHARS_PER_TOKEN
    
    def check_completion(self, 
                        response: Dict[str, Any], 
                        schema: Dict[str, Any],
                        must_have_fields: List[str]) -> float:
        if not response:
            return 0.0
                              
        is_schema_valid = self._validate_schema(response, schema)
                                   
        are_fields_present = True
        if must_have_fields:
            are_fields_present = all(
                f in response and response[f] and str(response[f]).strip()
                for f in must_have_fields
            )
                           
        if is_schema_valid and are_fields_present:
            return 1.0
        else:
            return 0.0
    
    def needs_retry(self, 
                   response: Dict[str, Any], 
                   schema: Dict[str, Any],
                   b_out: int,
                   output_text: Optional[str] = None,
                   goal: Optional[str] = None) -> bool:
        if 'error' in response:
            return True
        
        if not self._validate_schema(response, schema):
            return True
                         
        required = schema.get('required', [])
        missing = [f for f in required if f not in response]
        if missing:
            return True
                                                                
        empty_required = []
        for f in required:
            if f not in response:
                continue
            v = response.get(f)
            if v is None:
                empty_required.append(f)
                continue
            if isinstance(v, str) and not v.strip():
                empty_required.append(f)
                continue
        if empty_required:
            return True
                              
        if response.get('needs_more_info', False):
            return True

        if isinstance(required, list) and 'answer' in required and 'answer' in response:
            ans = response.get('answer')
            if isinstance(ans, str):
                a = ans.strip()
                g = (goal or "").strip().lower()
                                                           
                a_lower = a.lower()
                if a.startswith("{") or a.endswith("}") or "search error" in a_lower or "traceback" in a_lower:
                    return True
                                                                          
                if any(k in g for k in ["how many", "what year", "population", "how much", "what number"]):
                    if re.search(r"\d", a) is None:
                        return True
               
        return False
    
    def extract_missing_fields(self, 
                              response: Dict[str, Any],
                              schema: Dict[str, Any]) -> List[str]:
        required = schema.get('required', [])
        missing = [f for f in required if f not in response or not response[f]]
        return missing
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        if not JSONSCHEMA_AVAILABLE:
                               
            required = schema.get('required', [])
            return all(f in data for f in required)
        
        try:
            jsonschema.validate(data, schema)
            return True
        except jsonschema.ValidationError as e:
            return False
        except Exception as e:
            return False
    
    def get_config(self) -> Dict[str, Any]:
        return {
            'completion_threshold': self.completion_threshold,
            'output_length_threshold': self.output_length_threshold
        }
