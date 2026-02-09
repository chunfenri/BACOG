import json
from typing import List, Dict, Any, Set

                 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_infrastructure.clue import Clue

from .llm_client import LLMClient
from .prompt_builder import MSIRPromptBuilder


class MSIRPlanner:
    def __init__(self, 
                 llm_client: LLMClient,
                 include_few_shot: bool = False,
                 fallback_to_top_k: bool = True):
        self.llm_client = llm_client
        self.prompt_builder = MSIRPromptBuilder(include_few_shot=include_few_shot)
        self.fallback_to_top_k = fallback_to_top_k
              
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'fallback_calls': 0,
            'invalid_json_count': 0,
            'invalid_id_count': 0
        }
    
    def select_clues(self,
                    task_state: Dict[str, Any],
                    candidates: List[Clue],
                    k: int,
                    b_in: int = 512,
                    return_usage: bool = False):
        self.stats['total_calls'] += 1
                                                                             
        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        
        if not candidates:
            return ([], usage) if return_usage else []
        
        if k <= 0:
            return ([], usage) if return_usage else []
        
        if len(candidates) <= k:
            ids = [c.id for c in candidates]
            return (ids, usage) if return_usage else ids
                                                                                                   
        avg_tokens_per_clue = 45
                                                                                                
        min_window = max(10, 4 * k)
        num_from_budget = max(min_window, b_in // avg_tokens_per_clue)
                                  
        num_candidates_to_show = min(num_from_budget, len(candidates))
        top_candidates = candidates[:num_candidates_to_show]
                  
        try:
            prompt = self.prompt_builder.build_prompt(
                task_state=task_state,
                candidates=top_candidates,
                k=k
            )
        except Exception as e:
            ids = self._fallback_selection(candidates, k, reason="prompt_build_error")
            return (ids, usage) if return_usage else ids
                
        if not self.prompt_builder.validate_prompt_length(prompt):
            pass
                                                 
        try:
            if return_usage:
                result, usage = self.llm_client.call_with_json_with_usage(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.0
                )
            else:
                result = self.llm_client.call_with_json(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.0
                )
        except Exception as e:
            self.stats['invalid_json_count'] += 1
            ids = self._fallback_selection(candidates, k, reason="json_parse_error")
            return (ids, usage) if return_usage else ids
        
        selected_ids = result.get("keep", [])
                 
        if not isinstance(selected_ids, list):
            ids = self._fallback_selection(candidates, k, reason="invalid_keep_type")
            return (ids, usage) if return_usage else ids
                  
        valid_ids_set = {c.id for c in top_candidates}
        validated_ids = []
        invalid_ids = []
        
        for sid in selected_ids:
            if not isinstance(sid, str):
                invalid_ids.append(str(sid))
                continue
            
            if sid in valid_ids_set:
                validated_ids.append(sid)
            else:
                invalid_ids.append(sid)
                
        if invalid_ids:
            self.stats['invalid_id_count'] += len(invalid_ids)
                       
        if len(validated_ids) < k:
            validated_ids = self._fill_selection(
                validated_ids, 
                top_candidates,                                            
                k
            )
                                
        deduped_ids = []
        seen = set()
        for sid in validated_ids:
            if sid in seen:
                continue
            seen.add(sid)
            deduped_ids.append(sid)
        validated_ids = deduped_ids
                   
        final_ids = validated_ids[:k]
        self.stats['successful_calls'] += 1
        
        return (final_ids, usage) if return_usage else final_ids
    
    def _fallback_selection(self, 
                           candidates: List[Clue], 
                           k: int,
                           reason: str = "unknown") -> List[str]:
        if not self.fallback_to_top_k:
            return []

        self.stats['fallback_calls'] += 1
                                
        selected = [c.id for c in candidates[:k]]
        return selected
    
    def _fill_selection(self,
                      current_ids: List[str],
                      candidates: List[Clue],
                      target_k: int) -> List[str]:
        current_set = set(current_ids)
        fill_count = target_k - len(current_ids)
        
        if fill_count <= 0:
            return current_ids
                                   
        fill_ids = []
        for clue in candidates:
            if clue.id not in current_set:
                fill_ids.append(clue.id)
                if len(fill_ids) >= fill_count:
                    break
        
        result = current_ids + fill_ids
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.stats['total_calls']
        if total == 0:
            success_rate = 0.0
        else:
            success_rate = self.stats['successful_calls'] / total
        
        return {
            **self.stats,
            'success_rate': success_rate
        }
    
    def reset_stats(self):
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'fallback_calls': 0,
            'invalid_json_count': 0,
            'invalid_id_count': 0
        }
