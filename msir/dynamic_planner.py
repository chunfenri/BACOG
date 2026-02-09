import json
from typing import Dict, Any, List, Optional
from ..execution.task_state import TaskState, create_default_task_state
from .llm_client import LLMClient
from ..configs.system_config import config


class DynamicPlanner:
    def __init__(self, llm_client: LLMClient, avg_step_cost: float = None):
        self.llm_client = llm_client
        self.avg_step_cost = avg_step_cost if avg_step_cost is not None else config.msir.AVG_STEP_COST
        
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'fallback_calls': 0
        }
    
    def generate_plan_with_subqueries(self, 
                                      question: str, 
                                      max_steps: int = 4,
                                      candidates_preview: Optional[List[str]] = None,
                                      return_usage: bool = False):
        self.stats['total_calls'] += 1
                   
        prompt = self._build_decomposition_prompt(question, max_steps, candidates_preview)
        
        try:
            if return_usage:
                response, usage = self.llm_client.call_with_json_with_usage(prompt, max_tokens=500, temperature=0.0)
            else:
                response = self.llm_client.call_with_json(prompt, max_tokens=500, temperature=0.0)
                usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            
            plan = self._parse_subquery_response(response, max_steps)
            
            if plan:
                self.stats['successful_calls'] += 1
                return (plan, usage) if return_usage else plan
            else:
                raise ValueError("Empty plan returned")
                
        except Exception as e:
            self.stats['fallback_calls'] += 1
            plan = self._get_fallback_plan(question, max_steps)
            usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            return (plan, usage) if return_usage else plan
    
    def _build_decomposition_prompt(self, question: str, max_steps: int, 
                                    candidates_preview: Optional[List[str]]) -> str:
        candidates_section = ""
        if candidates_preview:
            preview_text = "\n".join([f"- {c[:150]}..." for c in candidates_preview[:5]])
            candidates_section = f"\n\nAvailable Information Sources:\n{preview_text}"
        
        prompt = f"""You are a Query Decomposition Expert for Multi-Hop Question Answering.

Question: {question}

Task: Decompose this question into at most {max_steps} sub-steps (sub-queries).{candidates_section}

CRITICAL RULES:
1. For COMPARISON questions (Are X and Y...?, Who is older/taller...?, Which came first...?):
   - Step 1: Search for information about X
   - Step 2: Search for information about Y  
   - Final: Compare and answer (QA)

2. For CHAIN questions (The director of X was born in...?):
   - Step 1: Search for X's director
   - Step 2: Search for director's birthplace
   - Final: Answer (QA)

3. For SIMPLE questions: 
   - Step 1: Search for information
   - Final: Answer (QA)

4. ALL steps EXCEPT the LAST must have intent="search"
5. The LAST step MUST have intent="qa"
6. Each search step should have a DIFFERENT, SPECIFIC sub-query

Output JSON array exactly like this:
[
  {{"query": "What is Scott Derrickson's nationality?", "intent": "search"}},
  {{"query": "What is Ed Wood's nationality?", "intent": "search"}},
  {{"query": "Compare the nationalities of Scott Derrickson and Ed Wood", "intent": "qa"}}
]

Output JSON ONLY (no markdown, no explanation):"""
        
        return prompt
    
    def _parse_subquery_response(self, response: Any, max_steps: int) -> Optional[List[Dict[str, str]]]:
        plan = None
        if isinstance(response, list):
            plan = response
        elif isinstance(response, dict):
                                                                             
            if "_list" in response and isinstance(response["_list"], list):
                plan = response["_list"]
                                                  
            for key in ['steps', 'plan', 'queries', 'sub_queries', 'decomposition']:
                if key in response and isinstance(response[key], list):
                    plan = response[key]
                    break
                                                                          
            if plan is None:
                if 'query' in response or 'sub_query' in response:
                                                                                  
                    single_query = response.get('query', response.get('sub_query', ''))
                    if single_query:
                        original_question = single_query                            
                        plan = [
                            {"query": single_query, "intent": "search"},
                            {"query": f"Based on the evidence, answer: {single_query}", "intent": "qa"}
                        ]
                else:
                    return None
        else:
            return None
        
        if not plan or len(plan) == 0:
            return None
        
              
        validated_plan = []
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                continue
            
            query = step.get('query', step.get('sub_query', ''))
            intent = step.get('intent', 'search').lower()
            
            if not query:
                continue
                        
            if i == len(plan) - 1:
                intent = 'qa'
                             
            elif intent not in ('search', 'qa'):
                intent = 'search'
            elif i < len(plan) - 1:
                intent = 'search'
            
            validated_plan.append({
                'query': query,
                'intent': intent
            })
        
        if len(validated_plan) < 2:
            return None
                       
        if len(validated_plan) > max_steps:
            validated_plan = validated_plan[:max_steps]
            validated_plan[-1]['intent'] = 'qa'
        
        return validated_plan
    
    def _get_fallback_plan(self, question: str, max_steps: int) -> List[Dict[str, str]]:
        return [
            {"query": question, "intent": "search"},
            {"query": f"Based on the evidence, answer: {question}", "intent": "qa"}
        ]
    
    
    def _detect_query_type(self, query: str) -> str:
        q_lower = (query or "").lower()
        comparison_patterns = (
            "compare", "vs", "versus", "than", "difference", 
            "same as", "different from", "which one", "which of"
        )
        if any(p in q_lower for p in comparison_patterns):
            return "comparison"
        return "factoid"
    
    def generate_state(self, query: str, total_budget: float = None) -> TaskState:
        if total_budget is None:
            total_budget = config.msir.DEFAULT_TOTAL_BUDGET
                   
        max_steps = max(2, int(total_budget / self.avg_step_cost))
                                                         
        query_type = self._detect_query_type(query)
                      
        plan_with_subqueries = self.generate_plan_with_subqueries(query, max_steps)
                           
        plan_steps = [step['query'] for step in plan_with_subqueries]
        
        return TaskState(
            goal=query,
            plan=plan_steps,
            step=1,
            next_allowed_intents=[],
            success_criteria={"must_have": ["answer"]},
            query_type=query_type,                                                  
            history=[{"subquery_plan": plan_with_subqueries}]
        )

