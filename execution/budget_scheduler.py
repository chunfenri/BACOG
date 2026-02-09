import math
from typing import Dict, Any, Tuple
from bacog.configs.system_config import config

class BudgetScheduler:
    def __init__(self, 
                 b_task_total: float = None, 
                 p_in: float = None, 
                 p_out: float = None, 
                 alpha: float = None):
        self.config = config.budget
        self.b_task_total = b_task_total if b_task_total is not None else self.config.B_TASK_TOTAL
        self.p_in = p_in if p_in is not None else self.config.P_IN
        self.p_out = p_out if p_out is not None else self.config.P_OUT
        self.alpha = alpha if alpha is not None else self.config.ALPHA
               
        if self.p_in > 0:
            self.gamma = self.p_out / self.p_in
        else:
            self.gamma = 1.0
            
        self.c_so_far = 0.0                 
        
    def update_cost(self, tokens_in: int, tokens_out: int):
        cost = tokens_in * self.p_in + tokens_out * self.p_out
        self.c_so_far += cost
        
    def calculate_b_in_cap(self, 
                          task_state: Dict[str, Any], 
                          intent_b_out: int, 
                          b_in_max: int = None) -> Tuple[int, bool]:
        if b_in_max is None:
            b_in_max = config.hesn.MAX_INPUT_TOKENS
                                     
        plan = task_state.get('plan', [])
        total_steps = len(plan) if plan else config.hesn.DEFAULT_TOTAL_STEPS
        current_step = task_state.get('step', 1)
        n_remain = max(1, total_steps - current_step + 1)
              
        b_rem_cost = self.b_task_total - self.c_so_far
                                             
        if self.p_in <= 0:
            return b_in_max
                      
        b_rem_eq = b_rem_cost / self.p_in
        c_out_worst_eq = self.gamma * intent_b_out
                                    
        numerator = b_rem_eq - c_out_worst_eq
        denominator = self.alpha * n_remain
        
        if numerator <= 0:
            return 128, True
            
        b_in_cap = math.floor(numerator / denominator)
                                     
        final_cap = max(0, min(b_in_max, b_in_cap))
                                                           
        if final_cap < 128:
            return 128, True
            
        return final_cap, False

    def get_status(self) -> Dict[str, Any]:
        return {
            "b_task_total": self.b_task_total,
            "c_so_far": self.c_so_far,
            "percent_used": self.c_so_far / self.b_task_total if self.b_task_total > 0 else 0,
            "gamma": self.gamma
        }
