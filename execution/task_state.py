from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class TaskState:
    goal: str                                   
    plan: List[str]                                 
    step: int                                       
    next_allowed_intents: List[str]                  
    success_criteria: Dict[str, Any]                               
                                                         
    query_type: str = "factoid"                                    
            
    completion: float = 0.0                         
    attempt_count: int = 0                         
    max_attempts: int = 2                       
          
    history: List[Dict[str, Any]] = field(default_factory=list)        
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "step": self.step,
            "plan": self.plan,
            "next_allowed_intents": self.next_allowed_intents,
            "success": self.success_criteria,
            "completion": self.completion,
            "query_type": self.query_type
        }
    
    def to_full_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskState':
        return cls(
            goal=data.get('goal', ''),
            plan=data.get('plan', []),
            step=data.get('step', 1),
            next_allowed_intents=data.get('next_allowed_intents', []),
            success_criteria=data.get('success_criteria', 
                                     data.get('success', {'must_have': []})),
            query_type=data.get('query_type', 'factoid'),
            completion=data.get('completion', 0.0),
            attempt_count=data.get('attempt_count', 0),
            max_attempts=data.get('max_attempts', 2),
            history=data.get('history', [])
        )
    
    def update_step(self, success: bool = True, result: Optional[Dict[str, Any]] = None):
        if result:
            self.history.append({
                'step': self.step,
                'attempt': self.attempt_count,
                'success': success,
                'result': result
            })
        
        if success:
                      
            self.step += 1
            self.attempt_count = 0
        else:
            self.attempt_count += 1
                               
            if self.attempt_count >= self.max_attempts:
                self.step = max(1, self.step - 1)        
                self.attempt_count = 0
    
    def is_completed(self) -> bool:
        return self.step > len(self.plan) or self.completion >= 0.9
    
    def get_current_step_name(self) -> str:
        if self.step > len(self.plan):
            return "completed"
        if self.step < 1:
            return "not_started"
        return self.plan[self.step - 1]
    
    def get_progress_ratio(self) -> float:
        if not self.plan:
            return 1.0
        return min(1.0, (self.step - 1) / len(self.plan))
    
    def __repr__(self) -> str:
        return (
            f"TaskState(step={self.step}/{len(self.plan)}, "
            f"completion={self.completion:.2f}, "
            f"current='{self.get_current_step_name()}')"
        )


def create_default_task_state(goal: str, 
                               plan: Optional[List[str]] = None,
                               allowed_intents: Optional[List[str]] = None,
                               must_have_fields: Optional[List[str]] = None) -> TaskState:
    if plan is None:
        plan = ["collect information", "analyze and process", "generate result"]
    
    if allowed_intents is None:
        allowed_intents = []
    
    if must_have_fields is None:
        must_have_fields = []
    
    return TaskState(
        goal=goal,
        plan=plan,
        step=1,
        next_allowed_intents=allowed_intents,
        success_criteria={"must_have": must_have_fields}
    )

