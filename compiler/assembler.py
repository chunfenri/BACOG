from typing import List, Tuple, Dict
from .templates import JsonTemplate


class Assembler:
    
    def __init__(self):
        self.json_template = JsonTemplate()
    
    def assemble(self,
                 task_state: dict,
                 compressed_clues: List[Tuple[str, str]],
                 schema: dict,
                 b_out: int) -> str:
        template = self.json_template
        
        prompt = template.build(
            task_state=task_state,
            compressed_clues=compressed_clues,
            schema=schema,
            b_out=b_out
        )
        
        return prompt

