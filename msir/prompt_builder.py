from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_infrastructure.clue import Clue


class MSIRPromptBuilder:
    def __init__(self, include_few_shot: bool = False):
        self.include_few_shot = include_few_shot
    
    def build_prompt(self, 
                    task_state: Dict[str, Any],
                    candidates: List[Clue],
                    k: int) -> str:
        if not candidates:
            raise ValueError("Candidates list is empty")
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
                          
        if len(candidates) < k:
            k = len(candidates)
               
        task_section = self._build_task_section(task_state)
        instruction_section = self._build_instruction_section(k)
        candidates_section = self._build_candidates_section(candidates)
        few_shot_section = self._build_few_shot_section() if self.include_few_shot else ""
        output_section = self._build_output_section()
                
        prompt_parts = [
            task_section,
            instruction_section,
        ]
        
        if few_shot_section:
            prompt_parts.append(few_shot_section)
        
        prompt_parts.extend([
            candidates_section,
            output_section
        ])
        
        return "\n\n".join(prompt_parts)
    
    def _build_task_section(self, task_state: Dict[str, Any]) -> str:
        goal = task_state.get('goal', 'Unknown goal')
        step = task_state.get('step', 1)
        plan = task_state.get('plan', [])
        total_steps = len(plan)

        global_goal = task_state.get('global_goal')
                                                                    
        section = f"# Task\nGoal: {goal}"
        
        if global_goal and global_goal != goal:
            section += f"\nGlobal Context: {global_goal}"
            
        section += f"\nStep: {step}/{total_steps}"
        
        return section
    
    def _build_instruction_section(self, k: int) -> str:
        return (
            "# Instruction\n"
            f"Pick exactly {k} IDs from Candidates.\n"
            "Output JSON ONLY (no markdown / no extra keys): {\"keep\":[\"id1\",\"id2\"]}\n"
            "Rules: use ONLY IDs shown in Candidates; do NOT use placeholders; no extra text."
        )
    
    def _build_candidates_section(self, candidates: List[Clue]) -> str:
        lines = ["# Candidates (showing id, type, brief peek, score)"]
        
        for clue in candidates:
            formatted_line = self._format_candidate(clue)
            lines.append(formatted_line)
        
        return "\n".join(lines)
    
    def _format_candidate(self, clue: Clue) -> str:
        _PEEK_MAX_LENGTH = 48
        _PEEK_TRUNCATE_LENGTH = 45
        _DEFAULT_SCORE = 0.5
                             
        peek = clue.peek
        if len(peek) > _PEEK_MAX_LENGTH:
            peek = peek[:_PEEK_TRUNCATE_LENGTH] + "..."
        
        peek = peek.replace('\n', ' ').replace('\r', '')
                                 
        score = clue.stats.sim if hasattr(clue.stats, 'sim') else _DEFAULT_SCORE
                                                        
        return f"[{clue.id}] s={score:.2f} {peek}"
    
    def _build_few_shot_section(self) -> str:
        return """# Example
{"keep":["Article_A_0","Article_C_0"]}"""
    
    def _build_output_section(self) -> str:
        return """# Output
JSON ONLY:"""
    
    def validate_prompt_length(self, prompt: str, max_tokens: int = None) -> bool:
        if max_tokens is None:
            max_tokens = 2000
            
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            token_count = len(enc.encode(prompt))
            
            if token_count > max_tokens:
                return False
            
            return True
        
        except ImportError:
            _CHARS_PER_TOKEN = 4
            estimated_tokens = len(prompt) // _CHARS_PER_TOKEN
            return estimated_tokens <= max_tokens
