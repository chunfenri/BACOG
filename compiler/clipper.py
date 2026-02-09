import re
from typing import List, Tuple

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class Clipper:
    def __init__(self):
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def clip_to_budget(self,
                      prompt: str,
                      budget: int,
                      compressed_clues: List[Tuple[str, str]],
                      clue_importance: List[float]) -> str:
        current_tokens = self._count_tokens(prompt)
        
        if current_tokens <= budget:
            return prompt
                                 
        prompt = self._shrink_sentences(prompt, compressed_clues)
        current_tokens = self._count_tokens(prompt)
        
        if current_tokens <= budget:
            return prompt
                    
        prompt = self._shrink_code_spans(prompt)
        current_tokens = self._count_tokens(prompt)
        
        if current_tokens <= budget:
            return prompt
        
        prompt = self._remove_clues_by_importance(
            prompt, 
            compressed_clues, 
            clue_importance, 
            budget
        )
        
        return prompt
    
    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
                          
        return len(text) // 4
    
    def _shrink_sentences(self, 
                         prompt: str,
                         compressed_clues: List[Tuple[str, str]]) -> str:
        new_prompt = prompt
        
        for clue_id, clue_text in compressed_clues:
                        
            if clue_text in prompt:
                                      
                lines = clue_text.split('\n')
                bullet_lines = [l for l in lines if l.strip().startswith('•')]
                
                if len(bullet_lines) >= 3:
                              
                    new_clue_text = '\n'.join(bullet_lines[:2])
                    new_prompt = new_prompt.replace(clue_text, new_clue_text)
        
        return new_prompt
    
    def _shrink_code_spans(self, prompt: str) -> str:
        lines = prompt.split('\n')
        new_lines = []
        in_code_block = False
        code_block_lines = []
        
        for line in lines:
            if re.match(r'L\d+:', line):
                in_code_block = True
                code_block_lines.append(line)
            else:
                if in_code_block:
                               
                    if len(code_block_lines) > 20:
                                    
                        window = 10                   
                        kept_lines = code_block_lines[:window] + ['...'] + code_block_lines[-window:]
                        new_lines.extend(kept_lines)
                    else:
                        new_lines.extend(code_block_lines)
                    code_block_lines = []
                    in_code_block = False
                
                new_lines.append(line)
        
                  
        if code_block_lines:
            if len(code_block_lines) > 20:
                window = 10
                kept_lines = code_block_lines[:window] + ['...'] + code_block_lines[-window:]
                new_lines.extend(kept_lines)
            else:
                new_lines.extend(code_block_lines)
        
        return '\n'.join(new_lines)
    
    def _remove_clues_by_importance(self,
                                    prompt: str,
                                    compressed_clues: List[Tuple[str, str]],
                                    clue_importance: List[float],
                                    budget: int) -> str:
        clues_with_importance = []
        for i, (clue_id, clue_text) in enumerate(compressed_clues):
            importance = clue_importance[i] if i < len(clue_importance) else 0.5
            clues_with_importance.append((importance, i, clue_id, clue_text))
                      
        clues_with_importance.sort(key=lambda x: x[0])
                    
        current_prompt = prompt
        for importance, idx, clue_id, clue_text in clues_with_importance:
            current_tokens = self._count_tokens(current_prompt)
            
            if current_tokens <= budget:
                break
                              
            current_prompt = self._remove_clue_from_prompt(current_prompt, clue_id, clue_text)
        
        return current_prompt
    
    def _remove_clue_from_prompt(self, prompt: str, clue_id: str, clue_text: str) -> str:
        if clue_text in prompt:
            new_prompt = prompt.replace(clue_text, '')
            return new_prompt
                                   
        lines = prompt.split('\n')
        filtered_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if clue_id in line:
                                  
                skip_next = True
                continue
            if skip_next and (line.strip().startswith('"content"') or 
                            line.strip().startswith('}') or
                            not line.strip()):
                       
                if line.strip() == '}' or line.strip() == '},':
                    skip_next = False
                continue
            else:
                skip_next = False
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)

