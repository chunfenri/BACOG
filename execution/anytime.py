from typing import List, Set, Tuple, Dict, Any
                 
from ..data_infrastructure.clue import Clue
from ..configs.system_config import config


class AnytimeHandler:
    def __init__(self, num_resend_clues: int = None):
        self.num_resend_clues = num_resend_clues if num_resend_clues is not None else config.anytime.NUM_RESEND_CLUES
        self.resend_stats = {
            'total_resends': 0,
            'successful_resends': 0
        }
    
    def get_resend_clues(self,
                        candidates: List[Clue],
                        selected_ids: Set[str],
                        missing_fields: List[str],
                        num_clues: int = None) -> List[Clue]:
        if num_clues is None:
            num_clues = self.num_resend_clues
                  
        remaining = [c for c in candidates if c.id not in selected_ids]
        
        if not remaining:
            return []
        
        if len(remaining) <= num_clues:
            return remaining
                     
        scored = []
        for clue in remaining:
            score = self._compute_resend_score(clue, missing_fields, selected_ids)
            scored.append((score, clue))
               
        scored.sort(reverse=True, key=lambda x: x[0])
                 
        selected = self._ensure_diversity([c for _, c in scored], num_clues)
        
        self.resend_stats['total_resends'] += 1
        
        return selected
    
    def build_incremental_prompt(self, 
                                 original_prompt: str,
                                 resend_clues: List[Tuple[str, str]],
                                 shell: str = 'json') -> str:
        if not resend_clues:
            return original_prompt
        
        lines = original_prompt.split('\n')
        insert_idx = None
                            
        for i, line in enumerate(lines):
            if '# Inputs' in line or '"clues"' in line or '## Inputs' in line:
                                    
                for j in range(i + 1, len(lines)):
                                       
                    if lines[j].startswith('#') and 'Constraints' in lines[j]:
                        insert_idx = j
                        break
                    elif lines[j].startswith('#') and j > i + 3:
                                     
                        insert_idx = j
                        break
                break
                                   
        if insert_idx is None:
            for i, line in enumerate(lines):
                if '# Constraints' in line or '## Constraints' in line:
                    insert_idx = i
                    break
                        
        if insert_idx is None:
                                       
            for i in range(len(lines) - 1, -1, -1):
                if 'Response' in lines[i] or 'Output' in lines[i]:
                    insert_idx = i
                    break
                        
        if insert_idx is None:
            _PROMPT_INSERT_OFFSET = 5
            insert_idx = max(0, len(lines) - _PROMPT_INSERT_OFFSET)
        
                
        additional_lines = [
            "",
            "## Additional Clues (Incremental Resend):",
            "# These clues are provided because your previous output was incomplete.",
            ""
        ]
        
        if shell == 'json':
            additional_lines.append('"additional_clues": [')
            for i, (cid, text) in enumerate(resend_clues):
                        
                comma = "," if i < len(resend_clues) - 1 else ""
                            
                text_escaped = text.replace('"', '\\"').replace('\n', '\\n')
                                                     
                trunc_len = config.anytime.RESEND_CLUE_LENGTH
                additional_lines.append(
                    f'  {{"id": "{cid}", "content": "{text_escaped[:trunc_len]}..."}}{comma}'
                )
            additional_lines.append(']')
        else:
                          
            trunc_len = config.anytime.RESEND_CLUE_LENGTH
            for cid, text in resend_clues:
                additional_lines.append(f"[{cid}] {text[:trunc_len]}...")
        
        additional_lines.append("")
            
        new_lines = lines[:insert_idx] + additional_lines + lines[insert_idx:]
        
        return '\n'.join(new_lines)
    
    def _compute_resend_score(self,
                             clue: Clue,
                             missing_fields: List[str],
                             already_selected: Set[str]) -> float:
        score = 0.0
        score += clue.stats.sim
        score += clue.stats.novel
                                      
        if clue.stats.risk in ['high', 'medium']:
            score += 0.5                        
                                    
        if missing_fields:
            peek_lower = clue.peek.lower()
            for field in missing_fields:
                if field.lower() in peek_lower:
                    score += 0.5                            
        
        return score
    
    def _ensure_diversity(self, 
                         candidates: List[Clue], 
                         target_count: int) -> List[Clue]:
        if len(candidates) <= target_count:
            return candidates
        
        selected = []
        type_counts = {}
                       
        for clue in candidates:
            if len(selected) >= target_count:
                break
            
            clue_type = clue.type
            current_count = type_counts.get(clue_type, 0)
                               
            _DIVERSITY_LIMIT = 2
            if current_count >= _DIVERSITY_LIMIT and len(selected) < target_count - 1:
                              
                has_other_types = any(
                    c.type != clue_type 
                    for c in candidates[len(selected):]
                )
                if has_other_types:
                    continue
            
            selected.append(clue)
            type_counts[clue_type] = current_count + 1
        
                    
        if len(selected) < target_count:
            for clue in candidates:
                if clue not in selected:
                    selected.append(clue)
                    if len(selected) >= target_count:
                        break
        
        return selected[:target_count]
    
    def record_resend_result(self, success: bool):
        if success:
            self.resend_stats['successful_resends'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.resend_stats['total_resends']
        if total > 0:
            success_rate = self.resend_stats['successful_resends'] / total
        else:
            success_rate = 0.0
        
        return {
            **self.resend_stats,
            'success_rate': success_rate
        }
    
    def reset_stats(self):
        self.resend_stats = {
            'total_resends': 0,
            'successful_resends': 0
        }

