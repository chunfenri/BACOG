import re
from typing import Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from bacog.configs.system_config import config


def _has_meta_params(model: Any) -> bool:
    try:
        for p in model.parameters():
            if getattr(p, "is_meta", False):
                return True
    except Exception:
        return False
    return False


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

class Compressor:
    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2", encoder: Optional[SentenceTransformer] = None):
                                                      
        self.encoder_model = encoder_model
        self._encoder = encoder
        self.config = config.compressor
    
    @property
    def encoder(self):
        if self._encoder is None:
            device = "cuda" if _torch_cuda_available() else "cpu"
            self._encoder = SentenceTransformer(self.encoder_model, device=device)
        if _has_meta_params(self._encoder):
            device = "cuda" if _torch_cuda_available() else "cpu"
            self._encoder = SentenceTransformer(self.encoder_model, device=device)
        return self._encoder
    
    def compress(self, clue, storage_reader, query: str = "") -> str:
        if getattr(clue, 'content', None):
            full_content = clue.content
        else:
                                                                  
            if hasattr(storage_reader, 'read'):
                full_content = storage_reader.read(clue.ptr)
            else:
                full_content = str(clue.ptr)           
        
        if not full_content:
            return getattr(clue, 'peek', "") or f"[{clue.id}]"
        
                       
        clue_type = getattr(clue, 'type', 'text')
        if clue_type == 'text':
            result = self.compress_text(full_content, query)
        elif clue_type == 'code':
            result = self.compress_code(full_content, getattr(clue, 'ptr', ''))
        elif clue_type == 'log':
            result = self.compress_log(full_content)
        else:
            result = full_content[:500] + "..." if len(full_content) > 500 else full_content
                                                      
        if not result or not result.strip():
                                                                      
            fallback = getattr(clue, 'peek', None) or full_content[:200] or f"[{clue.id}]"
            return fallback
        
        return result
    
    def compress_text(self, text: str, query: str) -> str:
        try:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if not paragraphs:
                return text[:self.config.FALLBACK_TEXT_LEN]
                
            if not query:
                return "\n\n".join(paragraphs[:self.config.DEFAULT_PARAGRAPHS])
                      
            para_embeddings = self.encoder.encode(paragraphs)
            query_embedding = self.encoder.encode(query)
            
            scores = []
            for i, para_emb in enumerate(para_embeddings):
                score = np.dot(query_embedding, para_emb)
                scores.append((score, i, paragraphs[i]))
            
                        
            scores.sort(key=lambda x: x[0], reverse=True)
            
            selected_paras = []
            current_len = 0
                       
            if scores:
                selected_paras.append(scores[0][2])
                current_len += len(scores[0][2])
                                     
            if current_len < self.config.FALLBACK_TEXT_LEN and len(scores) > 1:
                selected_paras.append(scores[1][2])
            
            return "\n\n".join(selected_paras)
            
        except Exception as e:
            return text[:self.config.FALLBACK_TEXT_LEN]

    def compress_code(self, code: str, ptr: str) -> str:
        try:
                                                               
            target_line = self._parse_line_range_from_ptr(ptr)
            return self._fallback_window(code, target_line)
        except Exception as e:
            return code[:self.config.FALLBACK_TEXT_LEN]

    def compress_log(self, log: str) -> str:
        lines = log.split('\n')
        return "\n".join(lines[:self.config.FALLBACK_LOG_LINES])

    def _parse_line_range_from_ptr(self, ptr: str) -> int:
        match = re.search(r'#L(\d+)', ptr)
        if match:
            return int(match.group(1))
        return 1

    def _fallback_window(self, code: str, target_line: int) -> str:
        lines = code.split('\n')
        window = self.config.FALLBACK_WINDOW_LINES
        start = max(0, target_line - 1 - window)
        end = min(len(lines), target_line - 1 + window + 1)
        return "\n".join(lines[start:end])
