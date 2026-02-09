from typing import List, Set, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from bacog.data_infrastructure.clue import Clue
from bacog.configs.system_config import config


class PoolManager:
    def __init__(self, 
                 max_size: int = None,
                 encoder_model_name: str = "all-MiniLM-L6-v2",
                 device: str = 'cpu',
                 encoder: Any = None):
        self.config = config.pool
        self.pool: List[Clue] = []
        self.max_size = max_size if max_size is not None else self.config.MAX_POOL_SIZE
        self.selected_ids: Set[str] = set()
        self.device = device
        self.encoder_model_name = encoder_model_name
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = SentenceTransformer(self.encoder_model_name, device=self.device)
    
    def add_clues(self, clues: List[Clue]):
        self.pool.extend(clues)
        if len(self.pool) > self.max_size:
            self.pool.sort(key=lambda c: c.stats.sim, reverse=True)
            self.pool = self.pool[:self.max_size]
    
    def score_and_rank(self, query: str, selected_ids: Set[str] = None):
        if selected_ids is None:
            selected_ids = self.selected_ids
            
        if query and hasattr(self, 'encoder'):
            try:
                query_emb = self.encoder.encode(query, convert_to_numpy=True, show_progress_bar=False)
                if len(query_emb.shape) == 2 and query_emb.shape[0] == 1:
                    query_emb = query_emb.flatten()
                
                clue_texts = [c.peek for c in self.pool]
                clue_embs = self.encoder.encode(clue_texts, convert_to_numpy=True, show_progress_bar=False)
                
                norm_query = np.linalg.norm(query_emb)
                norm_clues = np.linalg.norm(clue_embs, axis=1)
                
                norm_clues[norm_clues == 0] = 1e-9
                
                sims = np.dot(clue_embs, query_emb) / (norm_clues * norm_query)
                
                for i, clue in enumerate(self.pool):
                    clue.stats.sim = float(sims[i])
                    
            except Exception as e:
                pass

        selected_contents = [c.peek for c in self.pool if c.id in selected_ids]
        
        ranked_clues = []
        for clue in self.pool:
            if selected_ids and clue.id not in selected_ids:
                clue.stats.novel = self._compute_novel(clue, selected_contents)
            
            tier = 3
            
            if clue.stats.risk == 'high':
                tier = 1
            elif clue.stats.sim > 0.8:
                tier = 2
            
            if clue.id in selected_ids:
                tier = 4
            
            sort_key = (tier, -clue.stats.sim)
            ranked_clues.append((sort_key, clue))
            
        ranked_clues.sort(key=lambda x: x[0])
        
        self.pool = [clue for _, clue in ranked_clues]
    
    def get_top_k(self, k: int) -> List[Clue]:
        return self.pool[:k]
    
    def get_stats_summary(self) -> Dict[str, Any]:
        if not self.pool:
            return {'total_count': 0}
            
        lengths = [clue.stats.len for clue in self.pool]
        sims = [clue.stats.sim for clue in self.pool]
        
        high_risk_count = sum(1 for clue in self.pool if clue.stats.risk == 'high')
        
        return {
            'total_count': len(self.pool),
            'avg_length': np.mean(lengths) if lengths else 0,
            'sim_mean': np.mean(sims) if sims else 0,
            'high_risk_count': high_risk_count
        }
    
    def mark_as_selected(self, clue_ids: List[str]):
        self.selected_ids.update(clue_ids)
    
    def clear_selected(self):
        self.selected_ids.clear()
    
    def clear_pool(self):
        self.pool.clear()
        self.selected_ids.clear()
    
    def _compute_novel(self, clue: Clue, selected_contents: List[str]) -> float:
        if not selected_contents:
            return 1.0
        try:
            clue_emb = self.encoder.encode(clue.peek, convert_to_numpy=True, show_progress_bar=False)
            sel_embs = self.encoder.encode(selected_contents, convert_to_numpy=True, show_progress_bar=False)
            sims = np.dot(sel_embs, clue_emb) / (np.linalg.norm(sel_embs, axis=1) * np.linalg.norm(clue_emb))
            return 1.0 - np.max(sims)
        except:
            return 0.5

