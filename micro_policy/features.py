from transformers import AutoTokenizer
import torch

from bacog.configs.system_config import config

class RerankerPreprocessor:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", max_length=None, tokenizer=None):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length if max_length else config.hesn.MAX_INPUT_TOKENS
        
    def process_batch(self, queries, candidate_lists, task_contexts, device="cpu"):
        B = len(queries)
                                                                                        
        N = len(candidate_lists[0]) 
                                             
        pairs = []
        for i in range(B):
            q = queries[i]
            c_list = candidate_lists[i]
                                                                                
            for j in range(N):
                if j < len(c_list):
                    pairs.append([q, c_list[j]])
                else:
                                                      
                    pairs.append([q, ""])
                        
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].view(B, N, -1).to(device)
        attention_mask = encoded['attention_mask'].view(B, N, -1).to(device)
        
        steps = []
        goals = []
        for ctx in task_contexts:
                                          
            s = ctx.get('step', 1)
            t = ctx.get('total_steps', config.hesn.DEFAULT_TOTAL_STEPS)
            steps.append(s / max(1, t))
            goals.append(ctx.get('goal', ""))
            
        task_step = torch.tensor(steps, dtype=torch.float32).unsqueeze(1).to(device)         
                        
        goal_encoded = self.tokenizer(
            goals,
            padding=True,
            truncation=True,
            max_length=config.hesn.MAX_GOAL_TOKENS,                          
            return_tensors="pt"
        )
        goal_input_ids = goal_encoded['input_ids'].to(device)
        goal_attention_mask = goal_encoded['attention_mask'].to(device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "task_step": task_step,
            "goal_input_ids": goal_input_ids,
            "goal_attention_mask": goal_attention_mask
        }
