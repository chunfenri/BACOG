import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from bacog.configs.system_config import config


class EvidenceSetPolicy(nn.Module):
    def __init__(self, backbone_name="BAAI/bge-base-en-v1.5", num_intents=30):
        super().__init__()
        self.config = config.hesn
        self.backbone_name = backbone_name
                                                                        
        self.backbone = SentenceTransformer(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.hidden_dim = self.backbone.get_sentence_embedding_dimension()
                                                  
        self.query_to_set_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.config.NUM_HEADS,
            batch_first=True
        )
                                                                 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.config.NUM_HEADS,
            dim_feedforward=self.config.HIDDEN_DIM,
            batch_first=True
        )
        self.set_transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.config.SET_TRANSFORMER_LAYERS
        )
                              
        self.task_step_proj = nn.Sequential(
            nn.Linear(1, self.config.TASK_EMBED_DIM),
            nn.LayerNorm(self.config.TASK_EMBED_DIM),
            nn.Dropout(0.1)
        )
        self.task_goal_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.config.TASK_EMBED_DIM),
            nn.LayerNorm(self.config.TASK_EMBED_DIM),
            nn.Dropout(0.1)
        )
                            
        fusion_dim = self.hidden_dim + 2 * self.config.TASK_EMBED_DIM
        self.task_fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1)
        )
                                           
        self.seed_vector = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
                           
        self.k_values = self.config.K_VALUES
        self.k_head = nn.Linear(self.hidden_dim, len(self.k_values))
        self.b_in_head = nn.Linear(self.hidden_dim, 1)
                                                                                      
        self.routing_step_proj = nn.Linear(1, 64)                                          
        self.routing_head = nn.Linear(self.hidden_dim + 64, num_intents)
    
    def forward(self, input_ids, attention_mask, task_step, goal_input_ids, goal_attention_mask):
        B, N, L = input_ids.shape
        device = input_ids.device
                                              
        flat_input = input_ids.view(B * N, L)
        flat_mask = attention_mask.view(B * N, L)
        
        with torch.no_grad():
            model_output = self.backbone[0].auto_model(
                input_ids=flat_input,
                attention_mask=flat_mask,
                return_dict=True
            )
                                                                  
            token_embeddings = model_output.last_hidden_state
                                     
            input_mask_expanded = flat_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            candidate_embeddings = sum_embeddings / sum_mask            
            
            goal_output = self.backbone[0].auto_model(
                input_ids=goal_input_ids,
                attention_mask=goal_attention_mask,
                return_dict=True
            )
            goal_tokens = goal_output.last_hidden_state
            goal_mask_expanded = goal_attention_mask.unsqueeze(-1).expand(goal_tokens.size()).float()
            goal_sum = torch.sum(goal_tokens * goal_mask_expanded, 1)
            goal_mask_sum = goal_mask_expanded.sum(1)
            goal_mask_sum = torch.clamp(goal_mask_sum, min=1e-9)
            goal_embedding = goal_sum / goal_mask_sum          
                              
        set_embeds = candidate_embeddings.view(B, N, -1)
                                                                         
        set_context = self.set_transformer(set_embeds)             
                                     
        seed = self.seed_vector.expand(B, -1, -1)             
        
        step_feat = self.task_step_proj(task_step).unsqueeze(1)                          
        goal_feat = self.task_goal_proj(goal_embedding).unsqueeze(1)                          
        
        fusion_input = torch.cat([seed, step_feat, goal_feat], dim=-1)                              
        conditioned_seed = self.task_fusion(fusion_input)             
                                                                      
        scores = torch.matmul(conditioned_seed, set_context.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        global_state = torch.matmul(weights, set_context).squeeze(1)          
                           
        k_logits = self.k_head(global_state)
        b_in_pred = self.b_in_head(global_state)
                                                                               
        step_feat_routing = self.routing_step_proj(task_step)           
        routing_input = torch.cat([global_state, step_feat_routing], dim=-1)             
        routing_logits = self.routing_head(routing_input)
        
        return {
            "k_logits": k_logits,
            "b_in_pred": b_in_pred,
            "routing_logits": routing_logits
        }
    
    def forward_texts(self, queries, candidate_lists, task_steps, goals, device="cuda"):
        B = len(queries)
        N = len(candidate_lists[0])
                            
        all_candidates = []
        for c_list in candidate_lists:
            all_candidates.extend(c_list)
        
        with torch.no_grad():
            query_embeddings = self.backbone.encode(
                queries, 
                convert_to_tensor=True,
                device=device
            )
                                                     
            candidate_embeddings = self.backbone.encode(
                all_candidates,
                convert_to_tensor=True,
                device=device
            )
                                  
            goal_embeddings = self.backbone.encode(
                goals,
                convert_to_tensor=True,
                device=device
            )
                                         
        set_embeds = candidate_embeddings.view(B, N, -1)
                               
        set_context = self.set_transformer(set_embeds)             
                                     
        seed = self.seed_vector.expand(B, -1, -1).to(device)             
        
        step_feat = self.task_step_proj(task_steps.to(device)).unsqueeze(1)                          
        goal_feat = self.task_goal_proj(goal_embeddings).unsqueeze(1)                          
        
        fusion_input = torch.cat([seed, step_feat, goal_feat], dim=-1)
        conditioned_seed = self.task_fusion(fusion_input)             
                           
        scores = torch.matmul(conditioned_seed, set_context.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        global_state = torch.matmul(weights, set_context).squeeze(1)          
        
        k_logits = self.k_head(global_state)
        b_in_pred = self.b_in_head(global_state)
        routing_logits = self.routing_head(global_state)
        
        return {
            "k_logits": k_logits,
            "b_in_pred": b_in_pred,
            "routing_logits": routing_logits
        }
