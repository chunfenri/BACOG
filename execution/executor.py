import json
import time
from typing import Dict, Any, List, Optional

import torch

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

                                   
from ..data_infrastructure.pool_manager import PoolManager
from ..data_infrastructure.clue import Clue, ClueStats
from ..data_infrastructure.storage import StorageReader
from ..compiler.compressor import Compressor
from ..compiler.assembler import Assembler
from ..compiler.clipper import Clipper
from ..micro_policy.model import EvidenceSetPolicy
from ..micro_policy.features import RerankerPreprocessor
from ..msir.planner import MSIRPlanner

                          
from .task_state import TaskState
from .state_manager import StateManager
from .anytime import AnytimeHandler
from .budget_scheduler import BudgetScheduler
from .workers.base_worker import BaseWorker
from ..configs.system_config import config


class Executor:
    def __init__(self,
                 micro_policy: EvidenceSetPolicy,
                 preprocessor: RerankerPreprocessor,
                 msir_planner: MSIRPlanner,
                 compressor: Compressor,
                 assembler: Assembler,
                 clipper: Clipper,
                 storage_reader: StorageReader,
                 workers: Dict[str, BaseWorker],
                 contracts: List[Dict[str, Any]],
                 budget_scheduler: BudgetScheduler,
                 enable_anytime: bool = True,
                 device: str = 'cpu'):
        self.policy = micro_policy
        self.preprocessor = preprocessor
        self.msir_planner = msir_planner
        self.compressor = compressor
        self.assembler = assembler
        self.clipper = clipper
        self.storage_reader = storage_reader
        self.workers = workers
        self.contracts = contracts
        self.budget_scheduler = budget_scheduler
        self.enable_anytime = enable_anytime
        self.device = device
                                         
        self.state_manager = StateManager()
        self.anytime_handler = AnytimeHandler()
        
        self.search_results = []
               
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'retry_count': 0,
            'total_tokens_in': 0,
            'total_tokens_out': 0
        }
                                         
        if TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tokenizer = None
        else:
            self._tokenizer = None
    
    def reset_search_results(self):
        self.search_results = []
    
    def run(self, 
           query: str,
           pool_manager: PoolManager,
           task_state: TaskState) -> Dict[str, Any]:
        start_time = time.time()
        self.stats['total_executions'] += 1
        
        try:
            msir_tokens_in = 0
            msir_tokens_out = 0
            worker_tokens_in = 0
            worker_tokens_out = 0

            decision = self._run_micro_policy(query, pool_manager, task_state)
                                                     
            if decision['contract_idx'] >= len(self.contracts):
                decision['contract_idx'] = 0
                                                                               
            is_last_step = (task_state.step >= len(task_state.plan))
            if is_last_step and decision['contract_idx'] != 0:
                decision['contract_idx'] = 0
                                          
            selected_contract = self.contracts[decision['contract_idx']]
            intent = selected_contract['intents'][0]                                              
            worker_id = selected_contract['id']
                                                                           
            b_in_cap, is_oob = self.budget_scheduler.calculate_b_in_cap(
                task_state=task_state.to_dict(),
                intent_b_out=intent['b_out']
            )
            
            if is_oob:
                pass
            
            original_b_in = decision['b_in']
            decision['b_in'] = min(decision['b_in'], b_in_cap)
                                                                  
            estimated_clue_cost = 100
            base_prompt_cost = 300
            max_k_budget = max(1, (b_in_cap - base_prompt_cost) // estimated_clue_cost)
            
            plan_len = len(task_state.plan) if task_state.plan else 0
                                                                           
            k_floor = max(1, plan_len - 1)
            
            is_comparison = getattr(task_state, 'query_type', 'factoid') == 'comparison'
            if is_comparison:
                k_floor = k_floor + 1
                                                                   
            k_hesn = decision['k']
            decision['k'] = max(min(k_hesn, max_k_budget), min(k_floor, max_k_budget))
            
            if decision['k'] != k_hesn:
                pass
                                                                      
            is_qa_step = (decision['contract_idx'] == 0)                    
                                                       
            current_step_idx = task_state.step - 1             
            if current_step_idx < len(task_state.plan):
                step_query = task_state.plan[current_step_idx]
            else:
                step_query = query                              

            if is_qa_step and len(self.search_results) > 0:
                selected_clues = []
                selected_ids = []                
                                                                                
                for i, sr in enumerate(self.search_results):
                    clue_id = f"search_result_{i}"
                    if isinstance(sr, dict):
                        content = f"Q: {sr.get('query', 'Unknown')} → A: {sr.get('answer', '')}"
                    else:
                        content = str(sr)
                    clue = Clue(
                        id=clue_id,
                        type="text",
                        content=content,
                        peek=content[:100] + "..." if len(content) > 100 else content,
                        ptr=content,
                        stats=ClueStats(len=len(content), sim=1.0, novel=1.0, risk='low', ts=i)
                    )
                    selected_clues.append(clue)
                    selected_ids.append(clue_id)
                                                                              
                pool_manager.score_and_rank(query)
                raw_candidates = pool_manager.get_top_k(config.hesn.TOP_K_CANDIDATES)
                                                                                                    
                raw_candidates = [c for c in raw_candidates if not str(c.id).startswith("search_step_")]
                                                                                      
                extra = raw_candidates[:decision['k']]
                selected_clues.extend(extra)
                selected_ids.extend([c.id for c in extra])
                                                                                    
                candidates = raw_candidates
            else:
                rank_query = query if is_qa_step else step_query
                pool_manager.score_and_rank(rank_query)
                
                candidates = pool_manager.get_top_k(config.hesn.TOP_K_CANDIDATES)
                                                         
                filtered = [c for c in candidates if not str(c.id).startswith("search_step_")]
                if len(filtered) >= decision['k']:
                    candidates = filtered
                else:
                                                                                  
                    seen_ids = {str(c.id) for c in filtered}
                    candidates = filtered + [c for c in candidates if str(c.id) not in seen_ids]
                                                                     
                step_task_state = task_state.to_dict()
                step_task_state['global_goal'] = task_state.goal                                
                step_task_state['goal'] = rank_query                                    
                
                selected_ids, msir_usage = self.msir_planner.select_clues(
                    task_state=step_task_state,
                    candidates=candidates,
                    k=decision['k'],
                    b_in=decision['b_in'],                           
                    return_usage=True
                )
                msir_tokens_in += int(msir_usage.get('prompt_tokens', 0) or 0)
                msir_tokens_out += int(msir_usage.get('completion_tokens', 0) or 0)
                
                selected_clues = [c for c in candidates if c.id in selected_ids]
                                                                               
            compile_query = query if is_qa_step else step_query
            
            worker_budget = max(128, b_in_cap - msir_tokens_in)
            
            prompt, compressed_clues = self._compile_prompt(
                selected_clues=selected_clues,
                task_state=task_state,
                decision=decision,
                intent=intent,
                query=compile_query,                                    
                clip_budget=worker_budget                                         
            )
                                                                     
            _compiled_prompt_tokens_est = self._count_tokens(prompt)
            
            if worker_id not in self.workers:
                raise ValueError(f"Worker '{worker_id}' not found")
            
            worker = self.workers[worker_id]
            response = worker.execute(
                prompt=prompt,
                schema=intent['schema'],
                max_tokens=intent['b_out']
            )
            
            if '_meta_usage' in response:
                usage = response.pop('_meta_usage')
                worker_tokens_in += int(usage.get('prompt_tokens', 0) or 0)
                worker_tokens_out += int(usage.get('completion_tokens', 0) or 0)
            else:
                worker_tokens_out += self._count_tokens(json.dumps(response))
            
            retried = False
                                                         
            if self.enable_anytime:
                needs_retry = self.state_manager.needs_retry(
                    response=response,
                    schema=intent['schema'],
                    b_out=intent['b_out'],
                    output_text=json.dumps(response),
                    goal=task_state.goal
                )
                
                if needs_retry:
                    response, retry_usage = self._run_anytime_retry(
                        response=response,
                        candidates=candidates,
                        selected_ids=set(selected_ids),
                        original_prompt=prompt,
                        compressed_clues=compressed_clues,
                        decision=decision,
                        intent=intent,
                        worker=worker,
                        task_state=task_state,
                        original_tokens_in=_compiled_prompt_tokens_est
                    )
                                                         
                    worker_tokens_in += int(retry_usage.get('prompt_tokens', 0) or 0)
                    worker_tokens_out += int(retry_usage.get('completion_tokens', 0) or 0)
                    retried = True
                    self.stats['retry_count'] += 1
                                                            
            tokens_in = msir_tokens_in + worker_tokens_in
            tokens_out = msir_tokens_out + worker_tokens_out
                                                   
            self.budget_scheduler.update_cost(tokens_in, tokens_out)
                                             
            intent_required_fields = intent['schema'].get('required', [])
            
            completion = self.state_manager.check_completion(
                response=response,
                schema=intent['schema'],
                must_have_fields=intent_required_fields                                   
            )
            
            task_state.completion = completion
            success = (completion >= self.state_manager.completion_threshold)
            task_state.update_step(success=success, result=response)
                                                                      
            should_stop = success
            if decision['contract_idx'] == 0:
                                                                         
                should_stop = bool(is_last_step and success)
                                                                            
            if 'rationalized_query' in response and decision['contract_idx'] != 0:
                current_step_idx = task_state.step - 2                                   
                if current_step_idx < len(task_state.plan):
                    sub_query = task_state.plan[current_step_idx]
                else:
                    sub_query = query
                self.search_results.append({
                    "query": sub_query,
                    "answer": response['rationalized_query']
                })
                          
            self.stats['successful_executions'] += (1 if success else 0)
            self.stats['total_tokens_in'] += tokens_in
            self.stats['total_tokens_out'] += tokens_out
            
            execution_time = time.time() - start_time
            
            return {
                'response': response,
                'metadata': {
                    'tokens_in': tokens_in,
                    'tokens_out': tokens_out,
                    'tokens_in_msir': msir_tokens_in,
                    'tokens_out_msir': msir_tokens_out,
                    'tokens_in_worker': worker_tokens_in,
                    'tokens_out_worker': worker_tokens_out,
                    'selected_clues': selected_ids,
                    'retried': retried,
                    'contract_used': worker_id,
                    'intent_used': intent['name'],
                    'completion': completion,
                    'execution_time': execution_time,
                    'success': success,
                    'should_stop': should_stop
                }
            }
            
        except Exception as e:
            return {
                'response': {'error': 'execution_failed', 'message': str(e)},
                'metadata': {
                    'tokens_in': 0,
                    'tokens_out': 0,
                    'success': False
                }
            }

    def _run_micro_policy(self,
                         query: str,
                         pool_manager: PoolManager,
                         task_state: TaskState) -> Dict[str, Any]:
        candidates = pool_manager.get_top_k(config.hesn.TOP_K_CANDIDATES) 
        candidate_texts = [f"[{c.type}] {c.peek}" for c in candidates]
        
        task_context = {
            'step': task_state.step,
            'total_steps': len(task_state.plan),
            'goal': task_state.goal
        }
        
        inputs = self.preprocessor.process_batch(
            queries=[query],
            candidate_lists=[candidate_texts],
            task_contexts=[task_context],
            device=self.device
        )
                    
        self.policy.eval()
        with torch.no_grad():
            outputs = self.policy(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_step=inputs['task_step'],
                goal_input_ids=inputs['goal_input_ids'],
                goal_attention_mask=inputs['goal_attention_mask']
            )
            
        k_logits = outputs['k_logits'][0]
        k_idx = k_logits.argmax().item()
        k_val = self.policy.k_values[k_idx]
                           
        b_in_pred = outputs['b_in_pred'][0].item()
                     
        b_in_val = int(b_in_pred * config.hesn.MAX_INPUT_TOKENS)
                                                                     
        b_in_val = max(512, min(config.hesn.MAX_INPUT_TOKENS, b_in_val))
                                           
        b_in_val = (b_in_val // 16) * 16
                 
        routing_logits = outputs['routing_logits'][0]
        contract_idx = routing_logits.argmax().item()
        
        return {
            'contract_idx': contract_idx,
            'b_in': b_in_val,
            'k': k_val
        }
    
    def _compile_prompt(self,
                       selected_clues: List,
                       task_state: TaskState,
                       decision: Dict[str, Any],
                       intent: Dict[str, Any],
                       query: str = "",
                       clip_budget: int = None) -> tuple:
        compressed = []
        for clue in selected_clues:
            comp_text = self.compressor.compress(
                clue=clue,
                storage_reader=self.storage_reader,
                query=query 
            )
            compressed.append((clue.id, comp_text))
                  
        prompt = self.assembler.assemble(
            task_state=task_state.to_dict(),
            compressed_clues=compressed,
            schema=intent['schema'],
            b_out=intent['b_out']
        )
                                                                                    
        budget = clip_budget if clip_budget is not None else decision['b_in']
        clue_importance = [c.stats.sim for c in selected_clues]
        final_prompt = self.clipper.clip_to_budget(
            prompt=prompt,
            budget=budget,
            compressed_clues=compressed,
            clue_importance=clue_importance
        )
        
        return final_prompt, compressed
    
    def _run_anytime_retry(self,
                          response: Dict[str, Any],
                          candidates: List,
                          selected_ids: set,
                          original_prompt: str,
                          compressed_clues: List,
                          decision: Dict[str, Any],
                          intent: Dict[str, Any],
                          worker: BaseWorker,
                          task_state: TaskState,
                          original_tokens_in: int = 0) -> tuple:
        missing_fields = self.state_manager.extract_missing_fields(
            response, intent['schema']
        )
        
        resend_clues_objs = self.anytime_handler.get_resend_clues(
            candidates=candidates,
            selected_ids=selected_ids,
            missing_fields=missing_fields,
            num_clues=config.anytime.NUM_RESEND_CLUES
        )
        
        if not resend_clues_objs:
            return response, {'in': 0, 'out': 0}
        
        resend_compressed = []
        for clue in resend_clues_objs:
            comp_text = self.compressor.compress(
                clue=clue,
                storage_reader=self.storage_reader
            )
            resend_compressed.append((clue.id, comp_text))
        
        incremental_prompt = self.anytime_handler.build_incremental_prompt(
            original_prompt=original_prompt,
            resend_clues=resend_compressed
        )
        
        new_response = worker.execute(
            prompt=incremental_prompt,
            schema=intent['schema'],
            max_tokens=intent['b_out']
        )

        if isinstance(new_response, dict) and '_meta_usage' in new_response:
            usage = new_response.pop('_meta_usage')
            usage_dict = {
                'prompt_tokens': int(usage.get('prompt_tokens', 0) or 0),
                'completion_tokens': int(usage.get('completion_tokens', 0) or 0),
                'total_tokens': int(usage.get('total_tokens', 0) or 0),
            }
        else:
            usage_dict = {
                'prompt_tokens': 0,
                'completion_tokens': self._count_tokens(json.dumps(new_response)),
                'total_tokens': 0,
            }

        return new_response, usage_dict
    
    def _count_tokens(self, text: str) -> int:
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
                                            
        return len(text) // 4
