from typing import Optional, Union, List, Tuple, Any
import torch
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from spe_dec_tree import SpeculativeTree, prepare_tree_attention_batch

class DistributedLlamaForSpeculativeGeneration(DistributedLlamaForCausalLM, GenerationMixin):
    def __init__(self, config: DistributedLlamaConfig):
        DistributedLlamaForCausalLM.__init__(self, config)
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        streamer: Optional["BaseStreamer"] = None,
        beam_width: int = 3,
        max_tree_depth: int = 4,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        **model_kwargs,
    ) -> torch.LongTensor:
        
        generation_config = generation_config or getattr(self, "generation_config", GenerationConfig())
        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()

        generation_config.do_sample = False
        generation_config.return_dict_in_generate = False

        return self._sample(
            input_ids=input_ids,
            ssm=ssm,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=False,
            streamer=streamer,
            logits_warper=None,
            beam_width=beam_width,
            max_tree_depth=max_tree_depth,
            use_kv_cache=use_kv_cache,
            kv_cache_window=kv_cache_window,
            **model_kwargs,
        )
        
    def _sample(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList],
        beam_width: int = 3,
        max_tree_depth: int = 4,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        finished = False
        
        # 修复问题3：正确的KV cache管理
        past_key_values = None
        is_first_iteration = True
        
        while not finished:
            # 1. 构建推测树
            spec_trees = self._build_speculative_trees_batched(
                input_ids, ssm, beam_width, max_tree_depth
            )
            
            # 2. 验证树（使用改进的cache策略）
            verified_tokens, past_key_values = self._verify_trees_with_optimal_cache(
                input_ids, spec_trees, logits_processor, past_key_values, 
                is_first_iteration, use_kv_cache, kv_cache_window
            )
            
            is_first_iteration = False
            
            # 3. 应用停止条件
            if has_eos_stopping_criteria:
                verified_tokens = verified_tokens * unfinished_sequences + generation_config.pad_token_id * (
                    1 - unfinished_sequences
                )

            # 4. 更新输入序列
            input_ids = torch.cat([input_ids, verified_tokens], dim=-1)

            if streamer is not None:
                streamer.put(verified_tokens.cpu())

            # 5. 检查是否完成
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        return input_ids
    
    def _build_speculative_trees_batched(
        self, 
        input_ids: torch.LongTensor, 
        ssm: LlamaForCausalLM, 
        beam_width: int, 
        max_depth: int
    ) -> List[SpeculativeTree]:
        """构建推测树（已优化的版本）"""
        batch_size = input_ids.shape[0]
        trees = []
        
        for batch_idx in range(batch_size):
            root_token = input_ids[batch_idx, -1].item()
            tree = SpeculativeTree(root_token, f"req_{batch_idx}")
            
            for depth in range(max_depth):
                current_nodes = tree.get_nodes_at_depth(depth)
                if not current_nodes:
                    break
                
                # 批量构建contexts
                contexts = []
                for node in current_nodes:
                    path_to_node = node.get_path_from_root()
                    context = torch.cat([
                        input_ids[batch_idx, :-1],
                        torch.tensor([root_token] + path_to_node, device=input_ids.device)
                    ])
                    contexts.append(context)
                
                if not contexts:
                    break
                
                # 正确的padding和attention mask
                max_len = max(len(ctx) for ctx in contexts)
                padded_contexts = []
                attention_masks = []
                
                for ctx in contexts:
                    pad_len = max_len - len(ctx)
                    if hasattr(ssm.config, 'pad_token_id') and ssm.config.pad_token_id is not None:
                        pad_token_id = ssm.config.pad_token_id
                    else:
                        pad_token_id = 0
                    
                    padded = torch.cat([
                        torch.full((pad_len,), pad_token_id, dtype=torch.long, device=input_ids.device),
                        ctx
                    ])
                    
                    mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.bool, device=input_ids.device),
                        torch.ones(len(ctx), dtype=torch.bool, device=input_ids.device)
                    ])
                    
                    padded_contexts.append(padded)
                    attention_masks.append(mask)
                
                batch_contexts = torch.stack(padded_contexts)
                batch_masks = torch.stack(attention_masks)
                
                with torch.no_grad():
                    outputs = ssm(batch_contexts, attention_mask=batch_masks)
                    batch_logits = outputs.logits[:, -1, :]
                
                # 生成候选
                candidates_per_node = []
                for i in range(len(current_nodes)):
                    logits = batch_logits[i]
                    top_k_values, top_k_indices = torch.topk(logits, k=beam_width)
                    probs = torch.softmax(logits, dim=-1)
                    
                    candidates = []
                    for j in range(beam_width):
                        token_id = top_k_indices[j].item()
                        prob = probs[token_id].item()
                        candidates.append((token_id, prob))
                    
                    candidates_per_node.append(candidates)
                
                try:
                    new_nodes = tree.add_layer(current_nodes, candidates_per_node)
                    if not new_nodes:
                        break
                except ValueError:
                    break
            
            trees.append(tree)
        
        return trees
    
    def _verify_trees_with_optimal_cache(
        self,
        input_ids: torch.LongTensor,
        trees: List[SpeculativeTree],
        logits_processor: LogitsProcessorList,
        past_key_values: Optional[Any],
        is_first_iteration: bool,
        use_kv_cache: bool,
        kv_cache_window: int
    ) -> Tuple[torch.LongTensor, Any]:
        """
        修复问题3:实现方案B的optimal cache策略
        """
        
        full_sequence, attention_mask, batch_node_paths = prepare_tree_attention_batch(
            trees, input_ids, input_ids.device
        )
        
        if attention_mask is None:
            return self._fallback_generation_improved(input_ids, logits_processor), past_key_values
        
        new_tokens = full_sequence[:, input_ids.shape[1]:]
        
        with torch.no_grad():
            if not use_kv_cache:
                # 方案A：每轮都不用cache，直接full_seq
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=False
                )
                logits = outputs.logits[:, input_ids.shape[1]:, :]
                new_past_key_values = None
                
            elif is_first_iteration or past_key_values is None:
                # 方案B：首轮full_seq获取kv
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True
                )
                logits = outputs.logits[:, input_ids.shape[1]:, :]
                new_past_key_values = outputs.past_key_values
                
            else:
                # 方案B：后续只喂new_tokens，attention_mask=None
                # KV cache窗口管理
                if past_key_values and past_key_values[0][0].size(-2) > kv_cache_window:
                    # 剪裁KV cache
                    new_past_key_values = tuple(
                        (k[..., -kv_cache_window:, :], v[..., -kv_cache_window:, :]) 
                        for k, v in past_key_values
                    )
                else:
                    new_past_key_values = past_key_values
                
                outputs = self(
                    input_ids=new_tokens,
                    attention_mask=None,  # 依赖past_key_values
                    past_key_values=new_past_key_values,
                    use_cache=True
                )
                logits = outputs.logits
                new_past_key_values = outputs.past_key_values
        
        # 提取验证结果
        verified_tokens = self._extract_best_verified_paths_fixed(
            logits, batch_node_paths, input_ids, logits_processor
        )
        
        return verified_tokens, new_past_key_values
    
    def _extract_best_verified_paths_fixed(
        self,
        logits: torch.Tensor,
        batch_node_paths: List[List[List[TreeNode]]],
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList
    ) -> torch.Tensor:
        """
        修复问题4和5:正确处理空验证序列和最终token processor
        """
        batch_size = logits.shape[0]
        batch_verified = []
        
        # 修复问题4：安全的max_len计算
        for batch_idx in range(batch_size):
            node_paths = batch_node_paths[batch_idx]
            best_verified = []
            best_score = -1
            
            for node_path in node_paths:
                verified_tokens = []
                
                for node in node_path:
                    pos = node.position_in_sequence
                    if pos >= logits.shape[1]:
                        break
                    
                    predicted_token = torch.argmax(logits[batch_idx, pos]).item()
                    
                    if predicted_token == node.token_id:
                        verified_tokens.append(node.token_id)
                    else:
                        break
                
                if len(verified_tokens) > best_score:
                    best_score = len(verified_tokens)
                    best_verified = verified_tokens
            
            batch_verified.append(best_verified)
        
        # 修复问题4：安全的长度计算
        best_len = max((len(v) for v in batch_verified), default=0)
        if best_len == 0:
            return torch.zeros(batch_size, 0, dtype=torch.long, device=logits.device)
        
        # 填充到相同长度
        padded_verified = []
        for verified in batch_verified:
            padded = verified + [0] * (best_len - len(verified))
            padded_verified.append(padded)
        
        verified_tensor = torch.tensor(padded_verified, device=logits.device)
        
        # 修复问题5：正确处理最终token processor
        if verified_tensor.shape[1] > 0:
            # 使用新段的最后一个logits
            final_logits = logits[:, -1, :]  # 新段最后一个
            
            # processor输入：完整的当前序列
            processor_input = torch.cat([input_ids, verified_tensor], dim=1)
            processed_logits = logits_processor(processor_input, final_logits)
            
            next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
            
            # 拼接而非覆盖
            verified_tensor = torch.cat([verified_tensor, next_token], dim=1)
        
        return verified_tensor
    
    def _fallback_generation_improved(
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: LogitsProcessorList,
        temperature: float = 1.0
    ) -> torch.LongTensor:
        """改进的fallback生成"""
        with torch.no_grad():
            outputs = self(input_ids)
            logits = outputs.logits[:, -1, :]
            
            processed_logits = logits_processor(input_ids, logits)
            
            if temperature > 0:
                probs = torch.softmax(processed_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
            
            return next_token