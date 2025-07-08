from typing import Optional, Union, List, Tuple, Any
import torch
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from bloombee.models.llama.spe_dec_tree import SpeculativeTree, TreeNode, prepare_incremental_tree_batch
from bloombee.models.llama.config import DistributedLlamaConfig
from bloombee.models.llama.model import DistributedLlamaForCausalLM

from hivemind.utils.logging import get_logger

logger = get_logger()


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
<<<<<<< Updated upstream
        speculative_inference_iteration_size: int = 3,
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        A generate wrapper for speculative decoding.
        """

        # 如果没有传配置，则使用默认值
        generation_config = generation_config or getattr(self, "generation_config", GenerationConfig())

        # 初始化 logits_processor 和 stopping_criteria
        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()

        # 设置强制参数：关闭 do_sample 等
        generation_config.do_sample = False
        generation_config.return_dict_in_generate = False

        # 调用 _sample 来执行 speculative decoding
=======
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

>>>>>>> Stashed changes
        return self._sample(
            input_ids=input_ids,
            ssm=ssm,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=False,
            streamer=streamer,
<<<<<<< Updated upstream
            logits_warper=None,  # 暂不支持 warper
            speculative_inference_iteration_size=speculative_inference_iteration_size,
            **model_kwargs,
        )
=======
            logits_warper=None,
            beam_width=beam_width,
            max_tree_depth=max_tree_depth,
            use_kv_cache=use_kv_cache,
            kv_cache_window=kv_cache_window,
            **model_kwargs,
        )
        
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        speculative_inference_iteration_size: int = 3,
=======
        beam_width: int = 3,
        max_tree_depth: int = 4,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
>>>>>>> Stashed changes
        **model_kwargs,
    ) -> torch.LongTensor:
        logger.info("start sample!!!!")
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        finished = False
        
        # 高效KV cache管理
        past_key_values = None
        is_first_iteration = True
        step_idx = 0
        
        while not finished:
<<<<<<< Updated upstream
            # speculative_inference_iteration_size = min(
            #     speculative_inference_iteration_size, self.active_session._max_length - input_ids.shape[1]
            # )
            with torch.no_grad():
                speculative_outputs = ssm.generate(
                    input_ids,
                    max_new_tokens=speculative_inference_iteration_size,
                    do_sample=False,
                )
                # p_result, topk_tokens, topk_scores = self.small_model.generate_topk_proposals(input_ids, top_k=5)
                speculative_tokens = speculative_outputs[:, -speculative_inference_iteration_size:]
=======
            logger.info(f"\n==================== STEP {step_idx} ====================")
            input_ids_p = input_ids.tolist()
            logger.info(f"[DEBUG] Current prefix:  {input_ids_p} ")
            
            # 1. 构建推测树
            spec_trees = self._build_speculative_trees_batched(
                input_ids, ssm, beam_width, max_tree_depth
            )
            
            # 2. 验证树（方案B：高效增量KV cache）
            verified_tokens, past_key_values = self._verify_trees_incremental_efficient(
                input_ids, spec_trees, logits_processor, past_key_values, 
                is_first_iteration, use_kv_cache, kv_cache_window
            )
            
            is_first_iteration = False
            
            # 3. 应用停止条件
            if has_eos_stopping_criteria:
                verified_tokens = verified_tokens * unfinished_sequences + generation_config.pad_token_id * (
                    1 - unfinished_sequences
                )
>>>>>>> Stashed changes

            # 4. 更新输入序列
            input_ids = torch.cat([input_ids, verified_tokens], dim=-1)
            print("[DEBUG] Verified tokens appended:", verified_tokens.tolist())
            print("[DEBUG] New sequence:", input_ids.tolist())

            if streamer is not None:
                streamer.put(verified_tokens.cpu())

            # 5. 检查是否完成
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            finished = unfinished_sequences.max() == 0
            step_idx = step_idx + 1

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
        logger.info(f"batch_size: {batch_size}")
        for batch_idx in range(batch_size):
            root_token = input_ids[batch_idx, -1].item()
            tree = SpeculativeTree(root_token, f"req_{batch_idx}")
            logger.info(f"[DEBUG] (batch {batch_idx}) root token: {root_token}")
            
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
                
                logger.info(
                    f"[DEBUG] (batch {batch_idx}) depth {depth} SSM candidates: {candidates_per_node}"
                )
                try:
                    new_nodes = tree.add_layer(current_nodes, candidates_per_node)
                    if not new_nodes:
                        break
                except ValueError:
                    break
            
            logger.info(f"[DEBUG] batch {batch_idx} finished tree:\n {tree}")
            trees.append(tree)
        
        return trees
    
    def _verify_trees_incremental_efficient(
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
        方案B: 高效增量KV cache推理 - 核心修复
        """
        
        tree_tokens, attention_mask, batch_node_paths = prepare_incremental_tree_batch(
            trees, input_ids, input_ids.device
        )
        
        if attention_mask is None or tree_tokens.shape[1] == 0:
            return self._fallback_generation_improved(input_ids, logits_processor), past_key_values
        
        with torch.no_grad():
            if not use_kv_cache:
                # 不使用cache：直接处理树tokens
                outputs = self(
                    input_ids=tree_tokens,
                    attention_mask=None,
                    past_key_values=None,
                    use_cache=False
                )
                logits = outputs.logits
                new_past_key_values = None
                
            elif is_first_iteration or past_key_values is None:
                # 首轮：直接处理full_sequence获取完整KV
                full_sequence = torch.cat([input_ids, tree_tokens], dim=-1)
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=None,  # 首轮用标准下三角mask即可
                    past_key_values=None,
                    use_cache=True
                )
                # 提取树部分的logits
                logits = outputs.logits[:, input_ids.shape[1]:, :]
                # 只保留到prefix结尾的KV，丢弃树部分的KV
                prefix_len = input_ids.shape[1]
                # new_past_key_values = tuple(
                #     (k[..., :prefix_len, :], v[..., :prefix_len, :])
                #     for k, v in outputs.past_key_values
                # )
                
                print("DEBUG cache layer 0 type:", type(outputs.past_key_values[0]))
                print("DEBUG cache layer 0 elem0 type:", type(outputs.past_key_values[0][0]))

                # new_past_key_values = tuple(
                #     self._slice_layer_cache(layer, prefix_len)
                #     for layer in outputs.past_key_values
                # )
                self.inspect_kv_cache(outputs.past_key_values)
                new_past_key_values = outputs.past_key_values

                
            else:
                # 后续轮次：直接用cache处理树
                # KV cache窗口管理
                if past_key_values and past_key_values[0][0].size(-2) > kv_cache_window:
                    # 剪裁KV cache
                    trimmed_past_key_values = tuple(
                        (k[..., -kv_cache_window:, :], v[..., -kv_cache_window:, :]) 
                        for k, v in past_key_values
                    )
                    # 重新计算对应的past_len
                    past_len_trimmed = kv_cache_window
                    
                    # 重新构建attention mask以匹配trimmed cache
                    tree_tokens_retrimmed, attention_mask_retrimmed, _ = prepare_incremental_tree_batch(
                        trees, input_ids[:, -past_len_trimmed:], input_ids.device
                    )
                    
                    outputs = self(
                        input_ids=tree_tokens_retrimmed,
                        attention_mask=attention_mask_retrimmed,
                        past_key_values=trimmed_past_key_values,
                        use_cache=True
                    )
                else:
                    # 核心修复：只传树tokens，不传full_sequence
                    outputs = self(
                        input_ids=tree_tokens,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                self.inspect_kv_cache(outputs.past_key_values)
                logits = outputs.logits
                new_past_key_values = outputs.past_key_values
        
        # 提取验证结果
        verified_tokens = self._extract_best_verified_paths_fixed(
            logits, batch_node_paths, input_ids, logits_processor
        )
        logger.info(f"[DEBUG] Verified tokens (per batch): {verified_tokens.tolist()}")
        return verified_tokens, new_past_key_values
    
    def inspect_kv_cache(self, pk, max_layers=2, max_tokens=3):
        """
        pk = outputs.past_key_values
        只查看前 max_layers 层，避免输出过大。
        """
        for layer_idx, layer in enumerate(pk[:max_layers]):
            logger.info(f"\n===== layer {layer_idx} =====")
            
            # 1. (k, v) tuple 或 list
            if isinstance(layer, (tuple, list)) and len(layer) == 2 and torch.is_tensor(layer[0]):
                k, v = layer
                logger.info("format: (k, v) tuple/list")
                logger.info(f"  k.shape: {tuple(k.shape)}, v.shape: {tuple(v.shape)}")
                logger.info(f"  first k[0,0,:3]: {k[0,0,:max_tokens].tolist()}")
            
            # 2. list 包 tuple/list
            elif isinstance(layer, list) and len(layer) \
                and isinstance(layer[0], (tuple, list)):
                logger.info("format: nested list of (k,v)")
                for i, (k, v) in enumerate(layer[:1]):   # 只看第一个头
                    logger.info(f"  head {i}: k.shape {tuple(k.shape)}, v.shape {tuple(v.shape)}")
            
            # 3. list[Tensor]（paged-attention 索引或 stacked kv）
            elif isinstance(layer, list) and len(layer) == 1 and torch.is_tensor(layer[0]):
                t = layer[0]
                logger.info("format: list[Tensor]  —  stacked or meta-index")
                logger.info(f"  tensor.shape: {tuple(t.shape)} dim:  {t.dim()}")
            
            # 4. 直接是 Tensor（[2, n_heads, seq, dim]）
            elif torch.is_tensor(layer):
                logger.info("format: stacked Tensor")
                logger.info(f"  shape: {tuple(layer.shape)}")
            
            else:
                logger.info(f"format: unknown {type(layer)}")
    
    def _slice_layer_cache(self, layer, prefix_len):
        """
        Slice a past-kv *layer* down to prefix_len tokens, whatever shape HF
        decided to give us.  Returns something that your model will accept
        unchanged on the next `forward(use_cache=True)` call.
        """
        import torch

        # ---- 1. (k,v) tuple -------------------------------------------------
        if isinstance(layer, tuple) and len(layer) == 2 and torch.is_tensor(layer[0]):
            k, v = layer
            return (k[..., :prefix_len, :], v[..., :prefix_len, :])

        # ---- 2. list [k,v] (same as above but list) -------------------------
        if isinstance(layer, list) and len(layer) == 2 and torch.is_tensor(layer[0]):
            k, v = layer
            return (k[..., :prefix_len, :], v[..., :prefix_len, :])

        # ---- 3. Hugging Face Cache object -----------------------------------
        if hasattr(layer, "key") and hasattr(layer, "value"):
            return (layer.key[..., :prefix_len, :],
                    layer.value[..., :prefix_len, :])

        # ---- 4. nested list: [(k,v), (k,v), ...]  or  [[k,v], [k,v], ...] ---
        if isinstance(layer, list) and len(layer) and (
            isinstance(layer[0], (tuple, list))
        ):
            sliced = []
            for pair in layer:
                if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                    raise TypeError(
                        f"Nested cache item has unexpected shape {type(pair)} len={len(pair)}"
                    )
                k, v = pair
                sliced.append((k[..., :prefix_len, :], v[..., :prefix_len, :]))
            return sliced
        
        # --- ✱ 5. list [tensor]  --------------------------------------------
        if isinstance(layer, list) and len(layer) == 1 and torch.is_tensor(layer[0]):
            t = layer[0]
            if t.dim() >= 4 and t.size(0) == 2:
                k, v = t[0], t[1]
                return (k[..., :prefix_len, :], v[..., :prefix_len, :])

            # ── case 5b: something else (paged-attention meta, single-dim etc.) ─
            #   → don’t slice; just keep the original list wrapper
            return layer

        # ---- 5. stacked tensor shape [2, heads, seq, dim] -------------------
        if isinstance(layer, torch.Tensor) and layer.size(0) == 2:
            k, v = layer[0], layer[1]
            return (k[..., :prefix_len, :], v[..., :prefix_len, :])

        raise TypeError(f"Un-recognised cache format: {type(layer)}")


    
    def _extract_best_verified_paths_fixed(
        self,
        logits: torch.Tensor,
        batch_node_paths: List[List[List[TreeNode]]],
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList
    ) -> torch.Tensor:
        """
        正确处理空验证序列和最终token processor
        """
        batch_size = logits.shape[0]
        batch_verified = []
        
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
        
        # 安全的长度计算
        best_len = max((len(v) for v in batch_verified), default=0)
        if best_len == 0:
            # 没有验证通过的tokens，生成一个新token
            return self._fallback_generation_improved(input_ids, logits_processor)
        
        # 填充到相同长度
        padded_verified = []
        for verified in batch_verified:
            padded = verified + [0] * (best_len - len(verified))
            padded_verified.append(padded)
        
        verified_tensor = torch.tensor(padded_verified, device=logits.device)
        
        # 处理最终token processor
        if verified_tensor.shape[1] > 0:
            # 使用验证序列的最后位置logits来生成下一个token
            final_positions = []
            for v in batch_verified:
                pos = len(v) - 1
                if pos < 0:  # 保护：防止全失败但被pad的情况
                    pos = logits.shape[1] - 1
                final_positions.append(pos)
            
            final_logits = torch.stack([
                logits[i, pos] for i, pos in enumerate(final_positions)
            ])
            
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