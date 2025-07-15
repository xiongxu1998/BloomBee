from typing import Optional, Union, List, Tuple, Any
import torch
import numpy as np
import contextlib
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from bloombee.models.llama.spe_dec_tree import SpeculativeTree, TreeNode, prepare_incremental_tree_batch
from bloombee.models.llama.config import DistributedLlamaConfig
from bloombee.models.llama.model import DistributedLlamaForCausalLM
from bloombee.client.remote_generation import RemotePastKeyValues
from bloombee.client.inference_session import InferenceSession

from hivemind.utils.logging import get_logger

logger = get_logger()


class DistributedLlamaForSpeculativeGeneration(DistributedLlamaForCausalLM):
    def __init__(self, config: DistributedLlamaConfig):
        super().__init__(config)
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        streamer: Optional["BaseStreamer"] = None,
        beam_width: int = 2,
        max_tree_depth: int = 2,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 50,
        **model_kwargs,
    ) -> torch.LongTensor:
        
        generation_config = generation_config or getattr(self, "generation_config", GenerationConfig())
        logits_processor = logits_processor or LogitsProcessorList()
        stopping_criteria = stopping_criteria or StoppingCriteriaList()

        generation_config.do_sample = False
        generation_config.return_dict_in_generate = False

        # Calculate session max length - this is critical for distributed inference
        session_max_length = 128

        # Use inference session for proper distributed caching
        with self.transformer.h.inference_session(max_length=session_max_length) as session:
            return self._sample_with_session(
                input_ids=input_ids,
                ssm=ssm,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                session=session,
                streamer=streamer,
                beam_width=beam_width,
                max_tree_depth=max_tree_depth,
                use_kv_cache=use_kv_cache,
                kv_cache_window=kv_cache_window,
                max_new_tokens=max_new_tokens,
                **model_kwargs,
            )
        
    def _sample_with_session(
        self,
        input_ids: torch.LongTensor,
        ssm: LlamaForCausalLM,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        session: InferenceSession,
        streamer: Optional["BaseStreamer"],
        beam_width: int = 2,
        max_tree_depth: int = 2,
        use_kv_cache: bool = True,
        kv_cache_window: int = 2048,
        max_new_tokens: int = 50,
        **model_kwargs,
    ) -> torch.LongTensor:
        logger.info("Starting speculative decoding with distributed inference session!")
        
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        finished = False
        
        # Initialize past_key_values for session tracking
        past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(session.position)
        
        is_first_iteration = True
        step_idx = 0
        current_input_ids = input_ids
        
        
        while not finished and current_input_ids.shape[1] < input_ids.shape[1] + max_new_tokens:
            logger.info(f"\n==================== STEP {step_idx} ====================")
            logger.info(f"[DEBUG] Current sequence length: {current_input_ids.shape[1]}")
            logger.info(f"[DEBUG] Session position: {session.position}")
            
            # 1. Build speculative trees using SSM
            spec_trees = self._build_speculative_trees_batched(
                current_input_ids, ssm, beam_width, max_tree_depth
            )
            
            # 2. Verify trees using distributed inference - but through forward() call
            verified_tokens, verified_tokens_positions, past_key_values = self._verify_trees_with_forward(
                input_ids=current_input_ids,
                trees=spec_trees,
                logits_processor=logits_processor,
                past_key_values=past_key_values,
                is_first_iteration=is_first_iteration,
                use_kv_cache=use_kv_cache,
                kv_cache_window=kv_cache_window
            )
            
            past_key_values.set_kv_cache(verified_tokens_positions)
            
            is_first_iteration = False
            
            # 3. Apply stopping conditions
            if has_eos_stopping_criteria:
                verified_tokens = verified_tokens * unfinished_sequences + generation_config.pad_token_id * (
                    1 - unfinished_sequences
                )

            # 4. Update input sequence
            current_input_ids = torch.cat([current_input_ids, verified_tokens], dim=-1)
            logger.info(f"[DEBUG] Verified tokens appended: {verified_tokens.tolist()}")
            logger.info(f"[DEBUG] New sequence length: {current_input_ids.shape[1]}")

            if streamer is not None:
                streamer.put(verified_tokens.cpu())

            # 5. Check if finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(current_input_ids, None)
            finished = unfinished_sequences.max() == 0
            step_idx += 1

        if streamer is not None:
            streamer.end()

        return current_input_ids
    
    def _verify_trees_with_forward(
        self,
        input_ids: torch.LongTensor,
        trees: List[SpeculativeTree],
        logits_processor: LogitsProcessorList,
        past_key_values: RemotePastKeyValues,
        is_first_iteration: bool,
        use_kv_cache: bool,
        kv_cache_window: int,
    ) -> Tuple[torch.LongTensor, RemotePastKeyValues]:
        """
        Verify speculative trees using standard forward() call within the active session context
        """
        
        tree_tokens, attention_mask, batch_node_paths = prepare_incremental_tree_batch(
            trees, input_ids, input_ids.device
        )
        
        logger.info(f"[DEBUG] Tree tokens shape: {tree_tokens.shape}")
        logger.info(f"[DEBUG] Tree tokens: {tree_tokens}")
        logger.info(f"[DEBUG] Active session position: {self.transformer.h.active_session.position if self.transformer.h.active_session else 'None'}")
        
        if attention_mask is None or tree_tokens.shape[1] == 0:
            logger.warning("No tree tokens to verify, falling back to regular generation")
            return self._fallback_generation_with_forward(input_ids, logits_processor, past_key_values), past_key_values
        
        logger.info(f"attention_mask: {attention_mask}")
        tree_mask_packed = self.pack_bool_mask_to_int64(attention_mask)
        logger.info(f"tree_mask_packed: {tree_mask_packed}")
        
        with torch.no_grad():
            if not use_kv_cache:
                # No cache: process tree tokens directly
                logger.warning("Processing without KV cache, may cause error!!!")
                outputs = self(
                    input_ids=tree_tokens,
                    attention_mask=tree_mask_packed,
                    past_key_values=None,
                    use_cache=False
                )
                logits = outputs.logits
                new_past_key_values = past_key_values
                
            elif is_first_iteration or past_key_values is None:
                # First iteration: process full sequence to establish cache
                full_sequence = torch.cat([input_ids, tree_tokens], dim=-1)
                logger.info(f"[DEBUG] First iteration - processing full sequence of length: {full_sequence.shape[1]}")
                
                outputs = self(
                    input_ids=full_sequence,
                    attention_mask=tree_mask_packed,  # Let the session handle attention
                    past_key_values=None,  # Start fresh
                    use_cache=True
                )
                
                # Extract only the tree portion of the logits
                logits = outputs.logits[:, input_ids.shape[1]:, :]
                
                # Update past_key_values tracking
                if past_key_values is None:
                    new_past_key_values = RemotePastKeyValues()
                else:
                    new_past_key_values = past_key_values
                
                # The session will automatically handle the KV cache positioning
                if self.transformer.h.active_session:
                    new_past_key_values.update_seen(self.transformer.h.active_session.position)
                
                logger.info(f"[DEBUG] First iteration completed, session position: {self.transformer.h.active_session.position if self.transformer.h.active_session else 'None'}")
                
            else:
                # Subsequent iterations: use existing cache
                active_session = self.transformer.h.active_session
                if active_session is None:
                    raise ValueError("No active session available for cached inference")
                
                # Handle cache window management
                if active_session.position > kv_cache_window:
                    trim_amount = active_session.position - kv_cache_window
                    active_session.position = kv_cache_window
                    logger.info(f"Trimmed cache: reset position from {active_session.position + trim_amount} to {kv_cache_window}")
                
                logger.info(f"[DEBUG] Subsequent iteration - processing tree tokens of length: {tree_tokens.shape[1]}")
                
                # Process tree tokens with existing cache
                outputs = self(
                    input_ids=tree_tokens,
                    attention_mask=tree_mask_packed,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs.logits
                new_past_key_values = past_key_values
                new_past_key_values.update_seen(active_session.position)
                
                logger.info(f"[DEBUG] Subsequent iteration completed, session position: {active_session.position}")
        
        # Extract verification results
        verified_tokens, verified_tokens_positions = self._extract_best_verified_paths_fixed(
            logits, batch_node_paths, input_ids, logits_processor
        )
        logger.info(f"[DEBUG] Verified tokens (per batch): {verified_tokens.tolist()}")
        return verified_tokens, verified_tokens_positions, new_past_key_values
    
    def pack_bool_mask_to_int64(self, mask_bool: torch.Tensor) -> torch.Tensor:
        packed = np.packbits(mask_bool.cpu().numpy().astype(np.uint8), axis=-1)
        pad = (-packed.shape[-1]) % 8
        if pad:
            packed = np.pad(packed, [(0,0)]*(packed.ndim-1)+[(0,pad)], constant_values=0)
        packed = packed.reshape(*packed.shape[:-1], -1, 8)
        return torch.from_numpy(packed).to(mask_bool.device)
    
    def _fallback_generation_with_forward(
        self, 
        input_ids: torch.LongTensor, 
        logits_processor: LogitsProcessorList,
        past_key_values: RemotePastKeyValues,
        temperature: float = 1.0
    ) -> torch.LongTensor:
        """
        Fallback to regular generation using forward() call within active session
        """
        try:
            logger.info("[DEBUG] Using fallback generation")
            
            # Generate single token using standard forward call
            outputs = self(
                input_ids=input_ids[:, -1:],  # Just the last token
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]  # Last position logits
            
            # Apply logits processors
            processed_logits = logits
            for processor in logits_processor:
                processed_logits = processor(input_ids, processed_logits)
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(processed_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
            
            logger.info(f"[DEBUG] Fallback generated token: {next_token.tolist()}")
            return next_token
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            # Ultimate fallback - return EOS token
            eos_token_id = getattr(self.config, 'eos_token_id', 2)
            return torch.tensor([[eos_token_id]], device=input_ids.device)
    
    # Keep your existing methods with minimal changes
    def _build_speculative_trees_batched(
        self, 
        input_ids: torch.LongTensor, 
        ssm: LlamaForCausalLM, 
        beam_width: int, 
        max_depth: int
    ) -> List[SpeculativeTree]:
        """Build speculative trees using the small model (SSM)"""
        batch_size = input_ids.shape[0]
        trees = []
        logger.info(f"Building trees for batch_size: {batch_size}")
        
        for batch_idx in range(batch_size):
            root_token = input_ids[batch_idx, -1].item()
            tree = SpeculativeTree(root_token, f"req_{batch_idx}")
            logger.info(f"[DEBUG] (batch {batch_idx}) root token: {root_token}")
            
            for depth in range(max_depth):
                current_nodes = tree.get_nodes_at_depth(depth)
                if not current_nodes:
                    break
                
                # Build contexts for current nodes
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
                
                # Batch process contexts with SSM
                max_len = max(len(ctx) for ctx in contexts)
                padded_contexts = []
                attention_masks = []
                
                for ctx in contexts:
                    pad_len = max_len - len(ctx)
                    pad_token_id = getattr(ssm.config, 'pad_token_id', 0)
                    
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
                
                # Process with SSM (small model)
                with torch.no_grad():
                    outputs = ssm(batch_contexts, attention_mask=batch_masks)
                    batch_logits = outputs.logits[:, -1, :]
                
                # Generate candidates
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
                
                logger.info(f"[DEBUG] (batch {batch_idx}) depth {depth} SSM candidates: {candidates_per_node}")
                
                try:
                    new_nodes = tree.add_layer(current_nodes, candidates_per_node)
                    if not new_nodes:
                        break
                except ValueError as e:
                    logger.warning(f"Failed to add tree layer: {e}")
                    break
            
            logger.info(f"[DEBUG] batch {batch_idx} finished tree structure")
            trees.append(tree)
        
        return trees
    
    def _extract_best_verified_paths_fixed(
        self,
        logits: torch.Tensor,
        batch_node_paths: List[List[List[TreeNode]]],
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract best verified paths with proper error handling
        Returns: (verified_tokens, verified_tokens_positions)
        """
        batch_size = logits.shape[0]
        batch_verified = []
        batch_positions = []
        
        for batch_idx in range(batch_size):
            node_paths = batch_node_paths[batch_idx]
            best_verified = []
            best_positions = []
            best_score = -1
            
            for node_path in node_paths:
                verified_tokens = []
                verified_positions = []
                
                for node in node_path:
                    pos = node.position_in_sequence
                    if pos >= logits.shape[1]:
                        break
                    
                    predicted_token = torch.argmax(logits[batch_idx, pos]).item()
                    
                    if predicted_token == node.token_id:
                        verified_tokens.append(node.token_id)
                        # 计算在整个序列中的绝对位置
                        # 根节点位置 = input_ids.shape[1] - 1 (输入序列的最后一个位置)
                        # 树中节点的绝对位置 = 根节点位置 + 节点在树中的深度 + 1
                        root_position = input_ids.shape[1] - 1
                        absolute_position = root_position + node.depth + 1
                        verified_positions.append(absolute_position)
                    else:
                        break
                
                if len(verified_tokens) > best_score:
                    best_score = len(verified_tokens)
                    best_verified = verified_tokens
                    best_positions = verified_positions
            
            batch_verified.append(best_verified)
            batch_positions.append(best_positions)
        
        # Handle empty verification case
        best_len = max((len(v) for v in batch_verified), default=0)
        if best_len == 0:
            logger.warning("No tokens verified, generating single token using logits processor")
            # Generate a single token using the last position logits
            final_logits = logits[:, -1, :]
            processed_logits = final_logits
            for processor in logits_processor:
                processed_logits = processor(input_ids, processed_logits)
            next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
            
            # 生成单个token的位置
            next_token_positions = []
            for batch_idx in range(batch_size):
                # 下一个token的位置应该是当前输入序列长度
                next_pos = input_ids.shape[1]
                next_token_positions.append([next_pos])
            
            next_positions_tensor = torch.tensor(next_token_positions, device=logits.device)
            return next_token, next_positions_tensor
        
        # Pad verified tokens to same length
        padded_verified = []
        padded_positions = []
        for verified, positions in zip(batch_verified, batch_positions):
            padded_v = verified + [0] * (best_len - len(verified))
            padded_p = positions + [0] * (best_len - len(positions))  # 用0填充位置
            padded_verified.append(padded_v)
            padded_positions.append(padded_p)
        
        verified_tensor = torch.tensor(padded_verified, device=logits.device)
        positions_tensor = torch.tensor(padded_positions, device=logits.device)
        
        # Generate additional token using logits processor
        if verified_tensor.shape[1] > 0:
            # Use the position after last verified token
            final_positions = []
            next_token_positions = []
            
            for batch_idx, (verified, positions) in enumerate(zip(batch_verified, batch_positions)):
                if len(verified) < logits.shape[1]:
                    pos = len(verified)
                else:
                    pos = logits.shape[1] - 1
                final_positions.append(pos)
                
                # 计算下一个token的绝对位置
                if len(positions) > 0:
                    # 基于最后一个验证token的位置
                    next_pos = positions[-1] + 1
                else:
                    # 如果没有验证的token，则从当前输入序列的末尾开始
                    next_pos = input_ids.shape[1]
                next_token_positions.append(next_pos)
            
            final_logits = torch.stack([
                logits[i, pos] for i, pos in enumerate(final_positions)
            ])
            
            # Apply logits processor
            processor_input = torch.cat([input_ids, verified_tensor], dim=1)
            processed_logits = final_logits
            for processor in logits_processor:
                processed_logits = processor(processor_input, processed_logits)
            
            next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
            next_positions = torch.tensor([[pos] for pos in next_token_positions], device=logits.device)
            
            # 合并verified tokens和新生成的token
            verified_tensor = torch.cat([verified_tensor, next_token], dim=1)
            positions_tensor = torch.cat([positions_tensor, next_positions], dim=1)
        
        return verified_tensor, positions_tensor