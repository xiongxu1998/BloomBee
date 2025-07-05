from typing import Optional, Union, List, Tuple
import torch
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

# 导入你的树结构
# from spe_de import SpeculativeTree, 
from spe_dec_tree import SpeculativeTree, TreeNode

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
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        A generate wrapper for speculative decoding with tree structure.
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
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        assert not generation_config.do_sample, "sample is not working for speculative generation now"
        assert not synced_gpus, "synced_gpus is not working for speculative generation now"
        assert (
            not generation_config.return_dict_in_generate
        ), "return_dict_in_generate is not working for speculative generation now"

        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        finished = False
        firsts = True

        while not finished:
            # 1. 构建推测树
            spec_tree = self._build_speculative_tree(
                input_ids, ssm, beam_width, max_tree_depth
            )
            
            # 2. 获取所有候选路径
            candidate_paths = spec_tree.get_all_paths()
            
            if not candidate_paths:
                # 如果没有候选路径，fallback到原始方法
                verified_tokens = self._fallback_generation(input_ids, ssm, 1)
            else:
                # 3. 验证候选路径
                verified_tokens = self._verify_candidate_paths(
                    input_ids, candidate_paths, logits_processor, firsts
                )
            
            if firsts:
                firsts = False
            
            # 4. 应用停止条件
            if has_eos_stopping_criteria:
                verified_tokens = verified_tokens * unfinished_sequences + generation_config.pad_token_id * (
                    1 - unfinished_sequences
                )

            # 5. 更新输入序列
            input_ids = torch.cat([input_ids, verified_tokens], dim=-1)

            if streamer is not None:
                streamer.put(verified_tokens.cpu())

            # 6. 检查是否完成
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        return input_ids
    
    def _build_speculative_tree(
        self, 
        input_ids: torch.LongTensor, 
        ssm: LlamaForCausalLM, 
        beam_width: int, 
        max_depth: int
    ) -> SpeculativeTree:
        """构建推测树"""
        # 创建树，根节点是当前序列的最后一个token
        root_token = input_ids[0, -1].item()  # 假设batch_size=1
        tree = SpeculativeTree(root_token, "req_001")
        
        # 逐层构建树
        for depth in range(max_depth):
            # 获取当前深度的所有节点
            current_nodes = tree.get_nodes_at_depth(depth)
            if not current_nodes:
                break
            
            # 为每个节点生成候选
            candidates_per_node = []
            for node in current_nodes:
                # 构建到当前节点的完整路径
                path_to_node = node.get_path_from_root()
                context = torch.cat([
                    input_ids[0, :-1],  # 原始序列（除了最后一个token）
                    torch.tensor([root_token] + path_to_node, device=input_ids.device)
                ]).unsqueeze(0)
                
                # 使用SSM生成候选
                candidates = self._generate_candidates_for_node(context, ssm, beam_width)
                candidates_per_node.append(candidates)
            
            # 扩展树
            try:
                new_nodes = tree.add_layer(current_nodes, candidates_per_node)
                if not new_nodes:
                    break
            except ValueError:
                break
        
        return tree
    
    def _generate_candidates_for_node(
        self, 
        context: torch.LongTensor, 
        ssm: LlamaForCausalLM, 
        beam_width: int
    ) -> List[Tuple[int, float]]:
        """为单个节点生成候选tokens"""
        with torch.no_grad():
            outputs = ssm(context)
            logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
            
            # 获取top-k候选
            top_k_values, top_k_indices = torch.topk(logits, k=beam_width)
            probs = torch.softmax(logits, dim=-1)
            
            candidates = []
            for i in range(beam_width):
                token_id = top_k_indices[i].item()
                prob = probs[token_id].item()
                candidates.append((token_id, prob))
        
        return candidates
    
    def _verify_candidate_paths(
        self, 
        input_ids: torch.LongTensor, 
        candidate_paths: List[List[int]], 
        logits_processor: LogitsProcessorList,
        is_first: bool
    ) -> torch.LongTensor:
        """验证候选路径并返回最佳验证结果"""
        
        best_verified_tokens = []
        best_score = -1
        
        for candidate_path in candidate_paths:
            if not candidate_path:
                continue
                
            # 构建完整序列用于验证
            full_sequence = torch.cat([
                input_ids, 
                torch.tensor([candidate_path], device=input_ids.device)
            ], dim=-1)
            
            # 准备验证输入
            input_for_validation = full_sequence
            if not is_first:
                self.active_session.position = input_ids.shape[1] - 1
                input_for_validation = input_for_validation[:, -(len(candidate_path) + 1):]
            
            input_for_validation = input_for_validation[:, :-1]
            
            # 使用LLM验证
            with torch.no_grad():
                precise_model_outputs = self(input_for_validation)
            
            # 验证每个token
            verified_tokens = self._verify_token_sequence(
                precise_model_outputs.logits, 
                candidate_path, 
                input_for_validation, 
                logits_processor
            )
            
            # 选择最长的验证序列
            if len(verified_tokens) > best_score:
                best_score = len(verified_tokens)
                best_verified_tokens = verified_tokens
            
            del precise_model_outputs
        
        if not best_verified_tokens:
            # 如果没有验证通过的tokens，fallback到单token生成
            return self._fallback_generation(input_ids, None, 1)
        
        return torch.tensor([best_verified_tokens], device=input_ids.device)
    
    def _verify_token_sequence(
        self, 
        logits: torch.Tensor, 
        candidate_tokens: List[int], 
        input_for_validation: torch.LongTensor,
        logits_processor: LogitsProcessorList
    ) -> List[int]:
        """验证token序列"""
        verified_tokens = []
        
        for i, candidate_token in enumerate(candidate_tokens):
            # 获取对应位置的logits
            if i >= logits.shape[1]:
                break
                
            token_logits = logits[:, -(len(candidate_tokens) - i), :].clone()
            
            # 应用logits处理器
            current_input = input_for_validation[:, :-(len(candidate_tokens) - i - 1)] if i < len(candidate_tokens) - 1 else input_for_validation
            token_scores = logits_processor(current_input, token_logits)
            
            # 获取最佳token
            valid_token = torch.argmax(token_scores, dim=-1)
            
            # 检查是否匹配
            if valid_token.item() == candidate_token:
                verified_tokens.append(candidate_token)
            else:
                break
        
        return verified_tokens
    
    def _fallback_generation(
        self, 
        input_ids: torch.LongTensor, 
        ssm: Optional[LlamaForCausalLM], 
        num_tokens: int
    ) -> torch.LongTensor:
        """当树验证失败时的fallback方法"""
        with torch.no_grad():
            outputs = self(input_ids)
            logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            return next_token.unsqueeze(0)