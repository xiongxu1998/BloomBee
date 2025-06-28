from typing import Optional, Union

import torch
from transformers.generation import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

from bloombee.models.llama.config import DistributedLlamaConfig
from bloombee.models.llama.model import DistributedLlamaForCausalLM



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
        return self._sample(
            input_ids=input_ids,
            ssm=ssm,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=False,
            streamer=streamer,
            logits_warper=None,  # 暂不支持 warper
            speculative_inference_iteration_size=speculative_inference_iteration_size,
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
        speculative_inference_iteration_size: int = 3,
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

            full_sequence = torch.cat([input_ids, speculative_tokens], dim=-1)
            assert input_ids.shape[1] + speculative_inference_iteration_size == full_sequence.shape[1]

            input_for_validation = full_sequence
            if not firsts:
                self.active_session.position = input_ids.shape[1] - 1
                input_for_validation = input_for_validation[:, -speculative_inference_iteration_size - 1 :]
            else:
                firsts = False
            input_for_validation = input_for_validation[:, :-1]
            with torch.no_grad():
                precise_model_outputs = self(input_for_validation)
            full_token_logits = precise_model_outputs.logits[:, -speculative_inference_iteration_size:, :].clone()

            all_valid_tokens = []
            first_token = None
            for i in range(speculative_inference_iteration_size):
                token_logits = full_token_logits[:, i, :]
                token_scores = logits_processor(
                    input_for_validation[:, : -speculative_inference_iteration_size + 1 + i], token_logits
                )
                valid_token = torch.argmax(token_scores, dim=-1)

                if first_token is None:
                    first_token = valid_token

                if valid_token.item() == speculative_tokens[:, i].item():
                    all_valid_tokens.append(valid_token.unsqueeze(-1))
                else:
                    break

            if not all_valid_tokens and first_token is not None:
                all_valid_tokens.append(first_token.unsqueeze(-1))
            all_valid_tokens = torch.cat(all_valid_tokens, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                all_valid_tokens = all_valid_tokens * unfinished_sequences + generation_config.pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, all_valid_tokens], dim=-1)

            if streamer is not None:
                streamer.put(all_valid_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            finished = unfinished_sequences.max() == 0

            del precise_model_outputs

        if streamer is not None:
            streamer.end()

        return input_ids
