import contextlib
import dataclasses
from contextvars import ContextVar
from typing import Any, ContextManager, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from hivemind.utils.logging import get_logger
from torch import Tensor
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import ModelOutput
from transformers import LogitsProcessor

from bloombee.client.inference_session import InferenceSession
from bloombee.client.remote_sequential import RemoteSequential
from bloombee.utils.misc import DUMMY, docstring_from

logger = get_logger(__name__)


class RemotePastKeyValues(Cache):
    """only keeps the number of seen tokens. pretends to be a legit cache"""

    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.hypo_ids: Optional[torch.LongTensor] = None

    def __getitem__(self, _index: int) -> List[torch.Tensor]:
        return [DUMMY]  # For compatibility with BloomForCausalLM.prepare_inputs_for_generation()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def update_seen(self, new_seen: int) -> None:
        self._seen_tokens += new_seen

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search reordering is not implemented yet")


_skipped_tokens = ContextVar("skipped_tokens", default=0)


class _SkipTokensMixin:
    # This override is used in RemoteGenerationMixin by has to be defined in a class not named as "GenerationMixin"
    # due to how transformers.PreTrainedModel.can_generate() works
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> dict:
        input_ids = input_ids[:, _skipped_tokens.get() :]
        _skipped_tokens.set(0)
        return super().prepare_inputs_for_generation(input_ids, **kwargs)
    
class SpeculativeTopKCollector(LogitsProcessor):
    def __init__(self, k):
        self.k = k
        self.topk_tokens = []
        self.topk_logprobs = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        log_probs = torch.log_softmax(scores, dim=-1)
        topk = torch.topk(log_probs, self.k, dim=-1)
        self.topk_tokens.append(topk.indices)      # shape: [batch_size, k]
        self.topk_logprobs.append(topk.values)     # shape: [batch_size, k]
        return scores  # 不修改 logits


class RemoteGenerationMixin(_SkipTokensMixin):
    """
    This class is an upgrade to `transformers.GenerationMixin` that:

    - Designed to be compatible with most `transformers.GenerationMixin` strategies and options
    - Supports generation inside a remote InferenceSession, so that remote servers store your attention caches and
      you don't have to rerun the prefix through all the servers to generate each new token
    - Supports multiple `.generate()` calls inside one InferenceSession, so you can easily run interactive generation
      by showing tokens on the fly (multiple calls like `.generate(None, max_new_tokens=1, ...)`) or
      accept prompts from a user in a chat bot (multiple calls like `.generate(new_prompts, ...)`).
    - If there is no active session, `.generate()` will create a new InferenceSession with proper `max_length`.
      Otherwise, `.generate()` will use the active session. You can use the `session=...` argument to override that.
    """

    @docstring_from(RemoteSequential.active_session)
    @property
    def active_session(self) -> Optional[InferenceSession]:
        return self.transformer.h.active_session

    @docstring_from(RemoteSequential.use_session)
    def use_session(self, session: Optional[InferenceSession]) -> ContextManager[InferenceSession]:
        return self.transformer.h.use_session(session)

    @docstring_from(RemoteSequential.inference_session)
    def inference_session(self, **kwargs) -> ContextManager[InferenceSession]:
        return self.transformer.h.inference_session(**kwargs)

    @docstring_from(transformers.GenerationMixin.generate.__doc__)
    def generate(
        self, inputs: Optional[torch.Tensor] = None, *args, session: Optional[InferenceSession] = None, **kwargs
    ):
        self._fix_generate_kwargs(kwargs)
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)

        if session is not None:
            # If a session specified explicitly, use it
            context_manager = self.use_session(session)
        elif self.active_session is not None:
            # If there's an active session, don't do anything
            context_manager = contextlib.nullcontext(self.active_session)
        else:
            # If there's no active session, create a new one

            max_length = kwargs.get("max_length")
            max_new_tokens = kwargs.get("max_new_tokens")
            assert (max_length is None) != (
                max_new_tokens is None
            ), "You should set `max_length` or `max_new_tokens` (but not both) to reserve server-side attention caches"

            session_max_length = self.transformer.config.pre_seq_len
            if max_length is not None:
                session_max_length += max_length
            else:
                session_max_length += (inputs.shape[1] if inputs is not None else 0) + max_new_tokens
            context_manager = self.inference_session(max_length=session_max_length)

        with context_manager as session:
            # Prepend the tokens from the previous .generate() call
            n_prev_tokens = session.output_ids.shape[1] if session.output_ids is not None else 0
            if n_prev_tokens > 0:
                if kwargs.get("num_beams", 1) > 1:
                    logger.warning(
                        "Beam search will not work properly in the resumed petals.InferenceSession "
                        "since intermediate beam entries are lost"
                    )

                if inputs is not None:
                    inputs = torch.cat([session.output_ids, inputs], dim=1)
                else:
                    inputs = session.output_ids

                # Don't actually run all previous tokens through the transformer,
                # but keep them for transformers.GenerationMixin (e.g., to compute repetition_penalty)
                _skipped_tokens.set(max(0, n_prev_tokens - 1))

            if self._supports_cache_class and "past_key_values" not in kwargs:
                past_key_values = RemotePastKeyValues()
                past_key_values.update_seen(session.position)
                kwargs["past_key_values"] = past_key_values

            result = super().generate(inputs, *args, **kwargs)

            sequences = result.sequences if isinstance(result, ModelOutput) else result
            # Save tokens from this .generate() call
            session.output_ids = sequences
            # Crop the last tokens from the previous call
            sequences = sequences[:, n_prev_tokens:].clone()
            if isinstance(result, ModelOutput):
                result.sequences = sequences
            else:
                result = sequences

        return result
    
    def generate_topk_proposals(
        self,
        inputs: Optional[torch.LongTensor] = None,
        *args,
        top_k: int = 4,
        session: Optional[InferenceSession] = None,
        **kwargs,
    ):
        """
        Generate top-k token proposals for the next position.
        
        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            top_k: Number of top tokens to return
            attention_mask: Optional attention mask
            position_ids: Optional position IDs  
            past_key_values: Optional past key values for caching
            session: Optional inference session (if None, will create one)
            
        Returns:
            tuple: (topk_token_ids, topk_probs) with shapes [batch_size, top_k]
        """
        
        # === Session管理逻辑 (参考generate方法) ===
        self._fix_generate_kwargs(kwargs)
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)

        if session is not None:
            # If a session specified explicitly, use it
            context_manager = self.use_session(session)
        elif self.active_session is not None:
            # If there's an active session, don't do anything
            context_manager = contextlib.nullcontext(self.active_session)
        else:
            # If there's no active session, create a new one

            max_length = kwargs.get("max_length")
            max_new_tokens = kwargs.get("max_new_tokens")
            assert (max_length is None) != (
                max_new_tokens is None
            ), "You should set `max_length` or `max_new_tokens` (but not both) to reserve server-side attention caches"

            session_max_length = self.transformer.config.pre_seq_len
            if max_length is not None:
                session_max_length += max_length
            else:
                session_max_length += (inputs.shape[1] if inputs is not None else 0) + max_new_tokens
            context_manager = self.inference_session(max_length=session_max_length)

        with context_manager as session:
            # Prepend the tokens from the previous .generate() call
            n_prev_tokens = session.output_ids.shape[1] if session.output_ids is not None else 0
            if n_prev_tokens > 0:
                if kwargs.get("num_beams", 1) > 1:
                    logger.warning(
                        "Beam search will not work properly in the resumed petals.InferenceSession "
                        "since intermediate beam entries are lost"
                    )

                if inputs is not None:
                    inputs = torch.cat([session.output_ids, inputs], dim=1)
                else:
                    inputs = session.output_ids

                # Don't actually run all previous tokens through the transformer,
                # but keep them for transformers.GenerationMixin (e.g., to compute repetition_penalty)
                _skipped_tokens.set(max(0, n_prev_tokens - 1))

            if self._supports_cache_class and "past_key_values" not in kwargs:
                past_key_values = RemotePastKeyValues()
                past_key_values.update_seen(session.position)
                kwargs["past_key_values"] = past_key_values
            
            collector = SpeculativeTopKCollector(top_k)
            if "logits_processor" in kwargs:
                kwargs["logits_processor"].append(collector)
            else:
                kwargs["logits_processor"] = [collector]

            result = super().generate(inputs, *args, **kwargs)
            sequences = result.sequences if isinstance(result, ModelOutput) else result
            # Save tokens from this .generate() call
            session.output_ids = sequences
            # Crop the last tokens from the previous call
            sequences = sequences[:, n_prev_tokens:].clone()
            if isinstance(result, ModelOutput):
                result.sequences = sequences
            else:
                result = sequences

            return result, collector.topk_tokens[-1], collector.topk_logprobs[-1].exp()

    @staticmethod
    def _fix_generate_kwargs(kwargs: dict):
        # Suppress inappropriate "Both max_new_tokens and max_length" HF warning
        if "max_length" in kwargs and kwargs["max_length"] is None:
            del kwargs["max_length"]

        # Support do_sample = {0, 1} for backward compatibility with Petals < 2.1.0
        do_sample = kwargs.get("do_sample")
        if isinstance(do_sample, int):
            kwargs["do_sample"] = bool(do_sample)

    @staticmethod
    def _reorder_cache(past_key_values: RemotePastKeyValues, beam_idx: torch.LongTensor) -> RemotePastKeyValues:
        return dataclasses.replace(past_key_values, hypo_ids=beam_idx)
