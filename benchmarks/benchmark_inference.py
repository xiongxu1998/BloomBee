#!/usr/bin/env python3

import argparse
import multiprocessing as mp
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from hivemind.utils.logging import get_logger
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

logger = get_logger()


def validate_single_proposal(model, tokenizer, input_ids, topk_tokens, topk_probs, step, process_idx):
    """验证单次 proposal 生成的正确性"""
    try:
        with torch.no_grad():
            # 手动计算ground truth
            outputs = model.transformer(input_ids)
            last_hidden = outputs.last_hidden_state[:, -1, :]
            
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(last_hidden)
            else:
                lm_head = model.get_output_embeddings()
                logits = lm_head(last_hidden)
            
            probs_true = F.softmax(logits, dim=-1)
            topk_probs_true, topk_tokens_true = torch.topk(probs_true, k=topk_tokens.shape[1], dim=-1)
            
            # 验证
            tokens_match = torch.equal(topk_tokens, topk_tokens_true)
            probs_match = torch.allclose(topk_probs, topk_probs_true, atol=1e-6)
            probs_decreasing = torch.all(topk_probs[0, :-1] >= topk_probs[0, 1:])
            probs_in_range = torch.all((topk_probs >= 0) & (topk_probs <= 1))
            
            all_valid = tokens_match and probs_match and probs_decreasing and probs_in_range
            
            if all_valid:
                logger.debug(f"✅ P{process_idx} S{step}: Proposal validation passed")
            else:
                logger.warning(f"❌ P{process_idx} S{step}: Proposal validation failed!")
                logger.warning(f"   tokens_match={tokens_match}, probs_match={probs_match}")
                logger.warning(f"   probs_decreasing={probs_decreasing}, probs_in_range={probs_in_range}")
            
            return all_valid
            
    except Exception as e:
        logger.error(f"❌ P{process_idx} S{step}: Validation error: {e}")
        return False


def detailed_proposal_validation(model, tokenizer, input_ids, topk_tokens, topk_probs, step, process_idx):
    """详细的 proposal 验证（包含完整信息显示）"""
    logger.info(f"\n--- Detailed Proposal Validation: P{process_idx} S{step} ---")
    
    # 显示输入信息
    input_text = tokenizer.decode(input_ids[0].tolist())
    logger.info(f"Input text: '{input_text}'")
    logger.info(f"Input tokens: {input_ids[0].tolist()}")
    
    # 显示proposals
    logger.info(f"Generated Proposals (top-{topk_tokens.shape[1]}):")
    for i, (token_id, prob) in enumerate(zip(topk_tokens[0], topk_probs[0])):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        logger.info(f"  {i+1}. Token {token_id:5d} → '{decoded:20s}' | Prob: {prob:.6f}")
    
    # 验证正确性
    is_valid = validate_single_proposal(model, tokenizer, input_ids, topk_tokens, topk_probs, step, process_idx)
    
    # 额外统计信息
    prob_sum = topk_probs[0].sum().item()
    max_prob = topk_probs[0].max().item()
    min_prob = topk_probs[0].min().item()
    
    logger.info(f"Statistics:")
    logger.info(f"  Prob sum: {prob_sum:.6f}")
    logger.info(f"  Max prob: {max_prob:.6f}")
    logger.info(f"  Min prob: {min_prob:.6f}")
    logger.info(f"  Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    return is_valid


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--initial_peers", type=str, nargs="+", default=PUBLIC_INITIAL_PEERS, help="Initial peers")
    parser.add_argument("--torch_dtype", type=str, default="float32", help="Torch dtype")
    parser.add_argument("--n_processes", type=str, default=1, help="Number of concurrent processes")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=1, help="Number of warmup steps")
    
    # Proposal validation 特定参数
    parser.add_argument("--top_k", type=int, default=5, help="Number of proposals to generate")
    parser.add_argument("--validate_every", type=int, default=10, help="Run detailed validation every N steps")
    parser.add_argument("--validate_all", action="store_true", help="Validate every step (slower)")
    parser.add_argument("--no_validation", action="store_true", help="Skip all validation (faster)")
    
    # Speculative decoding 相关参数
    parser.add_argument("--varied_input", action="store_true", help="Use varied input prompts instead of just BOS token")
    parser.add_argument("--input_prompts", type=str, nargs="+", 
                       default=["<s>", "The capital of France is", "Hello world", "In machine learning"],
                       help="List of input prompts to cycle through")
    
    parser.add_argument("--prompt_len", type=int, default=2, help="Length of initial prompt")
    
    args = parser.parse_args()

    if args.n_processes == "n_gpus":
        args.n_processes = torch.cuda.device_count()
    else:
        args.n_processes = int(args.n_processes)

    pipe_recv, pipe_send = mp.Pipe(duplex=False)
    processes = [mp.Process(target=benchmark_inference, args=(i, args, pipe_send)) for i in range(args.n_processes)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()

    # 收集结果
    results = [pipe_recv.recv() for _ in range(args.n_processes)]
    speeds = [r['speed'] for r in results]
    validation_stats = [r['validation'] for r in results]
    
    # 汇总报告
    avg_speed = np.mean(speeds)
    total_validations = sum(v['total'] for v in validation_stats)
    passed_validations = sum(v['passed'] for v in validation_stats)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Average Speed: {avg_speed:.2f} proposals/sec")
    
    if total_validations > 0:
        success_rate = 100 * passed_validations / total_validations
        logger.info(f"Proposal Validation: {passed_validations}/{total_validations} passed ({success_rate:.1f}%)")
        
        if success_rate == 100.0:
            logger.info(f"🎉 ALL PROPOSALS VALIDATED SUCCESSFULLY! 🎉")
        else:
            logger.warning(f"⚠️  Some proposals failed validation. Check logs above.")
    else:
        logger.info(f"No validation performed (--no_validation was used)")


@torch.inference_mode()
def benchmark_inference(process_idx, args, result_pipe):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    logger.info(f"Created model: {process_idx=} {model.device=}")

    result = ""
    step_times = []
    validation_stats = {"total": 0, "passed": 0}
    
    
    if args.prompt_len > 1:
        # Create an initial prompt containing multiple tokens
        test_prompt = "Hello world from Xu, I am a master student."
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=False)
        # Ensure we have enough tokens, repeat if not enough
        while input_ids.shape[1] < args.prompt_len:
            input_ids = torch.cat([input_ids, input_ids], dim=1)
        # Truncate to specified length
        input_ids = input_ids[:, :args.prompt_len]
        logger.info(f"Using initial prompt with {args.prompt_len} tokens: {input_ids.shape}")
        
        print("Final input_ids:", input_ids)
        print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))
        print("Decoded prompt:", tokenizer.decode(input_ids[0]))
        
        # First process the initial multi-token input
        with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
            start_time = perf_counter()
            # 🔴 Key: Use multi-token input for the first inference
            # print("Initial -token input processed successfully!", args.prompt_len)
            p_result, topk_tokens, topk_scores = model.generate_topk_proposals(input_ids, top_k=args.top_k, session=sess)
            if topk_tokens.dim() == 2:
                top1_token_id = topk_tokens[0][0].item()
            else:
                top1_token_id = topk_tokens[0].item()
            top1_decoded = tokenizer.decode([top1_token_id])
            result += top1_decoded
            
            # step_times = [perf_counter() - start_time]
            # print("Initial -token input processed successfully!", args.prompt_len)
            
            # Continue generating remaining tokens
            num_initial_tokens = input_ids.shape[1]
            max_new_tokens = args.seq_len - num_initial_tokens
            for step in range(1, max_new_tokens):
                start_time = perf_counter()
                # input_ids = topk_tokens[:, 0:1]
                p_result, topk_tokens, topk_scores = model.generate_topk_proposals(max_new_tokens=1, top_k=args.top_k, session=sess)
                # top1_token_id = topk_tokens[0][0].item()
                if topk_tokens.dim() == 2:
                    top1_token_id = topk_tokens[0][0].item()
                else:
                    top1_token_id = topk_tokens[0].item()
                top1_decoded = tokenizer.decode([top1_token_id])
                result += top1_decoded
                print("p_result: ", p_result)
                logger.info(
                    f"test: P{process_idx} S{step}: "
                    f"topk={topk_tokens[0].tolist()[:5]} | "
                    f"scores={topk_scores[0].tolist()[:5]} | "
                    f"top1='{top1_decoded}' | "
                    f"partial_result='{result[-30:]}'"
                )

                
                if step >= args.warmup_steps:
                    step_times.append(perf_counter() - start_time)
                    speed = 1 / np.mean(step_times)
                    logger.info(f"{process_idx=} {step=} {speed=:.2f}")
    else:
        # Original single-token logic
        result = ""
        step_times = []
        with model.transformer.h.inference_session(max_length=args.seq_len) as sess:
            for step in range(args.seq_len):
                start_time = perf_counter()

                topk_tokens, topk_scores = model.generate_topk_proposals(input_ids, top_k=args.top_k)
                decoded = tokenizer.decode(topk_tokens[0].tolist())
                result += decoded

                if step >= args.warmup_steps:
                    step_times.append(perf_counter() - start_time)
                    speed = 1 / np.mean(step_times)
                    logger.info(f"{process_idx=} {step=} {speed=:.2f}")

    final_speed = 1 / np.mean(step_times) if step_times else 0.0
    result_pipe.send({
        "speed": final_speed,
        "validation": validation_stats
    })


if __name__ == "__main__":
    main()
