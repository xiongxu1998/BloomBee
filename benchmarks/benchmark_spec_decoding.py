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
from bloombee import (
    AutoDistributedConfig,
    AutoDistributedSpeculativeModel,
    DistributedLlamaForSpeculativeGeneration,
    RemoteSequential,
)
from bloombee.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = get_logger()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model")
    parser.add_argument("--ssm", type=str, required=True, help="Model")
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
    
    ssm = AutoModelForCausalLM.from_pretrained(args.ssm)
    
    model = AutoDistributedSpeculativeModel.from_pretrained(
        args.model, initial_peers=args.initial_peers, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    
    
    logger.info(f"Created model: {process_idx=} {model.device=}")

    result = ""
    step_times = []
    validation_stats = {"total": 0, "passed": 0}
    
    test_prompt = "Hello world from Xu, I am a master student."
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt", add_special_tokens=False)
    # Ensure we have enough tokens, repeat if not enough
    while input_ids.shape[1] < args.prompt_len:
        input_ids = torch.cat([input_ids, input_ids], dim=1)
    # Truncate to specified length
    input_ids = input_ids[:, :args.prompt_len]
    logger.info(f"Using initial prompt with {args.prompt_len} tokens: {input_ids.shape}")
    
    result = model.generate(input_ids=input_ids, ssm=ssm)

    final_speed = 1 / np.mean(step_times) if step_times else 0.0
    result_pipe.send({
        "speed": final_speed,
        "validation": validation_stats
    })


if __name__ == "__main__":
    main()