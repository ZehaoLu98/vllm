# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility functions for vllm_profile."""

import argparse


def parse_human_readable_int(value):
    """Convert human-readable integer strings to actual integers.
    
    Supports:
    - '8k' or '8K' -> 8192 (8 * 1024)
    - '2m' or '2M' -> 2097152 (2 * 1024 * 1024)
    - '1024' -> 1024 (plain integers)
    
    Args:
        value: String or int value to convert
        
    Returns:
        Integer value or None if input is None
    """
    if value is None:
        return None
    
    if isinstance(value, int):
        return value
    
    value_str = str(value).strip()
    
    # Check for 'k' or 'K' suffix (multiply by 1024)
    if value_str.lower().endswith('k'):
        try:
            num = float(value_str[:-1])
            return int(num * 1024)
        except ValueError:
            raise ValueError(f"Invalid format: {value}")
    
    # Check for 'm' or 'M' suffix (multiply by 1024*1024)
    if value_str.lower().endswith('m'):
        try:
            num = float(value_str[:-1])
            return int(num * 1024 * 1024)
        except ValueError:
            raise ValueError(f"Invalid format: {value}")
    
    # Plain integer
    try:
        return int(value_str)
    except ValueError:
        raise ValueError(f"Invalid integer format: {value}")


def parse_args():
    """Parse command line arguments for SchedulerConfig parameters."""
    parser = argparse.ArgumentParser(description="vLLM profiling script with SchedulerConfig options")

    # SchedulerConfig parameters
    parser.add_argument(
        "--max_num_batched_tokens",
        type=str,
        default=None,
        help="Maximum number of tokens to be processed in a single iteration. Supports human-readable format like '1k', '2M'."
    )

    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="Maximum number of sequences to be processed in a single iteration."
    )

    parser.add_argument(
        "--max_num_partial_prefills",
        type=int,
        default=1,
        help="For chunked prefill, the maximum number of sequences that can be partially prefilled concurrently."
    )

    parser.add_argument(
        "--max_long_partial_prefills",
        type=int,
        default=1,
        help="For chunked prefill, the maximum number of prompts longer than long_prefill_token_threshold that will be prefilled concurrently."
    )

    parser.add_argument(
        "--long_prefill_token_threshold",
        type=int,
        default=0,
        help="For chunked prefill, a request is considered long if the prompt is longer than this number of tokens."
    )

    parser.add_argument(
        "--scheduling_policy",
        type=str,
        default="fcfs",
        choices=["fcfs", "priority"],
        help="The scheduling policy to use: 'fcfs' (first come first served) or 'priority' (based on given priority)."
    )

    parser.add_argument(
        "--enable_chunked_prefill",
        type=lambda x: None if x.lower() == 'none' else x.lower() == 'true',
        default=False,
        help="If True, prefill requests can be chunked based on the remaining max_num_batched_tokens. (true/false/none)"
    )

    parser.add_argument(
        "--disable_chunked_mm_input",
        action="store_true",
        default=False,
        help="If set, do not partially schedule multimodal items when chunked prefill is enabled."
    )

    parser.add_argument(
        "--scheduler_cls",
        type=str,
        default=None,
        help="The scheduler class to use. Can be a class directly or the path to a class of form 'mod.custom_class'."
    )

    parser.add_argument(
        "--disable_hybrid_kv_cache_manager",
        type=lambda x: None if x.lower() == 'none' else x.lower() == 'true',
        default=None,
        help="If True, KV cache manager will allocate the same size for all attention layers. (true/false/none)"
    )

    parser.add_argument(
        "--async_scheduling",
        type=lambda x: None if x.lower() == 'none' else x.lower() == 'true',
        default=None,
        help="If False, disable async scheduling. Async scheduling helps avoid gaps in GPU utilization. (true/false/none)"
    )

    parser.add_argument(
        "--stream_interval",
        type=int,
        default=1,
        help="The interval (or buffer size) for streaming in terms of token length."
    )

    return parser.parse_args()
