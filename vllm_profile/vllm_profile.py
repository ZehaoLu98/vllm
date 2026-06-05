# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import os

from vllm import LLM, SamplingParams
from prompt_loader import load_prompts, get_default_prompts_path
from util import parse_human_readable_int, parse_args
from vllm.config import KVTransferConfig

enable_builtin_profiling = False

# Load prompts from external file
# You can override the prompts file path by setting VLLM_PROMPTS_FILE environment variable
prompts_file = os.environ.get('VLLM_PROMPTS_FILE', os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_generator", "under_8192_tokens.txt"))
prompts = load_prompts(prompts_file)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Convert human-readable format to integers
    max_num_batched_tokens_int = parse_human_readable_int(args.max_num_batched_tokens)

    # Print all argument settings
    print("=" * 80)
    print("SchedulerConfig Parameters:")
    print("=" * 80)
    print(f"  max_num_batched_tokens: {args.max_num_batched_tokens} -> {max_num_batched_tokens_int}")
    print(f"  max_num_seqs: {args.max_num_seqs}")
    print(f"  max_num_partial_prefills: {args.max_num_partial_prefills}")
    print(f"  max_long_partial_prefills: {args.max_long_partial_prefills}")
    print(f"  long_prefill_token_threshold: {args.long_prefill_token_threshold}")
    print(f"  scheduling_policy: {args.scheduling_policy}")
    print(f"  enable_chunked_prefill: {args.enable_chunked_prefill}")
    print(f"  disable_chunked_mm_input: {args.disable_chunked_mm_input}")
    print(f"  scheduler_cls: {args.scheduler_cls}")
    print(f"  disable_hybrid_kv_cache_manager: {args.disable_hybrid_kv_cache_manager}")
    print(f"  async_scheduling: {args.async_scheduling}")
    print(f"  stream_interval: {args.stream_interval}")
    print("=" * 80)
    print()

    if enable_builtin_profiling:
        profiler_config = dict(
            profiler="torch",
            torch_profiler_dir="./vllm_profile",
            torch_profiler_with_flops=True,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
        )
    else:
        profiler_config = None

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 64 * 1024 ** 3,  # 64 GiB of pinned CPU RAM for KV cache
            "single_gpu_tensor": True,  # Only 1 layer's KV on GPU at a time
        },
    )

    # Create an LLM with SchedulerConfig parameters
    llm = LLM(
        model="openai/gpt-oss-120b",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        profiler_config=profiler_config,
        enable_layerwise_nvtx_tracing=True,
        kv_cache_metrics=True,
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        # SchedulerConfig parameters
        max_num_batched_tokens=max_num_batched_tokens_int,
        max_num_partial_prefills=args.max_num_partial_prefills,
        max_long_partial_prefills=args.max_long_partial_prefills,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        scheduling_policy=args.scheduling_policy,
        enable_chunked_prefill=args.enable_chunked_prefill,
        disable_chunked_mm_input=args.disable_chunked_mm_input,
        scheduler_cls=args.scheduler_cls,
        disable_hybrid_kv_cache_manager=args.disable_hybrid_kv_cache_manager,
        async_scheduling=args.async_scheduling,
        stream_interval=args.stream_interval,
        disable_log_stats=False,
        # kv_transfer_config=ktc,
        max_model_len=8192,
    )

    if enable_builtin_profiling:
        llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    if enable_builtin_profiling:
        llm.stop_profile()

    # Calculate metrics
    ttfts = []
    tpots = []

    for output in outputs:
        ttft = output.metrics.first_token_latency
        ttfts.append(ttft)
        decode_time = output.metrics.last_token_ts - output.metrics.first_token_ts
        tpot = decode_time / (output.metrics.num_generation_tokens - 1)
        tpots.append(tpot)

    # Print the outputs.
    # print("-" * 50)
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    #     print("-" * 50)

    # Print performance metrics
    print("\n" + "=" * 80)
    print("Performance Metrics:")
    print("=" * 80)

    if ttfts:
        print(f"\nTime To First Token (TTFT):")
        print(f"  Average: {sum(ttfts) / len(ttfts):.4f} seconds")
        print(f"  Min:     {min(ttfts):.4f} seconds")
        print(f"  Max:     {max(ttfts):.4f} seconds")
    else:
        print(f"\nTime To First Token (TTFT): No data available")

    if tpots:
        print(f"\nTime Per Output Token (TPOT):")
        print(f"  Average: {sum(tpots) / len(tpots):.4f} seconds")
        print(f"  Min:     {min(tpots):.4f} seconds")
        print(f"  Max:     {max(tpots):.4f} seconds")
    else:
        print(f"\nTime Per Output Token (TPOT): No data available")

    print("=" * 80)
    print()

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()
