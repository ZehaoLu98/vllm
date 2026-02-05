# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams

enable_builtin_profiling = False

# Sample prompts.
prompts = [
    "What are the memory bandwidth requirements for different batch sizes?",
    "How does KV cache size scale with sequence length?",
    "What's the optimal tile size for GEMM operations in attention?",
    "How do you profile memory-bound versus compute-bound kernels?",
    "What causes memory bank conflicts in shared memory?",
    "How does tensor core utilization change with batch size?",
    "What's the difference between cuBLAS and cutlass for GEMM?",
    "How do you optimize warp-level reductions?",
    "What are the latency characteristics of different memory hierarchies?",
    "How does NVLink bandwidth compare to PCIe for multi-GPU?",
    "What's the impact of register spilling on kernel performance?",
    "How do you achieve memory coalescing in attention kernels?",
    "What are the trade-offs between flash attention variants?",
    "How does dynamic batching affect GPU occupancy?",
    "What's the overhead of CUDA kernel launches?",
    "How do you optimize MatMul for non-power-of-2 dimensions?",
    "What causes low achieved occupancy in compute-bound kernels?",
    "How does quantization affect memory bandwidth requirements?",
    "What's the performance impact of different CUDA stream strategies?",
    "How do you minimize PCIe transfers in inference pipelines?",
    "What are the characteristics of L2 cache persistence?",
    "How does prefetching improve transformer performance?",
    "What's the difference between sync and async CUDA operations?",
    "How do you profile device-to-device memory copies?",
    "What causes tail latency in batched inference?",
    "How does warp divergence impact attention kernels?",
    "What's the optimal block size for different GPU architectures?",
    "How do you balance compute and memory bandwidth utilization?",
    "What are the latency costs of different atomic operations?",
    "How does FP16 versus FP32 affect roofline analysis?",
    "What's the impact of kernel fusion on end-to-end latency?",
    "How do you optimize for different GPU generations?"

    "What's the performance difference between row-major and column-major layouts?",
    "How does batch size affect attention kernel occupancy?",
    "What are the trade-offs between online and offline softmax?",
    "How do you minimize global memory transactions in reductions?",
    "What's the impact of sequence length on memory bandwidth?",
    "How does cooperative groups improve kernel flexibility?",
    "What causes poor tensor core utilization in small batches?",
    "How do you profile multi-stream execution overlap?",
    "What's the overhead of cudaMemcpy versus unified memory?",
    "How does page-locked memory affect transfer performance?",
    "What are the latency characteristics of different synchronization primitives?",
    "How do you optimize strided memory access patterns?",
    "What's the difference between static and dynamic shared memory?",
    "How does warp scheduling affect instruction throughput?",
    "What causes high memory replay in Nsight Compute?",
    "How do you balance parallelism across SMs?",
    "What's the impact of different CUDA math library versions?",
    "How does kernel grid size affect launch overhead?",
    "What are the performance characteristics of half2 operations?",
    "How do you minimize bank conflicts in transpose operations?",
    "What's the difference between cuDNN and custom attention kernels?",
    "How does asynchronous copy improve pipeline efficiency?",
    "What causes serialization in atomic operations?",
    "How do you optimize for ampere versus hopper architectures?",
    "What's the impact of ECC on memory bandwidth?",
    "How does thread block clustering affect L2 cache hits?",
    "What are the latency costs of different reduction algorithms?",
    "How do you profile instruction pipeline stalls?",
    "What's the overhead of dynamic parallelism?",
    "How does MIG partitioning affect performance isolation?",
    "What causes poor scaling in multi-GPU inference?",
    "How do you optimize kernel parameters using occupancy calculator?"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)


def main():
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

    # Create an LLM.
    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        profiler_config=profiler_config,
        enable_layerwise_nvtx_tracing=True,
        kv_cache_metrics=True,
        enforce_eager=True,
        max_num_seqs=4,
    )

    if enable_builtin_profiling:
        llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    if enable_builtin_profiling:
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()
