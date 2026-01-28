# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams

enable_builtin_profiling = False

# Sample prompts.
prompts = [
    "Hello, my name is",
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
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        tensor_parallel_size=1,
        profiler_config=profiler_config,
        enable_layerwise_nvtx_tracing=True,
        kv_cache_metrics=True,
        enforce_eager=True,
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
