import os
import time
from multiprocessing import Event, Process, Queue
import multiprocessing as mp
from queue import Empty

from vllm import LLM, LLMEngine, EngineArgs, SamplingParams
from vllm.config import KVTransferConfig
from prompt_loader import load_prompts, get_default_prompts_path

"""
Disaggregated Prefill-Decode Pipeline using LLMEngine.step()

This script demonstrates a disaggregated architecture where:
- Prefill GPU (GPU 0): Runs prefill and sends completed requests immediately after each step
- Decode GPU (GPU 1): Receives prefilled requests and runs full decode

Key improvements:
- Uses LLMEngine.step() instead of LLM.generate() for fine-grained control
- Prefill sends requests to decode immediately after they finish (not in batches)
- Decode can start processing requests as they arrive (streaming mode)
- Better resource utilization and lower latency
"""

# Load prompts from external file
# You can override the prompts file path by setting VLLM_PROMPTS_FILE environment variable
prompts_file = os.environ.get('VLLM_PROMPTS_FILE', get_default_prompts_path())
prompts = load_prompts(prompts_file)

PREFILL_BATCH_SIZE = 2  # Number of prompts to prefill at once
DECODE_BATCH_SIZE = 2   # Number of prompts to decode at once
SENTINEL = "DONE"  # Signal to indicate no more prompts

# Switch to control decode behavior:
# If True: decode waits for all prompts to be prefilled before starting
# If False: decode processes prompts as they arrive (streaming mode)
WAIT_FOR_ALL_PREFILLS = True


def run_prefill(prefill_queue, decode_ready):
    """Prefill process: uses LLMEngine.step() to send requests immediately after prefill."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    )

    # Use LLMEngine instead of LLM for step-by-step control
    engine_args = EngineArgs(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        kv_transfer_config=ktc,
        enable_prefix_caching=False,
        max_num_seqs=128,
        max_num_batched_tokens=1024,
        enforce_eager=True,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    # Wait for decode instance to be ready
    decode_ready.wait()
    print("[Prefill] Decode ready, starting prefill...")

    # Add all requests to the engine
    print(f"[Prefill] Adding {len(prompts)} requests to engine...")
    for idx, prompt in enumerate(prompts):
        request_id = f"prefill-{idx}"
        engine.add_request(request_id, prompt, sampling_params)
    print(f"[Prefill] All {len(prompts)} requests added")

    # Step through and send finished prefills immediately
    finished_count = 0
    step_count = 0
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        step_count += 1
        
        for output in step_outputs:
            if output.finished:
                # Request finished prefilling, send to decode immediately
                finished_count += 1
                prefill_queue.put(output.prompt)
                print(f"[Prefill] Step {step_count}: Request {output.request_id} finished prefill ({finished_count}/{len(prompts)})")
    
    # Signal that all prompts have been sent
    prefill_queue.put(SENTINEL)
    print(f"[Prefill] All {finished_count} requests completed in {step_count} steps, sent DONE signal")

    # Keep running until decode finishes
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Prefill] Stopped by user.")


def run_decode(prefill_queue, decode_ready, all_done):
    """Decode process: uses LLMEngine.step() to process requests incrementally."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=512)

    ktc = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    )

    # Use LLMEngine for step-by-step control
    engine_args = EngineArgs(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        kv_transfer_config=ktc,
        enable_prefix_caching=False,
        max_num_seqs=128,
        max_num_batched_tokens=1024,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    # Signal that decode instance is ready
    decode_ready.set()
    print("[Decode] Ready, waiting for prefilled prompts...")
    print(f"[Decode] WAIT_FOR_ALL_PREFILLS mode: {WAIT_FOR_ALL_PREFILLS}")

    all_outputs = []
    request_counter = 0
    done_receiving = False
    
    if WAIT_FOR_ALL_PREFILLS:
        # Mode 1: Wait for all prompts to be prefilled first
        print("[Decode] Waiting for all prefill batches to complete...")
        
        # Collect all prompts until SENTINEL
        while True:
            try:
                prompt = prefill_queue.get(timeout=0.1)
                if prompt == SENTINEL:
                    print(f"[Decode] Received all prompts ({request_counter} total), starting decode...")
                    break
                # Add request to engine as soon as we receive it
                request_id = f"decode-{request_counter}"
                engine.add_request(request_id, prompt, sampling_params)
                request_counter += 1
            except Empty:
                pass  # Keep waiting
        
        # Now step through and process all decode requests
        finished_count = 0
        step_count = 0
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            step_count += 1
            
            for output in step_outputs:
                if output.finished:
                    finished_count += 1
                    generated_text = output.outputs[0].text
                    print(f"[Decode] Step {step_count}: {output.request_id} finished ({finished_count}/{request_counter})")
                    print(f"[Decode]   Prompt: {output.prompt!r}")
                    print(f"[Decode]   Generated: {generated_text[:100]}...")
                    all_outputs.append(output)
    else:
        # Mode 2: Process prompts as they arrive (streaming mode with LLMEngine)
        print("[Decode] Streaming mode: processing prompts as they arrive...")
        step_count = 0
        finished_count = 0
        
        while not done_receiving or engine.has_unfinished_requests():
            # Try to add new requests from queue
            try:
                prompt = prefill_queue.get_nowait()
                if prompt == SENTINEL:
                    done_receiving = True
                    print(f"[Decode] Received SENTINEL, no more prompts coming ({request_counter} total received)")
                else:
                    request_id = f"decode-{request_counter}"
                    engine.add_request(request_id, prompt, sampling_params)
                    request_counter += 1
                    print(f"[Decode] Added {request_id} to engine")
            except Empty:
                pass  # No new prompts available right now
            
            # Step the engine if there are unfinished requests
            if engine.has_unfinished_requests():
                step_outputs = engine.step()
                step_count += 1
                
                for output in step_outputs:
                    if output.finished:
                        finished_count += 1
                        generated_text = output.outputs[0].text
                        print(f"[Decode] Step {step_count}: {output.request_id} finished ({finished_count}/{request_counter})")
                        print(f"[Decode]   Prompt: {output.prompt!r}")
                        print(f"[Decode]   Generated: {generated_text[:100]}...")
                        all_outputs.append(output)
            else:
                # No requests to process, wait a bit before checking queue again
                time.sleep(0.01)

    print(f"[Decode] Complete! Processed {len(all_outputs)} requests in {step_count} steps")
    all_done.set()


if __name__ == "__main__":
    # Queue for passing prefilled prompts to decode
    prefill_queue = Queue()
    
    # Events for coordination
    decode_ready = Event()  # Signals decode LLM is initialized
    all_done = Event()      # Signals all decoding is complete

    prefill_process = Process(target=run_prefill, args=(prefill_queue, decode_ready))
    decode_process = Process(target=run_decode, args=(prefill_queue, decode_ready, all_done))

    # Start both processes concurrently
    prefill_process.start()
    decode_process.start()

    # Wait for decode to finish all work
    decode_process.join()
    
    # Clean up prefill process
    prefill_process.terminate()
    prefill_process.join()
    
    print("All processes complete!")