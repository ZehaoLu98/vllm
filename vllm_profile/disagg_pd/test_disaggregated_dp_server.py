#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark client for disaggregated prefill-decode proxy server.

This script loads prompts from various sources and sends them to a proxy server
at a configurable rate with different distribution options.

Usage Examples:
    # Basic usage with default settings (10 prompts, Poisson distribution at 1 req/s)
    python test_disaggregated_dp_server.py --proxy-url http://localhost:8192

    # Load prompts from ShareGPT dataset
    python test_disaggregated_dp_server.py \
        --proxy-url http://localhost:8192 \
        --dataset sharegpt \
        --dataset-path /path/to/sharegpt.json \
        --num-prompts 100 \
        --request-rate 5.0

    # Use Poisson distribution (exponential inter-arrival times)
    python test_disaggregated_dp_server.py \
        --proxy-url http://localhost:8192 \
        --num-prompts 50 \
        --request-rate 10.0 \
        --distribution poisson

    # Use uniform distribution (constant inter-arrival times)
    python test_disaggregated_dp_server.py \
        --proxy-url http://localhost:8192 \
        --num-prompts 50 \
        --request-rate 10.0 \
        --distribution uniform

    # Use gamma distribution with custom burstiness
    python test_disaggregated_dp_server.py \
        --proxy-url http://localhost:8192 \
        --num-prompts 50 \
        --request-rate 10.0 \
        --distribution gamma \
        --burstiness 0.5  # More bursty (< 1)

    # Linear ramp-up from 1 to 20 req/s
    python test_disaggregated_dp_server.py \
        --proxy-url http://localhost:8192 \
        --num-prompts 100 \
        --distribution ramp-linear \
        --ramp-start-rps 1 \
        --ramp-end-rps 20

    # Exponential ramp-up
    python test_disaggregated_dp_server.py \
        --proxy-url http://localhost:8192 \
        --num-prompts 100 \
        --distribution ramp-exponential \
        --ramp-start-rps 1 \
        --ramp-end-rps 50

    # Burst all at once (infinite request rate)
    python test_disaggregated_dp_server.py \
        --proxy-url http://localhost:8192 \
        --num-prompts 20 \
        --request-rate inf
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SampleRequest:
    """A single request sample."""
    prompt: str
    prompt_len: int = 0
    expected_output_len: int = 128
    request_id: str = ""


@dataclass
class RequestFuncOutput:
    """Output from a single request."""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # Inter-token latencies
    tpot: float = 0.0  # Time per output token (avg)
    prompt_len: int = 0
    error: str = ""


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    completed: int = 0
    failed: int = 0
    total_duration: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    # Decoder-specific metrics
    decoder_output_throughput: float = 0.0  # tokens/s during decode phase only
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    # TTFT (includes prefill + KV transfer + proxy overhead)
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0


# =============================================================================
# Prompt Loaders
# =============================================================================

def load_default_prompts(num_prompts: int, seed: int = 42) -> list[SampleRequest]:
    """Load default prompts when no dataset is specified."""
    random.seed(seed)
    
    default_prompts = [
        "Once upon a time in a land far away, there lived a",
        "The future of artificial intelligence is expected to",
        "In the year 2050, scientists discovered that",
        "The recipe for a perfect chocolate cake starts with",
        "Dear diary, today was an extraordinary day because",
        "The history of computing began in the early 1800s when",
        "A comprehensive guide to machine learning would start by explaining",
        "The most important scientific discovery of the 21st century was",
        "When planning a trip to Paris, you should first consider",
        "The process of photosynthesis is fascinating because it",
        "According to recent studies, the benefits of daily exercise include",
        "The development of the internet has transformed society by",
        "In a groundbreaking research paper, scientists revealed that",
        "The art of programming involves understanding concepts such as",
        "Climate change poses significant challenges to humanity because",
        "The exploration of Mars has revealed interesting facts about",
        "A detailed analysis of quantum computing shows that",
        "The principles of good software engineering include",
        "Throughout human history, civilizations have risen and fallen due to",
        "The relationship between nutrition and health is complex because",
    ]
    
    samples = []
    for i in range(num_prompts):
        prompt = default_prompts[i % len(default_prompts)]
        samples.append(SampleRequest(
            prompt=prompt,
            prompt_len=len(prompt.split()),  # Approximate token count
            expected_output_len=128,
            request_id=f"req-{i:06d}",
        ))
    
    return samples


def load_sharegpt_prompts(
    dataset_path: str,
    num_prompts: int,
    seed: int = 42,
    output_len: int | None = None,
) -> list[SampleRequest]:
    """Load prompts from ShareGPT dataset."""
    random.seed(seed)
    
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter entries with at least two conversation turns
    data = [
        entry for entry in data
        if "conversations" in entry and len(entry["conversations"]) >= 2
    ]
    
    random.shuffle(data)
    
    samples = []
    for i, entry in enumerate(data):
        if len(samples) >= num_prompts:
            break
        
        prompt = entry["conversations"][0]["value"]
        completion = entry["conversations"][1]["value"]
        
        samples.append(SampleRequest(
            prompt=prompt,
            prompt_len=len(prompt.split()),  # Approximate
            expected_output_len=output_len or len(completion.split()),
            request_id=f"req-{i:06d}",
        ))
    
    return samples


def load_sonnet_prompts(
    dataset_path: str,
    num_prompts: int,
    input_len: int = 550,
    output_len: int = 150,
    prefix_len: int = 200,
    seed: int = 42,
) -> list[SampleRequest]:
    """Load prompts from sonnet dataset."""
    random.seed(seed)
    
    with open(dataset_path, encoding="utf-8") as f:
        lines = f.readlines()
    
    # Calculate average tokens per line
    avg_tokens_per_line = sum(len(line.split()) for line in lines) / len(lines)
    
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    num_input_lines = max(1, int(input_len / avg_tokens_per_line))
    num_prefix_lines = max(0, int(prefix_len / avg_tokens_per_line))
    
    prefix_lines = lines[:num_prefix_lines]
    
    samples = []
    for i in range(num_prompts):
        extra_lines = random.choices(lines, k=num_input_lines - num_prefix_lines)
        prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
        
        samples.append(SampleRequest(
            prompt=prompt,
            prompt_len=len(prompt.split()),
            expected_output_len=output_len,
            request_id=f"req-{i:06d}",
        ))
    
    return samples


def load_custom_prompts(
    prompts_file: str,
    num_prompts: int,
    output_len: int = 128,
) -> list[SampleRequest]:
    """Load prompts from a custom file (one prompt per line or JSON array)."""
    with open(prompts_file, encoding="utf-8") as f:
        content = f.read()
    
    try:
        # Try to parse as JSON array
        prompts = json.loads(content)
        if isinstance(prompts, list):
            if isinstance(prompts[0], str):
                pass  # List of strings
            elif isinstance(prompts[0], dict):
                prompts = [p.get("prompt", p.get("text", str(p))) for p in prompts]
    except json.JSONDecodeError:
        # Treat as one prompt per line
        prompts = [line.strip() for line in content.split("\n") if line.strip()]
    
    samples = []
    for i in range(min(num_prompts, len(prompts))):
        prompt = prompts[i % len(prompts)]
        samples.append(SampleRequest(
            prompt=prompt,
            prompt_len=len(prompt.split()),
            expected_output_len=output_len,
            request_id=f"req-{i:06d}",
        ))
    
    # Repeat if needed
    while len(samples) < num_prompts:
        idx = len(samples)
        prompt = prompts[idx % len(prompts)]
        samples.append(SampleRequest(
            prompt=prompt,
            prompt_len=len(prompt.split()),
            expected_output_len=output_len,
            request_id=f"req-{idx:06d}",
        ))
    
    return samples


def load_prompts(args: argparse.Namespace) -> list[SampleRequest]:
    """Load prompts based on the specified dataset."""
    if args.dataset == "default":
        return load_default_prompts(args.num_prompts, args.seed)
    elif args.dataset == "sharegpt":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for ShareGPT dataset")
        return load_sharegpt_prompts(
            args.dataset_path, args.num_prompts, args.seed, args.output_len
        )
    elif args.dataset == "sonnet":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for sonnet dataset")
        return load_sonnet_prompts(
            args.dataset_path, args.num_prompts,
            args.input_len, args.output_len, args.prefix_len, args.seed
        )
    elif args.dataset == "custom":
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for custom dataset")
        return load_custom_prompts(
            args.dataset_path, args.num_prompts, args.output_len
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


# =============================================================================
# Request Rate Generation
# =============================================================================

def _get_current_request_rate(
    ramp_strategy: Literal["linear", "exponential"] | None,
    ramp_start_rps: float | None,
    ramp_end_rps: float | None,
    request_index: int,
    total_requests: int,
    base_request_rate: float,
) -> float:
    """Calculate request rate for ramp-up strategies."""
    if ramp_strategy and ramp_start_rps is not None and ramp_end_rps is not None:
        progress = request_index / max(total_requests - 1, 1)
        if ramp_strategy == "linear":
            increase = (ramp_end_rps - ramp_start_rps) * progress
            return ramp_start_rps + increase
        elif ramp_strategy == "exponential":
            ratio = ramp_end_rps / ramp_start_rps
            return ramp_start_rps * (ratio ** progress)
    return base_request_rate


async def generate_requests(
    input_requests: list[SampleRequest],
    request_rate: float,
    distribution: str,
    burstiness: float = 1.0,
    ramp_start_rps: float | None = None,
    ramp_end_rps: float | None = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    """
    Generate requests at the specified rate with the given distribution.
    
    Args:
        input_requests: List of requests to send
        request_rate: Target requests per second (use float('inf') for instant burst)
        distribution: One of 'poisson', 'uniform', 'gamma', 'ramp-linear', 'ramp-exponential'
        burstiness: For gamma distribution (1.0 = Poisson, <1 = more bursty, >1 = more uniform)
        ramp_start_rps: Starting RPS for ramp-up strategies
        ramp_end_rps: Ending RPS for ramp-up strategies
    
    Yields:
        Tuple of (request, current_request_rate)
    """
    total_requests = len(input_requests)
    
    # Determine ramp-up strategy
    ramp_strategy = None
    if distribution == "ramp-linear":
        ramp_strategy = "linear"
    elif distribution == "ramp-exponential":
        ramp_strategy = "exponential"
    
    # Precompute delays
    delay_times = []
    request_rates = []
    
    for i in range(total_requests):
        current_rate = _get_current_request_rate(
            ramp_strategy, ramp_start_rps, ramp_end_rps,
            i, total_requests, request_rate
        )
        request_rates.append(current_rate)
        
        if current_rate == float("inf"):
            delay_times.append(0.0)
        elif distribution == "uniform" or burstiness == float("inf"):
            # Constant inter-arrival time
            delay_times.append(1.0 / current_rate)
        elif distribution == "poisson" or burstiness == 1.0:
            # Exponential distribution (Poisson process)
            delay_times.append(np.random.exponential(1.0 / current_rate))
        else:
            # Gamma distribution
            theta = 1.0 / (current_rate * burstiness)
            delay_times.append(np.random.gamma(shape=burstiness, scale=theta))
    
    # Convert to cumulative delays
    cumulative_delays = [0.0]
    for delay in delay_times[:-1]:  # Don't need delay after last request
        cumulative_delays.append(cumulative_delays[-1] + delay)
    
    # Normalize if using ramp-up (ensure total time matches expectations)
    if ramp_strategy is None and cumulative_delays[-1] > 0 and request_rate != float("inf"):
        target_total = total_requests / request_rate
        factor = target_total / cumulative_delays[-1]
        cumulative_delays = [d * factor for d in cumulative_delays]
    
    # Send requests at the scheduled times
    start_time = time.time()
    for i, request in enumerate(input_requests):
        target_time = start_time + cumulative_delays[i]
        current_time = time.time()
        
        if target_time > current_time:
            await asyncio.sleep(target_time - current_time)
        
        yield request, request_rates[i]


# =============================================================================
# Request Sending Functions
# =============================================================================

async def send_request_openai_completions(
    request: SampleRequest,
    api_url: str,
    model: str,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """Send a request to OpenAI-compatible completions endpoint."""
    output = RequestFuncOutput()
    output.prompt_len = request.prompt_len
    
    payload = {
        "model": model,
        "prompt": request.prompt,
        "max_tokens": request.expected_output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    
    headers = {
        "Content-Type": "application/json",
    }
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    if request.request_id:
        headers["X-Request-Id"] = request.request_id
    
    generated_text = ""
    start_time = time.perf_counter()
    most_recent_time = start_time
    first_token_received = False
    
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        
                        # Each read may contain multiple SSE lines
                        for line in chunk_bytes.decode("utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            line = line.removeprefix("data: ").removeprefix("data:")
                            if line == "[DONE]":
                                continue
                            
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            
                            if choices := data.get("choices"):
                                text = choices[0].get("text", "")
                                timestamp = time.perf_counter()
                                
                                if not first_token_received:
                                    first_token_received = True
                                    output.ttft = timestamp - start_time
                                else:
                                    output.itl.append(timestamp - most_recent_time)
                                
                                most_recent_time = timestamp
                                generated_text += text
                            
                            if usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens", 0)
                    
                    output.latency = most_recent_time - start_time
                    output.generated_text = generated_text
                    output.success = first_token_received
                    
                    # Fix output token count: use streaming token
                    # count if usage field wasn't available
                    if not output.output_tokens and first_token_received:
                        # len(itl) intervals + 1 first token
                        output.output_tokens = len(output.itl) + 1
                    
                    # Compute TPOT (decode phase only)
                    if first_token_received and output.output_tokens > 1:
                        decode_duration = output.latency - output.ttft
                        output.tpot = decode_duration / (output.output_tokens - 1)
                    
                    if not first_token_received:
                        output.error = "No valid tokens received"
                else:
                    error_text = await response.text()
                    output.error = f"HTTP {response.status}: {error_text[:200]}"
                    output.success = False
    
    except asyncio.TimeoutError:
        output.error = "Request timed out"
        output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False
    
    if pbar:
        pbar.update(1)
    
    return output


async def send_request_openai_chat(
    request: SampleRequest,
    api_url: str,
    model: str,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """Send a request to OpenAI-compatible chat completions endpoint."""
    output = RequestFuncOutput()
    output.prompt_len = request.prompt_len
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.expected_output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    
    headers = {
        "Content-Type": "application/json",
    }
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    if request.request_id:
        headers["X-Request-Id"] = request.request_id
    
    generated_text = ""
    start_time = time.perf_counter()
    most_recent_time = start_time
    first_token_received = False
    
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        
                        # Each read may contain multiple SSE lines
                        for line in chunk_bytes.decode("utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            line = line.removeprefix("data: ").removeprefix("data:")
                            if line == "[DONE]":
                                continue
                            
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            
                            if choices := data.get("choices"):
                                delta = choices[0].get("delta", {})
                                text = delta.get("content", "")
                                timestamp = time.perf_counter()
                                
                                if text and not first_token_received:
                                    first_token_received = True
                                    output.ttft = timestamp - start_time
                                elif text:
                                    output.itl.append(timestamp - most_recent_time)
                                
                                if text:
                                    most_recent_time = timestamp
                                    generated_text += text
                            
                            if usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens", 0)
                    
                    output.latency = most_recent_time - start_time
                    output.generated_text = generated_text
                    output.success = first_token_received
                    
                    # Fix output token count: use streaming token
                    # count if usage field wasn't available
                    if not output.output_tokens and first_token_received:
                        # len(itl) intervals + 1 first token
                        output.output_tokens = len(output.itl) + 1
                    
                    # Compute TPOT (decode phase only)
                    if first_token_received and output.output_tokens > 1:
                        decode_duration = output.latency - output.ttft
                        output.tpot = decode_duration / (output.output_tokens - 1)
                    
                    if not first_token_received:
                        output.error = "No valid tokens received"
                else:
                    error_text = await response.text()
                    output.error = f"HTTP {response.status}: {error_text[:200]}"
                    output.success = False
    
    except asyncio.TimeoutError:
        output.error = "Request timed out"
        output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False
    
    if pbar:
        pbar.update(1)
    
    return output


# =============================================================================
# Benchmark Runner
# =============================================================================

async def run_benchmark(
    requests: list[SampleRequest],
    args: argparse.Namespace,
) -> tuple[list[RequestFuncOutput], float]:
    """Run the benchmark with the given requests."""
    
    # Determine API URL
    base_url = args.proxy_url.rstrip("/")
    if args.endpoint == "completions":
        api_url = f"{base_url}/v1/completions"
        request_func = send_request_openai_completions
    else:  # chat
        api_url = f"{base_url}/v1/chat/completions"
        request_func = send_request_openai_chat
    
    print(f"Sending {len(requests)} requests to {api_url}")
    print(f"Distribution: {args.distribution}")
    print(f"Request rate: {args.request_rate} req/s")
    if args.distribution in ["ramp-linear", "ramp-exponential"]:
        print(f"Ramp: {args.ramp_start_rps} -> {args.ramp_end_rps} req/s")
    if args.distribution == "gamma":
        print(f"Burstiness: {args.burstiness}")
    print()
    
    # Generate and send requests
    tasks = []
    pbar = tqdm(total=len(requests), desc="Sending requests")
    
    benchmark_start = time.perf_counter()
    
    async for request, current_rate in generate_requests(
        requests,
        args.request_rate,
        args.distribution,
        args.burstiness,
        args.ramp_start_rps,
        args.ramp_end_rps,
    ):
        task = asyncio.create_task(
            request_func(request, api_url, args.model, pbar)
        )
        tasks.append(task)
    
    # Wait for all requests to complete
    outputs = await asyncio.gather(*tasks)
    
    benchmark_end = time.perf_counter()
    total_duration = benchmark_end - benchmark_start
    
    pbar.close()
    
    return outputs, total_duration


def calculate_results(
    outputs: list[RequestFuncOutput],
    total_duration: float,
) -> BenchmarkResults:
    """Calculate aggregated benchmark results."""
    results = BenchmarkResults()
    
    ttfts = []
    latencies = []
    all_itls = []
    tpots = []
    decode_durations = []
    decode_output_tokens = []
    
    for output in outputs:
        if output.success:
            results.completed += 1
            results.total_input_tokens += output.prompt_len
            results.total_output_tokens += output.output_tokens
            ttfts.append(output.ttft)
            latencies.append(output.latency)
            all_itls.extend(output.itl)
            
            # Decoder-specific: decode phase = latency - ttft
            if output.output_tokens > 1:
                tpots.append(output.tpot)
                decode_dur = output.latency - output.ttft
                decode_durations.append(decode_dur)
                decode_output_tokens.append(output.output_tokens - 1)  # first token is part of TTFT
        else:
            results.failed += 1
    
    results.total_duration = total_duration
    
    if results.completed > 0:
        results.request_throughput = results.completed / total_duration
        results.output_throughput = results.total_output_tokens / total_duration
        
        # Decoder output throughput: tokens generated during decode phase
        # divided by total decode wall-clock time across all requests.
        # For concurrent requests, use the sum of per-request decode
        # durations to get the aggregate decode throughput.
        if decode_durations:
            total_decode_tokens = sum(decode_output_tokens)
            # Per-request average decoder throughput
            per_req_throughputs = [
                t / d for t, d in zip(decode_output_tokens, decode_durations) if d > 0
            ]
            results.decoder_output_throughput = float(np.mean(per_req_throughputs)) if per_req_throughputs else 0.0
        
        # TPOT stats (in ms)
        if tpots:
            results.mean_tpot_ms = np.mean(tpots) * 1000
            results.median_tpot_ms = np.median(tpots) * 1000
            results.p99_tpot_ms = np.percentile(tpots, 99) * 1000
        
        # TTFT stats (in ms)
        results.mean_ttft_ms = np.mean(ttfts) * 1000
        results.median_ttft_ms = np.median(ttfts) * 1000
        results.p99_ttft_ms = np.percentile(ttfts, 99) * 1000
        
        # Latency stats (in ms)
        results.mean_latency_ms = np.mean(latencies) * 1000
        results.median_latency_ms = np.median(latencies) * 1000
        results.p99_latency_ms = np.percentile(latencies, 99) * 1000
        
        # ITL stats (in ms)
        if all_itls:
            results.mean_itl_ms = np.mean(all_itls) * 1000
            results.median_itl_ms = np.median(all_itls) * 1000
            results.p99_itl_ms = np.percentile(all_itls, 99) * 1000
    
    return results


def print_results(results: BenchmarkResults, args: argparse.Namespace) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Proxy URL:        {args.proxy_url}")
    print(f"  Model:            {args.model}")
    print(f"  Num Prompts:      {args.num_prompts}")
    print(f"  Distribution:     {args.distribution}")
    print(f"  Request Rate:     {args.request_rate} req/s")
    
    print(f"\nRequest Statistics:")
    print(f"  Completed:        {results.completed}")
    print(f"  Failed:           {results.failed}")
    print(f"  Success Rate:     {results.completed / (results.completed + results.failed) * 100:.1f}%")
    
    print(f"\nThroughput:")
    print(f"  Total Duration:   {results.total_duration:.2f}s")
    print(f"  Request Rate:     {results.request_throughput:.2f} req/s")
    print(f"  Output Rate:      {results.output_throughput:.2f} tokens/s (end-to-end)")
    print(f"  Input Tokens:     {results.total_input_tokens}")
    print(f"  Output Tokens:    {results.total_output_tokens}")
    
    if results.completed > 0:
        print(f"\nDecoder Metrics:")
        print(f"  Decoder Throughput: {results.decoder_output_throughput:.2f} tokens/s (decode phase only)")
        if results.mean_tpot_ms > 0:
            print(f"  Mean TPOT:        {results.mean_tpot_ms:.2f}ms")
            print(f"  Median TPOT:      {results.median_tpot_ms:.2f}ms")
            print(f"  P99 TPOT:         {results.p99_tpot_ms:.2f}ms")
        
        print(f"\nTime to First Token (prefill + KV transfer + proxy):")
        print(f"  Mean TTFT:        {results.mean_ttft_ms:.2f}ms")
        print(f"  Median TTFT:      {results.median_ttft_ms:.2f}ms")
        print(f"  P99 TTFT:         {results.p99_ttft_ms:.2f}ms")
        
        print(f"\nEnd-to-End Latency:")
        print(f"  Mean Latency:     {results.mean_latency_ms:.2f}ms")
        print(f"  Median Latency:   {results.median_latency_ms:.2f}ms")
        print(f"  P99 Latency:      {results.p99_latency_ms:.2f}ms")
        
        if results.mean_itl_ms > 0:
            print(f"\nInter-Token Latency:")
            print(f"  Mean ITL:         {results.mean_itl_ms:.2f}ms")
            print(f"  Median ITL:       {results.median_itl_ms:.2f}ms")
            print(f"  P99 ITL:          {results.p99_itl_ms:.2f}ms")
    
    print("\n" + "=" * 60)


def save_results(
    results: BenchmarkResults,
    outputs: list[RequestFuncOutput],
    args: argparse.Namespace,
) -> None:
    """Save results to a JSON file."""
    if not args.output_file:
        return
    
    data = {
        "config": {
            "proxy_url": args.proxy_url,
            "model": args.model,
            "num_prompts": args.num_prompts,
            "distribution": args.distribution,
            "request_rate": args.request_rate,
            "burstiness": args.burstiness,
        },
        "results": {
            "completed": results.completed,
            "failed": results.failed,
            "total_duration": results.total_duration,
            "total_input_tokens": results.total_input_tokens,
            "total_output_tokens": results.total_output_tokens,
            "request_throughput": results.request_throughput,
            "output_throughput": results.output_throughput,
            "decoder_output_throughput": results.decoder_output_throughput,
            "mean_tpot_ms": results.mean_tpot_ms,
            "median_tpot_ms": results.median_tpot_ms,
            "p99_tpot_ms": results.p99_tpot_ms,
            "mean_ttft_ms": results.mean_ttft_ms,
            "median_ttft_ms": results.median_ttft_ms,
            "p99_ttft_ms": results.p99_ttft_ms,
            "mean_latency_ms": results.mean_latency_ms,
            "median_latency_ms": results.median_latency_ms,
            "p99_latency_ms": results.p99_latency_ms,
            "mean_itl_ms": results.mean_itl_ms,
            "median_itl_ms": results.median_itl_ms,
            "p99_itl_ms": results.p99_itl_ms,
        },
        "individual_results": [
            {
                "success": o.success,
                "latency": o.latency,
                "ttft": o.ttft,
                "tpot": o.tpot,
                "decode_duration": o.latency - o.ttft if o.success else None,
                "output_tokens": o.output_tokens,
                "generated_text": o.generated_text,
                "error": o.error if not o.success else None,
            }
            for o in outputs
        ],
    }
    
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")


def dump_generated_text(
    requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dump_path: str,
) -> None:
    """Dump prompts and generated text to a file."""
    if dump_path.endswith(".json"):
        data = []
        for req, out in zip(requests, outputs):
            data.append({
                "request_id": req.request_id,
                "prompt": req.prompt,
                "generated_text": out.generated_text if out.success else None,
                "success": out.success,
                "error": out.error if not out.success else None,
            })
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(dump_path, "w", encoding="utf-8") as f:
            for i, (req, out) in enumerate(zip(requests, outputs)):
                f.write(f"{'='*60}\n")
                f.write(f"Request {i} [{req.request_id}]\n")
                f.write(f"{'='*60}\n")
                f.write(f"PROMPT:\n{req.prompt}\n\n")
                if out.success:
                    f.write(f"GENERATED TEXT:\n{out.generated_text}\n")
                else:
                    f.write(f"FAILED: {out.error}\n")
                f.write(f"\n")
    
    print(f"\nGenerated text dumped to: {dump_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark client for disaggregated prefill-decode proxy server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Server configuration
    parser.add_argument(
        "--proxy-url",
        type=str,
        default="http://localhost:8192",
        help="URL of the proxy server (default: http://localhost:8192)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name to use for requests",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["completions", "chat"],
        default="completions",
        help="API endpoint type (default: completions)",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["default", "sharegpt", "sonnet", "custom"],
        default="default",
        help="Dataset to load prompts from (default: default)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file (required for sharegpt, sonnet, custom)",
    )
    parser.add_argument(
        "--num-prompts", "-n",
        type=int,
        default=10,
        help="Total number of prompts to send (default: 10)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=550,
        help="Input length for sonnet dataset (default: 550)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Expected output length (default: 128)",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=200,
        help="Prefix length for sonnet dataset (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    # Request rate configuration
    parser.add_argument(
        "--request-rate", "-r",
        type=float,
        default=1.0,
        help="Request rate in requests/second (use 'inf' for instant burst, default: 1.0)",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["poisson", "uniform", "gamma", "ramp-linear", "ramp-exponential"],
        default="poisson",
        help="Request interval distribution (default: poisson)",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for gamma distribution "
             "(1.0=Poisson, <1=more bursty, >1=more uniform, default: 1.0)",
    )
    parser.add_argument(
        "--ramp-start-rps",
        type=float,
        default=1.0,
        help="Starting request rate for ramp-up (default: 1.0)",
    )
    parser.add_argument(
        "--ramp-end-rps",
        type=float,
        default=10.0,
        help="Ending request rate for ramp-up (default: 10.0)",
    )
    
    # Output configuration
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--dump-text",
        type=str,
        default=None,
        help="Dump prompts and generated text to this file (text or .json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Handle inf request rate
    if isinstance(args.request_rate, str) and args.request_rate.lower() == "inf":
        args.request_rate = float("inf")
    
    return args


async def main() -> int:
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load prompts
    print(f"Loading prompts from {args.dataset} dataset...")
    try:
        requests = load_prompts(args)
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return 1
    
    print(f"Loaded {len(requests)} prompts")
    
    # Run benchmark
    try:
        outputs, total_duration = await run_benchmark(requests, args)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Calculate and print results
    results = calculate_results(outputs, total_duration)
    print_results(results, args)
    
    # Save results if requested
    if args.output_file:
        save_results(results, outputs, args)
    
    # Dump generated text if requested
    if args.dump_text:
        dump_generated_text(requests, outputs, args.dump_text)
    
    # Print failed request errors if verbose
    if args.verbose and results.failed > 0:
        print("\nFailed request errors:")
        for i, output in enumerate(outputs):
            if not output.success:
                print(f"  Request {i}: {output.error[:100]}")
    
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
