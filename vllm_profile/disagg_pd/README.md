# Disaggregated Prefill-Decode (PD) Pipeline

This directory contains scripts and tools for running and benchmarking **disaggregated prefill-decode** inference with vLLM. In this architecture, the prefill and decode phases are split across separate GPU instances and coordinated through a proxy server, enabling independent scaling and better hardware utilization.

---

## Architecture Overview

```
                        ┌──────────────────────┐
                        │    Proxy Server       │
     Client ──────────► │  (toy_proxy_server)   │
     Request            │    Port 8192          │
                        └───────┬───────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
               ▼                                 ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │   Prefiller (GPU 0) │          │   Decoder (GPU 1)   │
    │   Port 8100         │  ─────►  │   Port 8200         │
    │   kv_role=kv_both   │ KV xfer  │   kv_role=kv_both   │
    │   NixlConnector     │ (NIXL)   │   NixlConnector     │
    └─────────────────────┘          └─────────────────────┘
```

### How it works

1. **Client** sends a completion/chat request to the **proxy server**.
2. **Proxy** forwards the request to the **prefiller** with `max_tokens=1` and `kv_transfer_params: {do_remote_decode: True}`.
3. **Prefiller** computes the KV cache for the full prompt, generates one token, and returns `kv_transfer_params` containing remote block IDs and connection info.
4. **Proxy** takes the `kv_transfer_params` from the prefill response and attaches them to the original request, then forwards it to the **decoder**.
5. **Decoder** uses the NIXL connector to load the pre-computed KV cache from the prefiller's GPU memory and generates the full completion, streaming tokens back through the proxy to the client.

### Why disaggregate?

| Property | Prefill | Decode |
|---|---|---|
| **Bottleneck** | Compute-bound (processes all input tokens in parallel) | Memory-bound (auto-regressive, one token at a time) |
| **GPU utilization** | High compute, bursty | Low compute, sustained memory bandwidth |
| **Scaling** | Scale up with more/faster GPUs | Scale out with more instances |

By separating these phases, each can be independently optimized and scaled for its workload characteristics.

---

## Files

| File | Description |
|---|---|
| `test_disaggregated_dp_server.py` | **Benchmark client** — sends prompts to the disagg proxy server at configurable rates with various distributions. Supports multiple prompt sources, SSE streaming, and result dumping. |
| `test_disaggregate_dp_decrepted.py` | **Deprecated** — earlier approach using `LLMEngine.step()` directly with `ExampleConnector` and multiprocessing queues. Has a known shape mismatch bug (`[2, 16, 256]` vs `[2, 256, 256]`). Kept for reference. |
| `result.json` | Sample output from a benchmark run (generated text dump). |

The **proxy server** lives in the main vLLM repo at:
```
tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
```

---

## Quick Start

### Prerequisites

- 2 GPUs (e.g., NVIDIA H100)
- vLLM installed with NIXL support (`pip install nixl` or built from source)
- A model (examples below use `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

### 1. Start the Prefiller (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8100 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### 2. Start the Decoder (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8200 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

### 3. Start the Proxy Server

```bash
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 \
  --host 0.0.0.0 \
  --prefiller-host localhost --prefiller-port 8100 \
  --decoder-host localhost --decoder-port 8200
```

### 4. Run the Benchmark Client

```bash
python test_disaggregated_dp_server.py \
  --proxy-url http://localhost:8192 \
  --num-prompts 20 \
  --request-rate 5.0 \
  --dump-text result.json
```

---

## Benchmark Client Usage

### Basic Options

```
--proxy-url URL          Proxy server URL (default: http://localhost:8192)
--model MODEL            Model name for API requests
--endpoint {completions,chat}   API endpoint type (default: completions)
--num-prompts N          Number of prompts to send (default: 10)
--request-rate R         Requests per second (use 'inf' for burst, default: 1.0)
--output-file FILE       Save benchmark metrics to JSON
--dump-text FILE         Dump prompts + generated text to file (.json or .txt)
--verbose                Print detailed error info
```

### Prompt Sources

| Source | Flag | Description |
|---|---|---|
| Built-in | `--dataset default` | 20 continuation-style prompts (default) |
| ShareGPT | `--dataset sharegpt --dataset-path FILE` | Multi-turn conversation dataset |
| Sonnet | `--dataset sonnet --dataset-path FILE` | Poem lines for prefix caching tests |
| Custom | `--dataset custom --dataset-path FILE` | One prompt per line or JSON array |

### Request Rate Distributions

| Distribution | Flag | Description |
|---|---|---|
| Poisson | `--distribution poisson` | Exponential inter-arrival times (default) |
| Uniform | `--distribution uniform` | Constant inter-arrival times |
| Gamma | `--distribution gamma --burstiness 0.5` | Tunable burstiness (`<1` = bursty, `>1` = uniform) |
| Linear Ramp | `--distribution ramp-linear --ramp-start-rps 1 --ramp-end-rps 20` | Linearly increasing request rate |
| Exponential Ramp | `--distribution ramp-exponential --ramp-start-rps 1 --ramp-end-rps 50` | Exponentially increasing request rate |

### Examples

```bash
# Burst 50 requests at once, save results
python test_disaggregated_dp_server.py \
  --proxy-url http://localhost:8192 \
  --num-prompts 50 \
  --request-rate inf \
  --output-file benchmark_results.json \
  --dump-text generated.json

# Ramp from 1 to 20 req/s over 100 requests
python test_disaggregated_dp_server.py \
  --proxy-url http://localhost:8192 \
  --num-prompts 100 \
  --distribution ramp-linear \
  --ramp-start-rps 1 \
  --ramp-end-rps 20

# Bursty traffic with gamma distribution
python test_disaggregated_dp_server.py \
  --proxy-url http://localhost:8192 \
  --num-prompts 50 \
  --request-rate 10.0 \
  --distribution gamma \
  --burstiness 0.3

# Use ShareGPT dataset
python test_disaggregated_dp_server.py \
  --proxy-url http://localhost:8192 \
  --dataset sharegpt \
  --dataset-path /path/to/ShareGPT_V3_unfiltered.json \
  --num-prompts 200 \
  --request-rate 5.0

# Use chat endpoint
python test_disaggregated_dp_server.py \
  --proxy-url http://localhost:8192 \
  --endpoint chat \
  --num-prompts 10 \
  --request-rate 2.0
```

---

## Benchmark Metrics Reference

The benchmark client measures latency at the **client side** (outside the proxy), so every metric includes network round-trip and proxy overhead.

### Timing Diagram

```
                         TTFT
          ├──────────────────────────────┤
          │  prefill + KV xfer + proxy   │
request ──┤                              ├── T1 ── T2 ── T3 ── ... ── Tn ──► done
          │                              │   ├─┤  ├─┤  ├─┤
          │                              │    ITL  ITL  ITL
          │                              │
          │◄──────────────────── end-to-end latency ──────────────────────►│
          │                              │◄──── decode duration ──────────►│
```

### Metrics

| Metric | What it measures | Formula | Unit |
|---|---|---|---|
| **TTFT** | Time to first token — covers prefill, KV transfer (RDMA), proxy hops, and network | `first_token_time - request_start` | ms |
| **TPOT** | Time per output token — average decode-phase latency per token | `(latency - TTFT) / (output_tokens - 1)` | ms |
| **ITL** | Inter-token latency — gap between consecutive streamed tokens | `token[i]_time - token[i-1]_time` | ms |
| **End-to-end latency** | Total request duration from send to last token | `last_token_time - request_start` | ms |
| **Request throughput** | Completed requests per second over the benchmark | `completed / total_duration` | req/s |
| **Output throughput** | Total output tokens per second (end-to-end) | `total_output_tokens / total_duration` | tok/s |
| **Decoder throughput** | Mean per-request decode-phase token rate | `mean(decode_tokens / decode_duration)` | tok/s |

All latency metrics are reported as **mean**, **median (P50)**, and **P99**.

### Example Output

```
============================================================
BENCHMARK RESULTS
============================================================

Request Statistics:
  Completed:        5
  Failed:           0
  Success Rate:     100.0%

Throughput:
  Total Duration:   0.97s
  Request Rate:     5.14 req/s
  Output Rate:      623.81 tokens/s (end-to-end)
  Decoder Throughput: 680.42 tokens/s (decode phase only)
  Input Tokens:     50
  Output Tokens:    607

Decoder Metrics:
  Mean TPOT:        1.47ms
  Median TPOT:      1.45ms
  P99 TPOT:         1.62ms

Time to First Token (prefill + KV transfer + proxy):
  Mean TTFT:        71.70ms
  Median TTFT:      52.15ms
  P99 TTFT:         156.41ms

End-to-End Latency:
  Mean Latency:     260.30ms
  Median Latency:   245.12ms
  P99 Latency:      310.50ms

Inter-Token Latency:
  Mean ITL:         1.55ms
  Median ITL:       1.50ms
  P99 ITL:          2.10ms
============================================================
```

---

## Output Formats

### `--dump-text result.json`

```json
[
  {
    "request_id": "req-000000",
    "prompt": "Once upon a time in a land far away, there lived a",
    "generated_text": " young girl named Lily...",
    "success": true,
    "error": null
  }
]
```

### `--dump-text result.txt`

```
============================================================
Request 0 [req-000000]
============================================================
PROMPT:
Once upon a time in a land far away, there lived a

GENERATED TEXT:
 young girl named Lily...
```

### `--output-file metrics.json`

Contains full benchmark config, aggregate metrics, and per-request results including `generated_text`, `latency`, `ttft`, and `output_tokens`.

---

## Known Issues & Notes

1. **Base models require continuation-style prompts.** Models like TinyLlama (base, not instruction-tuned) may generate EOS immediately for question-style prompts (e.g., "What is the capital of France?") when run through the disaggregated pipeline. Use continuation prompts instead (e.g., "Once upon a time...").

2. **ExampleConnector shape mismatch.** The deprecated `test_disaggregate_dp_decrepted.py` uses `ExampleConnector` which has a known bug: `RuntimeError: shape mismatch: value tensor of shape [2, 16, 256] cannot be broadcast to indexing result of shape [2, 256, 256]`. Use the NIXL-based server approach instead.

3. **NIXL side-channel ports.** Each vLLM instance needs a unique `VLLM_NIXL_SIDE_CHANNEL_PORT`. The prefiller and decoder must be able to communicate over these ports for KV cache transfer.

4. **SSE streaming.** The benchmark client handles Server-Sent Events (SSE) with multi-line chunk splitting. Raw TCP reads from `aiohttp` can return multiple `data:` lines in a single chunk.

---

## Multi-Instance Scaling

The proxy server supports multiple prefiller and decoder instances with round-robin load balancing:

```bash
# 2 prefillers + 2 decoders
python toy_proxy_server.py \
  --port 8192 \
  --prefiller-host localhost localhost \
  --prefiller-port 8100 8101 \
  --decoder-host localhost localhost \
  --decoder-port 8200 8201
```

Each instance needs its own GPU and unique side-channel port.

---

## FAQ

### Does the decoder re-run prefill (recompute the KV cache)?

**No.** The decoder never re-computes the KV cache. Here's the exact flow:

1. The request arrives at the decoder with `kv_transfer_params` containing `do_remote_prefill=True` and the prefiller's remote block IDs.
2. The scheduler calls `get_num_new_matched_tokens()` in the NixlConnector, which reports **all prompt tokens as externally computed** (`count = len(token_ids) - num_computed_tokens`).
3. Because `load_kv_async=True`, the scheduler sets `num_new_tokens = 0` — no forward pass is scheduled. The request enters `WAITING_FOR_REMOTE_KVS` state.
4. The NIXL connector transfers the KV cache from the prefiller's GPU memory to the decoder's GPU memory via RDMA in the background, between scheduler steps.
5. Once the transfer completes, the request re-enters scheduling with `num_computed_tokens == num_prompt_tokens`. The scheduler classifies it as a **decode** request (not prefill), since `num_computed_tokens >= num_prompt_tokens`.
6. The decoder's first model forward pass processes just **1 token** against the pre-loaded KV cache — a standard decode step.

| Step | What happens | Forward pass? |
|---|---|---|
| Request arrives with `kv_transfer_params` | Scheduler marks all prompt tokens as externally computed | No |
| `WAITING_FOR_REMOTE_KVS` | RDMA transfers KV cache: prefiller GPU → decoder GPU | No |
| First scheduled step | 1 token, classified as **decode** | Yes (1 token) |

This is the whole point of disaggregation: the decoder only ever does decode steps (memory-bandwidth-bound, 1 token at a time), never the compute-intensive prefill.

### Why does the prefill run with `max_tokens=1`?

The proxy sends the request to the prefiller with `max_tokens=1` to force it to process all prompt tokens (computing the full KV cache) but generate only a single output token. The prefiller's job is to compute the KV cache and make it available — not to generate the full completion. The single output token is discarded; only the `kv_transfer_params` (containing remote block IDs and connection info) are extracted and forwarded to the decoder.

### How is data transferred between the prefiller and decoder?

The prompt text and KV cache metadata travel over **HTTP** through the proxy, while the heavy KV cache tensors are transferred directly **GPU-to-GPU** via NIXL (RDMA). No KV data passes through the proxy or CPU.

```
  Hop                       What moves                   Protocol
  ─────────────────────────────────────────────────────────────────
  Client    → Proxy         Prompt text (JSON)            HTTP
  Proxy     → Prefiller     Prompt + {do_remote_decode}   HTTP
  Prefiller → Proxy         1 token + KV block metadata   HTTP
  Proxy     → Decoder       Prompt + KV block metadata    HTTP
  Prefiller → Decoder       KV cache tensors              NIXL/RDMA
                            (GPU mem → GPU mem)
```

The proxy orchestrates the two-phase flow inside `_handle_completions()`:

1. **Prefill request** (`send_request_to_service`) — the proxy adds `kv_transfer_params: {do_remote_decode: True}`, sets `max_tokens=1` and `stream=False`, and sends the prompt to the prefiller over HTTP. The prefiller computes the full KV cache and returns a response containing `kv_transfer_params` with `remote_block_ids`, `remote_engine_id`, `remote_host`, and `remote_port`.

2. **Decode request** (`stream_service_response`) — the proxy extracts `kv_transfer_params` from the prefill response, attaches it to the original request body, and streams it to the decoder. The decoder uses these params to pull the KV cache from the prefiller's GPU memory via NIXL (RDMA), then generates tokens which stream back through the proxy to the client.

The prompt text is re-sent to the decoder for tokenization and token-ID validation, but the decoder **does not recompute the KV cache** — it loads the pre-computed blocks directly from the prefiller's GPU.

### What is `kv_role=kv_both` and why do both instances use it?

`kv_role=kv_both` means the instance can both **save** (send) and **load** (receive) KV caches. The prefiller saves KV caches after computing them; the decoder loads KV caches before generating. Both need the NIXL connector initialized for their respective role, and `kv_both` enables that on both sides. This also allows a single instance to serve as both prefiller and decoder (useful for testing).
