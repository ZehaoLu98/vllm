# Per-Layer KV Cache CPU Offloading

## Problem

For large models (e.g., Llama-70B) serving long-context requests, the KV cache
dominates GPU memory usage.  On memory-constrained GPUs this limits either the
maximum batch size or the maximum context length that can be served.

## Solution

This change introduces **per-layer KV cache CPU offloading** via the existing
LMCache connector infrastructure.  Instead of allocating KV cache on GPU for
**all** N transformer layers simultaneously, only a small GPU buffer (default:
2 layers, for double-buffering) is allocated.  The full KV cache for every
layer is stored in CPU pinned memory.

Before each layer's attention computation, the corresponding KV data is
asynchronously copied from CPU to the GPU buffer.  After attention completes,
the updated KV data is copied back to CPU.  Double-buffering overlaps the
data transfer of layer N+1 with the computation of layer N, hiding most of
the PCIe latency.

```
Per forward step:

  start_load_kv():
    async copy: cpu_kv[layer_0] -> gpu_buffer[slot 0]

  For each layer i = 0..N-1:
    wait_for_layer_load(layer_i):
      sync gpu_buffer[i%2] load complete
      async copy: cpu_kv[layer_{i+1}] -> gpu_buffer[(i+1)%2]   # prefetch

    [attention computes on gpu_buffer[i%2]]

    save_kv_layer(layer_i):
      async copy: gpu_buffer[i%2] -> cpu_kv[layer_i]            # save back

  wait_for_save():
    sync all pending saves
```

## Configuration

Enable via `kv_connector_extra_config` when using `LMCacheConnectorV1`:

```python
from vllm import LLM
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={
        "full_offload": True,
        "num_gpu_buffer_layers": 2,   # optional, default 2
    },
)
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    kv_transfer_config=ktc,
)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `full_offload` | `bool` | `False` | Enable per-layer CPU offloading mode |
| `num_gpu_buffer_layers` | `int` | `2` | Number of GPU buffer slots (>=2 for double-buffering) |

## Architecture

### Files changed

| File | Role |
|------|------|
| `vllm_/distributed/kv_transfer/kv_connector/v1/lmcache_integration/full_offload.py` | **New.** `FullOffloadEngine` — manages double-buffered async GPU<->CPU transfers using CUDA streams. |
| `vllm_/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` | Modified `LMCacheConnectorV1` to detect `full_offload` config and delegate worker-side methods to `FullOffloadEngine`. Scheduler-side methods return no-ops. |
| `vllm_/v1/worker/gpu/attn_utils.py` | Added `_allocate_kv_cache_with_offloading()` — allocates a small GPU buffer (2 layers) + full CPU pinned memory. |
| `vllm_/v1/worker/gpu_model_runner.py` | Added `_initialize_kv_cache_tensors_with_offloading()` — wires up the offloading allocation path and passes CPU tensors to the connector. |

### How it hooks into vLLM

The implementation reuses three existing integration points — no changes to
the attention layer code or the `@maybe_transfer_kv_layer` decorator are
required:

1. **`@maybe_transfer_kv_layer` decorator**
   (`vllm_/attention/utils/kv_transfer_utils.py`): already wraps every
   `unified_attention` call with `connector.wait_for_layer_load()` before
   and `connector.save_kv_layer()` after attention.

2. **`start_load_kv()` / `wait_for_save()`**: called by the model runner
   mixin (`kv_connector_model_runner_mixin.py`) before and after the forward
   pass respectively.

3. **`build_connector_meta()`**: returns a minimal
   `_FullOffloadConnectorMetadata` so that `has_connector_metadata()` is
   `True`, which is required for the per-layer hooks to fire.

### Memory layout

```
GPU (small buffer):
  buffer_slot_0: [num_blocks, 2, block_size, num_kv_heads, head_size]
  buffer_slot_1: [num_blocks, 2, block_size, num_kv_heads, head_size]

CPU pinned memory (one per logical layer):
  layer_0: [num_blocks, 2, block_size, num_kv_heads, head_size]
  layer_1: [num_blocks, 2, block_size, num_kv_heads, head_size]
  ...
  layer_N: [num_blocks, 2, block_size, num_kv_heads, head_size]
```

Layers sharing the same `KVCacheTensor.shared_by` group (e.g., hybrid
attention models) share both their GPU buffer slot and CPU backing tensor,
matching the existing non-offloading behaviour.

### CUDA stream synchronization

Each GPU buffer slot has a dedicated CUDA stream.  Events are used to
synchronize:

- **Load event**: recorded after CPU->GPU copy; the compute stream waits on
  it before running attention.
- **Save event**: recorded after GPU->CPU copy; the transfer stream for the
  next load waits on it before reusing the buffer slot.

## Trade-offs

| Aspect | Impact |
|--------|--------|
| GPU memory | Reduced from N layers to ~2 layers of KV cache |
| Throughput | Degraded by PCIe bandwidth bottleneck (~32 GB/s Gen4) |
| Latency | Double-buffering hides most transfer latency |
| Compatibility | Works with FlashAttention, FlashInfer; outside torch.compile/CUDA graph boundary |

## Existing related features

| Feature | Location | Difference |
|---------|----------|------------|
| Native CPU offloading | `vllm_/v1/kv_offload/` | Block-level eviction (LRU/ARC) when GPU is full; does **not** reduce peak GPU KV memory |
| LMCache layerwise mode | `vllm_v1_adapter.py` `use_layerwise=True` | Per-layer transfer for prefix cache reuse; GPU KV still fully allocated |
| **This change** | `full_offload.py` | Per-layer transfer for **all** active blocks; GPU KV reduced to buffer |
