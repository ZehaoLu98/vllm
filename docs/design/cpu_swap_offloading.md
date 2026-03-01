# CPU Swap KV Cache Offloading

This document describes the per-layer CPU swap implementation for KV cache
offloading added in the `explore` branch.

## Motivation

Standard vLLM stores the entire KV cache on GPU memory. This limits how many
sequences can be active at once to the number of blocks that fit in GPU VRAM.
CPU swap offloading trades bandwidth for capacity: KV data is kept on CPU RAM
(pinned memory) and swapped to/from the GPU as needed.

Unlike the existing `OffloadingConnector` (which bulk-transfers entire
requests' KV caches before/after a forward pass), the `SwapConnector` targets
an extreme regime: the entire KV cache lives on CPU, and only one layer's
worth of KV data is resident on GPU at any given moment during the forward
pass. This makes it possible to run very long sequences or very large batch
sizes with minimal GPU memory, at the cost of per-layer PCIe bandwidth.

---

## Key Concepts

### Offloaded block size vs GPU block size

The CPU stores KV blocks at `offloaded_block_size` tokens per block
(typically equal to `gpu_block_size`, but configurable to be a multiple).
`block_size_factor = offloaded_block_size // gpu_block_size` decides how many
GPU blocks map to one CPU block.

### GPU→CPU block mapping

The scheduler maintains a `gpu_to_cpu_block_map` dictionary that maps each
GPU block ID to its corresponding CPU block ID. This mapping is included in
`SwapConnectorMetadata` and shipped to the worker every step so the worker
knows where to read from / write to without extra coordination.

### Single-GPU-tensor mode

When `single_gpu_tensor: true` is set in `kv_connector_extra_config`, **all
transformer layers share a single GPU KV tensor** instead of one tensor per
layer. This is the most memory-aggressive option: the GPU holds only one
layer's worth of KV cache tokens at a time, and all layers time-multiplex that
buffer. The CPU holds the full KV state for every layer.

### No-eviction policy

`SwapManager` never evicts blocks. If the CPU runs out of space it raises an
assertion error. `CPUSwapSpec` validates at startup that
`cpu_bytes_to_use` is large enough to hold all possible KV blocks.

---

## File-by-File Description

### `vllm/v1/kv_offload/swap_manager.py` — `SwapManager`

A simple, no-eviction implementation of `OffloadingManager`.

| Method | Behaviour |
|---|---|
| `lookup` | Returns the count of consecutive ready blocks starting from the first hash. |
| `prepare_load` | Increments ref-counts and returns a backend load spec. |
| `prepare_store` | Allocates free CPU blocks; **asserts** instead of evicting. |
| `complete_store` | Marks blocks ready and emits `OffloadingEvent`s. |
| `complete_load` | Decrements ref-counts after a load is done. |
| `touch` | Moves recently-used hashes to the end of the ordered dict (LRU bookkeeping, no eviction). |
| `take_events` | Drains and returns accumulated `OffloadingEvent`s for KV cache event publishing. |

`SwapManager` wraps a pluggable `Backend` (currently `CPUBackend`) for the
actual block allocation and address resolution.

---

### `vllm/v1/kv_offload/cpu_swap.py` — `CPUSwapSpec`

`CPUSwapSpec` extends `OffloadingSpec` with CPU-swap-specific validation and
wiring.

**At initialisation it:**
1. Reads `cpu_bytes_to_use` from `kv_connector_extra_config`.
2. Computes `kv_bytes_per_offloaded_block` from page size, layer count, and
   tensor-parallel world size.
3. Derives `self.num_blocks = cpu_bytes_to_use // kv_bytes_per_offloaded_block`.
4. Asserts `self.num_blocks >= required_offloaded_blocks` — if GPU can hold
   *N* blocks then the CPU must hold at least *N* blocks of equivalent
   capacity, because blocks swap 1-to-1.

**`get_manager()`** returns a lazily-created `SwapManager(CPUBackend(...))`.

**`get_handlers()`** returns `CpuGpuOffloadingHandlers` for bulk GPU↔CPU
transfers (used for prefix-cache population; skipped in single-tensor mode).

---

### `vllm/distributed/kv_transfer/kv_connector/v1/swap_connector.py` — `SwapConnector`

The main connector class. It implements `KVConnectorBase_V1` and splits work
between a scheduler-side and a worker-side helper.

```
SwapConnector
├── SwapConnectorScheduler  (runs in the scheduler process)
└── SwapConnectorWorker     (runs in each GPU worker process)
```

#### `SwapConnectorMetadata`

Passed from scheduler to worker every step:

| Field | Description |
|---|---|
| `reqs_to_load` | Prefix-cache hit requests: CPU blocks → GPU blocks (bulk transfer). |
| `reqs_to_store` | Newly completed full blocks to persist on CPU (bulk transfer). |
| `active_gpu_block_ids` | Per-request list of GPU block IDs scheduled this step. |
| `gpu_to_cpu_block_map` | GPU block ID → CPU block ID for all live blocks. |
| `new_req_ids` | Requests with no KV on CPU yet (skip load phase). |

---

#### `SwapConnectorScheduler`

Runs on the scheduler. Mirrors `OffloadingConnectorScheduler` but also:

- Maintains `_gpu_to_cpu_block_map` — updated whenever a store spec is
  built, by zipping source GPU block IDs against destination CPU block IDs
  from the `BlockIDsLoadStoreSpec`.
- Tracks `_new_req_ids` — requests that appear for the first time in the
  current step. They have no KV on CPU yet, so the worker should skip loading
  for them.
- Tracks `_request_block_ids` — the list of all GPU block IDs allocated to
  each request (accumulates across steps as new tokens are generated).

Key methods:

| Method | Who calls it | Purpose |
|---|---|---|
| `get_num_new_matched_tokens` | KV cache manager | Returns number of tokens whose KV is already on CPU (prefix cache lookup). |
| `update_state_after_alloc` | KV cache manager | Called after GPU blocks are allocated; marks new requests, builds load spec for prefix hits. |
| `build_connector_meta` | Scheduler | Computes `reqs_to_store` for this step, builds `SwapConnectorMetadata`. |
| `update_connector_output` | Scheduler | Advances manager state when bulk jobs finish. |
| `request_finished` | Scheduler | Cleans up per-request state; returns whether the request is still waiting for a store. |

---

#### `SwapConnectorWorker`

Runs on each GPU worker. Owns two CUDA streams: `_load_stream` and
`_store_stream`.

**`register_kv_caches`** — called once at startup:
- Detects single-tensor mode (all layers point to the same GPU buffer).
- Allocates per-layer **pinned CPU tensors** sized `(num_cpu_blocks * block_size_factor, ...)`.
- Registers bulk transfer handlers (skipped in single-tensor mode).
- Precomputes `_block_size_bytes` for the `ops.swap_blocks` call.

**Per-forward-pass flow:**

```
start_load_kv()          ← submit deferred store jobs from prev step;
                            start bulk prefix-cache loads
  │
  │  (for each layer in the model)
  ▼
wait_for_layer_load(layer_name)   ← load_layer_from_cpu()
                                     copies CPU→GPU for this layer's blocks
  │
  │  [attention computation happens here on the GPU]
  │
  ▼
save_kv_layer(layer_name, ...)    ← store_layer_to_cpu()
                                     copies GPU→CPU asynchronously,
                                     records _store_event
  │
  │  (repeat for remaining layers)
  │
  ▼
wait_for_save()          ← synchronise _store_stream; submit bulk store jobs
```

**`load_layer_from_cpu`:**
1. Iterates `metadata.active_gpu_block_ids`, skipping `new_req_ids`.
2. Looks up each GPU block's CPU block ID in `gpu_to_cpu_block_map`.
3. Deduplicates and builds a `(cpu_id, gpu_id)` tensor for `ops.swap_blocks`.
4. Issues `ops.swap_blocks(cpu_tensor, gpu_tensor, ...)` on `_load_stream`.
5. **Synchronises** `_load_stream` — the attention kernel needs the data immediately.

**`store_layer_to_cpu`:**
1. Iterates `metadata.active_gpu_block_ids` (including new requests).
2. Builds a `(gpu_id, cpu_id)` tensor and calls `ops.swap_blocks` on
   `_store_stream`.
3. Records `_store_event` — does **not** synchronise — so the next layer's
   load can proceed in parallel with the store.

The `_load_stream` waits on `_store_event` before each load to guarantee that
older data is not overwritten before it has been copied to CPU.

---

### `vllm/v1/core/kv_cache_utils.py` — `_maybe_apply_single_gpu_tensor`

Called at the end of `get_kv_cache_config_from_groups`. When
`single_gpu_tensor: true` is in `kv_connector_extra_config`:

1. Collects all layer names across all KV cache groups.
2. Computes how many blocks fit in CPU (`cpu_num_blocks`) and GPU
   (`gpu_num_blocks`).
3. Sets `num_blocks = min(cpu_num_blocks, gpu_num_blocks)` and logs which is
   the bottleneck.
4. Replaces the N per-layer `KVCacheTensor` objects with a **single**
   `KVCacheTensor(size=page_size * num_blocks, shared_by=all_layer_names)`.

The result is that the GPU allocates only one tensor, and all layers'
`kv_caches[layer_name]` pointers all point to the same buffer.

---

### `vllm/distributed/kv_transfer/kv_connector/factory.py`

Added a registration entry:

```python
KVConnectorFactory.register_connector(
    "SwapConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.swap_connector",
    "SwapConnector",
)
```

### `vllm/v1/kv_offload/factory.py`

Added:

```python
OffloadingSpecFactory.register_spec(
    "CPUSwapSpec", "vllm.v1.kv_offload.cpu_swap", "CPUSwapSpec"
)
```

---

## Data Flow Summary

```
Scheduler process                        Worker process
─────────────────────────────────────    ─────────────────────────────────────
SwapConnectorScheduler                   SwapConnectorWorker
  │                                        │
  ├─ get_num_new_matched_tokens()          │
  │    └─ SwapManager.lookup()             │
  │                                        │
  ├─ update_state_after_alloc()            │
  │    └─ SwapManager.prepare_load()       │
  │                                        │
  ├─ build_connector_meta()                │
  │    ├─ SwapManager.prepare_store()      │
  │    └─ SwapConnectorMetadata ──────────►│
  │                                        │
  │                                        ├─ start_load_kv()
  │                                        │    └─ submit bulk loads/stores
  │                                        │
  │                                        ├─ wait_for_layer_load(L0)
  │                                        │    └─ CPU→GPU layer 0 (sync)
  │                                        │ [attention layer 0]
  │                                        ├─ save_kv_layer(L0)
  │                                        │    └─ GPU→CPU layer 0 (async)
  │                                        │
  │                                        ├─ wait_for_layer_load(L1)
  │                                        │    └─ CPU→GPU layer 1 (sync)
  │                                        │ [attention layer 1]
  │                                        ├─ save_kv_layer(L1)
  │                                        │    └─ GPU→CPU layer 1 (async)
  │                                        │   ...
  │                                        ├─ wait_for_save()
  │                                        │    └─ sync store stream
  │                                        │    └─ submit deferred bulk stores
  │                                       ◄│
  ├─ update_connector_output()             │
  │    └─ SwapManager.complete_store/load()│
  │                                        │
```

---

## Configuration

`SwapConnector` is enabled via `kv_transfer_config` in the vLLM config:

```python
from vllm.config import KVTransferConfig

KVTransferConfig(
    kv_connector="SwapConnector",
    kv_connector_extra_config={
        "spec_name": "CPUSwapSpec",
        # Total CPU memory budget for KV blocks (in bytes)
        "cpu_bytes_to_use": 32 * 1024**3,   # 32 GiB
        # Optional: put all layers in one GPU tensor
        "single_gpu_tensor": True,
    },
)
```

| Parameter | Required | Description |
|---|---|---|
| `cpu_bytes_to_use` | Yes | CPU pinned memory budget in bytes. Must be large enough to hold all KV blocks for all layers. |
| `spec_name` | No | Defaults to `"CPUOffloadingSpec"`. Set to `"CPUSwapSpec"` for no-eviction swap mode. |
| `single_gpu_tensor` | No | When `True`, all layers share one GPU KV buffer; only one layer's KV is on GPU at a time. Default: `False`. |

---

## Differences from `OffloadingConnector`

| | `OffloadingConnector` | `SwapConnector` |
|---|---|---|
| Transfer granularity | Entire request's KV (all layers at once) | One layer at a time, every forward pass |
| GPU memory | Holds all layers' KV for scheduled batch | Holds at most 1 layer's KV (in single-tensor mode) |
| CPU memory policy | LRU or ARC eviction when full | No eviction — asserts enough space exists |
| Prefix cache | Yes (bulk load before forward pass) | Yes (bulk load + per-layer swap) |
| CUDA stream overlap | Load stream overlaps with compute | Store of layer *N* overlaps with load of layer *N+1* |
| Primary use case | Disaggregated prefill, P/D separation | Extreme memory pressure (tiny GPU, large context) |
