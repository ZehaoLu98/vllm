# CPU Swap KV Cache Offloading — Deep Dive

This document provides a comprehensive explanation of how the per-layer CPU swap
offloading system works in vLLM using the `SwapConnector`. It covers the full
call stack from scheduler decisions through CUDA kernel execution, the
synchronization model, memory layout, and every relevant class and method.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture Overview](#architecture-overview)
3. [Key Concepts](#key-concepts)
4. [Initialization & Memory Setup](#initialization--memory-setup)
5. [Full Call Stack — Step by Step](#full-call-stack--step-by-step)
   - [Phase 1: Scheduler — Prefix Cache Lookup](#phase-1-scheduler--prefix-cache-lookup)
   - [Phase 2: Scheduler — Block Allocation & State Update](#phase-2-scheduler--block-allocation--state-update)
   - [Phase 3: Scheduler — Build Connector Metadata](#phase-3-scheduler--build-connector-metadata)
   - [Phase 4: Worker — Pre-Forward Transfers](#phase-4-worker--pre-forward-transfers)
   - [Phase 5: Worker — Per-Layer Swap Loop](#phase-5-worker--per-layer-swap-loop)
   - [Phase 6: Worker — Post-Forward Cleanup](#phase-6-worker--post-forward-cleanup)
   - [Phase 7: Scheduler — Completion Tracking](#phase-7-scheduler--completion-tracking)
6. [CUDA Stream Synchronization Model](#cuda-stream-synchronization-model)
7. [The `ops.swap_blocks` CUDA Kernel](#the-opsswap_blocks-cuda-kernel)
8. [Data Structures Reference](#data-structures-reference)
9. [File-by-File Description](#file-by-file-description)
10. [Configuration](#configuration)
11. [Differences from OffloadingConnector](#differences-from-offloadingconnector)

---

## Motivation

Standard vLLM stores the entire KV cache on GPU memory. This limits how many
sequences can be active at once to the number of blocks that fit in GPU VRAM.
CPU swap offloading trades bandwidth for capacity: KV data is kept on CPU RAM
(pinned memory) and swapped to/from the GPU as needed.

Unlike the existing `OffloadingConnector` (which bulk-transfers entire
requests' KV caches before/after a forward pass), the `SwapConnector` targets
an extreme regime: **the entire KV cache lives on CPU, and only one layer's
worth of KV data is resident on GPU at any given moment** during the forward
pass. This makes it possible to run very long sequences or very large batch
sizes with minimal GPU memory, at the cost of per-layer PCIe bandwidth.

**When to use `SwapConnector`:**
- Your model's KV cache exceeds available GPU VRAM
- You want to serve longer context lengths than GPU memory allows
- You have sufficient CPU RAM and PCIe bandwidth
- You are willing to accept higher latency per token in exchange for being able to serve at all

---

## Architecture Overview

The `SwapConnector` splits its logic across two processes:

```
SwapConnector (implements KVConnectorBase_V1)
├── SwapConnectorScheduler   (runs in the scheduler process)
│   ├── Decides what to load/store
│   ├── Maintains GPU↔CPU block mappings
│   └── Delegates to SwapManager → CPUBackend
└── SwapConnectorWorker      (runs in each GPU worker process)
    ├── Manages CUDA streams for async transfers
    ├── Owns per-layer CPU pinned-memory tensors
    └── Calls ops.swap_blocks for actual data movement
```

The scheduler builds a `SwapConnectorMetadata` object each step and ships it
to the worker via IPC. The worker uses this metadata to know exactly which
blocks to load from CPU before each layer and store back after.

---

## Key Concepts

### Offloaded block size vs GPU block size

The CPU stores KV blocks at `offloaded_block_size` tokens per block (typically
equal to `gpu_block_size`, but configurable to be a multiple).
`block_size_factor = offloaded_block_size // gpu_block_size` decides how many
GPU blocks map to one CPU block.

### GPU→CPU block mapping

The scheduler maintains a `gpu_to_cpu_block_map` dictionary that maps each GPU
block ID to its corresponding CPU block ID. This mapping is included in
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
assertion error. `CPUSwapSpec` validates at startup that `cpu_bytes_to_use` is
large enough to hold all possible KV blocks.

### New requests vs existing requests

A request that appears for the first time has no KV data on CPU. The worker
must skip the load phase for these requests (there is nothing to load) but
must still store their KV after attention computes it. The scheduler tracks
these via `_new_req_ids` / `SwapConnectorMetadata.new_req_ids`.

---

## Initialization & Memory Setup

Before any inference can happen, the system must allocate memory and wire up
the transfer pipeline. This happens in several stages:

### 1. `CPUSwapSpec.__init__()` — Capacity planning

**File:** `vllm/v1/kv_offload/cpu_swap.py`

When the engine starts, `CPUSwapSpec` is instantiated with the vLLM config
and `KVCacheConfig`. It:

1. Reads `cpu_bytes_to_use` from `kv_connector_extra_config`.
2. Computes bytes per offloaded block:
   ```
   page_size_bytes × num_layers × tensor_parallel_world_size × block_size_factor
   ```
3. Derives `num_blocks = cpu_bytes_to_use // kv_bytes_per_offloaded_block`.
4. Asserts `num_blocks >= required_offloaded_blocks` — the CPU must hold at
   least as many blocks as the GPU can address.

### 2. `SwapConnectorWorker.register_kv_caches()` — GPU & CPU tensor allocation

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/swap_connector.py:528`

Called once after GPU tensors are allocated. For each layer:

1. **Detect single-tensor mode:** If all GPU tensors share the same
   `data_ptr()`, enable single-tensor mode (skip bulk handlers).
2. **Probe tensor layout:** Call `attn_backend.get_kv_cache_shape(num_blocks=1234, ...)`
   to determine whether the tensor shape is `(num_blocks, ...)` or
   `(2, num_blocks, ...)` (the latter has a leading key/value dimension).
3. **Allocate pinned CPU tensor:** `torch.zeros(cpu_shape, dtype=..., device="cpu", pin_memory=True)`.
   Pinned memory enables DMA transfers via `cudaMemcpyAsync`.
4. **Compute block size in bytes:**
   `ref.element_size() * ref.stride(0)` — this is the stride of one block in
   the tensor, used by `ops.swap_blocks`.
5. **Register bulk handlers** (normal mode only) for `CpuGpuOffloadingHandlers`.

### 3. `SwapConnectorScheduler.__init__()` — Scheduler state

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/swap_connector.py:198`

Initializes:
- `_requests: dict[ReqId, Request]` — live requests
- `_request_block_ids: dict[ReqId, list[int]]` — GPU block IDs per request
- `_gpu_to_cpu_block_map: dict[int, int]` — persistent GPU→CPU mapping
- `_new_req_ids: set[ReqId]` — first-time requests
- `_reqs_to_load / _reqs_being_loaded / _reqs_being_stored` — transfer tracking
- `_next_stored_block_idx: dict[ReqId, int]` — next block to offload per request

---

## Full Call Stack — Step by Step

Each scheduler step follows a strict sequence. Below is the complete call
chain from top to bottom, with the exact methods and their responsibilities.

### Phase 1: Scheduler — Prefix Cache Lookup

```
Scheduler._schedule_waiting()
  └─► connector.get_num_new_matched_tokens(request, num_computed_tokens)
        └─► SwapConnectorScheduler.get_num_new_matched_tokens()
              ├─ Compute num_blocks = request.num_tokens // offloaded_block_size
              ├─ manager.touch(block_hashes)      # LRU bookkeeping
              ├─ manager.lookup(block_hashes)      # count consecutive CPU hits
              └─ Return (num_hit_tokens, True) or (0, False)
```

**What happens:** The scheduler is considering whether to schedule a waiting
request. It asks the connector: "How many of this request's tokens already
have KV data on CPU?" The `SwapManager.lookup()` walks the request's block
hashes in order and counts how many consecutive blocks are in `READY` state
(i.e., fully stored on CPU and not being transferred).

**Key detail:** The lookup starts from `start_block_idx = num_computed_tokens //
offloaded_block_size`, skipping any blocks already computed on GPU. A
minimum of `offloaded_block_size` tokens must be matchable for the hit to
count (partial blocks are not useful).

### Phase 2: Scheduler — Block Allocation & State Update

```
Scheduler._schedule_waiting()  (continued)
  ├─► KVCacheManager.allocate()  →  blocks: KVCacheBlocks
  └─► connector.update_state_after_alloc(request, blocks, num_external_tokens)
        └─► SwapConnectorScheduler.update_state_after_alloc()
              ├─ Register request in _requests and _request_block_ids
              ├─ If num_external_tokens == 0:
              │    └─ Mark as new: _new_req_ids.add(req_id)
              └─ If num_external_tokens > 0 (prefix cache hit):
                   ├─ manager.prepare_load(block_hashes) → src_spec (CPU)
                   ├─ GPULoadStoreSpec(pending_block_ids) → dst_spec (GPU)
                   ├─ _reqs_to_load[req_id] = (src_spec, dst_spec)
                   ├─ _reqs_being_loaded[req_id].update(block_hashes)
                   └─ _next_stored_block_idx[req_id] = num_blocks
```

**What happens:** After the KV cache manager allocates GPU blocks for the
request, the connector records which blocks are new and which need to be
bulk-loaded from CPU.

**Two cases:**
- **New request (no prefix hit):** Added to `_new_req_ids`. The worker will
  skip loading for this request since there is nothing on CPU to load.
- **Prefix cache hit:** A `TransferSpec` pair `(CPU src, GPU dst)` is built
  and stored in `_reqs_to_load`. The `SwapManager` increments reference
  counts on the CPU blocks to prevent them from being freed.

### Phase 3: Scheduler — Build Connector Metadata

```
Scheduler._make_scheduler_output()
  └─► connector.build_connector_meta(scheduler_output)
        └─► SwapConnectorScheduler.build_connector_meta()
              ├─ _get_reqs_to_store(scheduler_output)
              │    ├─ For each scheduled request:
              │    │    ├─ Track new GPU block IDs in _request_block_ids
              │    │    ├─ Compute total tokens and num_blocks
              │    │    ├─ manager.prepare_store(new_block_hashes)
              │    │    │    ├─ SwapManager.prepare_store()
              │    │    │    │    ├─ Filter already-stored hashes
              │    │    │    │    ├─ Assert enough free CPU blocks
              │    │    │    │    └─ backend.allocate_blocks() → CPU block IDs
              │    │    │    └─ Return PrepareStoreOutput with store_spec
              │    │    ├─ Build GPULoadStoreSpec(src_block_ids)
              │    │    ├─ Update _gpu_to_cpu_block_map for each (gpu_id, cpu_id)
              │    │    └─ Add to reqs_to_store
              │    └─ Return reqs_to_store
              ├─ Build active_gpu_block_ids for all scheduled requests
              ├─ Construct SwapConnectorMetadata:
              │    ├─ reqs_to_load      (prefix cache hits from Phase 2)
              │    ├─ reqs_to_store     (new blocks to persist)
              │    ├─ active_gpu_block_ids
              │    ├─ gpu_to_cpu_block_map
              │    └─ new_req_ids
              ├─ Clear _reqs_to_load and _new_req_ids
              └─ Handle preemptions: complete_store for preempted requests
```

**What happens:** This is the critical metadata-building step. The scheduler
examines every scheduled request to find blocks that have been filled with
new tokens and need to be stored to CPU. For each, it asks `SwapManager` to
allocate CPU blocks and returns a `TransferSpec` containing the GPU source
and CPU destination block IDs.

**The GPU→CPU block map update** is critical: when `prepare_store` returns a
`BlockIDsLoadStoreSpec` as the destination, the scheduler zips the GPU source
block IDs with the CPU destination block IDs and records the mapping. This
mapping persists across steps and is how the worker knows which CPU block
corresponds to each GPU block during per-layer swaps.

### Phase 4: Worker — Pre-Forward Transfers

```
ModelRunner.execute_model()
  └─► KVConnectorModelRunnerMixin._get_kv_connector_output()
        └─► connector.start_load_kv(forward_context)
              └─► SwapConnectorWorker.start_kv_transfers(metadata)
                    ├─ [single-tensor mode: return immediately]
                    ├─ Submit deferred store jobs from previous step:
                    │    for (job_id, spec) in _unsubmitted_store_jobs:
                    │        worker.transfer_async(job_id, spec)
                    │          └─► handler.transfer_async(job_id, spec)
                    │                └─► CpuGpuOffloadingHandler.transfer_async()
                    └─ Start prefix cache loads:
                         for req_id, spec in metadata.reqs_to_load:
                             worker.transfer_async(job_id, spec)
```

**What happens:** Before the forward pass begins, the worker submits any
deferred store jobs from the *previous* step (these were prepared but not
submitted to allow the forward pass to overlap) and kicks off bulk prefix
cache loads for any new requests with cache hits.

**Why deferred stores?** During the previous step's `wait_for_save()`, the
per-layer store stream was synchronized, meaning all GPU→CPU copies finished.
But the *bulk* store jobs (which persist the data to the `SwapManager`'s
accounting) are deferred to this point to avoid blocking the previous step's
critical path.

### Phase 5: Worker — Per-Layer Swap Loop

This is the heart of the SwapConnector. For each transformer layer in the
model, a load-compute-store cycle happens:

```
model.forward()  →  for each layer:
  └─► unified_attention() [decorated with @maybe_transfer_kv_layer]
        └─► wrapper():
              ├─ connector.wait_for_layer_load(layer_name)
              │    └─► SwapConnectorWorker.load_layer_from_cpu(layer_name, metadata)
              │          ├─ For each req_id in active_gpu_block_ids:
              │          │    ├─ Skip if req_id in new_req_ids (nothing on CPU)
              │          │    └─ For each gpu_id: look up cpu_id in gpu_to_cpu_block_map
              │          ├─ Deduplicate (gpu_id, cpu_id) pairs
              │          ├─ Build numpy src_to_dst mapping array
              │          ├─ Wait for previous store: _load_stream.wait_event(_store_event)
              │          ├─ on _load_stream:
              │          │    └─ ops.swap_blocks(cpu_tensor, gpu_tensor, block_size_bytes, mapping)
              │          │         └─► [CUDA kernel: cudaMemcpyAsync HostToDevice per block]
              │          └─ _load_stream.synchronize()  ← BLOCKING: attention needs the data
              │
              ├─ func() → unified_attention kernel executes on default stream
              │    └─► FlashAttention/PagedAttention reads from GPU KV cache
              │
              └─ connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
                   └─► SwapConnectorWorker.store_layer_to_cpu(layer_name, metadata)
                         ├─ For each req_id in active_gpu_block_ids:
                         │    └─ For each gpu_id: look up cpu_id in gpu_to_cpu_block_map
                         ├─ Deduplicate (gpu_id, cpu_id) pairs
                         ├─ Build numpy src_to_dst mapping array
                         ├─ _store_stream.wait_stream(current_stream)  ← wait for attention
                         ├─ on _store_stream:
                         │    └─ ops.swap_blocks(gpu_tensor, cpu_tensor, block_size_bytes, mapping)
                         │         └─► [CUDA kernel: cudaMemcpyAsync DeviceToHost per block]
                         └─ Record _store_event (NO synchronize — overlap with next layer)
```

**The decorator:** `@maybe_transfer_kv_layer` (in
`vllm/model_executor/layers/attention/kv_transfer_utils.py`) wraps the
attention function. It inspects the function signature to find the
`layer_name` parameter, then before execution calls `wait_for_layer_load()`
and after execution calls `save_kv_layer()`. If no connector is active, the
wrapper is a no-op.

**Load phase details:**
- The load skips `new_req_ids` because they have no KV data on CPU yet.
- Block IDs are deduplicated because multiple requests might share prefix
  cache blocks.
- The load **must synchronize** because the attention kernel on the default
  stream reads from the GPU KV cache immediately.
- Before loading, `_load_stream` waits on `_store_event` to ensure the
  previous layer's GPU→CPU store has read the GPU data before it gets
  overwritten.

**Store phase details:**
- The store includes ALL requests (including new ones) because after attention
  computes, even new requests have KV data that needs to be persisted to CPU.
- `_store_stream.wait_stream(current_stream)` ensures the store does not
  begin until the attention kernel has finished writing.
- The store does NOT synchronize — this is the key to overlap. The GPU→CPU
  copy runs concurrently with the next layer's CPU→GPU load.

**For tensors with a leading kv dimension** (shape `(2, num_blocks, ...)`),
the K and V halves are swapped separately:
```python
ops.swap_blocks(tensor[0], ...)  # K half
ops.swap_blocks(tensor[1], ...)  # V half
```

### Phase 6: Worker — Post-Forward Cleanup

```
[forward pass completes]
  └─► connector.wait_for_save()
        └─► SwapConnectorWorker.wait_for_all_stores()
              └─ _store_stream.synchronize()  ← wait for last layer's GPU→CPU copy
        └─► SwapConnectorWorker.prepare_store_kv(metadata)
              └─ [single-tensor mode: return]
              └─ For each req_id in metadata.reqs_to_store:
                   ├─ _jobs[job_id] = (req_id, True)
                   ├─ _store_jobs[req_id].add(job_id)
                   └─ _unsubmitted_store_jobs.append((job_id, spec))
                        ↑ deferred — will be submitted in next step's start_load_kv
  └─► connector.get_finished(finished_req_ids)
        └─► SwapConnectorWorker.get_finished()
              ├─ Collect results from worker.get_finished()
              │    └─ Each handler reports completed job_ids
              ├─ Update _jobs, _store_jobs, _load_job
              └─ Return (finished_sending, finished_recving)
```

**What happens:** After the entire forward pass:
1. **Synchronize the store stream:** The last layer's GPU→CPU copy must
   complete before we can proceed.
2. **Prepare bulk store jobs:** These are transfer specs for the
   `OffloadingWorker` bulk transfer system. They are NOT submitted yet — they
   are deferred to the next step's `start_load_kv()` to avoid blocking.
3. **Collect finished transfers:** Check which bulk load/store jobs completed
   since the last check.

### Phase 7: Scheduler — Completion Tracking

```
Scheduler.update_from_output()
  └─► connector.update_connector_output(kv_connector_output)
        └─► SwapConnectorScheduler.update_connector_output()
              ├─ For req_id in finished_sending:
              │    └─ manager.complete_store(block_hashes)
              │         └─ SwapManager.complete_store()
              │              └─ Mark blocks as READY (ref_cnt = 0)
              │              └─ Emit OffloadingEvent if events enabled
              └─ For req_id in finished_recving:
                   └─ manager.complete_load(block_hashes)
                        └─ SwapManager.complete_load()
                             └─ Decrement ref_cnt on loaded blocks
```

**What happens:** The scheduler receives the worker's report of which
transfers completed. It advances the `SwapManager` state accordingly:
- **Store completion:** Blocks transition from "being stored" to "ready".
  They are now available for future prefix cache lookups.
- **Load completion:** Reference counts are decremented, allowing the blocks
  to be freed if no longer needed.

When a request finishes:

```
Scheduler._connector_finished(request)
  └─► connector.request_finished(request, block_ids)
        └─► SwapConnectorScheduler.request_finished()
              ├─ Clean _gpu_to_cpu_block_map for request's blocks
              ├─ Clean _requests, _request_block_ids, _next_stored_block_idx
              ├─ Remove from _new_req_ids
              └─ Return (is_waiting_for_store, None)
```

---

## CUDA Stream Synchronization Model

The SwapConnector uses three CUDA streams with carefully ordered
synchronization points to maximize overlap:

```
Time ──────────────────────────────────────────────────────────────►

Default Stream   ║ attention L0 ║                ║ attention L1 ║
                 ║              ║                ║              ║
_load_stream     ╠══ CPU→GPU L0 ═╗              ╠══ CPU→GPU L1 ═╗
                 ║  (sync here) ──╬──►          ║  (sync here) ──╬──►
                 ║               ║              ║               ║
_store_stream    ║               ╠══ GPU→CPU L0 ══╗             ╠══ GPU→CPU L1 ═
                 ║               ║  (no sync)    ║             ║
                 ║               ║     event ────╬─► wait      ║
```

**Synchronization rules:**

1. **_load_stream.synchronize()** — After each CPU→GPU copy, the load stream
   is fully synchronized. This blocks the CPU thread until the copy
   completes, because the attention kernel on the default stream needs the
   data immediately.

2. **_store_stream.wait_stream(current_stream)** — Before each GPU→CPU copy,
   the store stream waits for the default stream's attention kernel to
   finish. This ensures the store reads the correct (post-attention) data.

3. **_load_stream.wait_event(_store_event)** — Before each CPU→GPU copy for
   the *next* layer, the load stream waits for the previous layer's GPU→CPU
   store to have read the GPU buffer. In single-tensor mode, the GPU buffer
   is shared, so we must not overwrite it before the store is done.

4. **_store_stream.synchronize()** in `wait_for_all_stores()` — At the end
   of the forward pass, we must ensure all stores have completed before
   proceeding to the next scheduler step.

---

## The `ops.swap_blocks` CUDA Kernel

**File:** `csrc/cache_kernels.cu:33`

```cpp
void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 int64_t block_size_in_bytes,
                 const torch::Tensor& block_mapping)
```

This is the low-level function that performs the actual data movement:

1. **Determine copy direction** from device types:
   - `cudaMemcpyHostToDevice` (CPU→GPU) for loads
   - `cudaMemcpyDeviceToHost` (GPU→CPU) for stores
2. **Iterate block_mapping** (a CPU tensor of shape `[N, 2]`):
   - `block_mapping[i][0]` = source block number
   - `block_mapping[i][1]` = destination block number
3. **Issue `cudaMemcpyAsync`** for each block:
   ```
   src_offset = src_block_number * block_size_in_bytes
   dst_offset = dst_block_number * block_size_in_bytes
   cudaMemcpyAsync(dst + dst_offset, src + src_offset,
                   block_size_in_bytes, memcpy_type, stream)
   ```

**Important:** The `block_mapping` tensor must be on CPU to avoid a
GPU-CPU synchronization for each `.item()` call. The function runs on
whichever CUDA stream is current (set by `torch.cuda.stream()`).

---

## Data Structures Reference

### `SwapConnectorMetadata`

Passed from scheduler to worker every step via IPC:

| Field | Type | Description |
|---|---|---|
| `reqs_to_load` | `dict[ReqId, TransferSpec]` | Prefix-cache hit requests: CPU blocks → GPU blocks (bulk transfer). |
| `reqs_to_store` | `dict[ReqId, TransferSpec]` | Newly completed full blocks to persist on CPU (bulk transfer). |
| `active_gpu_block_ids` | `dict[ReqId, list[int]]` | Per-request list of ALL GPU block IDs in use this step. |
| `gpu_to_cpu_block_map` | `dict[int, int]` | GPU block ID → CPU block ID for all live blocks. |
| `new_req_ids` | `set[ReqId]` | Requests with no KV on CPU yet (skip load, still store). |

### `TransferSpec`

A tuple `(src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec)` where each spec
contains a list of block IDs and a medium identifier ("GPU" or "CPU").

### `SwapManager` (no-eviction OffloadingManager)

| Method | Behaviour |
|---|---|
| `lookup(hashes)` | Count consecutive READY blocks from the start. |
| `prepare_load(hashes)` | Increment ref-counts; return backend load spec. |
| `prepare_store(hashes)` | Allocate free CPU blocks; **assert** instead of evicting. |
| `complete_store(hashes)` | Mark blocks READY; emit `OffloadingEvent`. |
| `complete_load(hashes)` | Decrement ref-counts. |
| `touch(hashes)` | LRU bookkeeping (move to end of OrderedDict). |

### `CPUBackend`

Manages a pool of CPU block slots. Each slot is `offloaded_block_size` tokens
worth of KV data. Provides `allocate_blocks()`, `free()`, and
`get_load_store_spec()`.

---

## File-by-File Description

### `vllm/v1/kv_offload/cpu_swap.py` — `CPUSwapSpec`

`CPUSwapSpec` extends `OffloadingSpec` with CPU-swap-specific validation and
wiring. At initialization it validates CPU memory is sufficient, and provides
factory methods for `SwapManager` and `CpuGpuOffloadingHandlers`.

### `vllm/v1/kv_offload/swap_manager.py` — `SwapManager`

A simple, no-eviction implementation of `OffloadingManager`. Wraps a pluggable
`Backend` (currently `CPUBackend`) for block allocation and address resolution.

### `vllm/distributed/kv_transfer/kv_connector/v1/swap_connector.py` — `SwapConnector`

The main connector class. Implements `KVConnectorBase_V1` and delegates to
`SwapConnectorScheduler` (scheduler process) and `SwapConnectorWorker`
(GPU worker process).

### `vllm/model_executor/layers/attention/kv_transfer_utils.py` — `@maybe_transfer_kv_layer`

The decorator that wraps `unified_attention()` and
`unified_attention_with_output()`. Before the attention function executes,
it calls `connector.wait_for_layer_load(layer_name)`. After execution, it
calls `connector.save_kv_layer(layer_name, kv_cache, attn_metadata)`.

### `vllm/v1/kv_offload/worker/worker.py` — `OffloadingWorker`

Manages multiple `OffloadingHandler`s for bulk async transfers. The
`SwapConnectorWorker` uses this for prefix-cache loads and deferred stores
(not for per-layer swaps, which use direct `ops.swap_blocks` calls).

### `csrc/cache_kernels.cu` — `swap_blocks()`

CUDA kernel that performs block-level `cudaMemcpyAsync` between CPU and GPU
tensors. Used on both `_load_stream` and `_store_stream`.

### `vllm/distributed/kv_transfer/kv_connector/factory.py`

Registration entry:

```python
KVConnectorFactory.register_connector(
    "SwapConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.swap_connector",
    "SwapConnector",
)
```

### `vllm/v1/kv_offload/factory.py`

Registration entry:

```python
OffloadingSpecFactory.register_spec(
    "CPUSwapSpec", "vllm.v1.kv_offload.cpu_swap", "CPUSwapSpec"
)
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
| Bulk transfer handler | Always used | Skipped in single-tensor mode |

---

## Complete Data Flow Diagram

```
Scheduler process                        Worker process
─────────────────────────────────────    ──────────────────────────────────────────
SwapConnectorScheduler                   SwapConnectorWorker
  │                                        │
  ├─ get_num_new_matched_tokens()          │
  │    ├─ SwapManager.touch()              │
  │    └─ SwapManager.lookup()             │
  │                                        │
  ├─ update_state_after_alloc()            │
  │    ├─ Track _request_block_ids         │
  │    ├─ Mark _new_req_ids (no hit)       │
  │    └─ SwapManager.prepare_load() (hit) │
  │                                        │
  ├─ build_connector_meta()                │
  │    ├─ _get_reqs_to_store()             │
  │    │    ├─ SwapManager.prepare_store() │
  │    │    └─ Update _gpu_to_cpu_block_map│
  │    └─ SwapConnectorMetadata ──────────►│
  │                                        │
  │                                        ├─ start_load_kv()
  │                                        │    ├─ submit deferred stores (prev step)
  │                                        │    └─ start bulk prefix-cache loads
  │                                        │
  │                                        ├─ [for each layer in model:]
  │                                        │    │
  │                                        │    ├─ wait_for_layer_load(Ln)
  │                                        │    │    ├─ _load_stream.wait_event(store_event)
  │                                        │    │    ├─ ops.swap_blocks(CPU→GPU) on _load_stream
  │                                        │    │    └─ _load_stream.synchronize()  ← BLOCKING
  │                                        │    │
  │                                        │    ├─ [attention kernel Ln on default stream]
  │                                        │    │
  │                                        │    └─ save_kv_layer(Ln)
  │                                        │         ├─ _store_stream.wait_stream(default)
  │                                        │         ├─ ops.swap_blocks(GPU→CPU) on _store_stream
  │                                        │         └─ record store_event  ← NO SYNC (overlap)
  │                                        │
  │                                        ├─ wait_for_save()
  │                                        │    ├─ _store_stream.synchronize()
  │                                        │    └─ prepare deferred bulk store jobs
  │                                        │
  │                                        ├─ get_finished()
  │                                       ◄│    └─ report finished load/store job IDs
  │                                        │
  ├─ update_connector_output()             │
  │    ├─ SwapManager.complete_store()     │
  │    └─ SwapManager.complete_load()      │
  │                                        │
  └─ request_finished() [on completion]    │
       └─ Clean _gpu_to_cpu_block_map      │
       └─ Clean all per-request state      │
```
