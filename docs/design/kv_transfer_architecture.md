# vLLM v1 KV Transfer Connector Architecture

## Overview

The KV transfer connector manages KV cache transfers between GPU/CPU/remote storage. It supports:
- **CPU offloading**: Swapping KV blocks between GPU and CPU pinned memory
- **Disaggregated prefill**: Transferring KV data between vLLM instances (prefill → decode)

---

## 1. KV Transfer Group

**Global singleton** accessed via `get_kv_transfer_group()` in [vllm/distributed/kv_transfer/kv_transfer_state.py](vllm/distributed/kv_transfer/kv_transfer_state.py).

```python
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
connector = get_kv_transfer_group()
```

The connector has two roles:
- `KVConnectorRole.SCHEDULER` — runs in the scheduler process
- `KVConnectorRole.WORKER` — runs in the worker process

Base API defined in [vllm/distributed/kv_transfer/kv_connector/v1/base.py](vllm/distributed/kv_transfer/kv_connector/v1/base.py).

---

## 2. Key Files

| File | Purpose |
|------|---------|
| [vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py](vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py) | OffloadingConnector (CPU↔GPU) — main connector implementation |
| [vllm/v1/worker/kv_connector_model_runner_mixin.py](vllm/v1/worker/kv_connector_model_runner_mixin.py) | Worker-side lifecycle orchestration |
| [vllm/v1/core/sched/scheduler.py](vllm/v1/core/sched/scheduler.py) | Scheduler — all connector hook call sites |
| [vllm/v1/engine/core.py](vllm/v1/engine/core.py) | Top-level engine step loop |
| [vllm/v1/worker/gpu_model_runner.py](vllm/v1/worker/gpu_model_runner.py) | execute_model(), uses connector context manager |
| [vllm/v1/kv_offload/worker/cpu_gpu.py](vllm/v1/kv_offload/worker/cpu_gpu.py) | SingleDirectionOffloadingHandler — actual CUDA DMA |
| [vllm/v1/kv_offload/worker/worker.py](vllm/v1/kv_offload/worker/worker.py) | OffloadingWorker, OffloadingHandler abstract |
| [vllm/v1/kv_offload/abstract.py](vllm/v1/kv_offload/abstract.py) | OffloadingManager (scheduler-side) abstract |
| [vllm/v1/kv_offload/mediums.py](vllm/v1/kv_offload/mediums.py) | GPULoadStoreSpec, CPULoadStoreSpec |
| [vllm/model_executor/layers/attention/kv_transfer_utils.py](vllm/model_executor/layers/attention/kv_transfer_utils.py) | `@maybe_transfer_kv_layer` decorator |

---

## 3. Engine Step Sequence

Top-level in [engine/core.py:386-402](vllm/v1/engine/core.py#L386):
```
scheduler.schedule()  →  model_executor.execute_model()  →  scheduler.update_from_output()
```

---

### Phase 1: `scheduler.schedule()`

| Step | Location | Connector API | What it does |
|------|----------|---------------|--------------|
| 1 | [scheduler.py:550](vllm/v1/core/sched/scheduler.py#L550) | `get_num_new_matched_tokens()` | Check how many prefix tokens are already offloaded/remote |
| 2 | [scheduler.py:614](vllm/v1/core/sched/scheduler.py#L614) | `update_state_after_alloc()` | After block allocation, notify connector of new block assignments |
| 3 | [scheduler.py:751](vllm/v1/core/sched/scheduler.py#L751) | `update_state_after_alloc()` | Same, for requests promoted from waiting |
| 4 | [scheduler.py:896](vllm/v1/core/sched/scheduler.py#L896) | `build_connector_metadata()` | Build the `ConnectorMetadata` object sent to the worker |

**Output**: `SchedulerOutput` contains `ConnectorMetadata` (load specs, store specs, job IDs).

---

### Phase 2: `worker.execute_model()`

Orchestrated by `_get_kv_connector_output()` context manager in [kv_connector_model_runner_mixin.py:84](vllm/v1/worker/kv_connector_model_runner_mixin.py#L84).

| Step | Connector API | What it does |
|------|---------------|--------------|
| 1 | `bind_connector_metadata(metadata)` | Worker receives ConnectorMetadata from scheduler |
| 2 | `start_load_kv(scheduler_output)` | Flush deferred stores from previous step, submit load DMA jobs via `ops.swap_blocks()` |
| 3 | `[model forward pass]` | Per-layer: `wait_for_layer_load()` (no-op) → attention → `save_kv_layer()` (no-op) |
| 4 | `wait_for_save(scheduler_output)` | Enqueue store jobs into `_unsubmitted_store_jobs` (deferred to next step) |
| 5 | `get_finished()` | Poll CUDA events non-blocking via `event.query()`, collect completed job IDs |
| 6 | `get_block_ids_with_load_errors()` | Check for any failed loads |
| 7 | `clear_connector_metadata()` | Clean up per-step state |

**Why `wait_for_layer_load` / `save_kv_layer` are no-ops in OffloadingConnector:**
The OffloadingConnector does bulk block transfers, not layer-by-layer streaming. The per-layer hooks exist for disaggregated prefill connectors that send KV data layer-by-layer over the network.

**Why store submission is deferred:**
`wait_for_save()` is called after token sampling. Submitting the store DMA immediately would risk stalling. Instead, stores are queued in `_unsubmitted_store_jobs` and flushed at the start of the next step's `start_load_kv()`.

---

### Phase 3: `scheduler.update_from_output()`

| Step | Location | Connector API | What it does |
|------|----------|---------------|--------------|
| 1 | [scheduler.py:1492](vllm/v1/core/sched/scheduler.py#L1492) | `update_connector_output(output)` | Process `finished_recving` and `finished_sending` job lists from worker |
| 2 | [scheduler.py:1996](vllm/v1/core/sched/scheduler.py#L1996) | `_update_waiting_for_remote_kv()` | For requests in `WAITING_FOR_REMOTE_KVS`: if load done → cache blocks, update `num_computed_tokens`, promote to RUNNING |
| 3 | [scheduler.py:1778](vllm/v1/core/sched/scheduler.py#L1778) | `request_finished(req)` | On request completion, notify connector to free any held resources |

---

## 4. Async Load Lifecycle

```
Step N:   scheduler sees request needs KV load
          → prepare_load() on OffloadingManager
          → ConnectorMetadata includes LoadSpec
          → worker submit DMA via ops.swap_blocks() on dedicated CUDA stream
          → request enters WAITING_FOR_REMOTE_KVS state (NOT forwarded this step)

Step N+k: get_finished() polls event.query() (non-blocking)
          → if complete: report job_id in finished_recving
          → scheduler._update_waiting_for_remote_kv() promotes request to RUNNING
          → request is forwarded with KV data already in GPU memory
```

**Why `event.query()` is non-blocking and correct:**
- `torch.Event.query()` returns `True/False` immediately without stalling the GPU
- The request consuming the loaded KV data is **never** scheduled for forward in the same step as its load submission
- By the time the request is forwarded, the CUDA event has already completed

---

## 5. SingleDirectionOffloadingHandler

[vllm/v1/kv_offload/worker/cpu_gpu.py](vllm/v1/kv_offload/worker/cpu_gpu.py) — handles one direction (CPU→GPU or GPU→CPU).

### Key design:
- Each transfer gets a **dedicated CUDA stream** (pooled for reuse)
- Transfers are **serialized**: each stream waits on the `end_event` of the previous transfer
- GPU→CPU transfers additionally wait on `torch.cuda.current_stream()` to let model computation finish first
- **Block size factor**: CPU blocks may be larger than GPU blocks; `expand_block_ids()` converts logical block IDs to kernel block IDs

### `transfer_async()` flow:
```python
# 1. Build src→dst block mapping
expand_block_ids(src_blocks, src_block_size_factor, src_to_dst[:, 0])
expand_block_ids(dst_blocks, dst_block_size_factor, src_to_dst[:, 1])

# 2. Get stream from pool (or create new)
stream = self._stream_pool.pop() or torch.cuda.Stream()

# 3. Serialize: wait for previous transfer to finish
stream.wait_event(last_transfer.end_event)

# 4. Submit CUDA kernel
with torch.cuda.stream(stream):
    start_event.record(stream)
    ops.swap_blocks(src_tensor, dst_tensor, block_size_in_bytes, src_to_dst_tensor)
    end_event.record(stream)
```

### `get_finished()` flow:
```python
while self._transfers and self._transfers[0].end_event.query():
    transfer = self._transfers.popleft()
    elapsed = start_event.elapsed_time(end_event) * 1e-3  # ms → s
    results.append(TransferResult(job_id, success=True, ...))
    # Return stream and events to pool
```

---

## 6. Scheduler-Side: OffloadingManager

Defined abstractly in [vllm/v1/kv_offload/abstract.py](vllm/v1/kv_offload/abstract.py).

| Method | Purpose |
|--------|---------|
| `lookup(block_hashes)` | Max prefix length that is offloaded. Returns `None` to delay scheduling. |
| `prepare_load(block_hashes)` | Pin blocks from eviction, return `LoadStoreSpec` for worker |
| `touch(block_hashes)` | Mark blocks as recently used (LRU update) |
| `complete_load(block_hashes)` | Unpin blocks after load completes |
| `prepare_store(block_hashes)` | Pin blocks, return `StoreSpec` + list of evicted blocks |
| `complete_store(block_hashes, success)` | Mark blocks as stored (now loadable) or remove if failed |
| `take_events()` | Yield `OffloadingEvent` objects (stored/removed) for external consumers |

---

## 7. Data Flow Summary

```
Scheduler                          Worker
─────────────────────────────────────────────────────────────────
OffloadingManager.prepare_load()
  → CPULoadStoreSpec(block_ids=[3,7,12])
  → ConnectorMetadata
                                   start_load_kv(ConnectorMetadata)
                                     → OffloadingWorker.transfer_async()
                                       → SingleDirectionOffloadingHandler
                                         → ops.swap_blocks() on CUDA stream
                                         → end_event recorded

                                   [forward pass — request NOT included yet]

                                   get_finished()
                                     → event.query() → True
                                     → returns [job_id=42]

update_connector_output(output)
  finished_recving=[42]
  → _update_waiting_for_remote_kv()
    → cache blocks, num_computed_tokens updated
    → request promoted WAITING → RUNNING

[next step: request forwarded with KV in GPU]
```
