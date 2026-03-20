# vLLM KV Cache GPU Management from CPU Processes

## Overview

vLLM's serving processes (scheduler, model runner) run entirely on the **CPU**. They manage GPU KV cache memory through integer block IDs and PyTorch's CUDA runtime — never touching GPU memory directly. The GPU holds a single pre-allocated tensor pool, and CPU-side bookkeeping decides which pages are free, allocated, or shared.

---

## 1. GPU Memory Pre-Allocation at Startup

At initialization, the GPU worker allocates **one large contiguous tensor** for all KV cache pages:

- `GPUModelRunner._allocate_kv_cache_tensors` (`vllm/v1/worker/gpu_model_runner.py`) calls `torch.zeros(size, dtype=torch.int8, device="cuda")` — a single flat byte buffer covering all blocks.
- These raw tensors are reshaped into per-layer KV cache views via `_reshape_kv_cache_tensors` and bound to each attention layer via `bind_kv_cache`.
- GPU memory is **fixed and pre-allocated** — no dynamic CUDA malloc happens at serving time.

**Key files:**
- `vllm/v1/worker/gpu_worker.py` — worker receives KV cache config and calls `model_runner.initialize_kv_cache()`
- `vllm/v1/worker/gpu_model_runner.py` — `_allocate_kv_cache_tensors`, `_reshape_kv_cache_tensors`, `bind_kv_cache`
- `vllm/v1/kv_cache_interface.py` — `KVCacheConfig` with `num_blocks` and `kv_cache_tensors`

---

## 2. CPU-Side Block Tracking (Block Pool and Free List)

The CPU scheduler never accesses GPU memory. It maintains a **block pool** of lightweight metadata objects:

- `BlockPool` (`vllm/v1/core/block_pool.py`) creates `[KVCacheBlock(id=0), KVCacheBlock(id=1), ...]` — one per physical GPU page.
- A doubly-linked **free list** (`FreeKVCacheBlockQueue` in `vllm/v1/core/kv_cache_utils.py`) enables O(1) allocate/free operations.
- **Allocation** = pop a block from the free list, increment `ref_cnt`.
- **Free** = decrement `ref_cnt`; when it hits 0, push back to the free list.
- **Prefix cache hits** use `touch` to remove a block from the free list and increment `ref_cnt` (preventing eviction).

The `block_id` integer **is** the physical page index into the pre-allocated GPU tensor. No indirection table is needed.

**Key data structures:**
- `KVCacheBlock` (`vllm/v1/core/kv_cache_utils.py`) — `block_id`, `ref_cnt`, hash metadata
- `FreeKVCacheBlockQueue` (`vllm/v1/core/kv_cache_utils.py`) — intrusive doubly-linked free list
- `BlockPool` (`vllm/v1/core/block_pool.py`) — manages all blocks and free queue
- `SingleTypeKVCacheManager` (`vllm/v1/core/single_type_kv_cache_manager.py`) — per-request block ownership via `req_to_blocks`

---

## 3. Scheduler Sends Block IDs to GPU Workers

When the scheduler runs a scheduling step:

1. `KVCacheManager.allocate_slots` (`vllm/v1/core/kv_cache_manager.py`) assigns block IDs to requests.
2. Block IDs are serialized into `SchedulerOutput`:
   - New requests: `NewRequestData.block_ids` (`vllm/v1/core/sched/output.py`)
   - Running requests: `CachedRequestData.new_block_ids` (`vllm/v1/core/sched/output.py`)
   - Freshly allocated blocks needing zeroing: `new_block_ids_to_zero`
3. The output is sent to GPU workers via `collective_rpc("execute_model", ...)` (`vllm/v1/executor/abstract.py`).

---

## 4. GPU Worker Builds Block Tables and Slot Mappings

The GPU model runner translates integer block IDs into GPU-resident tensors:

- `_update_states` appends new block IDs to a per-request **block table** (`vllm/v1/worker/gpu_model_runner.py`).
- `_prepare_inputs` commits the block table to a GPU tensor and computes **slot mappings**:
  ```
  slot = block_id × block_size + offset_within_block
  ```
  (see `vllm/v1/worker/block_table.py`)

---

## 5. Attention Kernels Use Block Tables to Index GPU Memory

The attention kernels receive two key inputs:

### Writing new K/V values (slot_mapping)
- `reshape_and_cache_flash` (`csrc/cache_kernels.cu`) uses:
  ```
  block_idx = slot / block_size
  offset    = slot % block_size
  ```
  to write into the paged cache tensor.

### Reading K/V during attention (block_table)
- FlashAttention passes `block_table` to `flash_attn_varlen_func(block_table=...)` (`vllm/v1/attention/backends/flash_attn.py`), which gathers the correct K/V pages for each sequence.

---

## 6. The Model Runner: A CPU Process Using PyTorch for GPU Execution

The `GPUModelRunner` is a **regular Python/CPU process**. It does not run on the GPU. Its role:

1. **Prepares inputs on CPU** — builds block tables, slot mappings, input token ID tensors, position arrays, etc.
2. **Copies them to GPU** — via PyTorch `.to(device)` or by writing into pre-allocated CUDA tensors.
3. **Launches GPU kernels asynchronously** — calls PyTorch operations (`model.forward(...)`, `flash_attn_varlen_func(...)`, `reshape_and_cache_flash(...)`). These are asynchronous: the CPU enqueues work on the GPU's CUDA stream and returns immediately.
4. **Synchronizes only when needed** — e.g., to read output logits back to CPU for sampling/scheduling decisions.

All matrix multiplications, attention computations, and KV cache reads/writes execute on the GPU. The CPU process is the orchestrator — it decides *what* to run (which requests, which blocks, which tokens) and dispatches the work via CUDA.

---

## End-to-End Flow Diagram

```
CPU Scheduler                          GPU Worker (CPU process)
─────────────                          ───────────────────────
BlockPool (free list of IDs)
    │
    ├─ allocate → pop block_id
    ├─ free     → push block_id
    │
    └─ SchedulerOutput{block_ids} ──RPC──► _update_states()
                                              │
                                        block_table[req][i] = block_id
                                        slot_mapping = block_id × block_size + offset
                                              │
                                        ┌─────┴─────┐
                                   write K/V     read K/V
                                (slot_mapping)  (block_table)
                                        │           │
                                   cache_kernels  flash_attn       ← GPU kernels
                                   .cu             kernel
                                        │           │
                                        └─────┬─────┘
                                              │
                                    Pre-allocated GPU KV Cache Tensor
                                    [num_blocks × block_size × num_heads × head_dim]
```

**Key insight:** The CPU never accesses GPU memory. It only manipulates integer block IDs. The block ID *is* the physical page index in the pre-allocated GPU tensor. The GPU worker (still a CPU process) receives those IDs, places them in a block table tensor on GPU, and CUDA kernels use simple integer arithmetic (`block_id × block_size + offset`) to convert IDs to memory offsets.
