# KV Cache Offloading Strategies in vLLM KV Connectors

This document summarizes the KV cache offloading strategies of each KV connector in vLLM (excluding `SwapConnector`), focusing on **when** and **how** data moves between GPU and CPU (or external storage).

---

## Architecture Overview

All connectors inherit from `KVConnectorBase_V1` and separate logic into two roles:

- **Scheduler-side**: Runs in the scheduler process. Decides *what* to load/store via metadata.
- **Worker-side**: Runs in GPU workers. Executes the actual data transfers.

### Lifecycle within a single engine step

```
Scheduler                              Worker (GPU Model Runner)
─────────                              ─────────────────────────
get_num_new_matched_tokens()
update_state_after_alloc()
build_connector_meta()
        ──── metadata ────>   bind_connector_metadata()
                              start_load_kv()            ← bulk async load
                              ┌─ for each layer:
                              │   wait_for_layer_load()  ← per-layer sync point
                              │   [attention computation]
                              │   save_kv_layer()        ← per-layer async save
                              └─
                              wait_for_save()            ← sync all saves
                              get_finished()             ← report completed transfers
                              clear_connector_metadata()
```

The `maybe_transfer_kv_layer` decorator on each attention layer calls `wait_for_layer_load()` before and `save_kv_layer()` after the attention computation, enabling per-layer pipelining.

---

## 1. OffloadingConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`

### Strategy: Bulk GPU ↔ CPU offloading with LRU eviction

The OffloadingConnector treats CPU memory as an LRU-managed extension of GPU KV cache. It transfers **all layers at once** (cross-layer blocks) rather than layer-by-layer.

### Swap-Out Timing (GPU → CPU)

| Phase | What happens |
|-------|-------------|
| `build_connector_meta()` | Scheduler calls `_get_reqs_to_store()` which identifies new complete blocks ready to offload. The `OffloadingManager.prepare_store()` decides which blocks to store (and which old blocks to evict via LRU). |
| `wait_for_save()` | Worker calls `prepare_store_kv()` which **defers** the store jobs — they are added to `_unsubmitted_store_jobs`. |
| Next step's `start_kv_transfers()` | The deferred store jobs are actually submitted. This ensures offloading starts **after** token sampling completes, avoiding delays to generation latency. |

**Key insight:** Store operations are intentionally delayed by one engine step. The store is prepared during `wait_for_save()` but not submitted until the *next* step's `start_kv_transfers()`, so GPU→CPU copies don't compete with the critical path of token generation.

### Swap-In Timing (CPU → GPU)

| Phase | What happens |
|-------|-------------|
| `get_num_new_matched_tokens()` | Scheduler queries `OffloadingManager.lookup()` to check if the request's prefix exists in CPU cache. Returns the number of hit tokens. |
| `update_state_after_alloc()` | After GPU blocks are allocated, `OffloadingManager.prepare_load()` pins the CPU blocks (prevents eviction). A `TransferSpec` (src=CPU, dst=GPU block IDs) is recorded. |
| `start_kv_transfers()` | Worker submits the load as an async job. The CPU→GPU copy runs concurrently with other work. |
| `get_finished()` | Worker reports which loads completed. Scheduler calls `complete_load()` to unpin blocks. |

### Properties
- `prefer_cross_layer_blocks = True` — all layers share one contiguous tensor per block
- Async, non-blocking transfers via `OffloadingWorker`
- LRU or ARC eviction policies available

---

## 2. LMCacheConnectorV1

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`

### Strategy: External KV cache system delegation

LMCacheConnector is a thin wrapper that delegates all operations to an external LMCache engine (either native or the latest `lmcache` library).

### Swap-Out Timing

| Phase | What happens |
|-------|-------------|
| `save_kv_layer()` | Called per-layer by the attention decorator. The LMCache engine handles the actual storage (may go to CPU, disk, or remote). |
| `wait_for_save()` | Blocks until LMCache confirms all saves are done. |
| `request_finished()` | Triggers async finalization in LMCache. |

### Swap-In Timing

| Phase | What happens |
|-------|-------------|
| `get_num_new_matched_tokens()` | Queries LMCache for prefix hits. Returns `(num_tokens, False)` — the `False` indicates **synchronous** loading. |
| `start_load_kv()` | Delegates to LMCache engine to begin loading matched KV data. |
| `wait_for_layer_load()` | Blocks until the specific layer's KV is ready (per-layer sync). |

### Properties
- Per-layer transfer granularity (uses the attention decorator hooks)
- Synchronous load model (`is_async = False`)
- Storage medium depends on LMCache configuration (CPU RAM, SSD, remote)
- Supports KV event aggregation across workers

---

## 3. P2pNcclConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

### Strategy: GPU-to-GPU peer transfer via NCCL (disaggregated prefill-decode)

This connector transfers KV cache **between GPUs** — from a prefill worker to a decode worker — not to CPU. It uses NCCL for high-bandwidth GPU↔GPU communication.

### Transfer Timing

| Phase | What happens |
|-------|-------------|
| `build_connector_meta()` | Scheduler packages request metadata (token IDs, block IDs) into `P2pNcclConnectorMetadata`. |
| `start_load_kv()` | **Consumer (decode):** Receives KV data from the prefill worker via `P2pNcclEngine`. |
| `save_kv_layer()` | **Producer (prefill):** Sends KV data per-layer to the decode worker after computing attention. |
| `wait_for_save()` | Blocks until all NCCL sends complete. |

### Properties
- Producer/consumer model controlled by `is_kv_producer` flag
- Supports chunked prefill (partial KV transfer for long sequences)
- No CPU involvement — pure GPU↔GPU via NCCL
- Per-layer transfer via the attention decorator

---

## 4. NixlConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`

### Strategy: RDMA-based disaggregated transfer with dynamic topology

NIXL (Negotiated Inter-process transfer) enables KV transfer between distributed vLLM instances (prefill, decode, and storage agents) using RDMA for high throughput.

### Transfer Timing

| Phase | What happens |
|-------|-------------|
| `get_num_new_matched_tokens()` | Scheduler checks if KV data is available from remote agents. |
| `update_state_after_alloc()` | Allocates local blocks and prepares remote fetch descriptors. |
| `start_load_kv()` | Worker initiates async RDMA reads from remote NIXL agents into local GPU memory. May use a host (CPU) staging buffer. |
| `save_kv_layer()` | No-op for most setups; KV is read directly from GPU memory by remote agents. |
| `get_finished()` | Polls NIXL transfer handles for completion. |

### Properties
- ZMQ-based metadata exchange for agent discovery
- Supports host (CPU) transfer buffers for cross-node communication
- Async, non-blocking RDMA transfers
- Complex multi-agent topology (prefill ↔ decode ↔ storage)
- Versioned protocol for P/D interoperability

---

## 5. MultiConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`

### Strategy: Tiered/composite caching via multiple child connectors

MultiConnector wraps multiple KV connectors and delegates operations to each in sequence. This enables **hierarchical caching** (e.g., GPU prefix cache → CPU offload → remote distributed cache).

### Transfer Timing

All lifecycle methods are forwarded to each child connector:

| Phase | What happens |
|-------|-------------|
| `get_num_new_matched_tokens()` | Queries each child connector; uses the best hit. |
| `start_load_kv()` | Each child starts its own loads. |
| `save_kv_layer()` | Each child saves to its own medium. |
| `wait_for_save()` | Waits for all children. |

### Properties
- Metadata is a tuple of child metadata objects (`MultiKVConnectorMetadata`)
- Stats aggregated per-connector
- Enables combining e.g., `OffloadingConnector` + `NixlConnector`

---

## 6. MoRIIOConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`

### Strategy: Modular distributed IO via MoRIIO engine

Uses the MoRIIO (Modular Remote IO) framework for disaggregated KV transfers between nodes.

### Transfer Timing

| Phase | What happens |
|-------|-------------|
| `start_load_kv()` | Initiates async remote KV fetch via MoRIIO engine. |
| `save_kv_layer()` | Sends KV data to remote nodes per-layer. |
| `get_finished()` | Polls `ThreadPoolExecutor` futures for completion. |

### Properties
- External `mori.io` library integration
- ZMQ-based handshaking between workers
- Async transfers via thread pool
- Supports multiple backend types

---

## 7. MooncakeConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

### Strategy: Mooncake transfer engine for prefill-decode disaggregation

Specialized for disaggregated serving using the Mooncake transfer engine for efficient cross-node KV movement.

### Transfer Timing

| Phase | What happens |
|-------|-------------|
| `start_load_kv()` | Decode worker fetches KV from prefill via Mooncake engine. |
| `save_kv_layer()` | Prefill worker stores KV for later retrieval. |
| `request_finished()` | Triggers cleanup in Mooncake engine. |

### Properties
- External Mooncake library (`mooncake.engine.TransferEngine`)
- HTTP/ZMQ bootstrap coordination
- Direct tensor memory addressing via base addresses

---

## 8. ExampleConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py`

### Strategy: Disk-based KV cache for debugging/reference

Saves and loads KV cache to/from disk files using `safetensors`. This is a reference implementation not meant for production.

### Transfer Timing

| Phase | What happens |
|-------|-------------|
| `start_load_kv()` | Reads KV tensors from disk files (synchronous). |
| `save_kv_layer()` | Writes KV tensors to disk files per-layer (synchronous). |

### Properties
- Synchronous, single-threaded
- Disk I/O via safetensors
- Hashes prompts for cache filenames
- Supports multimodal features

---

## 9. DecodeBenchConnector

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py`

### Strategy: Synthetic KV fill for benchmarking

Fills KV cache with dummy (random) values to simulate prefill-decode disaggregation without any real transfer. Used purely for decode performance benchmarking.

### Transfer Timing

| Phase | What happens |
|-------|-------------|
| `start_load_kv()` | Fills allocated KV blocks with random values (configurable mean/std). |

### Properties
- No real network or storage I/O
- Configurable fill distribution
- Emulates the decode-only workload pattern

---

## Comparison Summary

| Connector | Medium | Granularity | Async | Store Timing | Load Timing |
|-----------|--------|-------------|-------|-------------|-------------|
| **Offloading** | CPU RAM | All layers (bulk) | Yes | Deferred to next step | Before forward pass |
| **LMCache** | CPU/SSD/Remote | Per layer | Partial | During attention | Before attention |
| **P2pNccl** | GPU↔GPU (NCCL) | Per layer | Yes | After attention | Before forward |
| **Nixl** | RDMA network | Bulk | Yes | On-demand reads | Before forward |
| **Multi** | Mixed (composite) | Varies | Yes | Varies per child | Varies per child |
| **MoRIIO** | Network | Per layer | Yes | After attention | Before forward |
| **Mooncake** | Network | Per layer | Yes | After attention | Before forward |
| **Example** | Disk | Per layer | No | After attention | Before forward |
| **DecodeBench** | Synthetic | Bulk | No | N/A | Before forward |

---

## Key Design Patterns

1. **Deferred stores (OffloadingConnector):** GPU→CPU stores are queued during `wait_for_save()` but only submitted at the start of the *next* engine step. This avoids competing with token sampling on the critical path.

2. **Per-layer pipelining (LMCache, P2pNccl, etc.):** The `maybe_transfer_kv_layer` decorator intercepts each attention layer's forward, calling `wait_for_layer_load()` before and `save_kv_layer()` after. This enables overlapping computation of layer N with transfer of layer N-1.

3. **Cross-layer bulk transfer (OffloadingConnector):** By setting `prefer_cross_layer_blocks = True`, all layers' KV data for a given block is stored contiguously. This allows a single DMA operation to move an entire block's KV data across all layers, rather than N separate per-layer transfers.

4. **LRU eviction (OffloadingConnector):** The `OffloadingManager` uses LRU (or ARC) policy to decide which CPU-side blocks to evict when space is needed for new stores. Blocks being loaded are pinned to prevent eviction.
