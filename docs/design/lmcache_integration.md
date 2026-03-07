# LMCache Integration with vLLM

This document explains how LMCache integrates with vLLM, the features it provides, and the detailed KV cache movement between CPU and GPU during prefill and decode phases.

## Overview

LMCache is an external KV cache management system that integrates with vLLM through a connector-based abstraction. It enables **cross-request KV cache reuse** by persisting computed KV tensors to external storage (CPU memory, disk, or remote nodes) and loading them back when a new request shares a prefix with a previously seen one. This avoids redundant prefill computation and significantly reduces time-to-first-token (TTFT).

## Architecture

### Connector-Based Abstraction

LMCache plugs into vLLM via the **KV Connector** interface defined in `vllm/distributed/kv_transfer/kv_connector/v1/base.py`. The main entry point is `LMCacheConnectorV1` in `lmcache_connector.py`, which delegates to the core implementation in `lmcache_integration/vllm_v1_adapter.py`.

```
vLLM Engine
├── Scheduler Process
│   └── LMCacheConnectorV1Impl (SCHEDULER role)
│       └── LookupClient  ← prefix matching queries
│
└── Worker Process(es)
    └── LMCacheConnectorV1Impl (WORKER role)
        └── LMCacheEngine  ← actual KV data transfers
            ├── GPU Connector (paged-mem or buffer-based)
            └── Storage Backends (CPU / Disk / Remote via NIXL)
```

### Scheduler-Worker Split

The connector has two distinct roles that run in separate processes:

- **Scheduler Role**: Decides *what* to load and save. It queries the LMCache lookup client for prefix hits, creates load/save specifications, and passes metadata to workers.
- **Worker Role**: Executes *actual data transfers*. It initializes the LMCacheEngine, manages GPU connectors, and performs the physical movement of KV tensors between storage and GPU memory.

Communication between the two roles happens via `LMCacheConnectorMetadata`, which is built by the scheduler each engine step and consumed by the worker.

### Key Source Files

| File | Purpose |
|------|---------|
| `kv_connector/v1/lmcache_connector.py` | Public connector entry point |
| `lmcache_integration/vllm_v1_adapter.py` | Core implementation (~1400 lines) |
| `lmcache_integration/multi_process_adapter.py` | Multi-process support via ZMQ |
| `lmcache_integration/utils.py` | Helper utilities |

## Features

### 1. Cross-Request KV Cache Reuse

When a new request arrives, LMCache checks whether any prefix of its token sequence has been seen before. If a match is found, the corresponding KV tensors are loaded from storage instead of being recomputed. This is the primary feature.

- Supports both **token-based** matching (exact token ID sequences) and **hash-based** matching (block-level content hashes).
- Handles **multimodal** content by embedding visual feature hashes into token IDs for prefix matching.

### 2. Flexible Storage Backends

LMCache supports multiple storage tiers:

| Backend | Use Case |
|---------|----------|
| **Local CPU** | Fast in-process offloading via pinned memory |
| **Local Disk** | Persistent storage on SSD/NVMe |
| **Remote (NIXL)** | Disaggregated storage across nodes |
| **Hybrid** | Combination of the above via `remote_serde` config |

### 3. Chunk-Based Management

LMCache operates on fixed-size **chunks** (default 256 tokens). vLLM's paged blocks map to chunks via `blocks_in_chunk = chunk_size / block_size`. All load/save operations are chunk-aligned.

### 4. Layer-Wise Pipelining

When `use_layerwise=True`, KV data is loaded and saved one transformer layer at a time using async generators. This allows overlap: layer N+1's KV can be loaded from storage while layer N is computing attention, maximizing GPU utilization and hiding transfer latency.

### 5. Disaggregated Prefill

LMCache can split prefill and decode across different vLLM instances:
- A **prefiller** instance computes KV and stores it via LMCache.
- A **decoder** instance loads the KV from remote storage and runs decode only.
- Coordinated via `DisaggSpec` metadata and NIXL transport.

### 6. Decode-Phase KV Saving

Optionally (`save_decode_cache=True`), newly generated KV from decode tokens can also be saved. This allows future requests to reuse KV from completed generations, not just from shared prompts.

## Key Data Structures

### LoadSpec and SaveSpec

These specifications control what the worker does for each request in a given engine step:

```python
@dataclass
class LoadSpec:
    vllm_cached_tokens: int      # Tokens already in vLLM's local cache
    lmcache_cached_tokens: int   # Total tokens available in LMCache
    can_load: bool               # Whether blocks are allocated and ready

@dataclass
class SaveSpec:
    skip_leading_tokens: int     # Tokens already saved (skip these)
    can_save: bool               # Whether the scheduler allows saving
```

### RequestTracker

Scheduler-side state tracking per request:

```python
@dataclass
class RequestTracker:
    req_id: str
    prompt_len: int
    token_ids: list[int]           # All tokens scheduled so far
    allocated_block_ids: list[int] # GPU block IDs
    num_saved_tokens: int          # How many tokens saved to LMCache
    is_decode_phase: bool
    skip_save: bool
    # ... plus multimodal and disagg fields
```

## Detailed KV Cache Movement

### Engine Step Lifecycle

Each vLLM engine step follows this sequence, with LMCache hooks at each stage:

```
┌─────────────────────── SCHEDULER PROCESS ───────────────────────┐
│                                                                  │
│  1. get_num_new_matched_tokens()  → Query LMCache for prefix    │
│  2. update_state_after_alloc()    → Confirm blocks allocated     │
│  3. build_connector_meta()        → Package LoadSpec/SaveSpec    │
│                                                                  │
│  ──── LMCacheConnectorMetadata sent to worker ────────────────  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────── WORKER PROCESS ──────────────────────────┐
│                                                                  │
│  4. start_load_kv()               → Begin loading from storage   │
│                                                                  │
│  For each transformer layer L:                                   │
│  │  5. wait_for_layer_load(L)     → Block until layer L ready   │
│  │  6. [Attention computation]                                   │
│  │  7. save_kv_layer(L)           → Save layer L to storage     │
│                                                                  │
│  8. wait_for_save()               → Finalize saves, unpin       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Prefill Phase: KV Cache Movement

During prefill, both loading (from LMCache) and saving (to LMCache) occur:

#### Step 1: Prefix Lookup (Scheduler)

```
Request arrives with token_ids = [t0, t1, t2, ..., t99]

Scheduler calls: lookup_client.lookup(token_ids)
  → LMCache returns: 64 tokens matched (chunks 0 and 1 of size 32)

Scheduler creates LoadSpec:
  vllm_cached_tokens = 0     # nothing in vLLM's local cache
  lmcache_cached_tokens = 64 # 64 tokens found in LMCache
  can_load = True             # blocks allocated

Result: Only tokens [t64 ... t99] need prefill computation.
        Tokens [t0 ... t63] will be loaded from LMCache.
```

#### Step 2: KV Loading (Worker — GPU ← Storage)

```
Worker receives LoadSpec for the request.

token_mask construction:
  Position:  [0  1  2 ... 63  64  65 ... 99]
  Mask:      [1  1  1 ... 1   0   0  ... 0 ]
             ↑ Load from LMCache    ↑ Will be computed fresh

Data flow (layer-wise mode):

  Storage (CPU/Disk/Remote)
       │
       ▼  lmcache_engine.retrieve_layer(token_ids, mask, layer=L)
  ┌─────────────┐
  │ LMCache     │   Reads chunk data for layer L
  │ Engine      │   Converts to GPU tensor format
  └──────┬──────┘
         │
         ▼  GPU Connector copies into vLLM's paged KV buffer
  ┌─────────────────────────────────┐
  │ vLLM GPU Paged KV Cache         │
  │                                  │
  │ Block 0: [t0  ... t15]  ← loaded│
  │ Block 1: [t16 ... t31]  ← loaded│
  │ Block 2: [t32 ... t47]  ← loaded│
  │ Block 3: [t48 ... t63]  ← loaded│
  │ Block 4: [t64 ... t79]  ← empty (will be computed)
  │ Block 5: [t80 ... t99]  ← empty (will be computed)
  └─────────────────────────────────┘
```

#### Step 3: Attention + Save (Worker — GPU → Storage)

For each transformer layer, the worker:
1. Waits for loaded KV to be ready (`wait_for_layer_load`)
2. Runs attention over ALL tokens (loaded + freshly computed)
3. Saves the newly computed KV back to LMCache (`save_kv_layer`)

```
Layer L attention computation:

  Input: All 100 tokens (64 loaded + 36 new)
  Output: KV for tokens [t64 ... t99] (new ones)

  save_kv_layer(layer=L):
    store_mask construction:
      Position:  [0  1 ... 63  64  65 ... 99]
      Mask:      [0  0 ... 0   1   1  ... 1 ]
                  ↑ Already in LMCache    ↑ Save these

    Data flow:

    ┌─────────────────────────────────┐
    │ vLLM GPU Paged KV Cache         │
    │ Block 4: [t64 ... t79] (new KV) │──┐
    │ Block 5: [t80 ... t99] (new KV) │──┤
    └─────────────────────────────────┘  │
                                          ▼
                            lmcache_engine.store_layer()
                                          │
                                          ▼
                            Storage (CPU/Disk/Remote)
                            Chunk 2: [t64 ... t95]  ← saved
                            (partial chunk t96-t99 may be discarded
                             depending on discard_partial_chunks config)
```

### Decode Phase: KV Cache Movement

During decode, the KV cache movement is minimal:

```
Decode step: Generate token t100

┌─ Scheduler ─────────────────────────────────┐
│ get_num_new_matched_tokens() → returns 0    │
│ (no new prefix hits expected during decode)  │
│                                              │
│ build_connector_meta():                      │
│   SaveSpec.skip_leading_tokens = 100         │
│   SaveSpec.can_save = save_decode_cache      │
└──────────────────────────────────────────────┘

┌─ Worker ─────────────────────────────────────┐
│ start_load_kv(): No loads needed             │
│                                              │
│ For each layer L:                            │
│   wait_for_layer_load(): No-op               │
│   [Attention: use existing KV + new t100]    │
│   save_kv_layer():                           │
│     If save_decode_cache=False → skip        │
│     If save_decode_cache=True:               │
│       Save only t100's KV to LMCache         │
│                                              │
│ wait_for_save(): Minimal cleanup             │
└──────────────────────────────────────────────┘
```

When `save_decode_cache=False` (the default), decode steps incur **zero LMCache overhead** — all KV stays on GPU only.

When `save_decode_cache=True`, each decode step saves one token's KV per layer to LMCache storage. This allows future requests that share the same generation prefix to skip recomputation.

### Summary: Data Flow Diagram

```
                    PREFILL                              DECODE
                    ──────                               ──────

   Storage ──retrieve──▶ GPU KV Cache              GPU KV Cache
   (CPU/Disk/            (loaded tokens)            (all tokens)
    Remote)                   │                          │
                              ▼                          ▼
                     Attention (all tokens)      Attention (1 new token)
                              │                          │
                              ▼                          ▼
   Storage ◀──store─── GPU KV Cache              GPU KV Cache
   (CPU/Disk/           (new tokens only)        (optionally save
    Remote)                                       new decode token)
```

## Configuration

### LMCache Config File

Set via the `LMCACHE_CONFIG_FILE` environment variable (YAML):

```yaml
chunk_size: 256
save_decode_cache: false
enable_async_loading: true
use_layerwise: true
enable_blending: false
local_cpu: true
max_local_cpu_size: 10737418240   # 10 GB
local_disk: null
remote_serde: null
enable_nixl: false
```

### vLLM KVTransferConfig

```python
--kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "use_native": true,
        "discard_partial_chunks": false,
        "skip_last_n_tokens": 0,
        "lmcache.save_decode_cache": true,
        "lmcache.use_layerwise": true
    }
}'
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 256 | Number of tokens per LMCache chunk |
| `save_decode_cache` | false | Save KV from decode tokens |
| `use_layerwise` | false | Enable per-layer async pipelining |
| `enable_async_loading` | false | Non-blocking prefix lookups |
| `skip_last_n_tokens` | 0 | Skip N trailing tokens in lookup |
| `discard_partial_chunks` | false | Drop incomplete chunks on save |
| `enable_blending` | false | Blend cached and fresh KV |
| `enable_nixl` | false | Enable NIXL remote transport |
| `kv_role` | — | `kv_producer`, `kv_consumer`, or `kv_both` |
