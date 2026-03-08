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

### 7. Prefill vs Decode Differentiation

LMCache distinguishes prefill and decode requests at multiple levels throughout the pipeline:

#### Phase Detection

- **`RequestTracker.is_decode_phase`**: Set to `True` when a request is re-scheduled with exactly 1 new token (`len(new_token_ids) == 1`), indicating the transition from prefill to decode. Starts as `False` for new requests.
- **`ReqMeta.is_last_prefill`**: Set to `True` when all prompt tokens have been scheduled (`input_token_len == tracker.prompt_len`). This marks the final prefill step before decode begins.

#### Behavioral Differences

| Aspect | Prefill | Decode |
|--------|---------|--------|
| KV Load from LMCache | Yes (if cache hit found) | No (`load_spec` is always `None`) |
| KV Save to LMCache | Yes (always, unless explicitly skipped) | Only if `save_decode_cache=True` |
| Chunk alignment on save | Truncated to chunk boundary (unless last prefill with `discard_partial_chunks=False`) | Tokens accumulate until a chunk boundary is reached |
| Disagg transfer | Triggered on **every** prefill step (each chunk); `is_last_prefill` flag signals completion to the decoder | Not applicable |

#### Save Gating Logic

The `skip_save` decision in `ReqMeta.from_request_tracker()` explicitly checks the phase:

```python
skip_save = tracker.disagg_spec is None and (
    tracker.skip_save
    or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
    or (tracker.is_decode_phase and not save_decode_cache)  # decode-specific gate
    or request_skip
)
```

Note the `tracker.disagg_spec is None` guard: when a `DisaggSpec` is present, the entire `skip_save` expression is **always `False`**, meaning saving (i.e., transfer) is **never skipped** for disaggregated requests. This is because in disagg mode, "saving" is actually a remote transfer to the decoder — it must happen on every prefill step regardless of other skip conditions.

When `save_decode_cache=False` (the default), all decode steps are skipped for saving, incurring zero LMCache overhead. When `save_decode_cache=True`, each decode step saves the new token's KV (subject to chunk boundary alignment), enabling future requests to reuse generated output prefixes.

#### Chunk Alignment Differences

During chunked prefill (not the last chunk), token counts are always truncated to the chunk boundary since more tokens will arrive in the next step. On the last prefill step, partial chunks can optionally be saved (controlled by `discard_partial_chunks`). During decode, tokens accumulate one-by-one and are only flushed to storage when a full chunk boundary is crossed.

#### Disaggregated Transfer: Step-by-Step Detail

In a disaggregated (prefill/decode split) setup, the "store" operation on the prefiller is actually a **remote transfer** to the decoder. This happens incrementally starting from the **first** prefill step, not just the last one. Here is what happens at each stage:

**1. Request arrival (Scheduler — `update_state_after_alloc`):**
The incoming request carries `kv_transfer_params` with a `disagg_spec` dict containing the decoder's connection info (`receiver_host`, `receiver_init_port`, `receiver_alloc_port`). The scheduler extracts this into a `DisaggSpec` object and stores it in `tmp_disagg_tracker`.

**2. Request tracking (Scheduler — `build_connector_meta`):**
When the `RequestTracker` is created for the new request, the `DisaggSpec` is popped from `tmp_disagg_tracker` and attached to the tracker. It flows through to `ReqMeta` and ultimately into the `LMCacheConnectorMetadata` sent to the worker.

**3. Every prefill step (Worker — `wait_for_save`):**
On each prefill engine step, `wait_for_save()` processes the request:

```
For a kv_producer with disagg_spec present:

a) skip_leading_tokens = min(save_spec.skip_leading_tokens,
                              disagg_spec.num_transferred_tokens)
   → On the first step, both are 0, so all tokens are transferred.
   → On subsequent steps, only new (untransferred) tokens are sent.

b) Build store_mask: mark already-transferred tokens as False.

c) Check is_last_prefill:
   - If True:  set disagg_spec.is_last_prefill = True
               (signals to LMCache engine that this is the final chunk)
   - If False: truncate token_ids to chunk-aligned length
               (partial chunks are not transferred mid-prefill)

d) Call lmcache_engine.store(
       token_ids, mask=store_mask, kvcaches=kvcaches,
       slot_mapping=slot_mapping,
       transfer_spec=request.disagg_spec,  ← triggers NIXL send
       ...
   )

e) Update disagg_spec.num_transferred_tokens = len(token_ids)
```

**4. How many chunks are transferred per step?**

Each step does **not** transfer exactly one chunk. It transfers **all chunk-aligned untransferred tokens at once** in a single `lmcache_engine.store()` call. The number of chunks per step depends on how many tokens the scheduler computes in that step relative to the chunk size.

The key calculation is:

```
tokens_available  = total tokens scheduled so far (cumulative)
chunk_aligned_len = tokens_available // chunk_size * chunk_size  (unless is_last_prefill)
already_sent      = disagg_spec.num_transferred_tokens
tokens_to_send    = chunk_aligned_len - already_sent
chunks_to_send    = tokens_to_send / chunk_size   (can be 0, 1, 2, or more)
```

If the scheduler computes 600 tokens in one step with `chunk_size=256`, that's 2 full chunks (512 tokens) transferred in a single call, with 88 leftover tokens deferred to the next step.

**5. Examples:**

**Example A: One chunk per step** (scheduler computes 256 tokens/step, `chunk_size=256`)

```
Step 1: Compute t0..t255 (256 tokens total)
  chunk_aligned = 256, already_sent = 0
  Transfer: t0..t255 (1 chunk)
  num_transferred_tokens = 256

Step 2: Compute t256..t511 (512 total)
  chunk_aligned = 512, already_sent = 256
  Transfer: t256..t511 (1 chunk)
  num_transferred_tokens = 512

Step 3: Compute t512..t700 (701 total, is_last_prefill=True)
  No truncation (last prefill) → transfer t512..t700 (189 tokens, partial chunk)
  disagg_spec.is_last_prefill = True
```

**Example B: Multiple chunks per step** (scheduler computes 600 tokens/step, `chunk_size=256`)

```
Step 1: Compute t0..t599 (600 tokens total)
  chunk_aligned = 512 (= 600 // 256 * 256)
  already_sent = 0
  Transfer: t0..t511 (2 chunks in one store() call)
  num_transferred_tokens = 512
  Leftover: t512..t599 deferred

Step 2: Compute t600..t700 (701 total, is_last_prefill=True)
  No truncation → transfer t512..t700 (189 tokens)
  This includes 88 leftover tokens from step 1 + 101 from step 2
  disagg_spec.is_last_prefill = True
```

**Example C: Zero chunks transferred** (scheduler computes 200 tokens, `chunk_size=256`)

```
Step 1: Compute t0..t199 (200 tokens total)
  chunk_aligned = 0 (= 200 // 256 * 256)
  Nothing to transfer! (skip_leading_tokens == len(token_ids) → continue)
  num_transferred_tokens = 0

Step 2: Compute t200..t499 (500 total)
  chunk_aligned = 256
  Transfer: t0..t255 (1 chunk, includes tokens from step 1)
  num_transferred_tokens = 256

Step 3: Compute t500..t700 (701 total, is_last_prefill=True)
  No truncation → transfer t256..t700 (445 tokens, ~2 chunks)
  disagg_spec.is_last_prefill = True
```

**6. Data flow visualization (multi-chunk example):**

```
Prefiller GPU                              Decoder GPU
┌──────────────────┐                      ┌──────────────────┐
│ Step 1:          │                      │                  │
│ Computed 600 tok │                      │                  │
│ t0..t511         │── NIXL (2 chunks) ─▶ │ t0..t511         │
│ (chunks 0 & 1)   │   is_last=False      │                  │
│                  │   transferred=512    │                  │
│ t512..t599       │   (deferred)         │                  │
├──────────────────┤                      ├──────────────────┤
│ Step 2:          │                      │                  │
│ Computed 101 tok │                      │                  │
│ t512..t700       │── NIXL (partial) ──▶ │ t512..t700       │
│ (189 tokens)     │   is_last=TRUE       │ All KV received! │
│                  │   transferred=701    │                  │
└──────────────────┘                      └──────────────────┘
                                          Decoder begins
                                          decode phase
```

**Key points:**
- Transfer starts on the **first** prefill step, not just the last.
- Each step transfers **all chunk-aligned untransferred tokens** — this can be 0, 1, or multiple chunks depending on how many tokens the scheduler computed.
- The `store()` call receives the full cumulative token sequence but uses `store_mask` and `offset` to only read/send the new portion.
- The `is_last_prefill` flag on the `DisaggSpec` tells the LMCache engine (and the decoder) that the KV transfer is complete and decoding can begin.
- The `skip_save` logic is **bypassed** when `disagg_spec` is present — the transfer always happens regardless of other skip conditions.
- `discard_partial_chunks` must be `False` for disagg to work correctly, otherwise the tail tokens after the last chunk boundary are silently dropped.

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

### Disaggregated Prefill/Decode Scenario

In a disaggregated setup, prefill and decode run on **separate vLLM instances** connected via NIXL (Network Interface eXtended Library). A proxy routes each request to the prefiller first, then to the decoder.

```
                    PREFILLER                         DECODER
                (kv_producer)                     (kv_consumer)
              ┌──────────────────┐             ┌──────────────────┐
  Request ──▶ │ Compute KV for   │   NIXL      │ Load KV from     │
  (via proxy) │ all prompt tokens │───────────▶ │ remote prefiller │
              │                  │  transfer   │                  │
              │ Return: done     │             │ Run decode only  │──▶ Output
              └──────────────────┘             └──────────────────┘
```

#### DisaggSpec

Each disaggregated request carries a `DisaggSpec` that tells the prefiller where to send KV:

```python
@dataclass
class DisaggSpec:
    req_id: str                  # Request being processed
    receiver_id: str             # Unique ID of the decoder instance
    receiver_host: str           # Hostname/IP of the decoder
    receiver_init_port: int      # NIXL handshake port
    receiver_alloc_port: int     # NIXL data transfer port
    is_last_prefill: bool        # True on the final prefill iteration
    num_transferred_tokens: int  # Tokens transferred so far
```

The `DisaggSpec` is extracted from request-level `kv_transfer_params` on the scheduler side and attached to the `RequestTracker`.

#### Prefiller Side (kv_producer)

The prefiller computes KV and sends it to the decoder via NIXL:

```
1. Scheduler: Receive request with disagg_spec in kv_transfer_params
   └─ Create DisaggSpec with decoder host/port info
   └─ Attach to RequestTracker

2. Worker: Compute attention for all prompt tokens (normal prefill)

3. Worker: wait_for_save()
   └─ For each request with disagg_spec:
      └─ lmcache_engine.store(
            token_ids, mask, kvcaches, slot_mapping,
            transfer_spec=disagg_spec   ← triggers NIXL send
         )
      └─ Update disagg_spec.num_transferred_tokens
      └─ Set disagg_spec.is_last_prefill = True on final iteration

4. KV data flows:  Prefiller GPU → NIXL → Decoder GPU
```

Key behavior: when `disagg_spec` is present, **saving is never skipped** — the skip_save logic is overridden because the KV must always be transferred to the decoder.

#### Decoder Side (kv_consumer)

The decoder loads KV from the remote prefiller and runs decode:

```
1. Scheduler: get_num_new_matched_tokens()
   └─ Query LMCache for remotely available KV
   └─ Create LoadSpec with lmcache_cached_tokens = prompt_length

2. Worker: start_load_kv()
   └─ lmcache_engine.retrieve() or retrieve_layer()
   └─ KV loaded from remote prefiller via NIXL into local GPU pages

3. Worker: For each layer L:
   └─ wait_for_layer_load(L): Block until remote KV arrives
   └─ Attention: Use loaded KV (no local prefill computation needed)

4. Worker: wait_for_save()
   └─ kv_role == "kv_consumer" → skip save (decoder doesn't store back)

5. Proceed to decode phase with all KV loaded
```

#### Multi-Chunk Prefill

For long prompts that span multiple engine steps, the prefiller transfers KV incrementally:

```
Step 1: Prefill tokens [t0 ... t255]
  └─ Store + transfer chunk 0 via NIXL
  └─ disagg_spec.num_transferred_tokens = 256

Step 2: Prefill tokens [t256 ... t511]
  └─ Store + transfer chunk 1 via NIXL
  └─ disagg_spec.num_transferred_tokens = 512

Step 3: Prefill tokens [t512 ... t700]  (last prefill)
  └─ Store + transfer chunk 2 via NIXL
  └─ disagg_spec.is_last_prefill = True
  └─ Decoder can begin decoding
```

#### Example Deployment

```bash
# Prefiller instance (kv_producer)
UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_CONFIG_FILE=prefill_config.yaml \
vllm serve $MODEL --port 8100 \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "discard_partial_chunks": false,
      "lmcache_rpc_port": "producer1"
    }
  }'

# Decoder instance (kv_consumer)
UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_CONFIG_FILE=decode_config.yaml \
vllm serve $MODEL --port 8200 \
  --kv-transfer-config '{
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "discard_partial_chunks": false,
      "lmcache_rpc_port": "consumer1"
    }
  }'

# Proxy routes requests: client → prefiller → decoder → client
```

The prefiller's LMCache config enables NIXL as sender (`nixl_role: "sender"`, `enable_nixl: true`), while the decoder's config sets it as receiver (`nixl_role: "receiver"`).

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
## Why LMCache Disagg Requires All KV Before Decoding

In vLLM's disaggregated prefill/decode architecture, the **NIXL connector** allows the decoder to start processing other requests concurrently while KV is being transferred from the prefiller. The **LMCache connector** does not — the decoder only begins decoding a request after all of its KV cache has been fully transferred. This section explains why.

### Comparison with NIXL Connector

| Aspect | NIXL Connector | LMCache Connector |
|--------|---------------|-------------------|
| Transfer mechanism | Direct GPU-to-GPU RDMA | Chunk-based storage engine (CPU/disk/remote) |
| Decoder accepts request before prefill completes | Yes | No |
| Transfer initiation | Non-blocking `nixl_xfer` | Synchronous `lmcache_engine.retrieve()` |
| Completion tracking | Polls NIXL transfer handles via `get_finished()` | `get_finished()` returns `(None, None)` |
| Scheduler state used | `WAITING_FOR_REMOTE_KVS` | Not used |
| Decoder serves other requests during transfer | Yes | N/A (decoder doesn't see the request yet) |

### Root Causes

#### 1. Chunk-Based Storage Abstraction vs. Direct Memory Transport

NIXL operates at the **memory transport** level. During `register_kv_caches()`, the NIXL connector registers the actual GPU paged KV buffer memory regions with the NIXL library. When the decoder needs KV from the prefiller, it performs an RDMA read that lands data **directly into vLLM's paged KV cache** — same memory layout, zero intermediate copies.

LMCache operates at a **cache storage** level. It adds an indirection layer: the prefiller calls `lmcache_engine.store()` which serializes KV into LMCache's chunk-based format (potentially involving CPU memory, disk, or remote backends). The decoder calls `lmcache_engine.retrieve()` which reads from storage and copies data into vLLM's GPU pages through LMCache's GPU connector. Data does not land directly in vLLM's paged buffer — it passes through LMCache's internal buffer first.

#### 2. Synchronous Retrieve API

LMCache's `start_load_kv()` calls `lmcache_engine.retrieve()`, which is a **blocking** call that copies all KV for the request into GPU pages immediately:

```python
# vllm_v1_adapter.py — start_load_kv()
ret_token_mask = self.lmcache_engine.retrieve(
    tokens[:lmcache_cached_tokens],
    token_mask[:lmcache_cached_tokens],
    kvcaches=kvcaches,
    slot_mapping=slot_mapping[:lmcache_cached_tokens],
    request_configs=request.request_configs,
    req_id=request.req_id,
)
```

There is no "initiate transfer + poll for completion" pattern. The layerwise mode (`retrieve_layer`) uses a generator for per-layer pipelining but still operates synchronously within the forward pass.

In contrast, NIXL's `start_load_kv()` posts a non-blocking transfer and returns immediately:

```python
# nixl_connector.py — start_load_kv()
def start_load_kv(self, metadata: NixlConnectorMetadata):
    """Start loading by triggering non-blocking nixl_xfer.
    We check for these trnxs to complete in each step()."""
```

#### 3. No Async Completion Tracking

The LMCache connector's `get_finished()` always returns `(None, None)`:

```python
# vllm_v1_adapter.py
def get_finished(
    self, finished_req_ids: set[str]
) -> tuple[set[str] | None, set[str] | None]:
    return None, None
```

The scheduler relies on `get_finished()` returning completed request IDs to move requests from `WAITING_FOR_REMOTE_KVS` back to `WAITING` (see `_update_from_kv_xfer_finished()` and `_update_waiting_for_remote_kv()` in `scheduler.py`). Without this, the async scheduling flow is inoperative.

NIXL's worker tracks in-flight transfers in `_recving_transfers` and reports completions through `get_finished()` → `_pop_done_transfers()`.

#### 4. `get_num_new_matched_tokens()` Returns `False` for Async

The second return value of `get_num_new_matched_tokens()` signals whether the KV load will happen asynchronously. LMCache always returns `False`:

```python
# lmcache_connector.py
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    return self._lmcache_engine.get_num_new_matched_tokens(
        request, num_computed_tokens
    ), False  # ← always synchronous
```

When the scheduler sees `False`, it schedules the request immediately into the `RUNNING` state rather than placing it in `WAITING_FOR_REMOTE_KVS`. NIXL returns `True`, which triggers the async path:

```python
# scheduler.py
if load_kv_async:
    skipped_waiting_requests.prepend_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    continue
```

#### 5. Proxy Serializes the Flow

The LMCache disagg proxy `await`s the prefiller's full response before forwarding to the decoder:

```python
# disagg_proxy_server.py
# Send to prefiller and WAIT for completion
await send_request_to_service(
    app.state.prefill_client, "/completions", req_data
)
# Only THEN stream from decoder
return StreamingResponse(
    stream_service_response(app.state.decode_client, "/completions", req_data)
)
```

Even if the connector supported async loading, the decoder would not see the request until the prefiller has finished all prefill steps and KV transfers.

### What Would Be Needed for Concurrent Operation

To enable LMCache disagg to overlap prefill and decode (similar to NIXL):

1. **Async retrieve API in LMCacheEngine**: An "initiate + poll" interface (e.g., `retrieve_async()` + `is_retrieve_done()`) instead of the blocking `retrieve()`.
2. **Return `True` for async in `get_num_new_matched_tokens()`**: So the scheduler uses the `WAITING_FOR_REMOTE_KVS` path.
3. **Implement `get_finished()`**: Track in-flight transfers and report completed request IDs.
4. **Modify the proxy**: Send to both prefiller and decoder concurrently, or use `kv_transfer_params` to coordinate (as NIXL does).
5. **Direct GPU page writes**: Bypass LMCache's intermediate buffer and write RDMA data directly into vLLM's registered KV cache pages.