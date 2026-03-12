# KV Connector Engine Step Sequence

```
═══════════════════════════════════════════════════════════════════
  ENGINE CORE (core.py:386-402)
  ┌─ scheduler.schedule()
  │  scheduler.execute_model()
  │  scheduler.update_from_output()
  └─ return outputs
═══════════════════════════════════════════════════════════════════

╔═══════════════════════════════════════════════════════════════╗
║  PHASE 1: scheduler.schedule()                               ║
║  scheduler.py                                                ║
╚═══════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────┐
  │ 1a. Schedule RUNNING requests (handle preemptions etc)   │
  │     [no connector calls here]                            │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 1b. Per WAITING request loop                             │
  │                                                          │
  │  ┌─ If status == WAITING_FOR_REMOTE_KVS:                │
  │  │   _update_waiting_for_remote_kv()   ← :550           │
  │  │    └─ checks self.finished_recving_kv_req_ids         │
  │  │    └─ if ready: cache_blocks(), set num_computed →    │
  │  │       move to schedulable                   ← :2026   │
  │  │                                                       │
  │  ├─ get_computed_blocks (local prefix cache)   ← :608    │
  │  │                                                       │
  │  ├─ ★ connector.get_num_new_matched_tokens()   ← :614   │
  │  │   Args: (request, num_local_computed_tokens)          │
  │  │   Returns: (ext_tokens | None, load_kv_async: bool)   │
  │  │   • None → skip request, try later                    │
  │  │   • ext_tokens=0 → no offloaded hit                   │
  │  │   • ext_tokens>0, async=True → will load async        │
  │  │                                                       │
  │  ├─ If load_kv_async: num_new_tokens = 0       ← :650   │
  │  │   (no forward work, just allocate blocks)             │
  │  │                                                       │
  │  ├─ kv_cache_manager.allocate_slots()          ← :726   │
  │  │   (allocates GPU blocks for external tokens)          │
  │  │                                                       │
  │  ├─ ★ connector.update_state_after_alloc()     ← :751   │
  │  │   Args: (request, blocks, num_external_tokens)        │
  │  │   • Builds (src_spec=CPU, dst_spec=GPU) transfer spec │
  │  │   • Adds to self._reqs_to_load                       │
  │  │                                                       │
  │  └─ If load_kv_async:                          ← :769   │
  │     request.status = WAITING_FOR_REMOTE_KVS    ← :773   │
  │     (request does NOT enter self.running)                │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 1c. Build SchedulerOutput                      ← :860   │
  │                                                          │
  │  ★ connector.build_connector_meta(sched_output) ← :896  │
  │    • Computes reqs_to_store (which blocks to offload)    │
  │    • Packages reqs_to_load (computed in 1b)              │
  │    • Returns OffloadingConnectorMetadata                 │
  │    • Clears internal state (reqs_to_load)                │
  │    → attached to scheduler_output.kv_connector_metadata  │
  └──────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════╗
║  PHASE 2: Worker execute_model()                             ║
║  gpu_model_runner.py                                         ║
╚═══════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────┐
  │ 2a. _get_kv_connector_output() context manager   ← :84  │
  │     (kv_connector_model_runner_mixin.py)                 │
  │                                                          │
  │  ★ connector.bind_connector_metadata(meta)       ← :95  │
  │    • Stores metadata on connector instance               │
  │                                                          │
  │  ★ connector.start_load_kv(forward_context)      ← :101 │
  │    OffloadingConnectorWorker.start_kv_transfers():       │
  │    ├─ Submit deferred store jobs from LAST step  ← :625  │
  │    │  (offloading_connector.py)                          │
  │    └─ Submit new load jobs (CPU→GPU DMA async)   ← :630  │
  │       └─ SingleDirectionOffloadingHandler                │
  │          .transfer_async()                       ← :119  │
  │          (cpu_gpu.py)                                    │
  │          ├─ build src_to_dst block index mapping         │
  │          ├─ get CUDA stream from pool                    │
  │          ├─ stream.wait_event(last_transfer.end_event)   │
  │          └─ ops.swap_blocks() on the stream (async DMA)  │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 2b. Model forward pass                          ← :3620 │
  │     (gpu_model_runner.py)                                │
  │                                                          │
  │  Per attention layer (via @maybe_transfer_kv_layer       │
  │  decorator, kv_transfer_utils.py:14):                    │
  │                                                          │
  │    ★ connector.wait_for_layer_load(layer_name)   ← :50  │
  │      → NO-OP for OffloadingConnector             ← :158 │
  │        (offloading_connector.py)                         │
  │                                                          │
  │    [attention layer runs]                                │
  │                                                          │
  │    ★ connector.save_kv_layer(layer, kv, meta)    ← :56  │
  │      → NO-OP for OffloadingConnector             ← :165 │
  │        (offloading_connector.py)                         │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 2c. Exiting _get_kv_connector_output() context           │
  │     (kv_connector_model_runner_mixin.py:104)             │
  │                                                          │
  │  ★ connector.wait_for_save()                     ← :106 │
  │    OffloadingConnectorWorker.prepare_store_kv():          │
  │    └─ Enqueues store jobs to _unsubmitted_store_jobs     │
  │       (deferred to NEXT step's start_load_kv)    ← :646 │
  │                                                          │
  │  ★ connector.get_finished(finished_req_ids)      ← :109 │
  │    OffloadingConnectorWorker.get_finished():              │
  │    └─ Polls end_event.query() per transfer       ← :196 │
  │       (cpu_gpu.py, non-blocking CUDA event poll)         │
  │    Returns: (finished_sending, finished_recving)         │
  │                                                          │
  │  ★ connector.get_block_ids_with_load_errors()    ← :111 │
  │  ★ connector.get_kv_connector_stats()            ← :113 │
  │  ★ connector.get_kv_connector_kv_cache_events()  ← :114 │
  │  ★ connector.clear_connector_metadata()          ← :117 │
  │                                                          │
  │  → all packed into KVConnectorOutput                     │
  │  → attached to ModelRunnerOutput.kv_connector_output     │
  └──────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════╗
║  PHASE 3: scheduler.update_from_output()                     ║
║  scheduler.py:1291                                           ║
╚═══════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────┐
  │ 3a. Process model output per request             ← :1333│
  │                                                          │
  │  Per finished request:                                   │
  │    _free_request()                               ← :1783│
  │    └─ ★ connector.request_finished(req, blocks)  ← :1992│
  │         Returns: (delay_free, kv_transfer_params)        │
  │         • delay_free=True → blocks held until            │
  │           finished_sending reported                      │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 3b. Feed worker output back into connector       ← :1490│
  │                                                          │
  │  ★ connector.update_connector_output(kv_output)  ← :2054│
  │    OffloadingConnectorScheduler:                         │
  │    ├─ finished_sending → manager.complete_store() ← :505│
  │    └─ finished_recving → manager.complete_load()  ← :512│
  │                                                          │
  │  Scheduler bookkeeping:                                  │
  │    finished_recving → add to                             │
  │      self.finished_recving_kv_req_ids            ← :2062│
  │      (used by _update_waiting_for_remote_kv              │
  │       in next step's schedule())                         │
  │    finished_sending → _free_blocks()             ← :2069│
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │ 3c. Collect events                               ← :1498│
  │                                                          │
  │  ★ connector.take_events()                       ← :1499│
  │    Yields BlockStored / BlockRemoved events              │
  │    (for observability / KV cache event publishing)       │
  └──────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════
  → return EngineCoreOutputs to engine
  → next step begins from PHASE 1
═══════════════════════════════════════════════════════════════
```
