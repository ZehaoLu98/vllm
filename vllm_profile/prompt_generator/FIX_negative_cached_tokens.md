# Fix: Prometheus Counter Negative Increment with LMCache

## Error

```
ValueError: Counters can only be incremented by non-negative amounts.
```

Full traceback:

```
File "vllm/v1/engine/async_llm.py", line 700, in output_handler
    logger_ref[0].record(...)
File "vllm/v1/metrics/loggers.py", line 1316, in record
    stat_logger.record(...)
File "vllm/v1/metrics/loggers.py", line 1123, in record
    self.counter_prompt_tokens_cached[engine_idx].inc(pts.cached_tokens)
File "prometheus_client/metrics.py", line 339, in inc
    raise ValueError('Counters can only be incremented by non-negative amounts.')
```

## Root Cause

`Request.num_cached_tokens` is initialized to `-1` as a sentinel value meaning "not yet determined" (see `vllm/v1/request.py:150`). The sentinel is normally replaced with the actual count when a request is scheduled (`scheduler.py:806-807`).

However, there are code paths where `num_cached_tokens` remains `-1` when it reaches the output:

1. **Failed KV load path** (`scheduler.py:~1471`): When a KV transfer fails and `kv_load_failure_policy="fail"` (the default), the request is terminated immediately. The `EngineCoreOutput` is created with `num_cached_tokens=request.num_cached_tokens`, which may still be `-1` if the KV transfer never completed successfully.

2. **General propagation**: Any code path that emits an `EngineCoreOutput` before the sentinel is replaced will pass `-1` into `PromptTokenStats.update_from_output()`, which accumulates it into `cached_tokens`. A Prometheus Counter rejects the resulting negative value.

## Data Flow

```
Request.num_cached_tokens = -1  (initialized in request.py)
        ↓
scheduler creates EngineCoreOutput(num_cached_tokens=-1)
        ↓
output_processor._update_stats_from_output() called with is_prefilling=True
        ↓
PromptTokenStats.update_from_output(num_cached_tokens=-1)
        → self.cached_tokens += -1  → cached_tokens becomes -1
        ↓
loggers.py: counter.inc(pts.cached_tokens)  → inc(-1) → ValueError
```

## Fix Applied

Two changes in the installed vllm package (`site-packages/vllm/`):

### 1. Scheduler: clamp sentinel in failed-KV-load output (`v1/core/sched/scheduler.py`)

```python
# Before
num_cached_tokens=request.num_cached_tokens,

# After
num_cached_tokens=max(0, request.num_cached_tokens),
```

### 2. Stats: defense-in-depth clamp (`v1/metrics/stats.py`, `PromptTokenStats.update_from_output`)

```python
def update_from_output(self, num_cached_tokens, ...):
    # Added: clamp sentinel value (-1 means "not yet determined") to 0.
    num_cached_tokens = max(0, num_cached_tokens)
    ...
```

## Reproduction Context

- vllm server started with `--kv-offloading-backend lmcache --kv-offloading-size 120 --disable-hybrid-kv-cache-manager`
- No preemptions observed in the server log
- Error triggered during high-concurrency benchmark (360 prompts, `--gpu-memory-utilization 0.42`)
- The lmcache connector uses sync loading (`enable_async_loading=False` by default, `load_kv_async=False`)

## Files Modified

- `.venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py` (line ~1471)
- `.venv/lib/python3.12/site-packages/vllm/v1/metrics/stats.py` (line ~278)

## Notes

- These edits are in the installed package, not the vllm source tree. They will be lost on reinstall.
- A proper upstream fix should ensure `num_cached_tokens` is never `-1` when emitted in any `EngineCoreOutput`.
