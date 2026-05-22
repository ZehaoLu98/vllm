# Executor architecture: UniProcExecutor, MultiprocExecutor, and disaggregated PD

This document explains how vLLM's v1 executors dispatch work to workers, how
the shared-memory `MessageQueue` between the executor and worker processes is
built, what role ZMQ plays inside it, and how disaggregated prefill/decode
(PD) is layered on top.

All references are to files under `vllm/` unless otherwise noted.

---

## 1. The executor contract

Both executors implement the abstract `Executor` interface in
[abstract.py](../../vllm/v1/executor/abstract.py). The class
`Executor.get_class()` selects a concrete executor based on the
`distributed_executor_backend` config string
([abstract.py:46-86](../../vllm/v1/executor/abstract.py#L46-L86)):

- `"uni"` → `UniProcExecutor`
- `"mp"` → `MultiprocExecutor`
- `"ray"` → `RayDistributedExecutor`
- `"external_launcher"` → `ExecutorWithExternalLauncher`

Every public operation funnels through a single primitive:

```python
def collective_rpc(self, method, timeout=None, args=(), kwargs=None,
                   non_block=False) -> list | Future[list]
```

See [abstract.py:141-191](../../vllm/v1/executor/abstract.py#L141-L191).
`method` can be either a string (looked up via `getattr` on the worker) or a
callable (cloudpickled and shipped to the worker). The result is a list with
one entry per worker.

All other operations on the abstract class — `execute_model`,
`determine_available_memory`, `add_lora`, `sleep`, etc. — are thin wrappers
that call `collective_rpc("method_name", ...)`. See for example:

- [abstract.py:135-139](../../vllm/v1/executor/abstract.py#L135-L139) — `determine_available_memory`
- [abstract.py:210-216](../../vllm/v1/executor/abstract.py#L210-L216) — `execute_model`
- [abstract.py:285-301](../../vllm/v1/executor/abstract.py#L285-L301) — `add_lora`, `remove_lora`, `pin_lora`, `list_loras`

The only thing each subclass really specializes is **how to dispatch a call
to N workers and collect N results**.

The worker on the other side is `WorkerWrapperBase` from
[worker_base.py](../../vllm/v1/worker/worker_base.py) — the wrapper holds the
actual `Worker` (which owns the model, KV cache, etc.) and exposes methods
like `execute_model`, `init_device`, `load_model`,
`determine_available_memory`, etc.

---

## 2. UniProcExecutor — single worker, same process

File: [uniproc_executor.py](../../vllm/v1/executor/uniproc_executor.py)

This is the "no distribution" case: TP=1, PP=1, one GPU, one process.
Everything happens in the engine process itself.

### Initialization — [uniproc_executor.py:27-51](../../vllm/v1/executor/uniproc_executor.py#L27-L51)

```python
self.driver_worker = WorkerWrapperBase(rpc_rank=0)
...
self.driver_worker.init_worker(all_kwargs=[kwargs])
self.driver_worker.init_device()
self.driver_worker.load_model()
```

It creates one `WorkerWrapperBase` in-process and calls
`init_worker`/`init_device`/`load_model` directly — no IPC, no subprocess, no
message queue. Even though there's only one rank, it still sets up a
`distributed_init_method` (TCP URL at
[uniproc_executor.py:55](../../vllm/v1/executor/uniproc_executor.py#L55)) so
torch.distributed APIs don't choke when called with world_size=1.

### Dispatch — [uniproc_executor.py:65-98](../../vllm/v1/executor/uniproc_executor.py#L65-L98)

```python
def collective_rpc(self, method, ..., non_block=False, single_value=False):
    if not non_block:
        result = run_method(self.driver_worker, method, args, kwargs)
        return result if single_value else [result]
    ...
```

A direct function call via `run_method`. No serialization, no queue, no
waiting. The `non_block=True` branch
([uniproc_executor.py:81-98](../../vllm/v1/executor/uniproc_executor.py#L81-L98))
is a tiny optimization for async scheduling: if the worker returned an
`AsyncModelRunnerOutput` (output still being copied D→H), the `.get_output()`
call is offloaded to a `ThreadPoolExecutor` so the engine can continue
scheduling the next batch while the previous one finishes copying.

### ExecutorWithExternalLauncher — [uniproc_executor.py:133-179](../../vllm/v1/executor/uniproc_executor.py#L133-L179)

A subclass for the `torchrun` use case: you launch N independent vLLM engines
yourself with torchrun, each runs its own `UniProcExecutor` but reads
`RANK`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT` from the environment, so they
form one TP group. Each engine sees only "its" worker but they collectively
run a TP-parallel model.

---

## 3. MultiprocExecutor — N workers, N subprocesses

File: [multiproc_executor.py](../../vllm/v1/executor/multiproc_executor.py)

This is the in-the-box distributed case: TP>1 and/or PP>1, multiple GPUs on
one node (or coordinated across nodes via the inner-DP group). The engine
spawns one Python subprocess per rank.

### Initialization — [multiproc_executor.py:103-236](../../vllm/v1/executor/multiproc_executor.py#L103-L236)

The pieces, in order:

1. **Shared memory queues.** Create an `rpc_broadcast_mq`
   ([multiproc_executor.py:145-151](../../vllm/v1/executor/multiproc_executor.py#L145-L151))
   — a shared-memory `MessageQueue` from
   [shm_broadcast.py](../../vllm/distributed/device_communicators/shm_broadcast.py).
   The executor writes to it; **all** workers read from it (broadcast
   pattern). Each worker also gets its own `worker_response_mq` for replies.

2. **Spawn workers.** The loop at
   [multiproc_executor.py:168-184](../../vllm/v1/executor/multiproc_executor.py#L168-L184)
   calls `WorkerProc.make_worker_process(...)` once per local rank. Each spawn:

    - Creates two pipes: a `ready_pipe` (child → parent "I'm up") and a
      `death_pipe` (when parent exits, child gets EOFError and shuts down).
      See [multiproc_executor.py:632-678](../../vllm/v1/executor/multiproc_executor.py#L632-L678).
    - Spawns a `multiprocessing.Process` running `WorkerProc.worker_main`
      ([multiproc_executor.py:775-877](../../vllm/v1/executor/multiproc_executor.py#L775-L877)).

3. **Wait for ready.**
   [`WorkerProc.wait_for_ready`](../../vllm/v1/executor/multiproc_executor.py#L701-L737)
   blocks until every child sends back a `{"status": "READY", "handle": ...}`
   message containing the handle to its response MQ.

4. **Health monitor thread.**
   [`start_worker_monitor`](../../vllm/v1/executor/multiproc_executor.py#L257-L288)
   starts a background thread that `wait()`s on every worker process's
   sentinel; if any dies it sets `is_failed`, calls `shutdown()`, and
   notifies the engine via `failure_callback`.

### Inside each worker — [multiproc_executor.py:775-877](../../vllm/v1/executor/multiproc_executor.py#L775-L877)

`worker_main` constructs a `WorkerProc`
([multiproc_executor.py:569-630](../../vllm/v1/executor/multiproc_executor.py#L569-L630)),
which:

- Builds a `WorkerWrapperBase` and calls `init_device()` + `load_model()` on
  it (loads model weights, joins NCCL TP group, etc.).
- Connects to the `rpc_broadcast_mq` using the handle the parent gave it
  ([multiproc_executor.py:541-543](../../vllm/v1/executor/multiproc_executor.py#L541-L543)).
- Sends `READY` back through `ready_pipe`.
- Enters
  [`worker_busy_loop`](../../vllm/v1/executor/multiproc_executor.py#L914-L940):

```python
while True:
    method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
        indefinite=True)
    func = (getattr(self.worker, method) if isinstance(method, str)
            else partial(cloudpickle.loads(method), self.worker))
    output = func(*args, **kwargs)
    if output_rank is None or self.rank == output_rank:
        self.handle_output(output)
```

Every worker blocks on the same broadcast queue. When one RPC arrives, every
worker runs the method, but only the designated reply rank writes the result
back.

### Dispatch — [multiproc_executor.py:329-397](../../vllm/v1/executor/multiproc_executor.py#L329-L397)

```python
self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
...
def get_response():
    responses = []
    for mq in response_mqs:
        status, result = mq.dequeue(timeout=...)
        ...
    return responses[0] if output_rank is not None else responses
```

The executor enqueues **once** to the broadcast MQ, then dequeues from
either all workers' response MQs (true collective call) or just one
(`output_rank`-only). The latter is used by `execute_model`/`sample_tokens`
where only the first TP worker of the last PP stage holds the final output —
see `_get_output_rank` at
[multiproc_executor.py:474-488](../../vllm/v1/executor/multiproc_executor.py#L474-L488).

### Pipelining with `FutureWrapper`

When `non_block=True`, instead of waiting on the response, the executor
stashes the `(future, get_response)` pair in `futures_queue` and returns
immediately
([multiproc_executor.py:387-397](../../vllm/v1/executor/multiproc_executor.py#L387-L397)).
This is how PP gets `max_concurrent_batches = pp_size`
([multiproc_executor.py:468-472](../../vllm/v1/executor/multiproc_executor.py#L468-L472))
— the scheduler keeps pumping work in so the pipeline stays full. When the
user (or a later blocking call) finally awaits `future.result()`, the queue
is drained in order
([multiproc_executor.py:67-94](../../vllm/v1/executor/multiproc_executor.py#L67-L94)).

### Summary table

| | UniProc | Multiproc |
|---|---|---|
| Worker count | 1, in-process | `world_size`, each in its own subprocess |
| RPC transport | direct Python call | shared-memory `MessageQueue` broadcast |
| Method shipping | none | string name or cloudpickle'd callable |
| Result collection | direct return value | dequeue from per-worker response MQ |
| Concurrency | thread for async output | `FutureWrapper` queue for PP pipelining |
| Failure handling | none needed | sentinel-watching monitor thread |
| Supports PP | no (`supports_pp=False`) | yes (`supports_pp=True`) |

---

## 4. MessageQueue — the SHM ring buffer + ZMQ hybrid

File: [shm_broadcast.py](../../vllm/distributed/device_communicators/shm_broadcast.py)

The `MessageQueue` class is the IPC primitive that connects the
`MultiprocExecutor` (writer) to its `WorkerProc` subprocesses (readers).
It's a hybrid: **shared memory** for fast on-node transport, **ZMQ sockets**
for cross-node transport and wakeups.

### Two queues, two directions

From
[multiproc_executor.py:145-151](../../vllm/v1/executor/multiproc_executor.py#L145-L151)
and
[multiproc_executor.py:545-546](../../vllm/v1/executor/multiproc_executor.py#L545-L546):

- **`rpc_broadcast_mq`** — executor writes, all workers read. Carries
  `(method, args, kwargs, output_rank)` RPC envelopes. One writer → N
  readers.
- **`worker_response_mq`** — each worker has one; worker writes, executor
  reads. Carries `(status, result)` replies. One writer → one reader (per
  queue).

A single RPC: enqueue once into the broadcast queue, dequeue from each (or
just one) response queue.

### Layout of the SHM ring buffer — [shm_broadcast.py:204-339](../../vllm/distributed/device_communicators/shm_broadcast.py#L204-L339)

When writer and readers are on the same node, both queues are backed by
`ShmRingBuffer`. It's a lock-free SPMC (single-producer, multi-consumer)
ring buffer carved out of POSIX shared memory.

```
+------------------------------+----------------------------------------+
| chunk0 | chunk1 | ... chunkN | metadata0 | metadata1 | ... metadataN  |
+------------------------------+----------------------------------------+
  max_chunks × max_chunk_bytes   max_chunks × (1 + n_reader) bytes
```

Each chunk's metadata is a tiny `1 + n_reader`-byte array:

```
[ written_flag | reader0_flag | reader1_flag | ... | readerN_flag ]
```

That's the whole synchronization protocol — no mutex, no condvar.

### Producer — [`acquire_write`](../../vllm/distributed/device_communicators/shm_broadcast.py#L536-L602)

1. Look at slot `current_idx`. If `written_flag == 1` and
   `sum(reader_flags) != n_reader`, the slot is still in flight → spin/yield
   (`sched_yield`).
2. Once free: zero `written_flag`, hand the data region to the caller via
   `yield buf`.
3. Caller fills the buffer. Then the producer **first** clears all reader
   flags, **then** sets `written_flag = 1`. Order matters — the comments at
   [shm_broadcast.py:582-585](../../vllm/distributed/device_communicators/shm_broadcast.py#L582-L585)
   explain why reversing it would create a visible "already read" state.
4. `memory_fence()` calls before/after the flag flips guarantee cross-CPU
   visibility
   ([shm_broadcast.py:55-73](../../vllm/distributed/device_communicators/shm_broadcast.py#L55-L73)).
   In Python they just acquire-and-release a `threading.Lock`, which is
   sequentially-consistent across processes thanks to underlying OS
   semantics.
5. Advance `current_idx`.

### Consumer — [`acquire_read`](../../vllm/distributed/device_communicators/shm_broadcast.py#L648-L701)

1. Look at slot `current_idx`. If `written_flag == 0` or
   `my_reader_flag == 1`, the slot isn't ready → wait via
   `SpinCondition.wait()`.
2. When ready: yield buf to caller, then set `my_reader_flag = 1`. Advance
   `current_idx`.

Every reader walks the same slots in the same order. Readers don't compete —
each has its own flag column.

### Spinning + sleeping: `SpinCondition` — [shm_broadcast.py:92-201](../../vllm/distributed/device_communicators/shm_broadcast.py#L92-L201)

A pure busy-spin burns CPUs when there's no traffic. A pure socket notify
adds latency under load. `SpinCondition` does both:

- For `busy_loop_s = 1` second after the last successful read, `wait()`
  just calls `sched_yield()` — cheap and very low latency.
- After 1 second of idle, it falls into `zmq.Poller.poll()` on a `SUB`
  socket. The producer's `enqueue` ends with
  `self._spin_condition.notify()`
  ([shm_broadcast.py:743](../../vllm/distributed/device_communicators/shm_broadcast.py#L743)),
  which sends a 1-byte `PUB` message. The socket uses `zmq.CONFLATE=1`
  ([shm_broadcast.py:130](../../vllm/distributed/device_communicators/shm_broadcast.py#L130))
  so only the latest notification is kept — no backlog of pings.
- A separate `PAIR` socket pair lets the worker's death-monitor thread
  interrupt an idle reader during shutdown — that's what
  [`MessageQueue.shutdown()`](../../vllm/distributed/device_communicators/shm_broadcast.py#L529-L534)
  triggers.

This is why the worker's busy loop at
[multiproc_executor.py:914-940](../../vllm/v1/executor/multiproc_executor.py#L914-L940)
can call `dequeue(indefinite=True)` without burning a core when idle but
still respond in microseconds when batches are flowing.

### Large payloads and remote readers

The SHM ring has a fixed `max_chunk_bytes` (default 24 MiB). Two cases need
an alternative path:

**Overflow** —
[shm_broadcast.py:722-743](../../vllm/distributed/device_communicators/shm_broadcast.py#L722-L743):

```python
if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
    with self.acquire_write(timeout) as buf:
        buf[0] = 1  # overflow
    self.local_socket.send_multipart(all_buffers, copy=False)
```

The writer still claims a ring slot — but writes a single overflow flag
byte. The actual payload goes over the ZMQ `XPUB`→`SUB` socket. Readers see
`buf[0] == 1` and read the real data from the socket
([shm_broadcast.py:754-768](../../vllm/distributed/device_communicators/shm_broadcast.py#L754-L768)).
This keeps message ordering correct.

**Cross-node readers** —
[shm_broadcast.py:408-426](../../vllm/distributed/device_communicators/shm_broadcast.py#L408-L426)
and
[shm_broadcast.py:745-746](../../vllm/distributed/device_communicators/shm_broadcast.py#L745-L746):
workers on a different host can't see the shared memory at all. For them
the writer sets up a TCP `XPUB` socket; every `enqueue` always sends through
it. Remote `dequeue` just reads from the socket. A single `MessageQueue` can
have a mix of local readers (SHM path) and remote readers (TCP path) at the
same time.

### Out-of-band serialization

[`enqueue`](../../vllm/distributed/device_communicators/shm_broadcast.py#L703-L746)
uses pickle protocol 5 with `buffer_callback`:

```python
def oob_callback(buf: PickleBuffer) -> bool:
    raw_buf = buf.raw()
    if len(raw_buf) < 1024 * 1024:
        return True       # inline (copy into the pickle stream)
    all_buffers.append(raw_buf)  # out-of-band (zero-copy ref)
    ...
    return False
```

Buffers >1 MiB (e.g. grammar bitmask tensors) are appended as raw byte
ranges instead of being copied into the pickle stream. The wire format:

```
[overflow byte][2 bytes: buffer count][len(buf0)][buf0][len(buf1)][buf1]...
```

with the corresponding `pickle.loads(..., buffers=...)` reconstruction in
`dequeue`. This is the same trick that makes `torch.Tensor` pickling cheap.

### Handshake — `wait_until_ready` — [shm_broadcast.py:496-527](../../vllm/distributed/device_communicators/shm_broadcast.py#L496-L527)

Sockets exist but a freshly bound `PUB` won't deliver to a subscriber that
hasn't fully connected yet (the "slow joiner" problem in ZMQ).
`wait_until_ready` handles this:

- The writer uses `XPUB` sockets, so it can `recv()` exactly one
  subscription notification per reader.
- Once all subscriptions are seen, it sends a `b"READY"` message; every
  reader blocks until it receives it.

After the worker subprocess sends `"READY"` through the pipe at
[multiproc_executor.py:826-833](../../vllm/v1/executor/multiproc_executor.py#L826-L833),
the executor and the worker both call `wait_until_ready()`
([multiproc_executor.py:215-219](../../vllm/v1/executor/multiproc_executor.py#L215-L219)
and
[multiproc_executor.py:837-839](../../vllm/v1/executor/multiproc_executor.py#L837-L839))
— the comment notes the order must be consistent to avoid deadlock.

### Sharing the queue with a child — the `Handle` object — [shm_broadcast.py:342-350](../../vllm/distributed/device_communicators/shm_broadcast.py#L342-L350)

The executor lives in process A. The worker is `fork`/`spawn`-ed into
process B. How does B access the same ring buffer? Through `Handle` — a
small pickleable struct containing:

- The shm name + dimensions (`buffer_handle`), so `ShmRingBuffer(name=...)`
  in the child opens the existing block instead of creating a new one. See
  [shm_broadcast.py:281-303](../../vllm/distributed/device_communicators/shm_broadcast.py#L281-L303).
- The ZMQ socket addresses (`local_subscribe_addr`, `local_notify_addr`,
  `remote_subscribe_addr`).

The executor calls `self.rpc_broadcast_mq.export_handle()` at
[multiproc_executor.py:151](../../vllm/v1/executor/multiproc_executor.py#L151)
and passes the handle to each child as `input_shm_handle`. The child calls
`MessageQueue.create_from_handle(handle, rank)`
([shm_broadcast.py:448-494](../../vllm/distributed/device_communicators/shm_broadcast.py#L448-L494))
— which constructs a `MessageQueue` in reader mode, attaches to the
existing shm, and connects to the writer's sockets.

Each worker also creates its own response queue (`MessageQueue(1, 1)` at
[multiproc_executor.py:546](../../vllm/v1/executor/multiproc_executor.py#L546))
and ships its handle back to the executor through the ready pipe
([multiproc_executor.py:826-833](../../vllm/v1/executor/multiproc_executor.py#L826-L833)).
The executor reconstructs each response queue in reader mode at
[multiproc_executor.py:680-699](../../vllm/v1/executor/multiproc_executor.py#L680-L699).

### Why mix ZMQ with shared memory

1. **Wakeup.** SHM has no built-in "data ready" signal. You'd have to
   spin-poll forever. The `PUB`/`SUB` ping at
   [shm_broadcast.py:743](../../vllm/distributed/device_communicators/shm_broadcast.py#L743)
   is what lets idle readers actually sleep.
2. **Bounded slot size.** Ring chunks are fixed at construction. Anything
   larger uses the `XPUB`→`SUB` socket as overflow — no need to pre-size
   for the worst case.
3. **One API for local and remote.** A `MessageQueue` with both local and
   remote readers writes to the ring + emits one `send_multipart` on the
   remote `XPUB`. Remote workers don't care that SHM exists.

---

## 5. ZMQ sockets in detail

ZeroMQ (`pyzmq`, imported at
[shm_broadcast.py:17](../../vllm/distributed/device_communicators/shm_broadcast.py#L17))
is a high-performance messaging library — "sockets, but with built-in
patterns". A ZMQ socket has:

- **A type** (`PUB`, `SUB`, `XPUB`, `PAIR`, `REQ`, `REP`, `PUSH`, `PULL`,
  etc.) — determines the messaging pattern.
- **A transport** — pluggable: `tcp://`, `ipc://` (Unix domain socket),
  `inproc://` (intra-process between threads). One side does `bind()`, the
  other `connect()`.
- **Automatic framing** — `send()`/`recv()` operate on whole messages, not
  byte streams. `send_multipart()` sends a list of frames as one logical
  message.
- **Internal buffering and background I/O threads** — your `send()` doesn't
  necessarily go on the wire immediately; it's queued.

### Socket types used in `shm_broadcast.py`

**`PUB` / `SUB` — broadcast (one-to-many)**

A `PUB` socket fans out every message to every connected `SUB`. Subscribers
filter by topic prefix; `setsockopt_string(SUBSCRIBE, "")` means "subscribe
to everything". Key gotchas:

- `SUB` drops messages it isn't subscribed to, silently.
- `PUB` drops messages when the send queue is full (high water mark,
  `SNDHWM`). Fire-and-forget.
- "Slow joiner" problem: messages sent before a subscriber finishes
  connecting are lost.

Used in `SpinCondition`
([shm_broadcast.py:126-153](../../vllm/distributed/device_communicators/shm_broadcast.py#L126-L153))
for the wakeup ping. The writer's `notify()` sends a single byte; idle
readers wake on the `SUB` side. Drops are fine — the ring-buffer flags are
the source of truth, the ping is just "go check the buffer again." That's
also why the reader sets `zmq.CONFLATE=1`
([shm_broadcast.py:130](../../vllm/distributed/device_communicators/shm_broadcast.py#L130)):
it only ever needs to know "there was at least one notification."

**`XPUB` / `SUB` — broadcast with subscription visibility**

`XPUB` is `PUB` plus one feature: when a `SUB` subscribes or unsubscribes,
the `XPUB` receives a special message describing that event, readable with
`recv()`. With `XPUB_VERBOSE=True` you see every subscription, not just the
first per topic.

Used by the main data sockets at
[shm_broadcast.py:384-391](../../vllm/distributed/device_communicators/shm_broadcast.py#L384-L391)
and
[shm_broadcast.py:414-422](../../vllm/distributed/device_communicators/shm_broadcast.py#L414-L422).
The writer needs `XPUB` (not plain `PUB`) specifically to solve the
slow-joiner problem in `wait_until_ready` — every reader's `SUB.connect()`
triggers a subscription event the writer can count.

**`PAIR` — strict 1-to-1**

`PAIR` sockets are the simplest: exactly one `PAIR` connects to exactly one
other, bidirectional, no filtering, no HWM drops in the normal case. Used
in `SpinCondition` for the cancellation channel at
[shm_broadcast.py:136-145](../../vllm/distributed/device_communicators/shm_broadcast.py#L136-L145).
Both ends live in the same process (the reader process), so the transport
is `inproc://...` — a pure in-memory queue between threads, no kernel
involvement.

### `zmq.Poller`

[`Poller`](../../vllm/distributed/device_communicators/shm_broadcast.py#L143-L145)
is ZMQ's `select()`/`epoll()` — register multiple sockets, then
`.poll(timeout_ms)` blocks until any of them has an event. This is what
lets one `wait()` call sleep on either "data available" or "please cancel"
or a timeout — all in a single syscall.

### Transports used

| Transport | Where it's used | What it does |
|---|---|---|
| `inproc://...` | Cancel `PAIR` sockets in `SpinCondition` | In-memory between threads of one process |
| `ipc://...` | `local_subscribe_addr`, `local_notify_addr` | Between processes on the same host; no TCP/IP |
| `tcp://ip:port` | `remote_subscribe_addr` | Cross-host |

### High water mark and `send_multipart`

[shm_broadcast.py:152](../../vllm/distributed/device_communicators/shm_broadcast.py#L152):
`setsockopt(zmq.SNDHWM, 1)` on the notify `PUB`. A `PUB` socket's send queue
is bounded by the HWM. With HWM=1, anything queued beyond that is dropped —
exactly what you want for "I just need to know there was a ping." Combined
with `CONFLATE=1` on the reader, the wakeup channel is essentially
memoryless: latest event wins.

A ZMQ message is logically a list of frames. `send_multipart([a, b, c])`
sends three byte buffers as one atomic message; the receiver gets the same
list back from `recv_multipart()`. `copy=False` keeps zero-copy references
into the original buffers. The overflow path at
[shm_broadcast.py:726](../../vllm/distributed/device_communicators/shm_broadcast.py#L726)
uses this:

```python
self.local_socket.send_multipart(all_buffers, copy=False)
```

`all_buffers[0]` is the pickle stream; `all_buffers[1:]` are the
out-of-band tensor buffers. `pickle.loads(recv, buffers=recv_oob)` at
[shm_broadcast.py:781](../../vllm/distributed/device_communicators/shm_broadcast.py#L781)
stitches them back into one Python object.

### Summary table

| Socket | Lives in | Purpose | Pattern |
|---|---|---|---|
| `local_socket` (`XPUB`) | writer | broadcast data + handshake to local readers | `XPUB`→`SUB`, `ipc://` |
| `local_socket` (`SUB`) | each local reader | receive overflow data | `SUB`, `ipc://` |
| `remote_socket` (`XPUB`) | writer | broadcast data to remote readers | `XPUB`→`SUB`, `tcp://` |
| `remote_socket` (`SUB`) | each remote reader | receive data (no SHM) | `SUB`, `tcp://` |
| `local_notify_socket` (`PUB`) | writer (inside `SpinCondition`) | ping idle readers to check SHM | `PUB`→`SUB`, `ipc://`, HWM=1 |
| `local_notify_socket` (`SUB`) | each local reader | receive notify pings | `SUB`, `ipc://`, CONFLATE=1 |
| `write_cancel_socket` (`PAIR`) | reader process, monitor thread | tell our own reader to exit | `PAIR`, `inproc://` |
| `read_cancel_socket` (`PAIR`) | reader process, reader thread | receive own-process cancel | `PAIR`, `inproc://` |

---

## 6. Disaggregated prefill/decode (PD)

The `MultiprocExecutor` itself **does not know anything about disaggregated
PD**. P/D disaggregation is layered on top via a separate component called
a `KVConnector` that lives inside each worker, talks to its peer on the
other instance through its own out-of-band channel (NIXL RDMA, NCCL P2P,
Mooncake, etc.), and surfaces transfer status through the existing
`collective_rpc` reply machinery. The executor gains exactly one extra
responsibility: aggregating the per-worker connector status reports into a
single result for the scheduler.

### The two instances and what they exchange

In disaggregated PD you run two (or more) vLLM engines:

- **Prefill (P) instance** — does the prompt forward pass, produces a full
  KV cache for the prompt. After the forward, it makes the KV blocks
  available to its KV connector.
- **Decode (D) instance** — receives those KV blocks (read-pull or push,
  depending on connector), then runs the decode steps producing tokens.

What flows between them is just KV cache blocks. Requests are routed by an
external router (not by vLLM); both engines see the same `request_id` and
the router stamps `kv_transfer_params` onto the request (peer engine ID,
remote block IDs, etc.).

### KVConnector — two halves of the same class

File: [base.py](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py)

**Scheduler side** (lives in the engine process, next to the scheduler) —
methods like `get_num_new_matched_tokens`, `update_state_after_alloc`,
`build_connector_meta`, `request_finished`
([base.py:416-516](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L416-L516)).

When a request arrives on the D side, the scheduler asks the connector "how
many tokens of this prompt are already in the remote KV cache?" — if many,
it skips re-prefilling them. When scheduling that request, the scheduler
attaches connector metadata (which blocks to read where from) to the
`SchedulerOutput`. When a request finishes on the P side, the connector
decides whether to free the blocks now or hold them until the remote read
is done.

**Worker side** (lives in each `WorkerProc`) — methods like `start_load_kv`,
`wait_for_save`, `save_kv_layer`, `get_finished`
([base.py:300-356](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L300-L356)).

The actual data plane. On the P worker, save KV blocks (push or expose for
pull). On the D worker, load incoming KV blocks into the local cache before
the forward pass starts.

The per-worker connector uses its **own** transport — NIXL/NCCL/Mooncake/etc.
— which is set up during worker init. None of this traffic flows through
the executor's `MessageQueue`. That queue carries control messages only; KV
bytes go side-band on the connector's network.

### Forward-pass integration — [kv_connector_model_runner_mixin.py:93-128](../../vllm/v1/worker/kv_connector_model_runner_mixin.py#L93-L128)

Every model runner forward goes through `_get_kv_connector_output`:

```python
@contextmanager
def _get_kv_connector_output(scheduler_output, ...):
    output = KVConnectorOutput()
    kv_connector = get_kv_transfer_group()
    kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

    # Kick off pulls/loads from remote peer. May be async.
    kv_connector.start_load_kv(get_forward_context())
    try:
        yield output                   # ← the model forward runs here
    finally:
        if wait_for_save:
            kv_connector.wait_for_save()       # block until our saves finish
        output.finished_sending, output.finished_recving = (
            kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
        output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        ...
```

One forward pass:

1. **Pre-forward**: bind metadata, `start_load_kv` (D side begins pulling
   blocks).
2. **Forward**: the model runs. Attention layers call `save_kv_layer` to
   push freshly computed KV (P side) — save is per-layer interleaved with
   compute, hidden behind the GPU.
3. **Post-forward**: `wait_for_save` ensures saves are flushed;
   `get_finished` returns IDs of requests whose transfers fully completed
   this step.

The returned `KVConnectorOutput` is attached to the `ModelRunnerOutput`
this worker enqueues onto its `worker_response_mq`.

### Executor role — aggregation

`MultiprocExecutor.execute_model` at
[multiproc_executor.py:296-306](../../vllm/v1/executor/multiproc_executor.py#L296-L306):

```python
def execute_model(self, scheduler_output, non_block=False):
    return self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        unique_reply_rank=self.output_rank,
        non_block=non_block,
        timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
        kv_output_aggregator=self.kv_output_aggregator,  # ← here
    )
```

Normally `execute_model` only reads back from `output_rank`. But when a KV
connector is configured, `kv_output_aggregator` is non-`None`
([abstract.py:273-277](../../vllm/v1/executor/abstract.py#L273-L277),
[engine/core.py:148-149](../../vllm/v1/engine/core.py#L148-L149)) and the
dispatch branches:

```python
# multiproc_executor.py:350-357
if kv_output_aggregator is not None:
    output_rank = None              # read from ALL workers
    aggregate = partial(kv_output_aggregator.aggregate,
                        output_rank=unique_reply_rank or 0)
else:
    output_rank = unique_reply_rank
    aggregate = lambda x: x
```

With PD enabled, the executor reads `ModelRunnerOutput` from **every**
worker (each one has its own `KVConnectorOutput` filled in), then passes
the list to
[`KVOutputAggregator.aggregate`](../../vllm/distributed/kv_transfer/kv_connector/utils.py#L48-L158).

### What aggregation does — [`KVOutputAggregator.aggregate`](../../vllm/distributed/kv_transfer/kv_connector/utils.py#L48-L158)

The interesting field is `finished_sending` / `finished_recving`. A
transfer for one request involves potentially all TP/PP workers — block by
block, layer by layer. A request can't be "fully sent/received" until all
workers (or however many the connector requires) report it done.

```python
def update_finished_set(req_ids, remaining_count_dict, finished_set):
    for req_id in req_ids or ():
        remaining = remaining_count_dict.get(req_id, self._expected_finished_count)
        remaining_count_dict[req_id] = remaining - 1
        if remaining_count_dict[req_id] == 0:
            finished_set.add(req_id)
            del remaining_count_dict[req_id]
```

`_expected_finished_count` is the number of workers that must each report a
request before the scheduler sees it as truly finished. Default is
`world_size`, but a connector can override via
[`get_finished_count()`](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L568)
— e.g. MLA replicates KV so only one rank needs to send. See
[utils.py:59-61](../../vllm/distributed/kv_transfer/kv_connector/utils.py#L59-L61)
where it's wired up.

The counters are **stateful across steps**. A request that's only partially
done after step N has its remaining-count carried into step N+1 in
`self._send_remaining_count`/`self._recv_remaining_count`. Only when it
hits 0 does it appear in `finished_sending`/`finished_recving` *and that's
the step the scheduler sees the finish*. This is the bridge between
"worker N reported X done in step T" and "the scheduler can free request
X's blocks now."

Other things aggregated in the same pass:

- `kv_connector_stats` — merged via `KVConnectorStats.aggregate`.
- `kv_cache_events` — concatenated across workers.
- `invalid_block_ids` — union of load failures across workers.

The aggregated result replaces the `kv_connector_output` field on the
chosen rank's `ModelRunnerOutput`
([utils.py:146-156](../../vllm/distributed/kv_transfer/kv_connector/utils.py#L146-L156)),
and that is what `collective_rpc` returns.

### Initialization handshake — peer discovery

Workers in P and D instances need to find each other before any transfer
can happen.
[engine/core.py:156-175](../../vllm/v1/engine/core.py#L156-L175):

```python
kv_connector = self.scheduler.get_kv_connector()
if kv_connector is not None:
    xfer_handshake_metadata = (
        self.model_executor.get_kv_connector_handshake_metadata()
    )
    if xfer_handshake_metadata:
        content: dict[int, Any] = {}
        for worker_dict in xfer_handshake_metadata:
            if worker_dict is not None:
                content.update(worker_dict)
        kv_connector.set_xfer_handshake_metadata(content)
```

`get_kv_connector_handshake_metadata` is itself a `collective_rpc`
([abstract.py:193-196](../../vllm/v1/executor/abstract.py#L193-L196)) —
every worker reports its connector handshake info (NIXL agent name, ZMQ
endpoints, NCCL handles, whatever the connector needs), the engine merges
them into one dict keyed by TP rank, and stashes it on the scheduler-side
connector. That metadata is then exchanged with the peer engine through a
side channel (the router or an explicit registration call) so each side
knows which remote endpoint to talk to for which rank.

### End-to-end picture for one D-side request

1. Router sends a request to the D engine with
   `kv_transfer_params = {peer_engine_id, peer_block_ids, ...}` attached
   (carried on the request through to `Request.kv_transfer_params`).
2. D scheduler calls the connector's `get_num_new_matched_tokens`
   ([scheduler.py:611](../../vllm/v1/core/sched/scheduler.py#L611) →
   [base.py:417](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L417))
   — connector says "all prompt tokens are remotely cached." Scheduler
   skips prefilling them.
3. D scheduler allocates local blocks and notifies the connector via
   `update_state_after_alloc`
   ([scheduler.py:749](../../vllm/v1/core/sched/scheduler.py#L749) →
   [base.py:452](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L452)),
   then transitions the request to `WAITING_FOR_REMOTE_KVS`
   ([scheduler.py:771](../../vllm/v1/core/sched/scheduler.py#L771)) so it
   stays out of the running set until its KV lands.
4. D scheduler calls `build_connector_meta(scheduler_output)`
   ([scheduler.py:917](../../vllm/v1/core/sched/scheduler.py#L917) →
   [base.py:473](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L473))
   — connector emits a `KVConnectorMetadata` saying "load remote blocks
   A→local block X, B→Y, ...". This is attached to
   `SchedulerOutput.kv_connector_metadata`. Note that this happens every
   step regardless of whether the request is in the running set — that's
   how a `WAITING_FOR_REMOTE_KVS` request still reaches the worker.
5. Executor's `execute_model`
   ([multiproc_executor.py:296-306](../../vllm/v1/executor/multiproc_executor.py#L296-L306))
   calls `collective_rpc("execute_model", args=(scheduler_output,), kv_output_aggregator=...)`
   ([multiproc_executor.py:329-397](../../vllm/v1/executor/multiproc_executor.py#L329-L397));
   the broadcast message is enqueued onto `rpc_broadcast_mq` at
   [multiproc_executor.py:363](../../vllm/v1/executor/multiproc_executor.py#L363).
6. Each D `WorkerProc` dequeues the broadcast from its `rpc_broadcast_mq`
   (see `WorkerProc.worker_busy_loop`,
   [multiproc_executor.py:914-940](../../vllm/v1/executor/multiproc_executor.py#L914-L940)).
   The message decodes to `(method_name="execute_model", args=(scheduler_output,))`;
   the worker dispatches to `self.worker.execute_model(scheduler_output)`,
   which in turn calls into the model runner. The runner wraps the forward
   in the `_get_kv_connector_output` context manager
   ([kv_connector_model_runner_mixin.py:93-128](../../vllm/v1/worker/kv_connector_model_runner_mixin.py#L93-L128)),
   driving the connector through its full per-step lifecycle:

   1. **`bind_connector_metadata(scheduler_output.kv_connector_metadata)`** —
      the worker-side connector reads the `KVConnectorMetadata` that the
      D scheduler attached in step 4. For this D request it now knows:
      *for each local block `X` allocated above, pull from peer engine
      `peer_engine_id`, peer rank `r`, remote block `A`*.
   2. **`start_load_kv(forward_context)`** — the connector hands the
      transfer list to its underlying transport. For NIXL, this enqueues a
      batch of one-sided RDMA reads from the P workers' exposed KV memory
      regions into this D worker's local KV cache pages. The call is
      **non-blocking** — it returns as soon as the reads are posted; the
      NIC moves the bytes in the background while the GPU is free to start
      computing.
   3. **Forward pass** — the model's *full* forward (all layers) runs on
      whatever requests the scheduler put in the running set. **The forward
      does not wait for KV transfers.** Loading and execution are decoupled
      *across steps*: a request whose remote KV hasn't fully landed yet is
      held by the scheduler in `WAITING_FOR_REMOTE_KVS` state and is simply
      **not in this step's batch**
      ([scheduler.py:2017-2061](../../vllm/v1/core/sched/scheduler.py#L2017-L2061));
      it gets admitted in some later step, once `get_finished` (below)
      reports its transfer done. Meanwhile, `start_load_kv` from this step
      and prior steps is moving bytes in the background, completely
      independently. `save_kv_layer` is invoked from inside each attention
      layer as it computes (still part of the same forward), but is a
      **no-op on the D side** (D has nothing to push out).
   4. **`wait_for_save()`** — no-op on the D side; on the P side this is
      where the worker would block until all of its outbound layer saves
      have actually been pushed/exposed.
   5. **`get_finished(scheduler_output.finished_req_ids)`** — the connector
      polls per-request transfer state and returns two sets:
      `(finished_sending, finished_recving)`. On the D side,
      `finished_recving` contains the request IDs whose **entire** KV (all
      layers × all blocks × all ranks this worker is responsible for) has
      finished arriving *during this step*. **This is the only mechanism by
      which the D scheduler learns that an async transfer has completed** —
      the forward never blocked on the transfer, so without this report the
      scheduler would never know a `WAITING_FOR_REMOTE_KVS` request became
      ready. On the P side, `finished_sending` similarly tells the P
      scheduler "your peer has finished pulling — safe to free these
      blocks." A request that's still partway through stays out of both
      sets; the counters in `KVOutputAggregator` carry its remaining count
      into the next step.
   6. **`get_block_ids_with_load_errors()`** — local block IDs whose RDMA
      reads failed; these are reported up so the scheduler can reissue or
      abort.

   All of the above lands in a `KVConnectorOutput` which is attached to the
   `ModelRunnerOutput` this worker is about to return.
7. Each D worker writes its `ModelRunnerOutput` (with
   `kv_connector_output.finished_recving = {...}`) into its
   `worker_response_mq` via `WorkerProc.handle_output` →  `enqueue_output`
   ([multiproc_executor.py:898-906](../../vllm/v1/executor/multiproc_executor.py#L898-L906),
   [multiproc_executor.py:883](../../vllm/v1/executor/multiproc_executor.py#L883)).
8. Executor reads outputs from all workers (because `kv_output_aggregator`
   is non-`None`, `output_rank` is `None`, so it dequeues every rank's
   response —
   [multiproc_executor.py:350-357](../../vllm/v1/executor/multiproc_executor.py#L350-L357),
   [multiproc_executor.py:365-385](../../vllm/v1/executor/multiproc_executor.py#L365-L385))
   and runs `KVOutputAggregator.aggregate`
   ([utils.py:48-158](../../vllm/distributed/kv_transfer/kv_connector/utils.py#L48-L158)).
   Only requests reported by all (or `expected_finished_count`) workers
   end up in the final `finished_recving`; partial counts persist in
   `_recv_remaining_count` across steps.
9. Scheduler sees the finished set in `_update_from_kv_xfer_finished`
   ([scheduler.py:2063-2090](../../vllm/v1/core/sched/scheduler.py#L2063-L2090)),
   which calls `connector.update_connector_output(...)`
   ([base.py:487](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L487))
   on the scheduler-side connector and adds the IDs to
   `finished_recving_kv_req_ids`. On the next scheduling pass,
   `_update_waiting_for_remote_kv`
   ([scheduler.py:2017-2061](../../vllm/v1/core/sched/scheduler.py#L2017-L2061))
   caches the now-resident blocks and transitions the request out of
   `WAITING_FOR_REMOTE_KVS`, making it eligible to enter the next forward.

The P side has the symmetric flow with `finished_sending` instead — see
`request_finished`
([base.py:497](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L497))
and `finished_sending` handling at
[scheduler.py:2087-2090](../../vllm/v1/core/sched/scheduler.py#L2087-L2090),
where the P scheduler frees the request's blocks once its peer has
confirmed the pull.

### What the MultiprocExecutor actually does for PD

Three things, and only three:

1. **Broadcast control messages.** `SchedulerOutput` carries
   `kv_connector_metadata`, so when it goes over `rpc_broadcast_mq`
   ([multiproc_executor.py:363](../../vllm/v1/executor/multiproc_executor.py#L363)),
   every worker gets its instructions. No PD-specific code; the metadata is
   just part of the dataclass.
2. **Collect outputs from all ranks instead of just one.** Toggled by
   `kv_output_aggregator` being non-`None`
   ([multiproc_executor.py:350-357](../../vllm/v1/executor/multiproc_executor.py#L350-L357)).
3. **Apply the aggregator function** to the list of `ModelRunnerOutput`s
   before returning
   ([multiproc_executor.py:388](../../vllm/v1/executor/multiproc_executor.py#L388),
   [multiproc_executor.py:397](../../vllm/v1/executor/multiproc_executor.py#L397)).

The actual KV traffic — gigabytes per request, the hot path of
disaggregation — flies through the connector's own transport (NIXL RDMA,
NCCL P2P, etc.), entirely outside the executor's shared-memory queue. The
executor's job is just to make sure every step the scheduler gets a
coherent "who finished what" signal.

---

## 7. Recommended reading order

1. [abstract.py:36-216](../../vllm/v1/executor/abstract.py#L36-L216) — get
   the `collective_rpc` contract straight first.
2. All of [uniproc_executor.py](../../vllm/v1/executor/uniproc_executor.py)
   — short, shows the pattern with zero IPC noise.
3. [multiproc_executor.py:96-236](../../vllm/v1/executor/multiproc_executor.py#L96-L236)
   (executor init) →
   [multiproc_executor.py:914-940](../../vllm/v1/executor/multiproc_executor.py#L914-L940)
   (worker busy loop) →
   [multiproc_executor.py:329-397](../../vllm/v1/executor/multiproc_executor.py#L329-L397)
   (dispatch).
4. [shm_broadcast.py:204-339](../../vllm/distributed/device_communicators/shm_broadcast.py#L204-L339)
   — `ShmRingBuffer` (the protocol diagram in the docstring is gold).
5. [shm_broadcast.py:536-701](../../vllm/distributed/device_communicators/shm_broadcast.py#L536-L701)
   — `acquire_write` and `acquire_read`, side by side.
6. [shm_broadcast.py:92-201](../../vllm/distributed/device_communicators/shm_broadcast.py#L92-L201)
   — `SpinCondition` (spin → idle transition and cancellation).
7. [shm_broadcast.py:703-781](../../vllm/distributed/device_communicators/shm_broadcast.py#L703-L781)
   — `enqueue`/`dequeue` framing and the overflow path.
8. [shm_broadcast.py:353-527](../../vllm/distributed/device_communicators/shm_broadcast.py#L353-L527)
   — `MessageQueue.__init__`, `create_from_handle`, `wait_until_ready` (the
   handshake).
9. For PD:
   [kv_connector_model_runner_mixin.py:93-128](../../vllm/v1/worker/kv_connector_model_runner_mixin.py#L93-L128)
   (forward-pass hook) →
   [base.py:1-100](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L1-L100)
   (two-halves contract) →
   [utils.py:48-158](../../vllm/distributed/kv_transfer/kv_connector/utils.py#L48-L158)
   (`KVOutputAggregator`) →
   [multiproc_executor.py:296-357](../../vllm/v1/executor/multiproc_executor.py#L296-L357)
   (re-read with aggregator in mind).
10. A concrete PD connector:
    [nixl_connector.py](../../vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py)
    (NVIDIA NIXL/RDMA, the canonical disagg implementation).
