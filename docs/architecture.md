# Runtime ownership and ABI 2

Dart APIs call a thin C bridge over the pinned ORT C API. `createAsync` and model loading run on execution isolates; input preparation/submission runs on another short-lived execution isolate. Native inference runs in a Runtime worker pool. The caller isolate polls terminal task handles and handles cancellation independently of ORT Run.

| Resource | Owner and lifetime |
|---|---|
| OrtEnv/workers | Runtime; Sessions retain OrtEnv beyond Runtime owner's close |
| OrtSession/metadata | Session; accepted Tasks retain Session through result acknowledgement |
| Task/RunOptions/error | One per accepted submission, with native process-unique task ID; requestId is correlation only |
| Inputs/names | Copied at native submission; originals can be destroyed immediately afterwards |
| Output OrtValues/tensors/diagnostic | Task retains all results until explicit task_release; no ephemeral callback arrays |
| Dart output | Validated copy; no native buffer retained in public tensor |
| Sync error | Thread-local, consumed on the same thread before further native calls |
| Async error | Independent immutable terminal Error in Task, including status, ORT code and stage |

Each Session admits at most `maxConcurrentRuns + maxQueueDepth` tasks, including input preparation. The default Session executes one task and permits 16 pending tasks. Runtime additionally bounds total admission to its worker count plus 64. Copying occurs outside the scheduler mutex. Runtime workers skip saturated Sessions so other Sessions can progress. Public Session thread, optimization, memory-pattern, queue and concurrency settings reach native code.

Close uses **cancel and drain**. Nonblocking begin-close rejects new work and sets termination on accepted tasks. Dart awaits its submissions/results, then destroys native owners on execution isolates. Runtime waits for concurrent loads, closes its Sessions, joins workers/monitor off the UI isolate, and releases its environment owner reference. Native result handles may outlive both owners safely. Public close is idempotent and awaitable; native destroy releases one owner reference exactly once.

Timeouts include input preparation, queueing and native Run, use CLOCK_MONOTONIC, and are monitored independently of the Run thread. Each task has its own OrtRunOptions. Cancellation/timeout completes after native execution has returned; no claim is made that GPU/NPU work can be hard-stopped. Completion wins over late cancellation. Removable token listeners are removed in all terminal paths.

There is no NativeCallable callback to close prematurely. Helpers exit only after transferring an integer handle; the public Session retains pending work until explicit acknowledgement. Applications must await Runtime.close before terminating its owning isolate or hot restart. Abrupt isolate/process termination is outside this controlled shutdown contract; finalizers/GC do not cancel native work.

ABI 2 removes the unsafe callback/caller-request-ID cancellation ABI. The loader rejects old bridges and ORT version mismatches. `tools/generate_bindings.py` regenerates checked-in Dart bindings from the header; see migration documentation.
