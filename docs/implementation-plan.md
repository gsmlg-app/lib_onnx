# Implementation progress correction — 2026-11-04

The report at baseline7af242d declared P0–P4 complete, Hardware Observed on Apple M1 Max, Qwen validation and release readiness. Those conclusions exceeded its evidence: the source inferred hardware use from CPU names, run() invoked synchronous FFI, the example simulated inference, and adapter unit tests did not execute Qwen. This correction supersedes those claims without rewriting the historical commit or published0.1.0 changelog.

| Area | Current implementation and evidence | Open gate |
|---|---|---|
| Tensor/ownership | ABI2, checked dtype sizes/ranks, retained task inputs/results; real ORT/sanitizer regression harness | Sanitizers do not instrument prebuilt ORT/EP/system libraries |
| Async/lifecycle | Execution-isolate loads and native worker runs; per-task cancellation/deadlines; awaitable close; heartbeat/latch tests | Abrupt isolate exit/hot restart must use controlled application shutdown |
| Policies | Real strict CPU constraint; whitelist rejected; ordered providers; honest nullable evidence | Physical GPU/NPU trace collector unavailable |
| Models/example | Real small MLP; safe embedding adapter; pinned Qwen acquisition/import/tokenizer/reference workflow | Genuine Qwen ONNX export/checkpoint and full model runs not supplied |
| Delivery | Native-assets macOS route, checksums/licenses, external clean consumers and stable release gate | iOS/Android artifact, package and device acceptance remain separate |

The original design in ../lib_onnx-implementation-plan.md remains a target specification, not a record of completed acceptance. Exact commands, statuses, limitations and remaining prerequisites are in reviews/runtime-acceleration-remediation.md. Stable0.2.0 is scoped to macOS arm64 CPU inference with separate host delivery acceptance in reviews/stable-0.2.0.md. Accelerator and full model gates above remain open.
