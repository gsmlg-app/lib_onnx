# Correctness/stress evidence — 2026-11-04

The previous80–84µs /12,000runs/s report measured the old synchronous implementation and lacked raw benchmark evidence. It is retained in historical Git only and is not a benchmark for ABI2. The previous blanket "no leaks"/RSS/ANE statements exceeded the tests and are superseded.

Current `dart test` passes59 tests, including100 load/run/await-close cycles and1000 real CPU Identity executions with exact bytes. These are correctness/lifecycle tests, not an accelerator throughput benchmark. Real MLP results under CPU and strict CoreML compute-unit configurations match `[1.5,2.5,3.5,4.5]`; individual duration_us values in raw logs are samples, not statistically controlled performance claims. Native run duration excludes queue/input preparation; timeout includes them.

The native harness passes ASan+UBSan and a separate TSan build. macOS leaks reported0 leaked bytes for the checked non-sanitized process, with a restriction warning about read-only inspection. Only bridge/harness code is sanitizer-instrumented; prebuilt ORT, EPs and system frameworks are not. This evidence does not establish universal leak freedom, GPU/NPU utilization or Qwen performance.

No controlled warmup/repeat/thermal accelerator benchmark or real-text Qwen retrieval benchmark has run. Model-specific numeric and retrieval tolerances are fixed in the Qwen validation workflow before execution; no target is marked accepted from adapter unit tests.
