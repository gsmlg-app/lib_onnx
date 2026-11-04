# Execution policy and evidence

`providerPriority` is graph allocation order: all requested accelerators are registered in that order. CPU must be last; CPU-first order is explicitly rejected because ORT's implicit CPU EP is last. Only CPU, CoreML and QNN are in scope. There is no silent `break` after the first accelerator.

- `strict`: requested EP initialization must succeed; `session.disable_cpu_ep_fallback=1` forbids ORT CPU nodes when an accelerator is configured. An unsupported Loop graph is a regression fixture. This constraint does not prohibit CoreML internal CPU computation.
- `allowed`: EP initialization failure or unsupported graph nodes may use ORT CPU. Initialization failures are retained in diagnostics. Actual partial graph fallback is **unknown** when assignment data is unavailable, rather than falsely reported false.
- `whitelistedOps`: **UnsupportedPolicy**. Fused/control/subgraph CPU assignments cannot be reliably enumerated by this implementation, so no whitelist guarantee is made.
- CPU baseline is explicit execution, not fallback; its report is false.

Acceleration preference is independent of physical verification. Factories default to `HardwareRequirement.any`; `verifyHardware: true` opts into mandatory physical GPU/NPU evidence. The current backend has no per-run telemetry collector and rejects such requirements with `HardwareVerificationUnavailable`. CPU-only/hardware conflicts fail before loading. Extra options cannot replace protected policy/backend options.

CoreML uses ORT 1.31.1's generic provider API with real option names: MLComputeUnits, MLProgram format, RequireStaticInputShapes, EnableOnSubgraphs, ModelCacheDirectory. CPUAndNeuralEngine permits both CPU and ANE. Cache directories are namespaced by the actual model/external-data identity and ordered providers/compilation options, because ORT's own URL/IO-name hash is insufficient for model identity.

QNN uses `libQnnHtp.so`; CPU/GPU backend names are rejected. No Apple hardware probe is involved. Build capability comes only from ORT's actual provider enumeration. Missing provider, backend initialization, model load and verification failures retain their stage and ORT message. Android hardware/model acceptance still requires the matching QNN build and device run.

Every runWithDiagnostics result owns its report and task ID. lastDiagnosticReport means the last successfully decoded completion on that Dart isolate, and cannot identify concurrent callers' results. Evidence records distinguish registration and Run completion, with source, scope, model manifest hash, runtime version and run/task identity where applicable. Dependency/device initialization, graph distribution, compute plan and physical utilization remain unknown unless measured. Compute plans are estimates, not usage.

An EP registered in a Session may still receive zero nodes. activeProvider is nullable, actualProviders may be empty or contain multiple measured providers, nodeDistribution is empty without assignment data, and fallbackOccurred is nullable. `hardwareObserved` is never inferred from chip names, CPUOnly, provider registration or Session creation. isAccelerated requires physical run observation. The legacy enum values remain available for compatibility; current reports do not manufacture unavailable evidence levels.
