# Migration from0.1 to0.2.0-dev.1

This candidate changes ABI and public diagnostic/close semantics and therefore is not a patch to0.1.

- Rebuild/bundle bridge ABI2 with exact ORT1.31.1. Loader rejects ABI1, missing ABI symbols and mismatched ORT versions. Regenerate Dart bindings with `python3 tools/generate_bindings.py` then `dart format lib/src/ffi`.
- Use `await OnnxRuntime.createAsync()` and `await InferenceSession.loadFile/loadBytes(...)`. Former fromFile/fromBytes factories are deprecated blocking aliases; fromFileSync/fromBytesSync explicitly preserve blocking worker-side use. create() is also synchronous initialization.
- Await `session.close()` / `runtime.close()`. Runtime.dispose is a deprecated Future-returning alias. Close cancels and drains before releasing native owners and must precede isolate exit/hot restart.
- Prefer runWithDiagnostics(...).outputs/diagnostic for concurrent callers. run still returns the output map. Diagnostic activeProvider/modelHash/fallbackOccurred are nullable, actual/registered providers are separate, and unknown graph assignment is empty. lastDiagnosticReport means last successfully decoded completion.
- Caller requestId is correlation only; native task identity is unique. Cancel listener removal is real. Timeout includes preparation/queue time; cancellation completes after native computation returns and does not promise immediate GPU/NPU termination.
- Apple/QNN factories express acceleration preferences by default. Use explicit HardwareRequirement or `ExecutionPolicy.*(verifyHardware:true)` for mandatory physical observation; currently unavailable evidence fails explicitly. CPUAndNeuralEngine allows CPU+ANE. whitelist policy and legacy requireAne guarantees are unsupported.
- CPU must be last in providerPriority. Extra options cannot override protected compute-unit/backend/policy options. CoreML cache uses ModelCacheDirectory and a content/config namespace.
- Pure Dart consumers no longer require Flutter SDK. Normal builds use native assets, not CWD/Homebrew lookup. libraryPath is explicit development override with architecture/ABI/ORT checks. Platform acceptance remains per artifact/model/device.
