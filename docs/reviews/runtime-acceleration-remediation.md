# Runtime/acceleration remediation

Date:2026-11-04. Reviewed baseline:7af242db423c6f5ef992048b641bf6a23be0fbd9. Changes remain on main; no branches/worktrees or historical tags were created or rewritten. The supplied repair prompt is preserved as an unrelated untracked input and excluded from delivery.

This is the historical candidate repair report. Subsequent stable0.2.0 CPU-only
acceptance is recorded in [stable-0.2.0.md](stable-0.2.0.md); accelerator/model
gates below remain open.

**Software repair is implemented and verified as a0.2.0-dev.1 /ABI2 candidate. Platform accelerator release acceptance and Qwen acceptance remain open.** No new stable package/release is published. For GSMLG, this is suitable for scoped macOS candidate integration/safety evaluation, not production GPU/NPU/Qwen acceptance.

## Findings, fixes and regressions

| Original issue verified on current baseline | Repair locations / behavior | Actual evidence / limit |
|---|---|---|
| int8/uint8/bool output width guessed4 | native/src/lib_onnx_bridge.c dtype table and checked tensor_size; Dart validates before pointer reads | Native uint8 regression failed16vs4 before repair, passes4 after; all7 types run through real ORT |
| Scalars/zeros invalid; fixed16D truncation; unchecked overflow | Native rank queries/exact allocations, count/byte checks, scalar1/zero0; tensor Dart validation |28 dtype×shape real ORT cases (normal/scalar/empty/rank20); negative/overflow/wrong-byte tests |
| Unsupported output kind/type and partial result leaks | IsTensor/type/size/status checks; single cleanup of every OrtValue/tensor/shape/array; structured metadata errors | Actual tensor+sequence outputs fail after first output is created; unsupported double fails; sanitizer harness passes |
| Async tasks borrowed inputs/session/runtime | Task copies names/input bytes, retains Session, Session retains Runtime/OrtEnv; result/error owns lifetime | Inputs destroyed before worker use; results read after Session/Runtime owner destruction; early task acknowledgement tested |
| maxConcurrency/Session queue options ineffective | Bounded native scheduler, per-Session active+pending limits, total Runtime budget workers+64; thread/optimization/memory options applied | Two Sessions progress independently, one Session serializes by default, configured concurrent2 works, queue overflow tested |
| Thread/allocation/ORT failures unchecked | Checked initialization/thread creation, memory/status/type queries and error cleanup; no fixed provider-option limit | Invalid inputs and partial output failure tested; exhaustive allocator fault injection not claimed |
| Runtime/Session could close underneath tasks or UI-block joins | Nonblocking admission stop/cancel; task retention; awaitable cancel-and-drain close; joins/destruction on execution isolates | Active/queued close, Runtime with live Session, repeated Dart close, load-vs-close, delayed release tests |
| Shared global last error/cancel state | Thread-local sync errors; immutable per-task errors/stage/ORT code; native task IDs/independent RunOptions | Duplicate requestIds across Sessions/isolate calls stay independent; terminal task errors remain valid after owners close |
| Future run() called synchronous FFI; synchronous load/compile | createAsync/loadFile/loadBytes and input preparation on helper isolates; ORT Run on native pool; explicit deprecated Sync alternatives | Event-loop heartbeat during actual initialization/load/run; native latch confirms responsive control path |
| Future.ignore did not unsubscribe | Removable CancellationToken listeners, safe active guard and finally cleanup | Cancellation before execution, queue/dequeue gap, true long ORT Loop, after completion/Session/Runtime close; repeated tokens |
| Timeout unused; queue→Run race | CLOCK_MONOTONIC deadline from input preparation through queue/run; independent monitor; termination options exist before queueing | Latched queued timeout and real ORT Loop timeout; completion wins over late cancellation; resources await native return |
| Callback returned before Dart consumed buffers | ABI2 has no foreign completion callback: task/output/diagnostic getters borrow until explicit task_release | Delayed acknowledgement/early client-release tests; no NativeCallable lifetime assumption |
| Old bridge could accidentally load | ABI2 header + generated bindings + ABI/exact ORT loader check | Missing override/ABI1/wrong ORT regressions; corrupt/missing archive and wrong SDK/bundle checks |
| Apple chip name promoted to hardwareObserved | Removed hardware probing/inference; nullable actual provider/fallback; explicit evidence provenance and per-task result | CPUOnly never reports GPU/ANE; EP registration/run completion stay separate from physical observation; no node/hash fabrication |
| Model hash missing/hardcoded | Hash actual ONNX bytes and ordered external-data inventory; traverses nested graphs/sparse tensors/local functions/training graphs | Runtime-vs-manifest identity regression, external function/training regressions, real MLP/Identity hashes in raw logs |
| strict only checked EP registration | session.disable_cpu_ep_fallback=1 on selected ORT; CPU baseline remains distinct; CPU-first priority rejected | CoreML registers but unsupported Loop fails with ORT CPU-node fallback-disabled error; allowed mixed assignment remains unknown |
| Whitelist silently behaved as allowed | UnsupportedPolicy before loading; fused/control/subgraph coverage cannot be guaranteed | Explicit rejection regression; no unsafe whitelist workaround |
| Mandatory physical evidence conflated with preference | Factory preferences use any; verifyHardware opt-in; incompatible CPU/hardware configs reject | Mandatory hardware returns HardwareVerificationUnavailable, not invented usage; normal CoreML runs allowed |
| CoreML flags/options/cache ignored | Generic ORT1.31 API; CPUAndNeuralEngine, MLProgram, shape/subgraph options and ModelCacheDirectory; namespace by content/config/provider order | Strict real MLP succeeds for ALL/CPUAndGPU/CPUAndNeuralEngine/CPUOnly; physical unit still unobserved |
| QNN depended on Apple ANE; option truncation | No Apple probe; HTP backend basename enforced; dynamic option arrays; extras cannot override protected settings | Default HTP policy tests; actual macOS build reports qnn.initialize/provider unavailable; Android-specific SDK/model execution not-run |
| CoreML symbols unconditionally linked on every target | Generic ORT provider API only; no CoreML factory header/symbol; target-specific CMake input | Cross-target rejection tests; Linux CI configured; local Linux runtime not-run |
| Example delayed/faked successful inference and hardware | Actual bundled305-byte MLP, asynchronous native inference, real report, no simulated report export | Example analysis passed; clean Flutter release consumer built/launched actual ORT Identity separately |
| Adapter assumed ranks/masks/fp32 and padding token0 | hiddenDim/rank/type/mask validation, left/right last-valid pooling, no-token error, explicit IEEEfp16 conversion, finite stable L2 | Adapter regressions first failed;17 adapter+manifest tests passed; no hand-authored hidden states called Qwen acceptance |
| No genuine Qwen/tokenizer provenance | tools/export_embedding.py acquire/import pinned official revision, external-data inventory; prepare_qwen.py reference/tokenizer; validate_qwen.dart native/pooling/retrieval | Official revision97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3; tooling tests pass; actual weights/export/toolchain unavailable and model run blocked |
| CWD/Homebrew/native-copy consumer path | Dart native-assets hook builds bridge+verified exact ORT; pure Dart SDK boundary; explicit development override only | External hosted archive consumer, fresh Pub cache, unrelated CWD, JIT+release CLI+Flutter release launch pass |
| Apple branch used macOS dylib for iOS | Exact target/SDK native bundles; no macOS fallback into iOS; CMake cross-target guards | iOS wrong-artifact rejection passes; real device/simulator builds blocked pending separate inputs |
| Android16KB stated without actual artifact | Explicit NDK/ABI/QNN/HTP inputs; parse every ELF PT_LOAD and final APK/AAB; no proprietary SDK upload | Synthetic parser/control-flow checks pass; no Android package produced/audited; not platform acceptance |
| Artifact/release readiness claims exceeded evidence | Generated binary hashes and actual compiled providers; separate configured/compiled/packaged/consumer/device states; stable release gate in all publish workflows | Clean consumer receipts and artifact manifest; unaccepted platforms not stable; push only runs CI |

## Commands and retained evidence

Host: Apple M1 Max32GiB, macOS27.0.1, Xcode27.2, Dart3.13.7, Flutter3.47.9. Native headers and runtime are pinned1.31.1/API31. Normal macOS route minimum14 follows the actual upstream binary's load command.

| Command | Result |
|---|---|
| python3 tools/make_test_models.py | Real ONNX fixtures generated; no large weights |
| dart format --output=none --set-exit-if-changed lib hook test tools example/lib native/tests/lifecycle_test.dart | Passed |
| dart analyze lib hook test tools native/tests/lifecycle_test.dart | Passed |
| dart test -r expanded |59 passed, including100 lifecycle cycles/1000 exact real-ORT runs |
| tools/run_native_tests.sh | ASan+UBSan passed |
| tools/run_native_tests.sh thread | Separate TSan passed |
| tools/run_native_tests.sh none; MallocStackLogging=1 leaks -quiet --atExit -- .cache/native-tests/none/runtime_test "$PWD/test/fixtures" | Harness passed;0 leaks for measured process, restricted-inspection warning retained |
| tools/run_dart_native_tests.sh | Native dequeue latch/Dart control regression passed |
| python3 tools/test_distribution.py; python3 tools/test_qwen_profile.py |6+3 tooling tests passed |
| python3 tools/verify_hook_inputs.py |5 expected rejection paths passed, not hardware evidence |
| cd example && flutter analyze --no-pub | Passed |
| dart run tools/validate_mlp.dart --provider=CoreML --units=ALL/CPUAndGPU/CPUAndNeuralEngine/CPUOnly --fallback=strict | Four separate commands passed real MLP, no ORT CPU fallback; physical units unknown |
| dart run tools/validate_mlp.dart --provider=CPU --fallback=strict | Correct baseline, fallback=false |
| dart run tools/validate_mlp.dart --provider=QNN --fallback=strict | Expected failure code4/ORT2, qnn.initialize: not supported in this macOS build |
| python3 tools/verify_consumer.py --output build/consumer-evidence --ort-archive .cache/ort/onnxruntime-osx-arm64-1.31.1.tgz --flutter | Passed: exact Pub archive, external hosted consumer/fresh cache, JIT+release CLI+Flutter arm64 release build and launch; signatures checked |

Raw model outputs are in evidence/coreml-*.txt, cpu-baseline.txt and qnn-unavailable-on-mac.txt. Terminal excerpts and sanitizer limits: evidence/software-checks.txt. Full package delivery commands/results: evidence/packaging-clean-consumer.log, packaging-clean-consumer.json, packaging-flutter-launch.json; CMake and hook rejections are separately logged. The source fingerprint in the consumer receipt binds the tested native/Dart/hook inputs; its archive checksum describes that tested payload. Later documentation/test-only additions do not establish additional runtime/hardware acceptance.

Prebuilt ORT/EP/system frameworks are **not** sanitizer-instrumented. Leak-tool output is a checked harness result, not universal proof of no leaks. No exhaustive allocation/thread-failure injection or abrupt-isolate/hot-restart recovery guarantee is claimed. Applications must await controlled shutdown; there is no GC assumption.

## Remaining platform/model gates

- macOS: real CoreML configurations and clean release launch verified. Physical GPU/ANE tracing, full Qwen profile, Developer ID/notarization/store distribution remain not-run. Ad hoc signature validity is not production signing.
- iOS: a paired physical iPad Air M3 is available; no separate verified device/simulator ORT bridge bundle or signed clean consumer was supplied/built. Device/simulator packaging and CoreML runtime/physical evidence remain blocked, not passed.
- Android: Pocket_DS/SG8275 device and vendor HTP library are present. Missing verified application ORT-QNN/HTP/SoC dependency bundle and legal redistribution inputs block the application build; every ELF/final APK/AAB16KB audit and true HTP inference remain not-run. Host Apple detection no longer blocks QNN.
- Qwen: genuine official checkpoint/export was not supplied; Python3.11 attempts report missing huggingface-hub/onnx. Profile scripts specify locked dependencies and real acquisition/import/reference workflow. Full tokenizer→ONNX→pooling→normalization→retrieval is blocked until those inputs are present. Small-model and mock/control tests are not substituted.
- Linux: pinned CPU correctness route and CI are configured; local Linux artifact/consumer execution not-run. No new accelerator platforms are claimed.

## Historical claims and delivery

Previous P0–P4/all-platform-ready/Hardware Observed/Qwen-validated/blanket-no-leak claims are superseded in README, implementation progress, device matrix and benchmark report. Historical0.1 notes are retained with a correction, not silently deleted. See ../migration-0.2.md for breaking API/ABI/SDK changes.

This repair authorizes a scoped commit and ordinary push to the existing main branch. Final commit SHA and remote parity are reported in the delivery response, since a commit cannot contain its own hash. No stable publication or tag movement is performed; software and platform acceptance remain distinct in artifacts_manifest.json and release_gate.py.

The first pushed candidate, d86b074, exposed two hosted CI setup failures: standalone Dart recursively resolved the Flutter example, and formatting ran before the example's package language version was resolved. The workflows now use `dart pub get --no-example` for the core and resolve the example before formatting it. The original failed runs are [Tests](https://github.com/gsmlg-app/lib_onnx/actions/runs/41821114445) and [CI](https://github.com/gsmlg-app/lib_onnx/actions/runs/41821114356); follow-up CI status is reported with delivery.
