# lib_onnx

Dart/Flutter → thin C ABI → ONNX Runtime. **0.3.0 / bridge ABI 2** provides CPU inference on macOS arm64 and Android arm64-v8a with ONNX Runtime 1.31.1, and on iOS with ONNX Runtime 1.30.0 (the newest official iOS archive). CoreML and QNN HTP integration remain experimental; GPU/NPU acceptance and Qwen acceptance remain open. See the [0.3.0 acceptance scope](docs/reviews/0.3.0-mobile-cpu.md).

| Target | Pinned official ORT input | Bundled by the hook |
|---|---|---|
| macOS arm64 (macOS 14+) | `onnxruntime-osx-arm64-1.31.1.tgz` | bridge + `libonnxruntime.1.dylib` |
| Android arm64-v8a, x86_64 | `onnxruntime-android-1.31.1.aar` (Maven Central) | bridge + `libonnxruntime.so`, ELF 16 KB checked |
| iOS device, simulator arm64/x64 (iOS 15.1+) | `pod-archive-onnxruntime-c-1.30.0.zip` | bridge with ORT statically linked |
| Linux x64/arm64 | official Linux 1.31.1 archives | CPU correctness checks only |
| Any other target (Windows, Android armeabi-v7a, macOS x64, …) | none | nothing: the app builds, and `OnnxRuntime.create*` throws `UnsupportedError` |

The native-assets route downloads and verifies the pinned archive (SHA-256) and builds the bridge without CWD/Homebrew searches or copying libraries. It requires Dart 3.10+ and a C11 toolchain (Xcode for Apple targets, the Android NDK for Android). `hooks.user_defines.lib_onnx.native_bundle` still overrides any target with an explicitly verified bundle; see [distribution](docs/distribution.md).

```yaml
dependencies:
  lib_onnx: ^0.3.1
```

```dart
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';

Future<void> main() async {
  final runtime = await OnnxRuntime.createAsync();
  try {
    final session = await InferenceSession.loadFile(
      'mlp_smoke.onnx', // tools/make_test_models.py
      runtime: runtime,
      options: SessionOptions.cpuOnly(),
    );
    final input = OnnxTensor.fromFloat32(
      [1, 4], Float32List.fromList([1, 2, 3, 4]),
    );
    final result = await session.runWithDiagnostics({'input': input});
    print(result.outputs['output']!.asFloat32List());
    print(result.diagnostic); // Registered EPs, observed/unknown facts, actual model hash
    await session.close();
  } finally {
    await runtime.close();
  }
}
```

`preferAne` permits CoreML CPU and Neural Engine; it does not promise ANE-only execution. Strict policy disables ORT CPU EP fallback, independently of CoreML's internal compute units. A model with unsupported nodes fails. Physical hardware requirements are optional and currently fail explicitly with `HardwareVerificationUnavailable` because no per-run device trace is collected. Registration and a successful run do not establish GPU/NPU use.

`loadFile`, `loadBytes`, `createAsync`, `run`, and awaitable `close` keep native work/waits off the calling isolate. Inputs are snapshotted; accepted tasks own native copies and independent RunOptions. Timeouts include preparation and queueing on a monotonic clock; interruption is cooperative. Results are retained until Dart acknowledges them. Call `close` before stopping an owning isolate or hot restarting; GC/abrupt isolate termination is not a shutdown protocol.

Seven numeric types, scalars, empty tensors and arbitrary valid ranks are supported. Unsupported values, overflow and invalid buffers fail explicitly. `EmbeddingAdapter` validates pooling/masks/fp16/normalization; its tests do not establish Qwen execution. The [Qwen profile workflow](docs/model-profiles.md) pins genuine checkpoint/tokenizer provenance and requires real inference before acceptance.

```sh
dart pub get
python3 tools/make_test_models.py
dart test
tools/run_native_tests.sh             # ASan + UBSan; needs exact local ORT input
tools/run_native_tests.sh thread      # separate TSan build
tools/run_dart_native_tests.sh        # test-only dequeue latch
python3 tools/verify_consumer.py --help
```

The Flutter example uses a bundled real MLP and real diagnostics. Generate the desired host scaffold with Flutter tooling before running it; the archive consumer script creates a separate clean Flutter app for delivery checks.

[Repair evidence and remaining gates](docs/reviews/runtime-acceleration-remediation.md) · [API/ABI migration](docs/migration-0.2.md) · [Architecture](docs/architecture.md) · [Execution policy](docs/execution-policy.md) · [Device matrix](docs/validation/device_matrix.md)

MIT. ONNX Runtime's license and third-party notices are retained under `native/third_party/onnxruntime/`. Qualcomm SDK files are not redistributed here.
