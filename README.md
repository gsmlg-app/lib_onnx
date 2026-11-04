# lib_onnx

Dart/Flutter → thin C ABI → ONNX Runtime 1.31.1, with explicit CoreML and QNN HTP policies. The current candidate is **0.2.0-dev.1 / bridge ABI 2**. GPU/NPU platform acceptance and Qwen acceptance remain open; this candidate is not a production acceleration release.

The implemented native-assets route builds and bundles the macOS arm64 bridge and pinned ORT dependency without CWD/Homebrew searches or copying dylibs. It requires Dart 3.10+, a C11 compiler, and macOS 14+. Linux artifacts are for CPU correctness checks. iOS device/simulator and Android QNN bundles are separate experimental inputs; see [distribution](docs/distribution.md).

```yaml
dependencies:
  lib_onnx: 0.2.0-dev.1 # candidate source; not yet published by this repair
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
      options: SessionOptions.appleSilicon(
        preferAne: true,
        allowCpuFallback: false,
      ),
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
