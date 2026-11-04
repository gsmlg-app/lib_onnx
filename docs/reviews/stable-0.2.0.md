# Stable 0.2.0 acceptance — 2026-11-05

The stable support claim is CPU inference on macOS arm64, macOS 14+,
using bridge ABI2 and the pinned ONNX Runtime 1.31.1. CoreML remains
experimental; physical GPU/ANE observation, QNN/HTP, iOS/Android delivery,
and the full Qwen profile retain their existing open gates. Linux prebuilt
bundles remain CPU correctness inputs outside the stable platform claim.

The `device` receipt for this release covers real CPU inference on the
Apple M1 Max host, including independent launches of compiled CLI and Flutter
release consumers. It does not describe GPU/ANE observation, Developer ID
signing, notarization, or mobile device acceptance. `acceleration_claimed`
remains false; `hardware_observed` remains `not-run`.

Host: MacBook Pro / MacBookPro18,4, Apple M1 Max, 32 GB RAM, macOS 27.0.1,
Xcode 27.2, Dart 3.13.7 and Flutter 3.47.9. The native deployment target is
macOS 14, as required by the pinned upstream ORT library.

Fresh verification passed:

| Check | Observed result |
|---|---|
| Formatting, core analyzer and example analyzer | Passed |
| Dart tests | 59 passed, including real ORT inference and lifecycle/stress cases |
| Distribution and Qwen profile tooling | 11 + 3 passed; profile tooling does not establish Qwen execution |
| Hook input rejection checks | Five expected failures verified |
| Native ASan/UBSan and separate TSan harnesses | Passed; upstream ORT and system libraries are not instrumented |
| Native dequeue-latch/Dart lifecycle check | One test passed |
| Real MLP with strict CPU policy | Bridge 0.2.0 / ORT 1.31.1; output [1.5, 2.5, 3.5, 4.5]; fallback false |
| External hosted source-archive consumer with a fresh Pub cache | JIT, compiled CLI and Flutter arm64 release app passed real Identity inference |
| Independent release launches from an unrelated CWD | CLI and Flutter returned [1.0, 2.0, 3.0, 4.0] using CPU; no workspace dependency or manual native-library copying |
| Release app signature checks | Ad hoc signatures valid; Developer ID/notarization remain not-run |
| Unchanged `python3 tools/release_gate.py --version 0.2.0` | Passed |

Evidence: [software commands](evidence/stable-0.2.0-software.log),
[consumer receipt](evidence/stable-0.2.0-consumer.json), and
[consumer commands](evidence/stable-0.2.0-consumer.log). The receipt's source
fingerprint matches the released Dart/C/hook inputs. Its archive checksum
describes the tested payload before these evidence files were added.

`artifacts_manifest.json` records raw hook binary hashes and actual compiled
providers separately from the consumer's bundled binary hashes. The stable
target is `macos_arm64` with `accepted_execution_providers` limited to
`CPUExecutionProvider`; `device: passed` means the host CPU checks above.
Provider compilation or registration is not accelerated hardware acceptance.
