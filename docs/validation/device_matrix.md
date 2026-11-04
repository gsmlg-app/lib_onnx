# Device and model acceptance — 2026-11-04

Historical correction: the previous report inferred "Hardware Observed" from Apple chip names and treated synthetic embedding/adapter tests as model acceptance. Those claims are withdrawn here; the original Git history remains intact.

| Device / target | Live inventory | Actual check | Status and limit |
|---|---|---|---|
| MacBook Pro / Apple M1 Max /32GiB | macOS27.0.1, Xcode27.2, arm64 | Real MLP under strict CoreML ALL, CPUAndGPU, CPUAndNeuralEngine, CPUOnly; CPU baseline; native tests; clean release consumers | Software routes pass; **physical GPU/ANE observation not-run** |
| iPad Air13-inch M3 (iPad15,5) | available, paired physical device | devicectl inventory only | iOS ORT/CoreML device bundle, clean signed consumer and runtime acceptance **blocked**; device is present |
| iOS simulator arm64 | Xcode simulator inventory present | Wrong-platform bundle rejection test | Separate simulator artifact/build **not-run**; cannot establish ANE |
| AYANEO Pocket_DS /SG8275 /arm64-v8a | adb device, kernel pages4096; vendor libQnnHtp.so and libcdsprpc.so present | Read-only getprop/getconf/library inventory | No matching verified ORT QNN/HTP package input supplied; build/ELF/APK/device acceptance **blocked**; device is present |
| Linux x64/arm64 | No local target run | Pinned correctness artifacts / CI configuration | Local consumer/device **not-run** |

`tools/validate_mlp.dart` runs the actual305-byte MLP. Its source-file SHA256 is `cf26ee02305c2b808e9dd73ba908de34c5341ea690360ace4eaba53bdec3b775`, and model identity is `0d2cfad5ac8f58a6592d00055aa4b17f42a77cd0624c98b0aa8be697a3b5e407`. Logs in ../reviews/evidence/coreml-*.txt record requested units, strict policy, registered EP, task ID, actual unknowns and numerical output. Strict success prohibits ORT CPU-node fallback; it does not identify CoreML's internal physical unit.

The clean consumer runs a different102-byte Identity fixture, file SHA256 `d3dc6ebdfff0c3b32a814eb77868bda8f4cdb9dc1538b6d1878560613bfb21cf`, identity `d5ae9bdbcb472be787a1fb07a5f56450d1310e4c38804a6f21a7e708944e6515`. Neither fixture establishes Qwen support. No Qwen3-Embedding-0.6B backend/profile has passed actual model or hardware acceptance.

Inventory commands: `xcrun devicectl list devices --timeout 5`, `adb devices -l`, `adb -s 10.100.8.115:5555 shell getprop ro.soc.model`, `getprop ro.product.cpu.abi`, `getconf PAGE_SIZE`, and read-only vendor-library checks. Presence of a vendor HTP library is not permission to redistribute it and does not prove an application initialized that backend.

## Stable 0.2.0 CPU acceptance — 2026-11-05

The MacBook Pro / MacBookPro18,4 / Apple M1 Max / 32 GB / macOS 27.0.1 host
was checked again for bridge 0.2.0. Real CPU MLP inference, the full software
suite, and independently launched clean CLI and Flutter release consumers
passed. This establishes the macOS arm64 CPU-only stable target. GPU/ANE,
Qwen, iOS/Android and Linux device acceptance retain the separate limitations
above. Commands, hashes and receipts are in [stable-0.2.0.md](../reviews/stable-0.2.0.md).
