# Compatibility

| Component | Requirement / verified toolchain |
|---|---|
| Bridge | ABI2; package0.2.0; old bridges rejected |
| ORT | Exact1.31.1 / C API31; checksummed official archives |
| Dart | >=3.10<4, for native-assets hooks; tested3.13.7 |
| Flutter | Hook-capable toolchain; tested3.47.9; example minimum3.38 |
| Native | C11, pthreads; tested AppleClang21 / Xcode27.2 |
| macOS arm64 | Minimum14, dictated by actual pinned ORT LC_BUILD_VERSION |
| iOS device/simulator | Separate target-specific bundles required; deployment/signing must match supplied artifact |
| Android | arm64-v8a; explicit NDK/ORT QNN/HTP inputs; every ELF and final package must pass16KB audit |
| ONNX | Fixtures use opset17; acceptance is per actual model/profile, not every opset/operator |

Numeric bridge widths: float32/int32=4, float16=2, int64=8, int8/uint8/bool=1. Scalars contain one element; any zero dimension makes an empty tensor. Negative actual dimensions/overflow/byte mismatch fail. Model metadata may retain dynamic negative dimensions. Unsupported tensor dtypes, sequence/map/sparse outputs are rejected with explicit errors.

No Windows/CUDA/DirectML/OpenVINO/TPU/Web implementation is added by this repair. Linux CPU checks are correctness/packaging checks and do not replace accelerator acceptance. No iOS simulator or generic Apple chip assertion establishes ANE execution.
