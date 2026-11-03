# 工具链与兼容性规范

## 1. 软件版本锁（Version Pins）

| 组件 | 锁定版本 | 说明 |
|---|---|---|
| **ONNX Runtime** | `v1.31.1` (ORT C API Version 31) | 统一 C/C++ 动态链接库与头文件 |
| **Dart SDK** | `>= 3.3.0 < 4.0.0` | 支持现代 FFI 与内存模型 |
| **Flutter** | `>= 3.19.0` | 经过 Flutter 3.47+ 验证 |
| **C/C++ 标准** | `C11` / `C++17` | 原生 C ABI 桥接层无外部依赖 |
| **Apple Xcode / SDK** | Xcode 15+ / macOS SDK 14+ | 兼容 CoreML 6/7/8 API |
| **Android NDK** | `r25c` 或更高 | 具备 16 KB 分页支持 |
| **ONNX Opset** | `Opset 17` (兼容 14 ~ 21) | 官方主流模型导出推荐规范 |

---

## 2. 操作系统最低要求

- **macOS**：macOS Monterey (12.0) 及以上，原生支持 Apple Silicon arm64 架构。
- **iOS / iPadOS**：iOS 15.0 及以上真机（模拟器仅供编译测试，不作为加速依据）。
- **Android**：Android 10 (API Level 29) 及以上，64 位 ARM (`arm64-v8a`)。
- **Linux** (桌面扩展)：Ubuntu 20.04+ / Debian 11+ (glibc >= 2.28)。
- **Windows** (桌面扩展)：Windows 10 Version 2004 (Build 19041) 及以上。

---

## 3. 支持的数据类型（Numerical Tensors Only）

| 数据类型 | Dart 对应类型 | 原生字节大小 | 支持状态 |
|---|---|---|---|
| `float32` | `Float32List` | 4 字节 | 完全支持 |
| `float16` | `Uint8List` (IEEE 754 half) | 2 字节 | 存储与传输支持 |
| `int64` | `Int64List` | 8 字节 | 完全支持 (token IDs / mask) |
| `int32` | `Int32List` | 4 字节 | 完全支持 |
| `int8` | `Int8List` | 1 字节 | 完全支持 (量化模型) |
| `uint8` | `Uint8List` | 1 字节 | 完全支持 |
| `bool` | `List<bool>` / `Uint8List` | 1 字节 | 完全支持 |
| `string`, `sequence`, `map`, `sparse`, `bfloat16` | N/A | - | **明确不支持**（抛出 `UnsupportedError`） |
