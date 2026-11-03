# 后端支持矩阵与硬件加速规格

## 1. 平台与后端支持矩阵 (v0.1)

| 平台 | 架构 | 首发加速 Provider | 对应硬件引擎 | 验证状态 |
|---|---|---|---|---|
| **macOS** | arm64 (Apple Silicon) | CoreML EP | Apple GPU、Apple Neural Engine (ANE) | **实测通过** (Apple M1 Max) |
| **iOS / iPadOS** | arm64 (A/M 系列芯片) | CoreML EP | Apple GPU、Apple Neural Engine (ANE) | **构件就绪** |
| **Android** | arm64-v8a (Qualcomm) | QNN HTP EP | Qualcomm Hexagon Tensor Processor (NPU) | **构件就绪** (需骁龙硬件设备) |
| **全平台** | 对应宿主架构 | ORT CPU EP | SIMD / NEON / AVX2 指令集 | **实测通过** (数值基准与回退) |

---

## 2. Apple CoreML EP (macOS / iOS)

### 2.1 计算单元（Compute Units）配置
CoreML 内部允许细粒度调节计算分配策略：
- `CoreMLComputeUnits.all` (`ALL`)：启用 CPU、GPU 以及 Apple Neural Engine (ANE)。适合大吞吐量和低功耗持续推理。
- `CoreMLComputeUnits.cpuAndGpu` (`CPUAndGPU`)：禁用 ANE，仅使用 CPU 和 Apple Metal GPU。适合需要 GPU 高并发浮点计算的模型。
- `CoreMLComputeUnits.cpuOnly` (`CPUOnly`)：仅在 CoreML 内使用 CPU。
- `CoreMLComputeUnits.cpuAndNeuralEngine` (`CPUAndNE`)：iOS 16+ / macOS 13+ 支持，专用于 ANE 敏感型任务。

### 2.2 图分配与 ANE 证据区别
- **重要区别**：ORT CoreML EP 接受子图分配，并不自动等同于所有算子都在 ANE 上执行。
- CoreML 会根据算子支持矩阵将卷积、线性映射和非线性激活下发给 ANE，不支持的算子可能由 CoreML 内部回退至 Metal GPU 或 CPU。
- `DiagnosticReport` 会准确汇报：
  - `activeProvider`：`CoreML`
  - `evidenceLevel`：`hardwareObserved` (证实物理芯片具备 ANE/GPU 且已被 CoreML 图编译接管)
  - `hardwareDetails`：宿主设备型号（如 `Apple M1 Max`）

---

## 3. Qualcomm Snapdragon QNN EP (Android)

### 3.1 HTP (Hexagon Tensor Processor)
- 面向骁龙 8 Gen 1/2/3/4 及 X Elite 处理器的专有 NPU 架构。
- 要求模型在固定 shape 和量化（INT8 / FP16）下编译为 QNN context cache。
- 性能模式支持：`burst`、`sustained_high_performance`、`high_performance`、`balanced`、`power_saver`。

### 3.2 16 KB Page Compatibility
- Android 15 及后续平台要求原生二进制支持 16 KB 内存分页对齐。
- 本项目编译产物及链接依赖遵循 `-Wl,-z,max-page-size=16384` 规范，全面兼顾 4 KB 与 16 KB 页面架构。
