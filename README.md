# lib_onnx

面向 Dart 和 Flutter 的硬件加速 ONNX Runtime 推理库，支持 Apple CoreML（Apple Silicon GPU / Neural Engine）与高通骁龙 QNN HTP（Hexagon NPU）。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dart](https://img.shields.io/badge/Dart-3.3%2B-blue)](https://dart.dev)
[![Flutter](https://img.shields.io/badge/Flutter-3.19%2B-blue)](https://flutter.dev)

---

## 核心特性

- **硬件加速优先**：
  - Apple macOS / iOS：通过 CoreML EP 调度 Apple GPU 与 Apple Neural Engine (ANE) 进行低延迟、高能效推理；
  - Android：支持高通骁龙 QNN HTP EP（Hexagon NPU），兼容 16 KB 分页标准；
  - 桌面 CPU：支持跨平台 ORT CPU 基准与显式兼容回退。
- **三维执行策略**：
  - 严禁单一 `useGpu` 模糊开关，清晰解耦 Provider 优先级、CPU 回退策略（Strict / Allowed / WhitelistedOps）与物理硬件要求（GPU / NPU / Any）。
- **透明证据分级 (`EvidenceLevel`)**：
  - 提供 6 级结构化诊断证据，如实反映硬件芯片活动（`hardwareObserved`）、图切分状态（`graphPartitioned`）或未加速状态（`none`），绝不伪造利用率。
- **内存安全与纯数值张量**：
  - 原生 C ABI 隔离，指针不跨 Isolate 暴露；
  - 首批支持数值张量：`float32`、`float16`、`int64`、`int32`、`int8`、`uint8`、`bool`；未实现的非数值类型严格报错。
- **请求级独立取消与有界原生线程池**：
  - 支持 `CancellationToken` 协作式取消单个推理请求；
  - 原生工作线程池确保 Flutter UI Isolate 永远不被模型加载或推理计算阻塞。
- **Qwen3-Embedding 业务适配器**：
  - 内置 `EmbeddingAdapter`，实现符合官方规范的 **Last-token pooling** 与 **L2 归一化**，不依赖 Python/PyTorch。

---

## 快速上手

### 1. 添加依赖

在 `pubspec.yaml` 中添加：

```yaml
dependencies:
  lib_onnx: ^0.1.0
```

### 2. 运行推理与读取硬件诊断

```dart
import 'dart:typed_data';
import 'package:lib_onnx/lib_onnx.dart';

Future<void> main() async {
  // 1. 配置 Apple Silicon 硬件加速策略 (强制优先 ANE，禁止隐式 CPU 回退)
  final options = SessionOptions.appleSilicon(
    preferAne: true,
    allowCpuFallback: false,
  );

  // 2. 从文件或内存中加载 ONNX 模型
  final session = InferenceSession.fromFile(
    'path/to/model.onnx',
    options: options,
  );

  // 3. 构造数值张量
  final inputData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
  final inputTensor = OnnxTensor.fromFloat32([1, 4], inputData);

  // 4. 异步执行推理 (UI 线程安全)
  final outputs = await session.run({'input': inputTensor});
  final resultTensor = outputs['output']!;
  print('Inference Output: ${resultTensor.asFloat32List()}');

  // 5. 检查硬件加速证据
  final report = session.lastDiagnosticReport!;
  print('Active Provider: ${report.activeProvider.ortName}');
  print('Evidence Level: ${report.evidenceLevel.description}');
  print('Latency: ${report.durationMs.toStringAsFixed(2)} ms');
  print('Hardware: ${report.hardwareDetails}');

  // 6. 确定性释放资源
  session.close();
}
```

### 3. Qwen3-Embedding 向量提取

```dart
final adapter = const EmbeddingAdapter(hiddenDim: 1024);

final embeddings = adapter.extractEmbeddings(
  modelOutput: hiddenStatesTensor,
  attentionMask: attentionMaskTensor, // 自动执行 Last-token pooling
);

// 计算余弦相似度
final similarity = EmbeddingAdapter.cosineSimilarity(embeddings[0], embeddings[1]);
print('Cosine Similarity: $similarity');
```

---

## 编译与测试

### 编译原生 C ABI 桥接库

```bash
cd native
make
```

### 运行全套测试

```bash
dart test
```

### 启动示例应用

```bash
cd example
flutter run
```

---

## 架构与交付文档

- [系统架构与 ABI 规范](docs/architecture.md)
- [实施进展与交付报告](docs/implementation-plan.md)
- [后端支持矩阵与硬件规格](docs/backend-support.md)
- [工具链与兼容性规范](docs/compatibility.md)
- [执行策略与硬件证据规范](docs/execution-policy.md)
- [模型配置与 Qwen3 规范](docs/model-profiles.md)
- [构件分发与部署规范](docs/distribution.md)
- [真实设备验证矩阵](docs/validation/device_matrix.md)
- [性能基准与压力测试报告](docs/validation/benchmark_report.md)

---

## 许可证

本项目遵循 MIT 协议。
ONNX Runtime 核心由微软开发并遵循 MIT 协议。
Apple CoreML 为 Apple Inc. 系统框架。
Qualcomm QNN 为高通公司专有软件开发包。
