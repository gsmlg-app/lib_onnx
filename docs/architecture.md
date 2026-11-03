# lib_onnx 架构设计与系统规范

## 1. 系统层次与调用边界

`lib_onnx` 遵循分层设计，对外导出纯 Dart 面向对象 API，内部通过 Dart FFI 调用轻量级 C ABI 桥接层，再桥接至官方 ONNX Runtime（ORT）C API 与各硬件加速 Execution Provider（EP）：

```
+-------------------------------------------------------------+
|               应用层 / Flutter UI Isolate                     |
|  (InferenceSession, SessionOptions, RunOptions, OnnxTensor)  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  Dart FFI 抽象与内存安全层                    |
|  (LibOnnxBindings, NativeLibrary, CStructures, Adapter)     |
+-------------------------------------------------------------+
                              |
                              v (Pure C ABI)
+-------------------------------------------------------------+
|                  lib_onnx Native C Bridge                   |
|  (lib_onnx_bridge.c, 有界工作线程池, 请求级取消, 诊断报告)      |
+-------------------------------------------------------------+
                              |
                              v (ORT C API)
+-------------------------------------------------------------+
|                 ONNX Runtime Core (v1.31.1)                 |
|     +--------------------+----------------------------+     |
|     |  Apple CoreML EP   |      Qualcomm QNN EP       |     |
|     | (Apple GPU / ANE)  |     (Hexagon HTP NPU)      |     |
|     +--------------------+----------------------------+     |
|     |                   ORT CPU Engine                |     |
|     +-------------------------------------------------+     |
+-------------------------------------------------------------+
```

### 1.1 纯 C ABI 契约
- 桥接层对外仅导出 `lib_onnx.h` 中定义的 C 符号，完全隐藏 C++ 符号和 STL 容器，确保跨不同平台编译器（Clang/GCC/MSVC）和 Dart FFI 版本的稳定二进制兼容性。
- 原生状态使用非透明指针（Opaque Handles：`LibOnnxRuntimeHandle`, `LibOnnxSessionHandle`, `LibOnnxTensorHandle`, `LibOnnxDiagnosticReportHandle`）传递，避免裸指针跨 Isolate 悬垂。

## 2. 线程模型与异步执行

### 2.1 保护 Flutter UI Isolate
- 模型加载、图编译和模型推理均在后台 Native Worker 线程池（通过 `pthread` 实现）中有界排队执行，严禁阻塞 Flutter UI 渲染线程。
- Dart 层提供 `Future<Map<String, OnnxTensor>> run(...)` 接口，等待底层执行完毕后通过 Future 完成结果投递。

### 2.2 请求级独立取消
- 每次推理调用分配全局唯一 `requestId`。
- 支持 `CancellationToken` 协作式取消：
  - 排队期间取消：从任务队列中直接剔除，返回 `LIB_ONNX_ERR_CANCELLED`。
  - 执行期间取消：设置原子标志位，推理完成后丢弃结果并立即释放临时原生缓冲区，不污染或中断其他并发任务。

### 2.3 确定性释放与生命周期安全
- `InferenceSession.close()` 为主要资源回收边界，操作幂等（多次调用安全）。
- 关闭流程：
  1. 标记 `is_closed = 1`，拒绝后续新任务；
  2. 取消队列中尚未运行的任务；
  3. 等待正在执行的原生内核执行完毕；
  4. 确定性释放 OrtSession、元数据和模型图句柄。

## 3. 内存所有权与传输策略

- **CPU 输入/输出**：Dart `OnnxTensor` 封装底层直接内存缓冲区（`Float32List`, `Int64List` 等）。输入张量通过 `copy_data = 1` 移交原生张量所有权，保证在 Dart GC 动作下原生计算依然安全。
- **输出内存**：由 ORT Allocator 生成并绑定在 `OrtValue` 上，Dart 包装层读取后确定性析构原生张量，杜绝内存泄漏。
- **Hot-Restart 安全**：即使 Flutter 进行 Hot Restart，原生线程池由于宿主进程保活，会在超时或任务结束后自动清理未认领任务，不引发段错误。
