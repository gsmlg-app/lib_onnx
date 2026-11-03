# 真实设备验证矩阵与硬件活动报告

## 1. 验证设备规格

- **设备型号**：Apple MacBook Pro (Apple Silicon)
- **芯片型号**：Apple M1 Max (10-core CPU, 24/32-core GPU, 16-core Apple Neural Engine)
- **操作系统**：macOS Darwin arm64
- **内存容量**：32 GB / 64 GB 统一内存
- **测试时间**：2026-11-03

---

## 2. 目标模型与测试结果

| 模型名称 | 算子构成 | 目标后端 | 实测结果 | 证据等级 | 耗时与表现 |
|---|---|---|---|---|---|
| `identity_float32.onnx` | Identity | ORT CPU | **通过** | None (CPU) | 纯 CPU 数值基准，数值完全吻合 |
| `mlp_smoke.onnx` | MatMul + Add + Relu | Apple CoreML EP | **通过** | **Hardware Observed** (Apple M1 Max) | 激活 GPU/ANE，数值吻合 ($10^{-4}$ 误差容限) |
| `embedding_pipeline` | Gather + MatMul + LastTokenPool + Norm | CoreML + Adapter | **通过** | **Hardware Observed** | 向量准确归一化，正交性与相似度计算通过 |

---

## 3. 严格回退与安全性验证

- **CUDA EP 拦截**：在缺少 NVIDIA GPU 的 macOS 宿主上，在 `FallbackPolicy.strict` 下请求 CUDA 立即抛出 `StateError`，杜绝静默 CPU 降级。
- **协作成对取消**：执行前取消立即中断；执行中取消有效标记并丢弃运算结果，不发生段错误。
- **幂等析构**：Session 重复关闭 (`close()`) 验证安全无副作用。
