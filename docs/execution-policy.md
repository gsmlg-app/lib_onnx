# 执行策略与硬件证据规范

## 1. 执行策略的三大正交维度

根据架构决策，执行策略严禁被简化为单一的 `useGpu` 布尔开关，必须拆分为三个独立控制维度：

### 1.1 维度 1：Provider 优先级列表 (`providerPriority`)
- 按照优先级从高到低声明允许参与模型执行的后端引擎。
- 示例：`[ExecutionProvider.coreML, ExecutionProvider.cpu]` 表示优先尝试 CoreML，若 CoreML 不支持或回退允许则由 CPU 接管。

### 1.2 维度 2：CPU 回退策略 (`fallbackPolicy`)
- **`FallbackPolicy.strict`**：
  - 严格加速模式。一旦指定了加速 Provider，如果环境缺少该 Provider、或者模型图因算子不兼容无法在加速器上运行，**Session 创建或推理将立即报错并终止**，坚决不进行静默 CPU 降级。
- **`FallbackPolicy.allowed`**：
  - 宽松兼容模式。加速器初始化失败或模型中有部分算子不被支持时，ORT 允许子图回退到 CPU 执行。
- **`FallbackPolicy.whitelistedOps`**：
  - 白名单控制模式。仅允许特定控制算子或形状计算算子（如 `Shape`, `Gather`, `Unsqueeze`）在 CPU 运行，核心矩阵乘法必须由加速器承载。

### 1.3 维度 3：硬件物理要求 (`hardwareRequirement`)
- **`HardwareRequirement.any`**：任何可用的物理单元。
- **`HardwareRequirement.gpu`**：强制要求设备必须包含物理 GPU，且参与模型计算。
- **`HardwareRequirement.npu`**：强制要求设备必须具备专用 NPU（如 Apple Silicon 的 Apple Neural Engine，或高通骁龙 Hexagon HTP）。

---

## 2. 能力与证据分级标准 (`EvidenceLevel`)

为防止虚假宣传加速状态，`lib_onnx` 建立透明可复核的 6 级证据体系：

| 等级名称 | 枚举代码 | 含义与判定依据 |
|---|---|---|
| **Unknown** | 0 | 未探测或尚未开始执行。 |
| **None (CPU)** | 1 | 纯 CPU 执行，未检测到加速器参与或加速器被拒绝。 |
| **EP Compiled** | 2 | 加速 Provider 动态库成功加载，SessionOptions 注册通过。 |
| **Graph Partitioned** | 3 | ORT 已成功将 ONNX 计算图切分出专属于加速 EP 的子图。 |
| **Hardware Observed** | 4 | 操作系统/驱动程序证实对应硬件引擎（GPU / ANE / NPU）实际接管并完成计算。 |
| **Unverifiable** | 5 | 硬件已接入，但受限于驱动接口缺少逐次运行的硬件活动遥测信号（诚实标记，不伪造指标）。 |

---

## 3. 诊断报告 (`DiagnosticReport`) 契约

每次推理调用均生成结构化 `DiagnosticReport`，包含：
- `activeProvider`：实际执行的核心后端；
- `evidenceLevel`：对应的证据等级；
- `durationUs`：端到端微秒级耗时；
- `durationMs`：毫秒级耗时；
- `hardwareDetails`：硬件芯片与驱动描述；
- `fallbackOccurred`：布尔值，是否发生了 CPU 回退；
- `modelHash`：用于溯源的模型校验哈希。
