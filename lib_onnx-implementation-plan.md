# lib_onnx 实施计划：面向 Dart / Flutter 的硬件加速 ONNX 推理库

项目：`gsmlg-app/lib_onnx`  
发布包：`lib_onnx`  
日期：2026-10-31  
状态：待实施；以下为目标和验收要求，不代表已经具备相关能力。

## 1. 项目目标与边界

基于 ONNX Runtime（ORT）构建独立 Dart / Flutter 推理库，将真实 GPU/NPU 执行作为首版交付目标。应用可以配置执行后端、加载兼容模型、异步执行推理、检查加速与回退证据，并确定性释放原生资源。

CPU 用于数值基准和显式允许的兼容回退。只有 CPU 推理、只有 provider 枚举、只成功创建 Session，均不能视为项目完成。

库保持模型无关：不内置 Qwen 权重，不承担聊天、memory recall、向量数据库、模型下载服务或 tokenizer 业务。但仓库需要提供真实 embedding 示例和可重复的模型准备工具，以验证实际用途。

首个真实工作负载为 Qwen3-Embedding-0.6B；它必须从 PoC 开始验证，而不是等通用接口全部完成之后才尝试。

## 2. 版本范围

### v0.1：Apple 与 Android 的真实加速

| 平台 | 首批架构 | 必须交付的后端 | 验证目标 |
|---|---|---|---|
| macOS | arm64 | CoreML EP | 分别验证 Apple GPU、ANE 路径 |
| iOS / iPadOS | arm64 真机 | CoreML EP | 合格设备上的 GPU、ANE 执行 |
| Android | arm64-v8a | QNN EP / HTP | 明确支持矩阵中的 Qualcomm NPU |
| 上述平台 | 对应架构 | ORT CPU | 正确性基准、显式回退 |

QNN GPU 是单独的附加 profile，不是 HTP 的别名，也不能代表所有 Android GPU。仅在选定 SDK、ORT 构件、设备和模型组合通过验证后发布支持声明。当前 QNN 文档列出了 GPU 和 HTP 两类后端。[S2]

模拟器只承担编译、链接及 API 辅助测试，不作为 GPU/NPU 真机支持证据。Intel macOS、Android 32 位、Web 和 tvOS 不进入 v0.1。

### v0.2：桌面 GPU/NPU 扩展

Linux x86_64 首先接入 NVIDIA CUDA EP；Linux ARM64 / DGX 类目标另设构件和验证配置，不能复用 x86_64 的通过结论。

Windows 先完成 Windows ML 与独立 ORT/DirectML 集成路线的验证和选择，再交付明确系统范围的 GPU 支持；Intel OpenVINO、Windows Snapdragon QNN 等 NPU profile 分设备扩展。TensorRT、AMD Ryzen AI 不作为最初版本的泛化承诺。

DirectML 仍受支持，但官方说明新的 Windows 功能开发转向 Windows ML。CUDA 需要匹配 ORT、CUDA、cuDNN 与驱动组合。[S3][S4]

### TPU 与其他非 ORT 路线

不添加不能工作的通用 `tpu` provider。Pixel Tensor 及其他通过 LiteRT 接入的 NPU，留给独立通用 LiteRT 运行库；不要在 `lib_onnx` 内暗中切换推理引擎。应用层保留替换推理实现的边界即可。[S8]

## 3. 架构决策

### 3.1 调用边界

应用 / Embedding Adapter → 公共 Dart API → 自动生成的 FFI bindings → 轻量 C ABI Bridge → 固定版本 ORT → 合格 EP。

桥接实现使用 C 或少量 C++，对外只导出 C ABI。负责 opaque handle、版本协商、错误映射、原生任务与生命周期，不重复实现 Tensor 引擎、图优化或执行器。仅为了包装已有 C API，不额外引入 Rust。

配置使用不可变数据；后端选择、构件选择、模型兼容检查尽量写成可独立测试的纯函数。原生状态通过显式 handle 管理，不建立庞大的继承体系，也不为每个平台复制一套 Session API。

### 3.2 包结构

首版只发布一个公共 `lib_onnx` 包。内部按 runtime、session、tensor、provider、diagnostics、FFI、native bridge、build hook、模型工具、example 和测试分层。

只有平台构建工具确实需要时才拆出窄职责的集成包；不预先复制多包 federated 架构。普通桌面 Dart 程序不应依赖 Flutter engine。

Flutter example 遵循 GSMLG `flutter-app-template` 中适用的工程约定，但库本身不能变成完整应用模板。MethodChannel 仅用于确有需要的平台能力，不传输大批量 tensor。

### 3.3 原生集成

优先 `package_ffi` 和 build hooks。P0 验证 Apple 链接、传递依赖、符号、签名及 device/simulator 打包；如果固定工具链不能完整处理某项依赖，允许增加最小平台适配层，不能先假定 hooks 会解决一切。[S5]

每个目标应用选择一套一致的 ORT 构件组合。可选 EP 包不能各自引入互不兼容的 ORT。检查最终应用原生依赖图，并记录与其他 ORT 插件共存的限制。

## 4. API 与运行时契约

### 4.1 核心概念

Runtime 负责桥接/ORT 版本和能力查询；Session 负责模型加载、元信息、任务提交、诊断与关闭；Tensor 描述 dtype、shape、内存位置和所有权。

SessionOptions 包含 provider 顺序、执行策略、线程设置、优化、缓存及队列预算。RunOptions 包含请求 ID、deadline 和独立取消状态。DiagnosticReport 记录构件、后端、设备观察、图分配、回退和证据等级。

常规 API 不暴露原生指针。首批数值 tensor 支持 float32、float16 存储、int64、int32、int8、uint8、bool；未实现的 string、sequence、map、sparse、bfloat16 等返回明确错误，不能声称支持所有 ONNX 值。

支持文件路径和 ONNX external-data 文件。内存加载是额外路径，不能要求大模型先被完整、多次复制进 Dart 字节缓冲区。

### 4.2 执行策略必须拆成三个维度

1. Provider 顺序：允许哪些兼容后端参与执行。
2. ORT CPU 回退：是否允许 CPU 节点，以及模型专属的少量控制算子白名单。
3. 硬件要求：是否必须有指定 GPU/NPU 实际参与的合格证据。

默认应用路径优先合格加速器；CPU 兼容必须由配置明确允许。请求的加速器失败时，不能悄悄换成 CPU。

`RequireProvider(CoreML)` 不等于 `RequireAccelerator(ANE)`。CoreML 的计算单元配置允许 CPU 与 GPU/ANE 共同参与，ORT 图分配和 CoreML 内部硬件选择是不同层级。[S1]

严格硬件要求无法被可靠验证时，返回错误或明确的 unverifiable 状态。不要根据设备名称、provider 列表或 Session 创建成功伪造证据；也不要承诺每个算子都只在 NPU 上执行。

### 4.3 能力与证据分层

分别记录：EP 已编入构件、依赖存在、设备初始化成功、指定模型编译成功、图被分配到 EP、compute plan 报告的硬件分配、运行时 trace 验证的活动。

报告包含来源、模型 hash、后端版本和证据类型。无法获知的信息标为 unknown，不编造硬件利用率、分配百分比或可用状态。

优先稳定 API。解析日志只能是受版本约束的诊断方法，不能假装是稳定 ABI。CoreML compute plan 与估计耗时不等于逐次运行的实测利用率；ORT 节点数量也不等于计算耗时占比。[S1]

### 4.4 异步、取消与所有权

模型加载/编译和推理都不得阻塞 Flutter UI isolate。采用有界 native job executor，或经过验证的 ORT 异步集成，并保留执行期间可响应的控制路径。[S7]

禁止“在 worker isolate 同步 Run，同时期待同一个被阻塞 isolate 接收取消消息”的设计。

首版默认每 Session 一个在途任务、有界队列；只有完成后端并发验证后才增加并行度。还需 runtime 级总预算，避免多个 Session 各自创建过多原生线程和缓冲区。

每次 run 独立取消，不能用全局取消标志误伤其他请求。区分取消已请求、结果已丢弃、原生执行已结束。取消是协作式的，不承诺即时打断所有加速器 kernel。

关闭流程：拒绝新任务 → 取消排队任务 → 请求取消在途任务 → 等待原生完成 → 释放资源。超时也不能释放仍被 native 使用的内存。

显式 close 是主要资源边界，finalizer 只兜底。不跨 isolate 传递裸 handle；在途引用必须保活 tensor、Session 和 callback。覆盖 hot restart、迟到回调与重复关闭测试。

### 4.5 内存与数据传输

先使用安全的 CPU 输入/输出缓冲区及明确的所有权。生命周期正确后，再增加原生缓冲复用和后端专属 I/O binding；逐项说明同步与释放责任。不承诺跨设备通用零拷贝。[S7]

Embedding profile 应评估把 last-token 选取与归一化合入模型图，避免把完整 sequence hidden states 传回 Dart；只有语义一致性验证通过后才能采用。

## 5. 真实模型验证：从 P0 开始的并行主线

### 5.1 三层模型

小型确定性 ONNX fixture 用于 Tensor/FFI；小型真实神经网络用于快速加速器 smoke test；Qwen3-Embedding-0.6B 用于实际业务验证。

前两层通过不能证明 Qwen 通过。后端合格与模型 profile 合格必须分开记录。

### 5.2 Qwen 正确性

固定官方 checkpoint revision、tokenizer 文件、导出工具链、ONNX opset 和转换参数。量化前建立 PyTorch 参考结果。

验证 query instruction、document 输入、token IDs、padding、attention mask、最后有效 token pooling 和 L2 normalization。官方示例使用 last-token pooling，不能随意换成 mean/CLS pooling。[S6]

Tokenizer 与模型语义放在 example/adapter，不进入通用 runtime。模型准备可使用 Python；正式 Flutter 应用不得依赖 Python/PyTorch。

### 5.3 模型 profile

分别准备 CPU 参考、Apple CoreML 候选、QNN HTP 候选及后续 CUDA/Desktop profile。模型精度、量化格式与 shape 由目标后端决定，不要求一份 ONNX 文件运行所有设备。

先测试 batch=1 和 128/256/512 tokens 档位。它们是工程验证配置，不是模型上下文上限。按需加载并限制缓存，不同时常驻全部 shape 的 Session。长文本必须显式报错或由应用选择分块，不能静默截断。

QNN 候选按固定工具链支持的格式做校准量化与固定 shape；当前官方文档存在静态 shape 和算子限制，P0 必须针对所锁版本确认。[S2]

校准数据包含有代表性的中文、英文及代码；质量评估使用独立保留集，不只用随机数据“证明”量化正确。

Model manifest 记录所有文件/hash、external-data、checkpoint/tokenizer revision、输入输出契约、量化、shape 档位及后端合格结果。权重不进入 pub 包。

### 5.4 性能与质量

记录冷启动/编译、warm run、tokenization 到向量的端到端耗时、p50/p95、吞吐、峰值和稳态内存。只有具备可重复测量方法时才报告能耗。

尽可能同时进行同模型/同精度的 CPU-加速器比较，以及各后端最佳合格 profile 的端到端比较；明确标注精度、batch 和 shape 差异。

检验向量一致性及 Recall@10/NDCG@10 等检索指标。P0 固定各 profile 的容差和门槛，不能测试之后为过关随意放宽。

不预先承诺 NPU 总比 GPU/CPU 快。允许“功能可用但不推荐”的 profile；被推荐的配置必须在声明指标上有实测收益，且不越过质量/延迟/内存限制。

## 6. 实施阶段

### P0：兼容性与真实加速 PoC

锁定 ORT tag/C API、Flutter/Dart、Xcode/NDK、QNN SDK，建立设备矩阵。确认厂商运行库分发条件，再决定可发布的构件内容。

在真实 Apple 设备执行 CoreML 原生 PoC，在支持的 Android 设备执行 QNN HTP PoC。验证后端配置、模型加载、诊断和打包。同步进行 Qwen 导出、算子/shape 审计及真实模型尝试，不只测试 Add 模型。

在最小 Flutter consumer 中验证 Apple 原生链接和 Android 厂商依赖；建立精度、性能和内存基准。

交付：架构、兼容矩阵、版本锁、模型工具原型及设备证据。

验收：每个首版目标都有可重复的加速执行路径，或明确到算子、构件、设备、产物的阻塞与修复方向。阻塞项不能标为完成，也不能仅凭小模型通过宣称 Qwen 支持。

### P1：公共 FFI 运行时与生命周期

实现 C ABI、生成 bindings、Runtime/Session/Tensor、external-data 加载、结构化错误、有界异步任务、独立取消和确定性释放。增加 bridge/ORT 版本校验。

在 UI 之前实现能力与证据结构；配置不能缩减为单一 `useGpu` 开关。

验收：数值 fixture、非法输入、未支持类型、关闭/运行/取消竞态、迟到完成、重复生命周期和故障注入测试通过。加载与推理不阻塞 Flutter UI，P0 的加速路径已可由 Dart 调用。

### P2：Apple 与 Android 后端交付

Apple：实现计算单元配置、模型格式、shape 策略、编译缓存及诊断。分别验证 GPU 与 ANE profile。完成真实 macOS/iOS release 构建，不以模拟器取代真机。[S1]

Android：实现 QNN HTP 依赖、设备探测、固定 shape profile、profiling 及支持情况下的 compiled-context 缓存。记录 SDK、HTP 架构、设备/OS 和依赖；不支持时给出可操作错误。QNN GPU 单独验证。[S2]

验收：每个声明支持的平台/后端都有模型 hash、请求策略、证据等级、回退、精度、耗时和内存报告。缺少设备的目标标为未验证，不能用 mock 通过替代。

### P3：Qwen 验证闭环与 Flutter example

本工作从 P0 开始，与 P1/P2 并行；P3 是闭环，而不是此时才开始尝试 Qwen。

完成可重复导出/量化、模型 manifest、独立 tokenizer/预后处理 adapter 和 shape 档位验收。Flutter example 展示模型、请求后端、实际证据、文本 embedding、CPU 对比和报告导出。

UI 必须区分请求硬件、选中 EP、报告的硬件分配和运行时确认。混合执行或 unverifiable 状态不能显示为无条件“NPU 已启用”。

验收：所有声明支持的 Qwen/model/backend profile 端到端通过。Qwen 被阻塞的目标保留具体问题与已验证的通用 runtime 能力，但不能宣称 GSMLG embedding 目标完成，更不能悄悄换成小模型。

### P4：可重复分发、稳定性与 v0.1 发布

原生打包从 P0 开始；本阶段完成正式 consumer 和发布验证。

原生构件 manifest 记录 bridge ABI、ORT revision、EP/SDK、架构、系统要求、hash 与 notices。二进制放入版本化 Release 构件，build hook 按目标选择已验证组合；pub 包保存绑定、构建逻辑与必要元数据。

支持可信版本元数据中的 SHA-256 校验、原子安装、缓存、本地构件覆盖和离线构建。移动端启动后不下载可执行原生库；模型数据分发单独处理。

只分发允许再分发的厂商文件。必须由开发者提供的 SDK 要明确说明，不能把依赖不完整的路径宣传为开箱即用。

验证 Apple 签名/导出、Android release 打包及全部 native 依赖的 16 KB page compatibility。用仓库外干净 consumer 验证，不依赖未发布 path dependency 或未记录的本机 SDK。

覆盖 ABI 不匹配、缺少 backend/driver、损坏缓存、内存不足、重复加载/卸载、队列过载与应用暂停/恢复。支持生命周期通知不意味着保证无限后台执行。

验收：pub 发布校验、真实加速证据、Qwen 声明、构件清单、干净消费测试和文档一致。CPU-only 不是 v0.1 加速交付；preview 可明确未完成目标，但不能当作整个计划完成。

### P5：Windows/Linux GPU/NPU 扩展

实现 CUDA 构件、匹配依赖、设备选择、内存统计和合格 I/O binding，先验证 Linux。Linux ARM64 与 Windows CUDA 各自设置 profile。[S4]

完成 Windows ML 与 standalone ORT/DirectML 路线决策及系统兼容矩阵，交付选择的 GPU 路径。Intel/Qualcomm Windows NPU 只在硬件合格后加入。[S3]

验收：每个桌面目标满足与 v0.1 相同的数值、加速证据、模型、生命周期和发布标准。只有 API 一致不代表硬件支持。

## 7. 构件、模型与缓存规则

不使用未锁定的 latest。区分 Dart 包版本、bridge ABI、ORT 版本和模型 revision。

编译模型/context 缓存键包含模型及 external-data hash、shape、精度、ORT/EP/SDK、provider options 和相关设备/系统身份。按后端兼容规则处理，不能假设 compiled context 可跨 SoC 复制；损坏缓存需诊断并在可行时重建。[S1][S2]

由应用选择安装和使用的模型 profile；`lib_onnx` 不在设备上偷偷量化或转换模型。

## 8. 测试与证据要求

CI/报告区分纯 Dart 单测、native 集成测试、真实设备加速验证。硬件测试跳过仍然是跳过，不能变成通过。

建议初始压力负载：100 次加载/运行/关闭循环、1,000 次连续推理、各生命周期阶段取消、队列过载及持续运行。区分 warmup、缓存和 allocator retention 与泄漏；要求固定负载下内存稳定，而非机械要求 RSS 回到初始值。

每份设备报告记录 commit、构件 hash、设备/OS、驱动/SDK、model profile、输入、trace、精度指标、p50/p95、内存和未解决限制。Compute plan 估计与实际运行 trace 分栏保存。

支持状态分为：后端/设备组合支持，以及模型/shape/精度/后端 profile 支持；二者不能互相替代。

## 9. 文档交付

- `docs/architecture.md`：边界、ABI、线程和所有权。
- `docs/implementation-plan.md`：实施状态及对应证据。
- `docs/backend-support.md`：平台、EP 与设备矩阵。
- `docs/compatibility.md`：工具链、原生依赖和最低版本。
- `docs/execution-policy.md`：provider、CPU 回退、硬件要求与证据语义。
- `docs/model-profiles.md`：转换、量化、tokenizer 和 shape。
- `docs/distribution.md`：构件、构建集成、离线使用与 notices。
- `docs/validation/`：可重复的设备/模型验证报告。

不以乐观 README badge 代替缺失证据。不为实现方便创建无关分支/worktree 或修改兄弟 runtime 项目；共享 GSMLG 依赖问题应记录到上游，而不是静默绕过。

## 10. 完成定义

v0.1：指定 Apple 与 Qualcomm 目标可从干净 Flutter consumer 使用发布的包/构件，执行有证据的加速计算，诚实暴露回退，满足数值和生命周期要求，并具备该版本声明的真实 Qwen profile 验证。

完整项目：Windows/Linux 加速目标满足相同门槛。TPU 和其他推理引擎保持独立后续范围，除非明确追加需求。

最终交付报告必须分别说明：实现完成、硬件验证、模型验证、发布完成和仍存阻塞，不能混为一个“完成”。

## 官方参考

[S1] ORT CoreML EP：`https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html`

[S2] ORT QNN EP：`https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html`

[S3] ORT DirectML / Windows ML 指引：`https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html`

[S4] ORT CUDA EP：`https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html`

[S5] Flutter 原生绑定：`https://docs.flutter.dev/platform-integration/bind-native-code`

[S6] 官方 Qwen3-Embedding-0.6B：`https://huggingface.co/Qwen/Qwen3-Embedding-0.6B`

[S7] ORT C API：`https://onnxruntime.ai/docs/api/c/struct_ort_api.html`

[S8] LiteRT NPU：`https://developers.google.com/edge/litert/next/npu`

工具链与 SDK 的确切版本、平台最低要求均须在 P0 固定并复验；持续更新的在线文档不能替代指定版本的测试证据。
