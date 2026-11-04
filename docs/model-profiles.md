# 模型配置、真实 Qwen 验证与证据边界

截至 2026-11-04，仓库内的小模型与 Adapter 单测不能证明 Qwen3-Embedding 已运行，也不能证明 ANE/GPU/HTP 已使用。历史 `qwen3_embedding_fixture` 是随机权重 Gather/MatMul 合成图，不是 Qwen；旧生成器还在未执行模型时写入 `verified: true`，该路径已移除。历史实现说明可从 Git 历史查阅。

## 已实现的模型链路

Flutter 示例加载实际随包发布的 `example/assets/mlp_smoke.onnx`，调用异步 `loadBytes` 和 `runWithDiagnostics`，检查 `relu([-1, 0, 1, 2] + 0.5) = [0, 0.5, 1.5, 2.5]`，展示本次运行诊断。模型 SHA-256 为 `cf26ee02305c2b808e9dd73ba908de34c5341ea690360ace4eaba53bdec3b775`。该模型是最小执行链检查，没有 tokenizer 或语义检索能力。

`EmbeddingAdapter` 接受 fp32 或显式转换的 IEEE binary16 输出。`hasEmbeddedPooling: false` 要求 `[batch, sequence, hiddenDim]` 与二元 mask `[batch, sequence]`；`true` 要求 `[batch, hiddenDim]`。两者都校验真实 hiddenDim、rank、非空维度与有限值；全零 mask、零范数、NaN/Inf 会报错。mask 支持 int64/int32/uint8/bool；从后向前找最后有效 token，兼容左右 padding。没有 mask 时不会猜测 padding。缩放后计算 L2 范数，避免很小的有限向量被原样返回。

## 固定真实模型与工具链

官方模型：`Qwen/Qwen3-Embedding-0.6B`。checkpoint 与 tokenizer 均固定到同一真实提交：

`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`

来源为 [官方 HF API](https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B) 和 [该提交配置](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json)。配置为 28 层、hidden size 1024、原始权重 bf16。官方仓库不包含 ONNX，本轮采用“读取已导出的真实构件”路线，不提供未经运行验证的通用导出器。

验证工具链锁定在 `tools/qwen-validation-requirements.txt`：Python 3.11、PyTorch 2.7.1、Transformers 4.53.3、tokenizers 0.21.2、ONNX 1.18.0 等。脚本检查对应实际版本；这些版本是可复现输入，并不表示本机已经安装或完成导出。导入构件的默认 ONNX opset 必须是 17。生产者还必须提供其**实际**导出命令和精确工具链，不能把本仓库验证工具版本冒充为导出版本。

```sh
python3.11 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install -r tools/qwen-validation-requirements.txt
.venv-qwen/bin/python tools/export_embedding.py acquire --output .cache/qwen-checkpoint
```

`acquire` 只获取固定 revision 的配置、权重与 tokenizer，生成每个文件的实际 SHA-256 和 snapshot inventory。离线环境添加 `--offline`，需要对应内容已在本地 HF 缓存；缺失会明确失败。不要提交权重、external data 或本地虚拟环境。

## 导入已导出的 ONNX

准备真实模型及其 external data，和导出者的 `provenance.json`。后者必须包含 `checkpoint_revision`、`tokenizer_revision`（均为上面的固定提交）、非空 `toolchain` 对象（精确版本）及实际 `command`。这些是生产者声明；只有随后与真实 checkpoint 的数值比较才能建立推理一致性，清单不会据此自动标记验证通过。

```sh
.venv-qwen/bin/python tools/export_embedding.py import \
  --model /absolute/path/to/export/model.onnx \
  --provenance /absolute/path/to/export/provenance.json \
  --output /absolute/path/to/export/manifest.json \
  --output-name last_hidden_state \
  --profile-id cpu-fp32-b1-s128 --target-backend cpu --precision fp32 \
  --sequence-length 128
```

`--output-name` 必须来自实际 ONNX 元信息；包含 pooling 的 rank-2 导出需显式添加 `--embedded-pooling`。工具读取图中全部输入，支持实际声明的 `input_ids`、`attention_mask`、`position_ids`、`token_type_ids`，未知输入或 KV-cache 契约会明确拒绝。不会静默省略第三个输入。输入 rank、类型、静态 batch/sequence、输出 rank/hiddenDim、外部数据路径/offset/length 均检查；ONNX checker 也必须通过。

清单记录原始 ONNX SHA-256、全部 external data SHA-256、真实输入输出、opset、导出者声明与待验证 profile。所有导入 profile 的 `verified` 初始为 false。`sha256` 仅覆盖 ONNX 文件；`model_identity_sha256` 覆盖 ONNX 与所有 external data。后者与 Dart Session 相同：主文件路径归一为 `model.onnx`，主文件在前、external data 路径排序；每条记录以 `path,size,sha256` 顺序编码为紧凑 UTF-8 JSON 数组，再求 SHA-256。符号链接逃逸、绝对路径、`..`、缺文件和内容变更均拒绝。

## 真实文本到 Dart/native 检索验证

```sh
.venv-qwen/bin/python tools/prepare_qwen.py \
  --manifest /absolute/path/to/export/manifest.json \
  --model-directory /absolute/path/to/export \
  --checkpoint .cache/qwen-checkpoint \
  --padding-side left --output .cache/qwen-left-reference.json

dart run tools/validate_qwen.dart \
  --manifest /absolute/path/to/export/manifest.json \
  --model-directory /absolute/path/to/export \
  --reference .cache/qwen-left-reference.json \
  --backend cpu --report .cache/qwen-cpu-left-report.json
```

准备脚本校验真实 checkpoint/tokenizer 文件 inventory；使用官方查询指令格式、两个查询和两个文档，实际 tokenizer 生成输入，实际 PyTorch CPU fp32 eager 权重产生数值基准。按实际导出元信息构造 position_ids；若图要求 tokenizer 没有生成的 token_type_ids，会拒绝，不能伪造。超出 profile 长度直接失败，不截断。使用固定 batch 1、sequence 128（可另建 256/512 profile），不声称动态形状已经通过加速器编译。

Dart 脚本重新计算所有构件哈希，加载实际模型，逐个调用 ONNX 推理、`EmbeddingAdapter` last-valid-token pooling、L2 normalization，再与真实 checkpoint 向量比较。默认 cosine >= 0.999 且最大绝对误差 <= 0.01；阈值可显式提供并记入报告，不能将误差调宽后声称原精度通过。最后检查两条查询各自的正确 top-1 文档，并记录真实耗时。该四文本检查是正确性/检索 smoke test，不是 MTEB 分数。

对右 padding 重跑准备脚本 `--padding-side right` 和 Dart 验证。对 CoreML 使用 `--backend coreml`；对 QNN HTP 使用 `--backend qnn`，还需要相应真实构件、SDK/库及设备。策略为禁止 ORT CPU fallback 的加速偏好，报告保留未知硬件证据；数值通过、EP 注册或芯片名称都不升级为 ANE/GPU/NPU 运行观测。物理硬件验收仍需设备、task 对应 trace/compute-plan/运行证据。CPU 基准通过也不代表加速平台通过。

## 本轮实际状态与可重复检查

- Adapter 和 ModelManifest 的 Dart 单测覆盖 rank/hiddenDim/mask/左右 padding/全零 mask/fp16/非有限值/零范数，以及外部数据缺失校验。
- `python3 tools/test_qwen_profile.py` 只检查 inventory 安全和 external data 变更检测；不加载 Qwen。
- `python3 tools/export_embedding.py --help`、`python3 tools/prepare_qwen.py --help` 与静态分析只验证工具入口/代码。
- 本机 Python 3.11.15 可用，但没有 torch、Transformers、ONNX、HF Hub、NumPy，也没有真实 Qwen ONNX export、checkpoint/tokenizer snapshot。`acquire --offline` 报缺依赖；本机经配置代理访问 HF API 超时。服务端只读连接器获取元信息不代表本机已取得权重。因此真实 Qwen CPU/CoreML/QNN 数值与检索均为 **not-run / blocked**，不存在模型哈希或性能结果可填。

缺构件时继续通用运行库修复；出现实际算子、shape、精度或内存限制时保留原始错误与 profile，停止对应模型后端验收，不以手写 hidden states 或合成图代替。
