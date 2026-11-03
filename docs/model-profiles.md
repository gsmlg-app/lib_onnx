# 模型配置与 Qwen3-Embedding 规范

## 1. 真实目标工作负载：Qwen3-Embedding-0.6B

`lib_onnx` 将 `Qwen3-Embedding-0.6B` 作为 P0 到 P3 真实业务验证的首要工作负载。

### 1.1 输入契约
- `input_ids`：维度 `[batch_size, sequence_length]`，类型 `int64`；
- `attention_mask`：维度 `[batch_size, sequence_length]`，类型 `int64`（有效 token 处为 1，padding 处为 0）。

### 1.2 输出契约与 Pooling 规范
- **关键规则**：Qwen 官方 Embedding 模型采用 **Last-token pooling**（取每个序列最后一个有效 non-padding token 的 hidden state），**严禁随意替换为 Mean pooling 或 CLS pooling**。
- `EmbeddingAdapter` 实现 Last-token 提取：
  ```
  for each sequence b in batch:
      last_valid_idx = max(index where attention_mask[b][index] == 1)
      embedding[b] = hidden_states[b, last_valid_idx, :]
      normalize_l2(embedding[b])
  ```
- **L2 归一化**：向量必须除以欧几里得范数 $\|v\|_2$，以便后续通过点积直接计算余弦相似度（Cosine Similarity）。

---

## 2. 形状档位（Shape Tiers）

针对端侧硬件加速特性（尤其是 NPU 编译需要固定静态 shape），模型推荐采用档位化编译：
- **Tier 128**：`batch_size = 1`, `sequence_length = 128` (短文本、关键词、指令)
- **Tier 256**：`batch_size = 1`, `sequence_length = 256` (常规段落)
- **Tier 512**：`batch_size = 1`, `sequence_length = 512` (长文档片段)

> 规则：长文本输入必须显式抛出超出上下文错误或由应用层主动分块，严禁静默截断丢失语义。

---

## 3. 模型清单规范 (`manifest.json`)

每个用于生产部署的模型必须附带 `ModelManifest` 清单：
- 记录模型名、官方权重 revision、tokenizer revision；
- 记录 ONNX 文件名及 SHA-256 完整性哈希；
- 记录外部数据文件（external data files）；
- 记录经实测验证合格的 Profile 列表（包含目标后端、精度、批次大小与验证状态）。
