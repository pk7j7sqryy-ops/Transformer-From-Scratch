
## 一、文件结构总览

```
src/transformer_from_scratch/
├── config.py        ← 超参数配置
├── masks.py         ← Mask 生成
├── attention.py     ← 注意力机制
├── embeddings.py    ← 词向量 + 位置编码
├── layers.py        ← 层归一化、残差连接、Encoder/Decoder
├── model.py         ← 整体外壳、Generator
├── loss.py          ← Label Smoothing 损失
├── training.py      ← 训练循环
└── decoding.py      ← 贪心解码
```

## 二、模块说明

### model.py — 整体外壳

| 类 | 职责 | 输入 → 输出 |
|---|---|---|
| `EncoderDecoder` | 组装 encoder + decoder + embed + generator | `(src, tgt, src_mask, tgt_mask)` → `[B,T,D]` |
| `Generator` | 线性层 + log_softmax，映射回词表 | `[B,T,D]` → `[B,T,V]` |

### layers.py — 层组件

| 类/函数 | 职责 |
|---|---|
| `clones()` | 深拷贝 N 份模块，给 Encoder/Decoder 堆叠用 |
| `LayerNorm` | 层归一化，把 D 维向量拉回均值0方差1 |
| `SublayerConnection` | 残差连接壳：norm → 子层 → dropout → 加残差 |
| `EncoderLayer` | 单层 Encoder：Self-Attention + FFN |
| `Encoder` | N 个 EncoderLayer 堆叠 |
| `DecoderLayer` | 单层 Decoder：Masked Self-Attention + Cross-Attention + FFN |
| `Decoder` | N 个 DecoderLayer 堆叠 |
| `PositionwiseFeedForward` | FFN：两层线性层中间夹 ReLU |

### attention.py — 注意力机制

| 类/函数 | 职责 |
|---|---|
| `attention()` | 核心函数：Q·K/√d → mask → softmax → ·V |
| `MultiHeadAttention` | 多头注意力，拆成 H 个头并行算再拼回 |

### embeddings.py — 嵌入层

| 类 | 职责 |
|---|---|
| `Embeddings` | token ID → 词向量，再乘 √d_model |
| `PositionalEncoding` | 正弦余弦位置编码，加到词向量上 |

### masks.py — Mask 生成

| 函数 | 职责 | 输出 shape |
|---|---|---|
| `make_padding_mask` | 遮 PAD 填充位 | `[B,1,1,S]` |
| `make_causal_mask` | 遮未来 token | `[1,1,T,T]` |
| `make_decoder_mask` | 两者合并 & 操作 | `[B,1,T,T]` |

### config.py — 配置

| 类 | 职责 |
|---|---|
| `TransformerConfig` | 所有超参数 + 校验 |

### loss.py — 损失函数

| 类 | 职责 |
|---|---|
| `LabelSmoothing` | Label Smoothing 损失函数 |

### training.py — 训练

| 函数 | 职责 |
|---|---|
| 训练循环 | 前向传播 → loss → 反向传播 → 更新参数 |

### decoding.py — 推理

| 函数 | 职责 |
|---|---|
| 贪心解码 | 推理时逐 token 生成 |

## 三、依赖顺序（从底到顶）

```
config
  ↓
masks    embeddings     ← 并列
  ↓         ↓
attention  |
  ↓         ↓
layers ←───┘
  ↓
model（组装 layers）
  ↓
loss
  ↓
training → decoding
```

## 四、项目整体数据流

### 训练阶段

```
输入:
  src [B,S]   源语言 token ID（如中文）
  tgt [B,T]   目标语言 token ID（右移一位，如英文）

Step 1: 生成 Mask
  src [B,S] → make_padding_mask   → src_mask [B,1,1,S]
  tgt [B,T] → make_decoder_mask   → tgt_mask [B,1,T,T]

Step 2: Encoder
  src [B,S]
    → Embeddings + PositionalEncoding → [B,S,D]
    → Encoder (N层 × {Self-Attention(src_mask) + FFN})
    → memory [B,S,D]

Step 3: Decoder
  tgt [B,T]
    → Embeddings + PositionalEncoding → [B,T,D]
    → Decoder (N层 × {
        Masked Self-Attention(tgt_mask)    ← 自己看自己，防偷看
        Cross-Attention(src_mask)          ← 看 encoder 的 memory
        FFN
      })
    → hidden [B,T,D]

Step 4: Generator
  hidden [B,T,D]
    → Linear(d_model → vocab_size)
    → log_softmax
    → log_probs [B,T,V]

Step 5: Loss
  log_probs [B,T,V] + labels [B,T]
    → LabelSmoothing → loss
    → loss.backward() → 更新参数
```

### 推理阶段

```
Step 1: Encoder 只跑一次
  src [B,S] → Embeddings → Encoder → memory [B,S,D]

Step 2: Decoder 循环生成
  初始: tgt = [BOS]

  循环:
    tgt → Decoder(memory) → Generator
        → 取最后一个位置 argmax → 新 token
    tgt = tgt + 新 token
    直到生成 EOS 或达到最大长度

  最终: tgt = [BOS, token1, token2, ..., EOS]
```

## 五、Encoder vs Decoder 对比

| | Encoder | Decoder |
|---|---|---|
| 输入 | 源语言完整序列 | 目标语言（生成中的） |
| 子层数 | 2 个 | 3 个 |
| 第1层 | Self-Attention + src_mask | Masked Self-Attention + tgt_mask |
| 第2层 | FFN | Cross-Attention(Q=decoder, K/V=memory) |
| 第3层 | — | FFN |
| 作用 | 把源序列编码成 memory | 逐步生成目标 token |
| 推理时 | 只跑一次 | 循环调用 |

## 六、Mask 使用对照

| Mask | shape | 用在哪 | 遮什么 |
|---|---|---|---|
| src_mask (padding) | `[B,1,1,S]` | Encoder 每层 Self-Attention | PAD 填充位 |
| tgt_mask (padding+causal) | `[B,1,T,T]` | Decoder 第1层 Masked Self-Attention | PAD + 未来 token |
| src_mask (padding) | `[B,1,1,S]` | Decoder 第2层 Cross-Attention | PAD 填充位 |

---
## 七、开发流程
Phase 1: 搭骨架（先看到森林）
  ├── model.py: EncoderDecoder        ← ✅ 已完成
  └── model.py: Generator             ← ✅ 已完成

Phase 2: 基础组件（每棵树的树干）
  ├── layers.py: clones + LayerNorm + SublayerConnection
  ├── layers.py: PositionwiseFeedForward
  └── layers.py: EncoderLayer + Encoder + DecoderLayer + Decoder

Phase 3: 核心机制（树的细节）
  ├── attention.py: attention() + MultiHeadAttention
  └── embeddings.py: Embeddings + PositionalEncoding

Phase 4: 辅助工具
  ├── masks.py: make_padding_mask + make_causal_mask + make_decoder_mask
  └── config.py: TransformerConfig

Phase 5: 训练与推理
  ├── loss.py: LabelSmoothing
  ├── training.py: 训练循环
  └── decoding.py: 贪心解码

Phase 6: 组装运行
  ├── scripts/train_copy.py: 跑 Copy Task
  └── scripts/infer_copy.py: 推理验证