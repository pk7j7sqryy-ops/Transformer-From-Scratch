# 从零手写 Transformer：工程开发指南

> 基于 Harvard NLP 的 [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)，目标是用现代 PyTorch 从零实现一个可训练、可测试的经典 Encoder–Decoder Transformer。

## 1. 文档定位

这不是一份可以直接复制运行的成品源码，而是一份开发规格书。你需要按照文件顺序亲手完成代码中的 `TODO`，再用对应测试验证。

完成本项目后，你应该能够：

- 不使用 `torch.nn.Transformer` 和 `torch.nn.MultiheadAttention`，手写完整 Transformer。
- 解释每个模块的输入、输出和张量 shape。
- 独立实现 padding mask、causal mask 和 decoder mask。
- 独立实现 Scaled Dot-Product Attention 和 Multi-Head Attention。
- 解释 Encoder self-attention、Decoder self-attention 和 cross-attention。
- 跑通 synthetic copy task，并用 greedy decoding 复原输入序列。
- 使用单元测试定位 shape、mask、数值和训练问题。

## 2. V1 范围

### 2.1 必须实现

- Encoder–Decoder 架构
- Token Embedding
- Sinusoidal Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Position-wise Feed-Forward Network
- 自定义 LayerNorm
- Pre-LN 残差结构
- EncoderLayer / DecoderLayer
- Encoder / Decoder 堆叠
- Generator
- Label Smoothing
- Transformer warmup 学习率
- Synthetic copy task
- Greedy decoding
- 单元测试和集成测试

### 2.2 暂不实现

- BPE 和真实语料分词
- torchtext、spaCy 和 WMT 数据集
- Beam Search
- KV Cache
- RoPE
- Decoder-Only Transformer
- 混合精度和分布式训练

先保证经典 Transformer 主链路正确，再扩展这些内容。

## 3. 与原教程的约定差异

本项目保留原教程的核心架构，但使用更清晰的现代写法：

| 项目 | 原教程 | 本项目 |
|---|---|---|
| Mask 类型 | 部分使用 `uint8` | 全部使用 `torch.bool` |
| Mask 含义 | `1` 表示可见 | `True` 表示可见 |
| Mask shape | 进入 MHA 后再扩维 | Mask 函数直接返回四维可广播 shape |
| 梯度处理 | 部分使用 `.data` | 禁止使用 `.data` |
| LayerNorm | 手写 | 手写，并使用 `unbiased=False` |
| 残差结构 | Pre-LN | Pre-LN |
| 训练设备 | 偏 CUDA 环境 | CPU、CUDA、MPS 均可 |
| 数据任务 | 翻译 + copy task | V1 只做 copy task |

本项目约定：所有布尔 Mask 中，`True` 表示该位置允许参与 Attention，`False` 表示该位置必须被屏蔽。

## 4. 项目目录

```text
transformer-from-scratch/
├── pyproject.toml
├── README.md
├── .gitignore
├── DEVELOPMENT_GUIDE.md
│
├── docs/
│   ├── architecture.md
│   └── tensor_shapes.md
│
├── src/
│   └── transformer_from_scratch/
│       ├── __init__.py
│       ├── config.py
│       ├── masks.py
│       ├── attention.py
│       ├── embeddings.py
│       ├── layers.py
│       ├── model.py
│       ├── loss.py
│       ├── training.py
│       └── decoding.py
│
├── scripts/
│   ├── train_copy.py
│   └── infer_copy.py
│
├── tests/
│   ├── test_config.py
│   ├── test_masks.py
│   ├── test_attention.py
│   ├── test_embeddings.py
│   ├── test_layers.py
│   ├── test_model.py
│   ├── test_loss.py
│   └── test_copy_task.py
│
└── artifacts/                 # 运行时生成，不提交 Git
    ├── checkpoints/
    └── logs/
```

## 5. 环境和工程配置

### 5.1 创建环境

```bash
cd 04_Other_Project/transformer-from-scratch
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

### 5.2 `pyproject.toml`

先创建以下配置，再执行可编辑安装：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "transformer-from-scratch"
version = "0.1.0"
description = "A tested encoder-decoder Transformer implemented from scratch"
requires-python = ">=3.10"
dependencies = [
    "torch",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"

[tool.ruff]
line-length = 88
target-version = "py310"
```

安装：

```bash
pip install -e ".[dev]"
pytest
```

### 5.3 `.gitignore`

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.DS_Store
artifacts/
```

不要把 checkpoint、日志、缓存和虚拟环境提交进仓库。

## 6. 全局编码约定

### 6.1 维度命名

| 名称 | 含义 |
|---|---|
| `B` | batch size |
| `S` | source sequence length |
| `T` | target sequence length |
| `D` | `d_model` |
| `H` | attention heads |
| `Dk` | 每个 head 的维度，`D // H` |
| `Vsrc` | source vocabulary size |
| `Vtgt` | target vocabulary size |

### 6.2 主要张量 shape

```text
src_tokens          [B, S]
tgt_tokens          [B, T]

src_embedding       [B, S, D]
tgt_embedding       [B, T, D]

src_mask            [B, 1, 1, S]
causal_mask         [1, 1, T, T]
tgt_mask            [B, 1, T, T]

query/key/value      [B, H, L, Dk]
attention_scores    [B, H, Lq, Lk]
attention_weights   [B, H, Lq, Lk]

encoder_memory      [B, S, D]
decoder_hidden      [B, T, D]
log_probs           [B, T, Vtgt]
```

### 6.3 特殊 Token

V1 copy task 使用：

```text
PAD = 0
BOS = 1
普通 token = 2 ... vocab_size - 1
```

V1 使用固定长度解码，暂时不需要 EOS。扩展真实文本任务时再加入 EOS。

### 6.4 每个模块的开发循环

每写一个模块都执行：

1. 先写函数签名和 docstring。
2. 在 docstring 写清输入、输出和 shape。
3. 先写对应单元测试。
4. 再完成实现。
5. 只运行当前测试文件。
6. 当前模块通过后才进入下一个模块。

例如：

```bash
pytest tests/test_masks.py -q
pytest tests/test_attention.py -q
```

## 7. M0：配置模块

文件：`src/transformer_from_scratch/config.py`

### 7.1 职责

- 集中保存模型超参数。
- 在模型创建前检查非法配置。
- 避免在不同文件中硬编码 `d_model`、head 数和 dropout。

### 7.2 代码骨架

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    src_vocab_size: int = 11
    tgt_vocab_size: int = 11
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 128
    pad_id: int = 0
    bos_id: int = 1

    def __post_init__(self) -> None:
        # TODO 1: 检查词表大小是否大于 2。
        # TODO 2: 检查 d_model 是否能被 num_heads 整除。
        # TODO 3: 为简化位置编码，检查 d_model 是否为偶数。
        # TODO 4: 检查层数、d_ff、max_seq_len 是否为正数。
        # TODO 5: 检查 dropout 是否在 [0, 1) 范围内。
        pass
```

因为 dataclass 设置了 `frozen=True`，非法配置应直接抛出 `ValueError`，不要尝试修改字段。

### 7.3 测试要求

文件：`tests/test_config.py`

至少测试：

- 默认配置可以成功创建。
- `d_model=127, num_heads=4` 抛出 `ValueError`。
- `d_model` 为奇数时抛出 `ValueError`。
- `dropout=1.0` 抛出 `ValueError`。
- 词表大小小于等于 2 时抛出 `ValueError`。

验收：

```bash
pytest tests/test_config.py -q
```

## 8. M1：Mask 模块

文件：`src/transformer_from_scratch/masks.py`

### 8.1 `make_padding_mask`

目标：屏蔽序列中的 PAD token。

接口：

```python
import torch
from torch import Tensor


def make_padding_mask(tokens: Tensor, pad_id: int) -> Tensor:
    """Return visible-key mask with shape [B, 1, 1, L]."""
    # tokens: [B, L]
    # TODO: tokens != pad_id 得到 [B, L]
    # TODO: 在 head 和 query 维度插入两个维度
    raise NotImplementedError
```

示例：

```text
tokens = [[1, 5, 8, 0, 0]]
mask   = [[[[T, T, T, F, F]]]]
shape  = [1, 1, 1, 5]
```

这里只屏蔽 Key 的 PAD 位置。Query 对应 PAD 的 loss 会在损失函数中忽略。

### 8.2 `make_causal_mask`

目标：Decoder 的位置 `i` 只能看到 `0...i`，不能看到未来位置。

接口：

```python
def make_causal_mask(length: int, device: torch.device | None = None) -> Tensor:
    """Return lower-triangular visible mask with shape [1, 1, L, L]."""
    # TODO: 创建 [L, L] 的 bool 下三角矩阵
    # TODO: 增加 batch 和 head 两个维度
    raise NotImplementedError
```

长度为 4 时结果应为：

```text
T F F F
T T F F
T T T F
T T T T
```

推荐使用 `torch.tril`，不要使用 Python 双重循环。

### 8.3 `make_decoder_mask`

目标：同时屏蔽 target padding 和未来 token。

```python
def make_decoder_mask(tokens: Tensor, pad_id: int) -> Tensor:
    """Return combined target mask with shape [B, 1, T, T]."""
    # padding_mask: [B, 1, 1, T]
    # causal_mask:  [1, 1, T, T]
    # 利用广播按位与，得到 [B, 1, T, T]
    raise NotImplementedError
```

### 8.4 测试骨架

文件：`tests/test_masks.py`

```python
import torch

from transformer_from_scratch.runner.masks import (
    make_causal_mask,
    make_decoder_mask,
    make_padding_mask,
)


def test_padding_mask_shape_and_values() -> None:
    tokens = torch.tensor([[1, 5, 0], [1, 0, 0]])
    mask = make_padding_mask(tokens, pad_id=0)

    assert mask.dtype == torch.bool
    assert mask.shape == (2, 1, 1, 3)
    assert mask[0, 0, 0].tolist() == [True, True, False]
    assert mask[1, 0, 0].tolist() == [True, False, False]


def test_causal_mask() -> None:
    mask = make_causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    assert mask.shape == (1, 1, 4, 4)
    assert torch.equal(mask[0, 0], expected)


def test_decoder_mask_is_broadcast_combination() -> None:
    tokens = torch.tensor([[1, 7, 0, 0]])
    mask = make_decoder_mask(tokens, pad_id=0)

    assert mask.shape == (1, 1, 4, 4)
    assert not mask[..., 2].any()
    assert not mask[..., 3].any()
```

## 9. M2：Scaled Dot-Product Attention

文件：`src/transformer_from_scratch/attention.py`

### 9.1 数学公式

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

### 9.2 接口

```python
from collections.abc import Callable

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Args:
        query: [B, H, Lq, Dk]
        key: [B, H, Lk, Dk]
        value: [B, H, Lk, Dv]
        mask: broadcastable to [B, H, Lq, Lk], True means visible

    Returns:
        context: [B, H, Lq, Dv]
        weights: [B, H, Lq, Lk]
    """
    # TODO 1: 读取 Dk。
    # TODO 2: 计算 query @ key.transpose(-2, -1)。
    # TODO 3: 除以 sqrt(Dk)。
    # TODO 4: 将 mask=False 的 score 填成极小值。
    # TODO 5: 对最后一个维度做 softmax。
    # TODO 6: 可选 dropout。
    # TODO 7: weights @ value。
    raise NotImplementedError
```

关键提示：

```python
key.transpose(-2, -1)
torch.finfo(scores.dtype).min
scores.masked_fill(~mask, ...)
torch.softmax(scores, dim=-1)
```

不要写死 `-1e9`，因为不同浮点 dtype 的范围不同。

### 9.3 必须手算一次的例子

设：

```text
Q = [[1, 0]]
K = [[1, 0], [0, 1]]
V = [[10, 0], [0, 20]]
Dk = 2
```

手工完成：

1. `QKᵀ`
2. 除以 `√2`
3. softmax
4. 权重乘 `V`

然后把结果写进测试，用 PyTorch 实现与手算结果比较。

### 9.4 测试要求

文件：`tests/test_attention.py`

至少测试：

- context 和 weights shape 正确。
- weights 最后一维求和约等于 1。
- mask 为 `False` 的权重约等于 0。
- 输入、Mask 在同一 device 上时可以运行。
- 反向传播后 query、key、value 都能得到梯度。

示例断言：

```python
assert context.shape == (2, 4, 5, 16)
assert weights.shape == (2, 4, 5, 5)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones_like(weights.sum(dim=-1)),
    atol=1e-6,
)
```

## 10. M3：Multi-Head Attention

仍在 `attention.py` 中实现。

### 10.1 数据流

```text
[B, L, D]
   │ Linear
   ▼
[B, L, H, Dk]
   │ transpose(1, 2)
   ▼
[B, H, L, Dk]
   │ Attention
   ▼
[B, H, L, Dk]
   │ transpose + contiguous + reshape
   ▼
[B, L, D]
   │ output projection
   ▼
[B, L, D]
```

### 10.2 代码骨架

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attention_weights: Tensor | None = None

    def _split_heads(self, x: Tensor) -> Tensor:
        """Convert [B, L, D] to [B, H, L, Dk]."""
        # TODO
        raise NotImplementedError

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Convert [B, H, L, Dk] to [B, L, D]."""
        # TODO: transpose 后必须 contiguous，再 reshape/view。
        raise NotImplementedError

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        # TODO 1: Q、K、V 分别线性投影。
        # TODO 2: 分成多个 heads。
        # TODO 3: 调用 scaled_dot_product_attention。
        # TODO 4: 保存 detach 后的 attention weights，供调试使用。
        # TODO 5: 合并 heads。
        # TODO 6: output projection。
        raise NotImplementedError
```

### 10.3 Cross-Attention 的 shape

Cross-attention 中 source 和 target 长度可能不同：

```text
query  来自 Decoder：[B, T, D]
key    来自 Encoder：[B, S, D]
value  来自 Encoder：[B, S, D]
score               ：[B, H, T, S]
output              ：[B, T, D]
```

不要假设 `Lq == Lk`。

### 10.4 测试要求

- self-attention：`Lq == Lk`。
- cross-attention：`Lq != Lk`。
- d_model 不能被 heads 整除时抛异常。
- 输出 shape 恢复成 `[B, Lq, D]`。
- 所有线性层参数均能收到梯度。

## 11. M4：Embedding 和位置编码

文件：`src/transformer_from_scratch/embeddings.py`

### 11.1 TokenEmbedding

```python
import math

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        # TODO: 查表并乘以 sqrt(d_model)
        raise NotImplementedError
```

输入 `[B, L]`，输出 `[B, L, D]`。

### 11.2 SinusoidalPositionalEncoding

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # TODO 1: pe = zeros([max_seq_len, d_model])
        # TODO 2: position = arange(max_seq_len).unsqueeze(1)
        # TODO 3: 计算偶数维对应的 div_term
        # TODO 4: 偶数维填 sin，奇数维填 cos
        # TODO 5: 增加 batch 维度，得到 [1, max_seq_len, d_model]
        # TODO 6: 使用 register_buffer 保存 pe

    def forward(self, x: Tensor) -> Tensor:
        # TODO 1: 取出与当前序列长度相同的 pe
        # TODO 2: 转为与 x 相同 dtype
        # TODO 3: x + pe 后做 dropout
        raise NotImplementedError
```

核心公式：

\[
PE(pos, 2i)=\sin\left(pos/10000^{2i/d_{model}}\right)
\]

\[
PE(pos, 2i+1)=\cos\left(pos/10000^{2i/d_{model}}\right)
\]

`pe` 不是可训练参数，所以必须用 `register_buffer`，不能用 `nn.Parameter`。

### 11.3 测试要求

- TokenEmbedding 输出 shape 正确。
- 同一个 token 在查表阶段得到相同向量。
- 加位置编码后，同一个 token 出现在不同位置时表示不同。
- `pe.requires_grad` 为 `False`。
- 模型迁移到可用 device 后，`pe` 自动迁移。
- 输入序列超过 `max_seq_len` 时给出清晰异常。

## 12. M5：基础网络层

文件：`src/transformer_from_scratch/layers.py`

### 12.1 自定义 LayerNorm

```python
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # TODO 1: 最后一个维度求 mean。
        # TODO 2: 最后一个维度求 variance，unbiased=False。
        # TODO 3: 标准化。
        # TODO 4: weight * normalized + bias。
        raise NotImplementedError
```

标准化使用：

```text
(x - mean) / sqrt(variance + eps)
```

不要把 `eps` 加在平方根外。

### 12.2 PreNormResidual

教程代码采用：

```text
x + Dropout(Sublayer(LayerNorm(x)))
```

```python
from collections.abc import Callable


class PreNormResidual(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        # TODO
        raise NotImplementedError
```

### 12.3 PositionwiseFeedForward

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Linear(D -> Dff) -> ReLU -> Dropout -> Linear(Dff -> D)
        raise NotImplementedError
```

FFN 对每个位置独立应用相同参数，不会在 sequence 维度之间交换信息。

### 12.4 EncoderLayer

```python
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attention: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.attention_residual = PreNormResidual(d_model, dropout)
        self.ffn_residual = PreNormResidual(d_model, dropout)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        # TODO 1: self-attention 中 Q=K=V=x。
        # TODO 2: 再经过 FFN 残差分支。
        raise NotImplementedError
```

数据流：

```text
x
├─ residual ───────────────────────────────┐
└─ LayerNorm -> SelfAttention -> Dropout ──┤ +
                                           ▼
                                           x1
                                           ├─ residual ───────────┐
                                           └─ LN -> FFN -> Dropout ┤ +
                                                                  ▼
                                                                  output
```

### 12.5 DecoderLayer

```python
class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        # TODO: 保存三个子层
        # TODO: 创建三个独立的 PreNormResidual

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        # TODO 1: masked self-attention，Q=K=V=x，使用 tgt_mask。
        # TODO 2: cross-attention，Q=x，K=V=memory，使用 src_mask。
        # TODO 3: FFN。
        raise NotImplementedError
```

三个子层的顺序不能交换：

```text
masked self-attention -> cross-attention -> FFN
```

### 12.6 测试要求

- LayerNorm 输出最后一维均值接近 0、方差接近 1。
- PreNormResidual 在 sublayer 返回全 0 时，输出等于输入。
- FFN 输入输出 shape 一致。
- EncoderLayer 输入输出 `[B, S, D]`。
- DecoderLayer 支持 `S != T`。
- Encoder 和 Decoder 中的不同层不能共享同一个参数对象。

## 13. M6：完整模型

文件：`src/transformer_from_scratch/model.py`

### 13.1 克隆层

```python
import copy


def clone_module(module: nn.Module, count: int) -> nn.ModuleList:
    """Create independent deep copies of one module."""
    # TODO: 使用 copy.deepcopy，不能重复引用同一个 module。
    raise NotImplementedError
```

错误写法：

```python
nn.ModuleList([module] * count)
```

它会让所有层共享同一套参数。

### 13.2 Encoder

```python
class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, num_layers: int, d_model: int) -> None:
        super().__init__()
        self.layers = clone_module(layer, num_layers)
        self.final_norm = LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        # TODO: 依次通过每一层，最后做 final_norm。
        raise NotImplementedError
```

### 13.3 Decoder

```python
class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, num_layers: int, d_model: int) -> None:
        super().__init__()
        self.layers = clone_module(layer, num_layers)
        self.final_norm = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        # TODO: 依次通过 DecoderLayer，最后做 final_norm。
        raise NotImplementedError
```

### 13.4 Generator

```python
class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Linear 后对词表维做 log_softmax。
        raise NotImplementedError
```

输出是 log-probabilities，不是普通 probabilities。

### 13.5 Transformer

```python
class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: nn.Module,
        tgt_embedding: nn.Module,
        generator: Generator,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.generator = generator

    def encode(self, src_tokens: Tensor, src_mask: Tensor) -> Tensor:
        # TODO
        raise NotImplementedError

    def decode(
        self,
        memory: Tensor,
        src_mask: Tensor,
        tgt_tokens: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        # TODO
        raise NotImplementedError

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        # TODO: encode -> decode
        # 返回 decoder hidden states [B, T, D]，暂不经过 generator。
        raise NotImplementedError
```

### 13.6 `build_transformer`

```python
def build_transformer(config: TransformerConfig) -> Transformer:
    """Construct and initialize a Transformer from config."""
    # TODO 1: 创建 attention、FFN、position encoding 原型。
    # TODO 2: 创建 EncoderLayer 和 DecoderLayer。
    # TODO 3: 创建 Encoder 和 Decoder。
    # TODO 4: source/target embedding = TokenEmbedding + PositionEncoding。
    # TODO 5: 创建 Generator。
    # TODO 6: 组装 Transformer。
    # TODO 7: 对 dim > 1 的参数使用 xavier_uniform_ 初始化。
    raise NotImplementedError
```

创建 DecoderLayer 时，self-attention 和 cross-attention 必须是两个独立实例。

### 13.7 模型测试

文件：`tests/test_model.py`

```python
import torch

from transformer_from_scratch.config.config import TransformerConfig
from transformer_from_scratch.runner.masks import make_decoder_mask, make_padding_mask
from transformer_from_scratch.model import build_transformer


def test_transformer_forward_shapes() -> None:
    torch.manual_seed(0)
    config = TransformerConfig(
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
    )
    model = build_transformer(config)

    src = torch.tensor([[1, 2, 3, 0], [1, 4, 5, 6]])
    tgt = torch.tensor([[1, 2, 0], [1, 4, 5]])
    src_mask = make_padding_mask(src, config.pad_id)
    tgt_mask = make_decoder_mask(tgt, config.pad_id)

    hidden = model(src, tgt, src_mask, tgt_mask)
    log_probs = model.generator(hidden)

    assert hidden.shape == (2, 3, 32)
    assert log_probs.shape == (2, 3, config.tgt_vocab_size)
    assert torch.isfinite(hidden).all()
    assert torch.allclose(
        log_probs.exp().sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
    )
```

再增加一个 backward 测试：对 `log_probs.sum()` 调用 `backward()`，确认模型关键参数存在梯度。

## 14. M7：Batch 和 Copy Task 数据

文件：`src/transformer_from_scratch/training.py`

### 14.1 Batch

训练 target 必须右移：

```text
完整 target：   [BOS, 5, 8, 3, 2]
decoder input： [BOS, 5, 8, 3]
labels：        [5, 8, 3, 2]
```

代码骨架：

```python
from dataclasses import dataclass

from torch import Tensor


@dataclass
class Batch:
    src_tokens: Tensor
    decoder_input: Tensor
    labels: Tensor
    src_mask: Tensor
    tgt_mask: Tensor
    non_pad_tokens: Tensor

    @classmethod
    def from_tokens(
        cls,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        pad_id: int,
    ) -> "Batch":
        # TODO 1: decoder_input = tgt_tokens[:, :-1]
        # TODO 2: labels = tgt_tokens[:, 1:]
        # TODO 3: 创建 src_mask 和 tgt_mask
        # TODO 4: 统计 labels 中非 PAD token 数量
        raise NotImplementedError
```

### 14.2 Copy Task 数据

```python
def generate_copy_batches(
    vocab_size: int,
    batch_size: int,
    sequence_length: int,
    num_batches: int,
    pad_id: int,
    bos_id: int,
    device: torch.device,
) -> Iterator[Batch]:
    # TODO 1: 随机生成 [B, L]，普通 token 范围为 [2, vocab_size)。
    # TODO 2: 将每行第一个 token 设置为 bos_id。
    # TODO 3: src 和 target 使用独立 clone。
    # TODO 4: yield Batch.from_tokens(...)
    raise NotImplementedError
```

不要对 token ID 设置 `requires_grad=True`。Embedding 的权重需要梯度，整数 token ID 不需要梯度。

## 15. M8：Loss 和 Label Smoothing

文件：`src/transformer_from_scratch/loss.py`

### 15.1 LabelSmoothingLoss

输入：

```text
log_probs [N, V]
targets   [N]
```

代码骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        smoothing: float,
    ) -> None:
        super().__init__()
        # TODO: 检查参数并保存 confidence = 1 - smoothing。

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        # TODO 1: 检查 log_probs 为 [N, V]。
        # TODO 2: no_grad 下创建 true_distribution。
        # TODO 3: 非正确、非 PAD 类分配 smoothing / (V - 2)。
        # TODO 4: 正确类设置为 confidence。
        # TODO 5: PAD 列设置为 0。
        # TODO 6: target 本身为 PAD 的整行设置为 0。
        # TODO 7: F.kl_div(..., reduction="sum")。
        raise NotImplementedError
```

第一次跑 copy task 时建议：

```python
smoothing = 0.0
```

链路稳定后再设置为 `0.1`，这样更容易区分“代码错误”和“正则化带来的不确定性”。

### 15.2 `compute_loss`

```python
def compute_loss(
    decoder_hidden: Tensor,
    labels: Tensor,
    non_pad_tokens: Tensor,
    generator: nn.Module,
    criterion: nn.Module,
) -> Tensor:
    # TODO 1: generator(hidden) 得到 [B, T, V]。
    # TODO 2: flatten 为 [B*T, V]。
    # TODO 3: labels flatten 为 [B*T]。
    # TODO 4: criterion / non_pad_tokens。
    raise NotImplementedError
```

返回的是“每个非 PAD token 的平均 loss”，用于直接 `backward()`。

### 15.3 测试要求

- smoothing=0 时，正确类别概率越高，loss 越低。
- PAD target 不产生 loss。
- `true_distribution` 每个有效行之和约等于 1。
- `compute_loss` 返回标量。
- backward 后 Generator 和模型参数有梯度。

## 16. M9：学习率和训练循环

仍在 `training.py` 中实现。

### 16.1 Warmup 学习率

\[
lr = factor \cdot d_{model}^{-0.5}
\cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})
\]

```python
def transformer_learning_rate(
    step: int,
    d_model: int,
    factor: float,
    warmup_steps: int,
) -> float:
    # TODO: step=0 时按 1 处理，避免负指数异常。
    raise NotImplementedError
```

趋势应该是：

- warmup 阶段逐步增大。
- 超过 warmup 后按 `step^-0.5` 下降。

### 16.2 TrainState

```python
@dataclass
class TrainState:
    steps: int = 0
    examples: int = 0
    tokens: int = 0
```

### 16.3 `run_epoch`

```python
def run_epoch(
    model: Transformer,
    batches: Iterable[Batch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_state: TrainState,
    max_grad_norm: float | None = 1.0,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    # TODO 1: 遍历 batch。
    # TODO 2: 训练时先 optimizer.zero_grad(set_to_none=True)。
    # TODO 3: forward 得到 decoder hidden。
    # TODO 4: compute_loss。
    # TODO 5: 训练时 backward、clip_grad_norm_、optimizer.step、scheduler.step。
    # TODO 6: 累计 raw token loss 和 token 数。
    # TODO 7: 更新 TrainState。
    # TODO 8: 返回 epoch 平均 token loss。
    raise NotImplementedError
```

验证阶段应使用：

```python
with torch.no_grad():
    validation_loss = run_epoch(..., optimizer=None, scheduler=None, ...)
```

不要在验证阶段执行 backward 或 optimizer step。

### 16.4 优化器

Copy task 建议从教程参数开始：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1.0,
    betas=(0.9, 0.98),
    eps=1e-9,
)
```

将 `transformer_learning_rate` 传给 `LambdaLR`。小模型可将 warmup 从 400 开始调试，不必一开始使用论文的 4000。

## 17. M10：Greedy Decoding

文件：`src/transformer_from_scratch/decoding.py`

### 17.1 解码过程

```text
1. Encoder 只运行一次，得到 memory。
2. 初始 target = [BOS]。
3. 使用当前 target 生成 causal mask。
4. Decoder 输出最后一个位置的 hidden state。
5. Generator 得到下一个 token 的 log-probabilities。
6. argmax 选择 next token。
7. 拼接到 target 尾部。
8. 重复直到达到 max_length。
```

### 17.2 代码骨架

```python
import torch
from torch import Tensor


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src_tokens: Tensor,
    src_mask: Tensor,
    max_length: int,
    bos_id: int,
) -> Tensor:
    model.eval()

    # TODO 1: memory = models.encode(...)
    # TODO 2: 创建 [B, 1] 的 BOS 起始序列。
    # TODO 3: 循环 max_length - 1 次。
    # TODO 4: 每轮根据当前输出创建 causal mask。
    # TODO 5: models.decode(...)
    # TODO 6: 只取最后一个位置，送入 generator。
    # TODO 7: argmax 得到 [B, 1] next token。
    # TODO 8: 在 sequence 维拼接。
    raise NotImplementedError
```

实现必须支持 `batch_size > 1`，不要像旧教程那样只读取 batch 中第一个元素。

### 17.3 测试要求

- 返回 shape 为 `[B, max_length]`。
- 第一列全部为 BOS。
- 解码期间模型处于 eval 模式。
- 输出 dtype 与输入 token dtype 一致。
- 不建立梯度图。

## 18. M11：训练脚本

文件：`scripts/train_copy.py`

### 18.1 执行流程

```text
解析参数
  ↓
设置随机种子
  ↓
选择 CPU / CUDA / MPS
  ↓
创建 config 和 model
  ↓
创建 criterion / optimizer / scheduler
  ↓
循环 epoch
  ├─ 生成训练 batches
  ├─ run_epoch(train)
  ├─ 生成验证 batches
  └─ run_epoch(eval)
  ↓
保存 checkpoint
  ↓
执行一条 greedy decode 示例
```

### 18.2 推荐初始参数

```text
vocab_size = 11
sequence_length = 10
d_model = 128
num_heads = 4
encoder_layers = 2
decoder_layers = 2
d_ff = 512
dropout = 0.1
batch_size = 64
train_batches_per_epoch = 20
validation_batches = 5
epochs = 20
warmup_steps = 400
label_smoothing = 0.0
```

如果只想验证代码链路，先用：

```text
d_model = 32
num_heads = 4
d_ff = 64
layers = 1
batch_size = 8
epochs = 2
```

### 18.3 Device 选择函数

优先级建议：

```text
CUDA -> MPS -> CPU
```

不要在模型内部写死 device。模型、输入、Mask 必须由训练入口统一迁移。

### 18.4 Checkpoint 内容

使用 `pathlib.Path` 创建目录，保存：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": asdict(config),
    "train_state": asdict(train_state),
    "epoch": epoch,
}
```

保存路径：

```text
artifacts/checkpoints/copy_task.pt
```

## 19. M12：推理脚本

文件：`scripts/infer_copy.py`

流程：

1. 读取 checkpoint。
2. 使用 checkpoint 中的 config 重建模型。
3. 加载 `model_state_dict`。
4. 创建输入 token。
5. 创建 source mask。
6. 调用 `greedy_decode`。
7. 打印输入和输出。

期望结果：

```text
input : [1, 5, 8, 3, 2, 7, 4, 9, 6, 2]
output: [1, 5, 8, 3, 2, 7, 4, 9, 6, 2]
```

Copy task 使用固定长度，所以推理时 `max_length` 等于 source sequence length。

## 20. Copy Task 集成测试

文件：`tests/test_copy_task.py`

集成测试不要求每次训练完整模型，可以用一个固定小 batch 做过拟合测试：

```text
1. 固定随机种子。
2. 创建 d_model=16、heads=4、每侧 1 层的小模型。
3. dropout=0。
4. 创建一个固定 batch。
5. 记录第一次 loss。
6. 对同一 batch 训练若干步。
7. 记录最终 loss。
8. 断言最终 loss 明显小于初始 loss。
```

建议断言不要过分严格，避免不同硬件产生轻微差异：

```python
assert final_loss < initial_loss * 0.7
```

如果测试过慢，可使用自定义标记：

```python
@pytest.mark.slow
def test_model_can_overfit_one_copy_batch() -> None:
    ...
```

日常快速测试：

```bash
pytest -m "not slow"
```

完整测试：

```bash
pytest
```

## 21. README 应记录的内容

`README.md` 至少包含：

1. 项目目标。
2. 架构图或数据流说明。
3. 安装命令。
4. 测试命令。
5. 训练命令。
6. 推理命令。
7. Copy task 最终输出。
8. 已完成模块。
9. 与原论文/教程的差异。
10. 后续扩展计划。

不要在 README 里复制本开发文档的全部细节，只保留用户实际需要的运行说明。

## 22. 推荐开发与提交顺序

### 阶段 A：工程骨架

```text
chore(transformer): scaffold python project
```

完成：

- `pyproject.toml`
- `.gitignore`
- 包目录
- 测试目录
- README 骨架

### 阶段 B：Mask

```text
feat(transformer): implement attention masks
```

完成：

- `masks.py`
- `test_masks.py`

### 阶段 C：Attention

```text
feat(transformer): implement multi-head attention
```

完成：

- scaled attention
- multi-head attention
- attention tests

### 阶段 D：Embedding 和基础层

```text
feat(transformer): add embeddings and transformer layers
```

### 阶段 E：完整模型

```text
feat(transformer): assemble encoder-decoder model
```

### 阶段 F：训练和 Loss

```text
feat(transformer): add copy-task training pipeline
```

### 阶段 G：推理

```text
feat(transformer): add greedy decoding
```

### 阶段 H：集成和文档

```text
test(transformer): add copy-task integration test
docs(transformer): document architecture and usage
```

## 23. 常见 Bug 排查表

### 23.1 Attention 输出全是 NaN

检查：

- 是否整行全部被 Mask。
- score 是否在 softmax 前出现 NaN。
- 是否忘记除以 `sqrt(d_k)`。
- 输入 dtype 是否为浮点数。

### 23.2 Mask 没有效果

检查：

- 是否统一约定 `True=可见`。
- 是否写成了 `masked_fill(mask, ...)`，而不是 `masked_fill(~mask, ...)`。
- Mask 是否能广播到 `[B, H, Lq, Lk]`。
- causal mask 是否为下三角。

### 23.3 `view` 报 stride 错误

`transpose` 后张量通常不连续：

```python
x.transpose(1, 2).contiguous().view(...)
```

也可以使用 `reshape`，但仍应理解 contiguous 的原因。

### 23.4 Cross-Attention shape 错误

不要假设 source length 等于 target length：

```text
Q: [B, H, T, Dk]
K: [B, H, S, Dk]
score: [B, H, T, S]
```

### 23.5 多层参数意外共享

错误：

```python
nn.ModuleList([layer] * count)
```

正确方向：使用 `copy.deepcopy`。

### 23.6 训练 Loss 不下降

按顺序检查：

1. 先固定同一个 batch 反复训练，能否过拟合。
2. target 是否正确右移。
3. labels 是否与 decoder input 错开一位。
4. Generator 输出是否为 log-probabilities。
5. criterion 是否和 Generator 输出匹配。
6. `optimizer.zero_grad()` 是否执行。
7. scheduler 是否每步更新。
8. 模型是否意外处于 eval 模式。
9. dropout 是否过大。
10. Mask 是否把合法位置错误屏蔽。

### 23.7 推理始终重复同一个 Token

检查：

- 是否每轮都重新生成 causal mask。
- 是否只取 `decoder_hidden[:, -1]`。
- 是否把 next token 拼回已有 target。
- 是否加载了训练后的 checkpoint。
- BOS 是否和训练时一致。

### 23.8 训练正确但推理失败

最常见原因是 teacher forcing 和自回归推理链路不一致。单独测试：

- `encode()`
- `decode()`
- Generator
- 单步 greedy decode
- 多步拼接

## 24. 每个阶段的验收命令

```bash
# M0
pip install -e ".[dev]"

# M1
pytest tests/test_config.py tests/test_masks.py -q

# M2-M3
pytest tests/test_attention.py -q

# M4-M5
pytest tests/test_embeddings.py tests/test_layers.py -q

# M6
pytest tests/test_model.py -q

# M7-M9
pytest tests/test_loss.py -q

# 全部快速测试
pytest -m "not slow" -q

# 格式和静态检查
ruff check .

# 训练
python scripts/train_copy.py

# 推理
python scripts/infer_copy.py

# 最终完整测试
pytest -q
```

## 25. 最终完成标准

以下条件全部满足，V1 才算完成：

- [ ] 没有使用 `nn.Transformer`。
- [ ] 没有使用 `nn.MultiheadAttention`。
- [ ] 所有 Mask 使用 `torch.bool`。
- [ ] 每个模块的 docstring 写清 shape。
- [ ] 所有单元测试通过。
- [ ] Ruff 检查通过。
- [ ] 模型前向输出没有 NaN/Inf。
- [ ] 完整 backward 可以运行。
- [ ] 单 batch 可以被模型过拟合。
- [ ] Copy task 训练 loss 明显下降。
- [ ] Greedy decoding 可以复原输入序列。
- [ ] Checkpoint 可以保存和重新加载。
- [ ] README 包含安装、训练和推理命令。
- [ ] 你能脱离源码画出完整数据流。

## 26. 手搓过程中必须能回答的问题

### Mask

- padding mask 和 causal mask 分别解决什么问题？
- 为什么 source mask 可以从 `[B,1,1,S]` 广播到 `[B,H,T,S]`？
- 为什么 causal mask 是下三角？

### Attention

- `QKᵀ` 每个元素表示什么？
- 为什么除以 `sqrt(d_k)`？
- 为什么 softmax 在最后一个维度执行？
- 为什么权重乘的是 V，而不是 K？

### Multi-Head Attention

- 为什么 `d_model` 必须能被 head 数整除？
- 为什么 split heads 后要交换 sequence 和 head 维？
- 为什么 merge heads 前需要 contiguous？

### Encoder / Decoder

- Encoder self-attention 的 Q/K/V 来自哪里？
- Decoder masked self-attention 为什么需要 causal mask？
- Cross-attention 中为什么 Q 来自 Decoder，K/V 来自 Encoder？

### 训练

- 为什么 decoder input 和 labels 要错开一位？
- 为什么训练能并行处理 target，而推理必须逐 token 生成？
- Label Smoothing 为什么可能提高准确率却降低概率置信度？
- Warmup 为什么先升高学习率再衰减？

如果某一题回答不清楚，就回到对应模块重新检查 shape 和数据流。

## 27. V1 完成后的扩展顺序

建议按以下顺序继续，不要一次同时改多个核心机制：

1. 添加 EOS，支持提前结束解码。
2. 添加 Beam Search。
3. 添加 attention weights 可视化。
4. 接入一个小型真实翻译数据集。
5. 将 ReLU 替换成 GELU 或 SwiGLU。
6. 将正弦位置编码替换成 RoPE。
7. 删除 Encoder 和 cross-attention，改造成 Decoder-Only 模型。
8. 为 Decoder-Only 模型加入 KV Cache。
9. 加入混合精度和梯度累积。

每次扩展都应保留原测试，并新增覆盖新行为的测试。

## 28. 正式开工顺序

第一次开发只做以下内容：

```text
1. 创建目录和 pyproject.toml
2. 创建 src/transformer_from_scratch/__init__.py
3. 实现 config.py
4. 编写 test_config.py
5. 实现 masks.py
6. 编写 test_masks.py
7. 确认测试通过
8. 再进入 attention.py
```

不要在 Mask 测试通过前提前写完整模型。项目的核心难点不是代码行数，而是每层 shape、Mask 语义和训练/推理数据流始终一致。

## 29. 参考资料

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Harvard NLP annotated-transformer 源码](https://github.com/harvardnlp/annotated-transformer)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)

