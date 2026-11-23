# Transformer From Scratch
## 很久之前手搓了transformer，最近复习下，同时新增手搓的一个decoder的架构图加上之前的手搓代码，分享给大家！
<img width="2926" height="9861" alt="Transformer_decoder" src="https://github.com/user-attachments/assets/bb9772b4-fb7c-4452-836d-cb04ab55e681" />



使用 PyTorch 从零实现 Encoder–Decoder Transformer，并以英中翻译作为完整训练与推理任务。项目重点是把论文中的组件拆开实现、观察张量形状，并走通数据预处理、训练、评估和服务流程；分词器使用 Hugging Face Transformers 提供的预训练 tokenizer，模型主体由本项目代码实现。

## 项目结构

```text
Transformer-From-Scratch/
├── src/
│   ├── config/       # 模型、数据和训练参数
│   ├── model/        # Embedding、位置编码、Attention、Encoder/Decoder
│   ├── proprocess/   # 数据集加载与预处理（目录名沿用现有代码）
│   ├── runner/       # Mask、损失、训练、评估、预测
│   ├── web/          # FastAPI 推理服务
│   └── main.py       # 命令入口
├── docs/             # 架构、张量形状和学习复盘
├── Drawing_Transformer/ # Transformer 结构图
├── data/             # 本地原始及预处理数据（运行时生成）
├── logs/             # TensorBoard 训练日志（运行时生成）
├── models/           # checkpoint（运行时生成）
├── DEVELOPMENT_GUIDE.md
└── pyproject.toml
```

## 实现内容

- Token Embedding 与正弦位置编码。
- Scaled Dot-Product Attention、多头自注意力和 Encoder–Decoder Cross-Attention。
- Padding/Causal Mask、残差连接、LayerNorm、前馈网络及 Encoder/Decoder 堆叠。
- Label Smoothing、训练循环、checkpoint、评估和贪心解码。
- 使用 OPUS-100 英中语料完成翻译任务，并提供 FastAPI 推理接口。

## 环境与运行

需要 Python 3.10 或更高版本。完整数据与训练流程除 PyTorch 外还使用 `datasets`、`transformers`、`tqdm`、`tensorboard` 和 `sacrebleu`；Web 服务还需要 `fastapi`、`uvicorn`。这些运行依赖目前尚未全部写入 `pyproject.toml`。

```bash
cd 01_Ai_Project/Transformer-From-Scratch
python -m venv .venv
source .venv/bin/activate
python -m pip install torch datasets transformers tqdm tensorboard sacrebleu fastapi uvicorn
```

从项目根目录运行各任务：

```bash
python src/main.py preprocess  # 下载/预处理数据
python src/main.py train       # 训练模型
python src/main.py evaluate    # 评估模型
python src/main.py predict     # 命令行翻译
python src/main.py serve       # 启动推理服务
```

也可以只研究模型结构，优先阅读 `src/model/attention.py`、`src/model/embeddings.py`、`src/model/layers.py` 和 `src/model/model.py`。训练参数集中在 `src/config/config.py`。

## 文档导航

- [架构说明](docs/architecture.md)：模块职责和前向流程。
- [张量形状](docs/tensor_shapes.md)：主要中间张量维度。
- [学习复盘](docs/复盘_20260820.md)：数据、训练与实现过程记录。
- [开发指南](DEVELOPMENT_GUIDE.md)：从组件到集成的分阶段实现路线。

## 备注

当前仓库保留本地数据、训练日志和模型权重供继续实验使用；这些内容属于可再生成的运行产物，已在 `.gitignore` 中排除。若需要分享训练结果，请单独选择要发布的 checkpoint，而不要提交整个数据目录或日志目录。
