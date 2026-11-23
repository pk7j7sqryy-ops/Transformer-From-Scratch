from dataclasses import dataclass
from typing import Optional
import torch

'''
当前上下文 (decoder输入)
    ↓
Encoder + Decoder → hidden [B, T, D]     ← 提取上下文特征,找出各隐藏特征
    ↓
Linear → scores [B, T, V]                ← 对每个词打分
    ↓
log_softmax → log_probs [B, T, V]        ← 分数变概率-已知上下文，这个词是正确答案的概率
    ↓
LabelSmoothing → loss                    ← 和正确答案比，算损失
    ↓
loss.backward() → 梯度                    ← 找出往哪个方向调
    ↓
optimizer.step() → 更新参数               ← 调整参数，让下次预测更准
'''
'''
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
'''
from pathlib import Path
# 目录
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
LOGS_DIR = ROOT_DIR / 'logs'

RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# 文件
RAW_TRAIN_DATA = 'train.parquet'
RAW_TEST_DATA = 'test.parquet'
RAW_VALID_DATA = 'validation.parquet'
# LABELS_FILE = 'labels.txt'
MODEL_NAME_OR_PATH = 'google-bert/bert-base-chinese'
MODEL_NAME_EN_PATH ='google-bert/bert-base-uncased'

BEST_MODEL = 'models.pt'
MODEL_DATA='Helsinki-NLP/opus-100'
MODEL_DATA_TYPE='en-zh'


@dataclass(frozen=True)
class TransformerConfig:
    train_samples: int = 20_000
    data_seed: int = 42
    # 10万条数据、batch_size=32 时，每个 epoch 大约3125步
    #词表
    src_vocab_size: int = 21128
    tgt_vocab_size: int = 30522
    #模型结构
    d_model:int=256
    num_heads:int=4
    num_encoder_layers:int=4
    num_decoder_layers:int=4
    d_ff:int=1024#d_ff 一般 = 4×d_model）
    dropout: float = 0.1
    max_seq_len:int=256
    #特殊token
    pad_id:int=0
    bos_id:int=1
    #模型训练超参数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 0.001
    batch_size: int =  32
    epoch: int = 100
    log_steps: int = 25
    limit:Optional[int]=None  # None=全量数据，数字=限制条数（调试用）
    resume: bool = True
    early_stopping_patience: int = 8  # 早停容忍度
    warmup_steps: int = 500#前多少步线性增长（一般 4000，小数据可以设 500~1000）
    factor: float = 1.0#缩放系数，控制峰值 lr 大小
    max_grad_norm: float = 1.0
    # CPU 上跑 float16 AMP 不仅不加速，反而更慢——CPU 没有浮点16硬件，每个操作都要在 float16↔float32 之间反复转换
    use_amp: bool = False  # 混合精度（先关掉，后面再开）

    def __post_init__(self):
        # 校验1: 词表 > 2（至少 PAD + BOS + 1个普通token）
        if self.src_vocab_size <= 2:
            raise ValueError(f"src_vocab_size 必须 > 2, 当前 {self.src_vocab_size}")
        if self.tgt_vocab_size <= 2:
            raise ValueError(f"tgt_vocab_size 必须 > 2, 当前 {self.tgt_vocab_size}")

        # 校验2: d_model 能被 num_heads 整除
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model({self.d_model}) 必须能被 num_heads({self.num_heads}) 整除")

        # 校验3: d_model 是偶数（位置编码需要 sin/cos 配对）
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model 必须是偶数, 当前 {self.d_model}")

        # 校验4: 层数、d_ff、max_seq_len 为正数
        if self.num_encoder_layers <= 0:
            raise ValueError(f"num_encoder_layers 必须 > 0, 当前 {self.num_encoder_layers}")
        if self.num_decoder_layers <= 0:
            raise ValueError(f"num_decoder_layers 必须 > 0, 当前 {self.num_decoder_layers}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff 必须 > 0, 当前 {self.d_ff}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len 必须 > 0, 当前 {self.max_seq_len}")

        # 校验5: dropout 在 [0, 1) 范围内
        if not (0 <= self.dropout < 1):
            raise ValueError(f"dropout 必须在 [0, 1) 范围内, 当前 {self.dropout}")
