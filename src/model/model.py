'''
    各种模型类定义
'''

from torch.nn.functional import log_softmax
from model.attention import *
from model.embeddings import *
from model.layers import *
import copy

'''
make_model() 被调用
    ↓
创建模板: attn, ff, position
    ↓
EncoderLayer(c(attn), c(ff)) × N → Encoder
DecoderLayer(c(attn), c(attn), c(ff)) × N → Decoder
Embeddings + c(position) → src_embed
Embeddings + c(position) → tgt_embed
Generator(d_model, tgt_vocab)
    ↓
EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
    ↓
Xavier 初始化所有权重
    ↓
返回 models
config → masks → attention → embeddings → layers → models → loss → training → decoding
'''
#定义编码器和解码器，
#整体外壳，组装 encoder+decoder+embed+generator
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super().__init__()
        #定义编码器，输入[B,S,D]+mask-》[B,S,D]
        #Self-Attention(用 mask 屏蔽 PAD) → FFN → [B,S,D]，v
        self.encoder=encoder

        #定义解码器，输入[B,T,D]+masks+marrey-》[B,T,D],到最后一层linear前
        self.decoder=decoder

        #定义输入embedding，输入[B, S]-》[B,S,D]
        self.src_embed=src_embed

        #定义目标target，输入[B,T] → [B,T,D]
        self.tgt_embed=tgt_embed

        #把向量映射回词表概率[B,T,D] → [B,T,V]
        self.generator = generator

    #前向传播
    def forward(self,src,tgt,src_mask,tgt_mask):
        '''

        :param src: 源语言 token ID，shape [B, S]
        :param tgt: 目标语言 token ID（训练时右移一位后的输入），shape [B, T]
        :param src_mask:源序列的 padding mask，True=可见，shape [B, 1, 1, S]
        :param tgt_mask: 目标序列的 mask（padding + causal 的组合），shape [B, 1, T, T]
        :return:
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    #编码器入口
    def encode(self,src,src_mask):
        '''
        #数据流：src token [B,S] → src_embed 变成 [B,S,D] → encoder 处理 → memory [B,S,D]
        #Encoder 的输出会被 Decoder 的 cross-attention 反复读取，就像"记忆"一样供 Decoder 查询，所以叫 memory。
        :param src: 源语言 token ID，shape [B, S]
        :param src_mask: 源序列的 padding mask，True=可见，shape [B, 1, 1, S]
        :return: memory [B,S,D]
        '''

        return self.encoder(self.src_embed(src),src_mask)

    #解码器入口
    def decode(self,memory,src_mask,tgt,tgt_mask):
        '''
        数据流：tgt token [B,T] → tgt_embed 变成 [B,T,D] → 和 memory 一起送入 decoder → hidden [B,T,D]
        因为推理时 encoder 只跑一次，memory 复用。每次生成新 token 时只需要调用 decode，不需要重新跑 encoder。
        :param memory: encoder 的输出，shape [B, S, D]
        :param src_mask: 源序列的 padding mask，shape [B, 1, 1, S]
        :param tgt: 目标语言 token ID（训练时右移一位后的输入），shape [B, T]
        :param tgt_mask: 目标序列的 mask（padding + causal 的组合），shape [B, 1, T, T]
        :return: decoder 输出的 hidden states，shape [B, T, D]
        '''
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)


#定义词表linear和通过softmax生成概率
# 线性层 + log_softmax，把[B, T, D]映射到[B, T, V]
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        '''
        最后一个线形层，把d_model投影到词表范围内
        :param d_model: decoder 输出的 hidden 维度（比如 128）
        :param vocab_size:词表大小（比如 21118）
        '''
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        '''
        [B, T, D] → nn.Linear → [B, T, V] → log_softmax → [B, T, V]
        :param x:
        :return:
        '''
        return log_softmax(self.proj(x),dim=-1)


#模型构建
def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    '''
    0、EncoderDecoder把以下包起来组装
    1、先过emb（中，英的词表大小和dmodel和位置编码）
    2、先过encoder，输入（MHA+add/norm+ffn+add/norm），encoder层数，输出memory [B, S, D]
    3、在过decoder，输入（MMHA+add/norm+CHMA+add/norm+ffn+add/norm），decoder层数，输出最终 x [B,T,D]
    4、再过Generator，通过最终评分生成概率，后续model类单独调用，self.generator 只是存着，forward() 里没调它
    :param src_vocab:源语言词表大小
    :param tgt_vocab:目标语言词表大小
    :param N:Encoder/Decoder 层数
    :param d_model:模型维度
    :param d_ff:FFN 中间层维度（4倍 d_model）
    :param h:注意力头数
    :param dropout:dropout 概率
    :return:
    '''
    c=copy.deepcopy
    attn=MultiHeadAttention(h, d_model)
    ff =PositionwiseFeedForward(d_model,d_ff,dropout)
    position = PositionalEncoding(d_model,dropout)
    model=EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),c(ff), dropout), N),
        # nn.Sequential把两个模块串起来，调用时自动依次执行：
        nn.Sequential(Embeddings(d_model,src_vocab),c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model,tgt_vocab)
    )
    # 遍历所有参数，只对二维以上的（权重矩阵）做Xavier初始化，跳过一维的（bias）。
    #xavier_uniform_原地修改 ，默认的正态初始化会被 Xavier 初始化覆盖
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model


# 从 config 构建模型，参数统一从 TransformerConfig 读取，不再硬编码
def build_model(config):
    """
    从 TransformerConfig 构建模型，所有参数来自 config，避免硬编码
    :param config: TransformerConfig 实例
    :return: EncoderDecoder 模型
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(config.num_heads, config.d_model)
    ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
    position = PositionalEncoding(config.d_model, config.dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(config.d_model, c(attn), c(ff), config.dropout), config.num_encoder_layers),
        Decoder(DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout), config.num_decoder_layers),
        nn.Sequential(Embeddings(config.d_model, config.src_vocab_size), c(position)),
        nn.Sequential(Embeddings(config.d_model, config.tgt_vocab_size), c(position)),
        Generator(config.d_model, config.tgt_vocab_size)
    )
    model.generator.proj.weight = (
        model.tgt_embed[0].lut.weight
    )
    #Xavier正态分布初始化，
    # 后标准差大约是：，sqrt(2 / (词表大小 + dmodel))，
    # sqrt(2 / (21128 + 256)) ≈ 0.0097，在乘根号dmodel，补偿后0.0097 × 16 ≈ 0.155
    # 为什么不用he初始化？
    # 因为He / Kaiming初始化主要为“线性层后接ReLU”设计，
    # 而你的这段初始化会作用于模型中所有二维权重（We，Wqkv，ffnw1，ffnw2），不只是FFN。
    # Transformer中大部分投影层并没有直接接ReLU，而且还有LayerNorm、残差连接，所以全局Xavier比全局He更合适。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
