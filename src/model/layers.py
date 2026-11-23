import torch
from torch import nn
import copy
from torch.nn.functional import relu

#辅助函数，深拷贝 N 份模块，给 Encoder/Decoder 堆叠用
def clones(module,N):
    '''
    为什么用 copy.deepcopy 而不是 [module] * N？
    因为 [module] * N 只是创建 N 个引用指向同一个对象，改一个全变。
    deepcopy 才能创建 N 个独立的副本，各自有不同的权重。
    :param module: encoder和decoder
    :param N: 复制n份
    :return: 里
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#LayerNorm层归一化，把 D 维向量拉回均值0方差1
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        '''
        把每个位置的 D 维向量归一化到均值0、方差1，再用可学习参数缩放和偏移。
        xi = xi−μ /（σ+ε）
        LayerNorm（xi）=γi⋅xi+βi，让模型可以学习在归一化后的基础上进行适当的调整，保证归一化不会限制模型的表示能力
        :param features:D 维度大小（如 128）	对每个 D 维向量做归一化
        :a_2:缩放参数γ，初始值全1， a_2 = [1.0, 1.0, 1.0, 1.0]  ← torch.ones
        :b_2:偏移参数β，初始值全0 b_2 = [0.0, 0.0, 0.0, 0.0]  ← torch.zeros
        :param eps:ε为一个小的常数，防止除0
        '''
        super().__init__()
        self.a_2=nn.Parameter(torch.ones(features))#Parameter表示这个张量是可训练参数，反向传播时要更新它
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps
    def forward(self,x):
        '''
        xi = xi−μ /（σ+ε）
        LayerNorm（xi）=γi⋅xi+βi，让模型可以学习在归一化后的基础上进行适当的调整，保证归一化不会限制模型的表示能力
        :param x:x [B, S, D]
          → mean: 对最后一维 D 求均值 → [B, S, 1]
          → std:  对最后一维 D 求标准差 → [B, S, 1]
          → (x - mean) / (std + eps): 归一化到均值0方差1 → [B, S, D]
          → a_2 * 归一化 + b_2: 缩放和偏移 → [B, S, D]
        :return:a_2 * 归一化 + b_2: 缩放和偏移 → [B, S, D]
        '''
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        normalized = (
                (x - mean)
                / torch.sqrt(variance + self.eps)
        )

        return self.a_2 * normalized + self.b_2
        # mean = x.mean(-1,keepdim=True)
        # std=x.std(-1,keepdim=True)
        # return self.a_2*(x-mean)/(self.eps+std)+self.b_2

#残差连接和LayerNorm，残差连接壳：norm → 子层 → dropout → 加残差
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super().__init__()
        self.norm=LayerNorm(size)#层归一化
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        '''
        论文原文3.1：我们在每个子层周围使用残差连接 [11]，然后进行层归一化 [1]。
                    也就是说，每个子层的输出是 LayerNorm(x + Sublayer(x))，其中Sublayer(x) 是子层本身实现的函数。
        但是这里是先归一化在过子层：
                    训练更稳定。先归一化再过子层，子层收到的输入永远是均值0方差1的稳定值，梯度不容易爆炸。
                    先归一化 → 再过 Attention/FFN → 再 dropout → 最后加回原始 x。这样子层处理的永远是稳定的归一化输入
        :param x:x [B, S, D]
                  → norm(x): LayerNorm 归一化 → [B, S, D]
                  → sublayer(norm(x)): 过 Attention 或 FFN → [B, S, D]
                  → dropout(...): 随机丢弃 10% 神经元 → [B, S, D]
                  → x + ...: 残差连接，原始 x 直接加回来 → [B, S, D]
        :param sublayer:
        :return:[B, S, D]
        '''
        return x+self.dropout(sublayer(self.norm(x)))

#编码器各EncoderLayer，单层 Encoder：Self-Attention + FFN
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        '''
        自注意力 + 前馈网络
        :param size:d_model 维度（如 512）
        :param self_attn:自注意力层（后面实现）
        :param feed_forward:ffn前馈网络（后面实现）
        :param dropout:	dropout 概率（如 0.1）
        '''
        super().__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer1=SublayerConnection(size,dropout)
        self.sublayer2=SublayerConnection(size,dropout)
        self.size=size

    def forward(self,x,mask):
        '''
        :param x:x [B, S, D]
                  → sublayer1: Self-Attention（Q=K=V=x，因为是自注意力）
                  → sublayer2: Feed-Forward
                  → 输出 [B, S, D]
        :param mask:
        :return:输出 [B, S, D]
        '''
        x = self.sublayer1(x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer2(x,lambda x:self.feed_forward(x))

#N 个 EncoderLayer 堆叠
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layers=clones(layer,N)# n=6 返回 6 个 EncoderLayer
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        '''

        :param x: x [B, S, D]
                  → Layer 1: Self-Attention + FFN → [B, S, D]
                  → Layer 2: Self-Attention + FFN → [B, S, D]
                  → ... 重复 N 次
                  → 最后再过一次 LayerNorm
                  → 输出 memory [B, S, D]
        :param mask:
        :return:memory [B, S, D]
        '''
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

#单层 Decoder：Masked Self-Attention + Cross-Attention + FFN
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        '''

        :param size:d_model 维度（如 512）
        :param self_attn:自注意力（带 causal mask，防偷看未来）
        :param src_attn:交叉注意力（Q 来自 decoder，K/V 来自 encoder 的 memory）
        :param feed_forward:前馈网络
        :param dropout:dropout 概率（如 0.1）
        '''
        super().__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer1=SublayerConnection(size,dropout)
        self.sublayer2=SublayerConnection(size,dropout)
        self.sublayer3=SublayerConnection(size,dropout)
    def forward(self,x,memory,src_mask,tgt_mask):
        '''
        输入 x [B, T, D]    memory [B, S, D]

            第1层：Masked Self-Attention
              x [B,T,D] → sublayer1(x, self_attn) → 新的 x [B,T,D]
              （decoder 自己看自己，用 tgt_mask 防偷看）

            第2层：Cross-Attention
              上一步的 x [B,T,D] → sublayer2(x, src_attn) → 新的 x [B,T,D]
              （decoder 去看 encoder 的 memory，用 src_mask 遮 PAD）
              ↑ 注意：x 变了，memory 没变

            第3层：FFN
              上一步的 x [B,T,D] → sublayer3(x, ffn) → 最终 x [B,T,D]
        :param x:
        :param memory:
        :param src_mask:
        :param tgt_mask:
        :return: [B,T,D]
        '''
        x=self.sublayer1(x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x=self.sublayer2(x,lambda x:self.src_attn(x,memory,memory,src_mask))
        return  self.sublayer3(x,lambda x:self.feed_forward(x))

# N 个 DecoderLayer 堆叠
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)  # n=6 返回 6 个 EncoderLayer
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory,src_mask,tgt_mask):
        '''

        :param x: x [B, S, D]
                  → Layer 1: Self-Attention + FFN → [B, S, D]
                  → Layer 2: Self-Attention + FFN → [B, S, D]
                  → ... 重复 N 次
                  → 最后再过一次 LayerNorm
                  → 输出 memory [B, S, D]
        :param mask:
        :return:memory [B, S, D]
        '''
        for layer in self.layers:
            x = layer(x, memory,src_mask,tgt_mask)
        return self.norm(x)

#FFN：两层线性层中间夹 ReLU
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        '''
        第1步: self.w_1(x)              # x 是 d_model → 出来变成 d_ff
        第2步: F.relu(...)              # d_ff → d_ff
        第3步: self.dropout(...)        # d_ff → d_ff
        第4步: self.w_2(...)            # d_ff 进去 → 出来变成 d_model
        :param x:
        :return:
        '''
        return self.w_2(self.dropout(relu(self.w_1(x))))








