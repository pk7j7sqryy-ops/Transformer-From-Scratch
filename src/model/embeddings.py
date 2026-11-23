import torch
from torch import nn
import math

class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        '''
        lut: lookup table，查表把 token ID 变成向量
        :param d_model:	向量维度
        :param vocab:词表大小
        '''
        super().__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model
    def forward(self,x):
        '''
        为什么要乘 √d_model？ embedding 初始化后值很小（均值0方差1），但位置编码的值在 [-1, 1] 范围。
        如果不放大 embedding，加上位置编码后位置信息会盖过词义信息。乘 √d_model 把 embedding 放大到和位置编码同一量级。
                创建Embedding：默认 N(0, 1)
                ↓
        build_model：覆盖成 Xavier Uniform，
        乘 sqrt(d_model) 用于放大 token embedding，
        避免与 [-1, 1] 范围的位置编码相加时，
        token 语义信号相对过弱。
                ↓
        forward：Embedding结果 × √d_model
        :param x:
        :return:
        '''
        return self.lut(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        '''
        P E(pos,2i) = sin(pos/10000^2i/dmodel)
        P E(pos,2i+1) = cos(pos/10000^2i/dmodel)
        # 先建一个空表 [5000, 128]、
         # 位置索引 [0,1,2,...,4999] → [5000, 1]
         频率衰减项-div_term 就是公式里的 1 / 10000^(2i/d_model) 这部分
        :param d_model:
        :param dropout:
        :param max_len:
        '''
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        #这步应用到了a^b = exp(b * log(a))，和(e^x)^y = e^(x*y)，a = e^(ln a)等公式
        div_term=torch.exp(
            torch.arange(0,d_model,2) * -(math.log(10000.0)/d_model)
        )
        # print(position,div_term)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        # print(pe)
        #位置编码不需要训练，但要跟着模型一起存到 checkpoint，
        # register_buffer 就是告诉 PyTorch："这个张量虽然不训练，但请帮我管理它的设备迁移和存储。所以用 buffer
        self.register_buffer('pe',pe)

    def forward(self,x):
        '''
        截取位置编码，截取到和emb一样长，也就是用emb的行x.size(1)去截取
        这里pe 的 shape 是 [1, max_len, d_model]，第二个维度就是位置维度
        :param x:
        :return:
        '''
        x = x +self.pe[:,:x.shape[1]].requires_grad_(False)
        return self.dropout(x)

