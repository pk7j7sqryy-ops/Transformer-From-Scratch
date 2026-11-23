import torch
from torch import nn
from torch.nn.functional import softmax
import math
# from layers import clones

def attention(query,key,value,mask=None,dropout=None):
    '''
            mask == 0:  [False, False, True, True]
                                            ↑  ↑  这些位置是 True
                scores:    [ 2.1,   1.8,   3.5,  2.9]
                结果:      [ 2.1,   1.8,  -1e9, -1e9]    # 标记的位置被替换成 -1e9
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    '''
    d_k=query.size(-1)
    scores=(query @ key.mT)/math.sqrt(d_k)
    if mask is not None:
        #masked_fill="找到指定位置，替换成指定值"。-1e9（负无穷）
        # -1e9在float16里溢出成 - inf,softmax 会出 NaN。torch.finfo(scores.dtype).min 会自动适配当
        scores = scores.masked_fill(mask==0, torch.finfo(scores.dtype).min)
    scores_attn=softmax(scores,dim=-1)
    if dropout is not None:
        scores_attn=dropout(scores_attn)
    attn_result=scores_attn @ value
    return attn_result,scores_attn

class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k=d_model//h
        self.h=h
        #过4个线形层输出qkv和一个out
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,query,key,value,mask=None):
        B=query.size(0)

        # 第1步：Q/K/V 各过自己的线性层（内容变了，维度没变）
        q = self.w_q(query)  # [B, L, d_model]
        k = self.w_k(key)  # [B, L, d_model]
        v = self.w_v(value)  # [B, L, d_model]

        # 第2步：拆头，把 d_model 拆成 h 个 d_k,因为h+dk=dmodel
        q = q.view(B, -1, self.h, self.d_k)  # [B, L, h, d_k]
        k = k.view(B, -1, self.h, self.d_k)
        v = v.view(B, -1, self.h, self.d_k)

        # 第3步：把头维度换到前面，让每个头独立算
        q = q.transpose(1, 2)  # [B, h, L, d_k]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 第4步：调 attention 函数，h 个头并行算
        out, attn = attention(q, k, v, mask, self.dropout)  # [B, h, L, d_k]

        # 第5步：拼回来，把 h 个头拼成 d_model
        '''
        张量在内存里是一维连续排列的。transpose 只是改了"怎么看"数据，底层内存没动,
        view 按内存顺序读，transpose 改了 shape 但没改内存，两者不匹配就报错。contiguous 把内存重新排成和 shape 一致，view 就能正常切分了。
        transpose 后（内存不连续）:
          shape [3, 2]: [[0, 3], [1, 4], [2, 5]]
          内存: [0, 1, 2, 3, 4, 5]  ← 跳着读才能得到正确结果
        
        contiguous() 后（内存连续了）:
          shape [3, 2]: [[0, 3], [1, 4], [2, 5]]
          内存: [0, 3, 1, 4, 2, 5]  ← 重新排列，顺序读就对
        '''
        out = out.transpose(1, 2)           # [B, L, h, d_k]
        out = out.contiguous()               # 内存连续化
        out = out.view(B, -1, self.h * self.d_k)  # [B, L, d_model]

        # 第6步：过输出线性层，融合多头信息，拼回只是物理操作，w_o 才是信息融合
        return self.w_o(out)  # [B, L, d_model]

    #以下是后续进阶写法
    # def __init__(self,h,d_model,dropout=0.1):
    #     '''
    #
    #     :param h:头数
    #     :param d_model:模型维度
    #     :param dropout:每个头的维度 = d_model // h
    #     '''
    #     super().__init__()
    #     assert d_model % h==0
    #     self.d_k=d_model // h
    #     self.h = h
    #     self.linears=clones(nn.Linear(d_model,d_model),4)
    #     self.attn=None
    #     self.dropout=nn.Dropout(dropout)
    #
    # def forward(self,query,key,value,mask=None):
    #     if mask is not None:
    #         mask=mask.unsqueeze(1)
    #     nbatches=query.size(0)
    #
    #     #计算qkv
    #     '''
    #     zip(self.linears, (query, key, value))
    #         # → (Linear0, query)
    #         #   (Linear1, key)
    #         #   (Linear2, value)
    #         #   Linear3 没有配对，被忽略
    #     # 第一次循环，这里的qkv的最后维度还是dmodel，原始的embedding维度[B, S, D]或[B, T, D]
    #         l = Linear0,  x = query     → Linear0(query)
    #         # 第二次循环
    #         l = Linear1,  x = key        → Linear1(key)
    #         # 第三次循环
    #         l = Linear2,  x = value      → Linear2(value)
    #     '''
    #     query,key,value=[
    #         l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
    #         for l,x in zip(self.linears,(query,key,value))
    #     ]
    #
    #     #算注意力
    #     attn_result,self.attn=attention(query,key,value,mask,self.dropout)
    #
    #     #拼回
    #     x = attn_result.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k)
    #
    #     #过输出线性层
    #     return self.linears[-1](x)


