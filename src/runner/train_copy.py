'''

def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):

'''

import torch

from config.config import TransformerConfig
from model.model import build_model
from torchinfo import summary


'''
模型构建
'''

config=TransformerConfig()
model=build_model(config)



'''
参数查看
'''
# print(models)
def show_model_summary(model, config, batch_size=6, seq_len=10):
    B, S, T = batch_size, seq_len, seq_len
    src = torch.randint(0, config.src_vocab_size, (B, S))
    tgt = torch.randint(0, config.tgt_vocab_size, (B, T))
    src_mask = torch.ones(B, 1, 1, S, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(1, 1, T, T, dtype=torch.bool)).expand(B, 1, T, T)
    summary(model, input_data=[src, tgt, src_mask, tgt_mask])

# 用法
config = TransformerConfig()
show_model_summary(model, config)