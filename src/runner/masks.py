import torch
'''
填充+因果Mask 生成
为什么前面dataset已经填充0了 mask还要在标记一次 直接按照0是pad算不可以吗？
PAD 填 0 是数据层面的处理，
mask 是计算层面的屏蔽。Embedding 层把 0 当普通 token 查表得到向量，
必须用 mask 告诉 attention 层"这些位置是 PAD，权重设为 0"。
'''
import torch
# token填充掩码
def make_padding_mask(tokens,pad_id=0):
    '''
    tokens [B, S]: [[5, 8, 0, 0]]
    tokens != 0:    [[True, True, False, False]]    ← [B, S]
    unsqueeze(1):   [[[True, True, False, False]]]  ← [B, 1, S]
    unsqueeze(1):   [[[[True, True, False, False]]]] ← [B, 1, 1, S]
    后续和评分相加进行填充，
    scores: [B, H, Lq, Lk]
    mask:   [B, 1, 1, S]
             ↑  ↑  ↑   ↑
           batch H Lq  Lk
    1 在 H 位置 → 所有 head 共用同一份 mask
    1 在 Lq 位置 → 所有 query 位置共用同一份 mask
    :param tokens: src_tokens [B, S]
    :param pad_id: 0
    :return:[B, 1, 1, S]
    '''
    mask = (tokens != pad_id).unsqueeze(1).unsqueeze(1)
    return mask
# tokens=torch.randn(2,3)
# tokens=torch.tensor([[1,2,3],[1,1,0]])
# print(make_padding_mask(tokens))

#因果掩码填充
def make_causal_mask(size, device=None):
    '''
    torch.tril 是什么？ lower triangular，只保留下三角和对角线，上三角全设 0：
    :param size:tgt_tokens[B, T]的T
    :param device: 可选，指定创建mask的设备（CPU/GPU）
    :return: → [1，1，T, T]
    '''
    mask = torch.tril(torch.ones(size,size,dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)
    return mask

#token+因果掩码
def make_decoder_mask(tokens,pad_id=0):
    '''
    scores+两个掩码叠加
    :param tokens:
    :param pad_id:
    :return:
    '''
    padding_mask=make_padding_mask(tokens,pad_id)
    causal_mask=make_causal_mask(tokens.shape[1], device=tokens.device)
    return padding_mask & causal_mask


