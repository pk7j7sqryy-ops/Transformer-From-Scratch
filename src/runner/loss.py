'''
Label Smoothing 损失
'''

import torch
from torch import nn

class LabelSmoothing(nn.Module):
    '''
    标签平滑做什么？ 正常标签是 one-hot：[0, 1, 0, 0]，模型拼命把正确词的概率推到 100%，容易过拟合。
    标签平滑把它变成 [0.025, 0.9, 0.025, 0.025]，给其他词留一点概率，防止模型过度自信。
    为什么用 KLDivLoss 不用 CrossEntropyLoss？
    因为 Generator 已经做了 log_softmax，输出的是 log 概率。KLDivLoss 专门比较两个概率分布的差异，输入正好是 log 概率，和 Generator 的输出匹配。
    '''
    def __init__(self,size,padding_idx=0,smoothing=0.1):
        '''

        :param size:词表大小 V
        :param padding_idx:PAD 位置，不参与 loss
        :param smoothing:平滑量，默认 0.1
        :param confidence:	正确词的权重 = 1 - 0.1 = 0.9
        '''
        super().__init__()
        self.criterion=nn.KLDivLoss(reduction='sum')
        self.padding_idx=padding_idx
        self.confidence=1.0-smoothing
        self.smoothing=smoothing
        self.size=size
        self.true_dist=None

    def forward(self,x,target):
        '''
        CrossEntropyLoss = -log(p_正确词)
        KLDivLoss = Σ p_真实(x) * log(p_真实(x)/q_预测(x))
        当真实标签是 one-hot 时（[0, 0, 1, 0]），只有正确词位置非零，KLDivLoss 简化成：
                = 1 * log(1/q_正确词)
                = -log(q_正确词)
                = CrossEntropyLoss    ← 完全一样
        :param x: Generator 输出的 log 概率 [B*T, V]
        :param target:
        :return:
        '''
        assert x.size(1) == self.size
        # 第1步：复制 x 的形状，作为构造平滑标签的容器
        true_dist = x.data.clone()
        # 第2步：全部填上平滑值
        true_dist.fill_(self.smoothing / (self.size - 2))
        #scatter_(1, index, value) 的意思：在第1维（行内）找到 index 指定的列，填入 value。
        # scatter后: [0.033, 0.033, 0.9, 0.033, 0.033]    ← index = 2位置改成0.9
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        #PAD 位置不参与 loss，清零 [0.033, 0.033, 0.9, 0.033, 0.033]  →  [0, 0.033, 0.9, 0.033, 0.033]
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
