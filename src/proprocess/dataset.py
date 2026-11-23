"""
    取数据 — get_dataset() 从 data/processed/train/ 加载预处理好的数据（每条是 {'src_ids': [...], 'tgt_ids': [...]}）
    按批填充 — collate_batch() 把一个 batch 内所有序列按最大长度对齐，短的用 0 填充
    组装 DataLoader — get_dataloader() 把上面两步组合起来，返回一个可迭代的 DataLoader
"""
import torch
from functools import partial
from datasets import load_from_disk
from torch.utils.data import DataLoader
from config.config import *

def get_dataset(ds_type='train'):
    """获取从 preprocess 保存好的数据集"""
    path = PROCESSED_DATA_DIR / ds_type
    dataset = load_from_disk(path)
    return dataset

def collate_batch(batch, pad_id=0):
    """
    把一个 batch 的 (src_ids, tgt_ids) 列表 → padding 后的 dict
    batch 是 list，每个元素是 {'src_ids': [...], 'tgt_ids': [...]}
    """
    src_list = [item['src_ids'] for item in batch]
    tgt_list = [item['tgt_ids'] for item in batch]

    src_max = max(len(s) for s in src_list)
    tgt_max = max(len(t) for t in tgt_list)

    src_padded = [s + [pad_id] * (src_max - len(s)) for s in src_list]
    tgt_padded = [t + [pad_id] * (tgt_max - len(t)) for t in tgt_list]

    return {
        'src': torch.tensor(src_padded, dtype=torch.long),
        'tgt': torch.tensor(tgt_padded, dtype=torch.long),
    }

def get_dataloader(ds_type='train', batch_size=10, pad_id=0,limit=None):
    """创建 DataLoader，每次迭代返回 padding 后的 dict"""
    dataset = get_dataset(ds_type)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    collate_fn = partial(collate_batch, pad_id=pad_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(ds_type == 'train'),
        collate_fn=collate_fn
    )
    return dataloader

if __name__ == '__main__':
    config=TransformerConfig()
    dataloader = get_dataloader('train', batch_size=config.batch_size,pad_id=config.pad_id)
    print(dataloader)
    for batch in dataloader:
        print(f"src -> {batch['src']}")
        print(f"tgt -> {batch['tgt'].shape}")
        break