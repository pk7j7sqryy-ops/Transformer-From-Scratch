'''
数据预处理脚本
'''

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from config.config import *


def preprocess():
    config = TransformerConfig()

    raw_opus_dir = (
        RAW_DATA_DIR
        / f'opus100_{MODEL_DATA_TYPE}_{config.train_samples}'
    )

    # 第一阶段：获取未分词的原始数据
    if raw_opus_dir.exists():
        print(f'加载本地原始数据：{raw_opus_dir}')
        raw_dataset = load_from_disk(raw_opus_dir)

    else:
        print('正在下载 OPUS-100 en-zh 数据集...')

        dataset_online = load_dataset(
            MODEL_DATA,
            MODEL_DATA_TYPE,
        )

        train_size = min(
            config.train_samples,
            len(dataset_online['train']),
        )

        train_dataset = (
            dataset_online['train']
            .shuffle(seed=config.data_seed)
            .select(range(train_size))
        )

        raw_dataset = DatasetDict({
            'train': train_dataset,
            'valid': dataset_online['validation'],
            'test': dataset_online['test'],
        })

        RAW_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 保存未分词的原始中英文数据
        raw_dataset.save_to_disk(raw_opus_dir)

        print(f'原始数据已保存：{raw_opus_dir}')

    print('原始数据集：')
    print(raw_dataset)

    # 第二阶段：创建分词器
    src_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH
    )

    tgt_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_EN_PATH
    )

    # 第三阶段：批量分词
    def encode(batch):
        translations = batch['translation']

        zh_texts = [
            item['zh']
            for item in translations
        ]

        en_texts = [
            item['en']
            for item in translations
        ]

        src_ids = src_tokenizer(
            zh_texts,
            truncation=True,
            max_length=config.max_seq_len,
        )['input_ids']

        tgt_ids = tgt_tokenizer(
            en_texts,
            truncation=True,
            max_length=config.max_seq_len,
        )['input_ids']

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
        }

    print('编码前：')
    print(raw_dataset['train'][3])

    processed_dataset = raw_dataset.map(
        encode,
        batched=True,
        batch_size=1000,
        remove_columns=['translation'],
        desc='Tokenizing OPUS-100',
    )

    print('编码后：')
    print(processed_dataset['train'][3])
    print(processed_dataset)

    # 保存分词后的训练数据
    processed_dataset.save_to_disk(
        PROCESSED_DATA_DIR
    )

    print(f'分词数据已保存：{PROCESSED_DATA_DIR}')
    print('数据预处理完成！')


if __name__ == '__main__':
    preprocess()