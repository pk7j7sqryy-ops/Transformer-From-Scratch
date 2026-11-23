'''
    评估脚本
    使用 BLEU score + test loss 评估翻译模型
    数据流: 测试集 → 逐条贪心解码生成翻译 → 和参考翻译对比 → 算 BLEU
'''
from enum import StrEnum
from sacrebleu import corpus_bleu
from tqdm import tqdm
from transformers import AutoTokenizer
from config.config import *
from model.model import build_model
from proprocess.dataset import get_dataloader
from runner.loss import LabelSmoothing
from runner.masks import make_padding_mask
from runner.predict import Predictor
from runner.training import Batch


class Metric(StrEnum):
    BLEU='bleu'
    LOSS='loss'


#评估模型 生成翻译
def evaluate_model(model,dataloader,src_tokenizer,tgt_tokenizer,device, loss_fn=None, config=None):
    model.eval()
    predictor=Predictor(model,src_tokenizer,tgt_tokenizer,device)

    all_hypotheses = []    # 生成的翻译
    all_references = []    # 参考翻译
    all_sources = []       # 原始中文（打印样例用）
    total_loss = 0
    total_tokens = 0

    pad_id=config.pad_id

    for batch in tqdm(dataloader,desc='eval'):
        #拿到对应的中英文原词,数据在前面dataset的时候就已经填充多余位置0了
        # shape [4, 10]  4条中文，最长10个token
        '''
        src = [
          [101, 872, 1962, 102, 0,   0,   0,   0,   0,   0],   # 第0条: 你好世界 [PAD填充]
          [101, 639, 2828, 102, 0,   0,   0,   0,   0,   0],   # 第1条: 我喜欢你
          [101, 791, 1921, 6843, 102, 0,   0,   0,   0,   0],  # 第2条: 今天天气很好
          [101, 800, 1962, 102, 0,   0,   0,   0,   0,   0],   # 第3条: 世界你好
        ]
        '''
        # shape [4, 8]   4条英文，最长8个token
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        #掩码填充，标记哪些位置是真实token，哪些是PAD
        src_mask=make_padding_mask(src,pad_id=pad_id)
        #不计算梯度
        with torch.no_grad():
            #贪心解码
            #取第第一个维度，也就是batchsize
            for i in range(src.size(0)): # i = 0, 1, 2, 3
                #取每一条中文，0, 1, 2, 3
                single_src=src[i:i+1] # shape [1, 10]  ← 只取1条
                single_src_mask=src_mask[i:i+1]# shape [1, 1, 1, 10]  ← 对应的mask

                #贪心解码 输入每一条中文和掩码，生成英文
                output_ids=predictor._greedy_decode(single_src,single_src_mask)

                # 生成 → 英文文本
                # skip_special_tokens=True 会去掉 101(CLS) 和 102(SEP)
                # 结果: "hello world"
                hyp_text = tgt_tokenizer.decode(output_ids, skip_special_tokens=True)

                ## 参考 → 英文文本（从 tgt 里取真实答案）
                # tgt[0] = [101, 3403, 102, 0, 0, 0, 0, 0]
                # 去掉特殊token和PAD后: "hello world"
                ref_text = tgt_tokenizer.decode(tgt[i].tolist(), skip_special_tokens=True)

                #中文原文（打印样例用）
                src_text = src_tokenizer.decode(src[i].tolist(), skip_special_tokens=True)
                # 结果: "你好世界"

                #存数据
                all_hypotheses.append(hyp_text)
                all_references.append(ref_text)
                all_sources.append(src_text)
            #算损失
            if loss_fn:
                #处理 右移动一位
                batch_data=Batch(src,tgt,pad_id=pad_id)
                # 前向传播 [4, T, 256]
                out=model(batch_data.src, batch_data.tgt, batch_data.src_mask, batch_data.tgt_mask)
                # 映射到词表 [4, T, 30522]
                out=model.generator(out)
                ## [4*7, 30522]， ← 28 个位置，每个位置有 30522 个类的概率，[4*7]， ← 28 个位置的正确答案（类索引），传入loss
                loss = loss_fn(out.reshape(-1, out.size(-1)), batch_data.tgt_y.reshape(-1))
                #损失求和
                total_loss += loss.item()
                total_tokens += batch_data.ntokens.item()
    bleu = corpus_bleu(all_hypotheses, [all_references])
    results = {Metric.BLEU: bleu.score}
    if loss_fn:
        results[Metric.LOSS] = total_loss / max(1, total_tokens)
    return results, all_sources, all_hypotheses, all_references

def run_evaluation():
    '''
    载入模型 → 加载测试集 → 评估 → 打印结果和翻译样例
    '''
    config = TransformerConfig()
    device = config.device

    #1、加载模型
    model = build_model(config)
    model.load_state_dict(
        torch.load(MODELS_DIR/BEST_MODEL,map_location=device,weights_only=False)

    )
    #2、加载分词器
    src_tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    tgt_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_EN_PATH)

    #3、加载测试集
    test_loader=get_dataloader('test',config.batch_size, config.pad_id, limit=config.limit)

    #4、定义loss
    loss_fn=LabelSmoothing(
        size=config.tgt_vocab_size,
        padding_idx=config.pad_id,
        smoothing=0.1,
    )

    #5、评估
    results, sources, hypotheses, references=evaluate_model(
        model,test_loader,src_tokenizer,tgt_tokenizer,device,loss_fn,config
    )
    # 6. 打印评估结果
    print("\n======= 评估结果 =======")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    print("========================\n")

    # 7. 打印翻译样例
    print("======= 翻译样例 =======")
    for i in range(min(10, len(sources))):
        print(f"中文: {sources[i]}")
        print(f"生成: {hypotheses[i]}")
        print(f"参考: {references[i]}")
        print("---")


if __name__ == '__main__':
    run_evaluation()