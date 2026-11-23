'''
    实现预测器类及预测脚本
    推理脚本：贪心解码 + 翻译
    数据流: 中文句子 → 分词 → encoder → memory → decoder逐步生成 → 英文句子
'''
import torch
from transformers import AutoTokenizer
from config.config import *
from model.model import build_model
from runner.masks import make_padding_mask, make_decoder_mask

# 自定义预测器类
class Predictor:
    def __init__(self,model,src_tokenizer,tgt_tokenizer,device):
        self.model=model.to(device)
        self.src_tokenizer = src_tokenizer     # 编码中文输入
        self.tgt_tokenizer = tgt_tokenizer     # 解码英文输出

        self.config = TransformerConfig()
        self.device=device
        self.bos_id=tgt_tokenizer.cls_token_id
        self.eos_id=tgt_tokenizer.sep_token_id


    def _greedy_decode(self,src,src_mask,max_len=100):
        '''
        single_src [1, 10] → encoder → memory [1, 10, 256]  （编码中文，只跑一次）
            循环生成英文：
              ys = [101]                                    （只有BOS）
              → decoder(memory, ys) → generator → 取最后一位 → 预测: 3403 (hello)

              ys = [101, 3403]                              （BOS + hello）
              → decoder(memory, ys) → generator → 取最后一位 → 预测: 102 (EOS)
              → 停止

            output_ids = [101, 3403, 102]  ← 生成的英文 token IDs
        :param src:
        :param src_mask:
        :param max_len:
        :return:
        '''
        self.model.eval()
        with torch.no_grad():
            #编码器获取encode
            memory=self.model.encode(src,src_mask)
            #初始化decoder，初始化 ys 为 [BOS]，后面循环里每步把新词拼上去
            ys=torch.ones(1,1).fill_(self.bos_id).type_as(src)
            # 推理第1步: ys = [101]                       ← 只有BOS
            # → decoder → 预测出
            # 8024(hello)
            # 推理第2步: ys = [101, 8024]                  ← BOS + 上一步的词
            # → decoder → 预测出
            # 1037(world)
            #
            # 推理第3步: ys = [101, 8024, 1037]
            # → decoder → 预测出
            # 102(EOS) → 停止

            for i in range(max_len-1):
                #生成因果mask
                tgt_mask=make_decoder_mask(ys,pad_id=self.config.pad_id)
                #decoder跑一次，拿到hidden
                out=self.model.decode(memory,src_mask,ys,tgt_mask)
                #取最后一个位置，过generator得到概率【1，v】,只用通过最后一个词的评分来预测词表概率
                out=self.model.generator(out[:,-1])
                #取概率最高的
                _,next_word=out.max(dim=-1)
                #拼凑到ys后面
                ys=torch.cat([ys,next_word.unsqueeze(-1)],dim=1)
                #生成eos就停止
                if next_word.item()==self.eos_id:
                    break
        return ys.squeeze(0).tolist()


    # 预测脚本
    def predict(self,texts:str|list[str]):
        #传入中文-返回英文
        is_str=isinstance(texts,str)
        if is_str:
            texts=[texts]
        results=[]
        for text in texts:
            # 中文输入 → 用 src_tokenizer
            token_ids = self.src_tokenizer.encode(text)
            #转tensor+srcmask
            src=torch.tensor([token_ids]).to(self.device)
            src_mask=make_padding_mask(src,pad_id=self.config.pad_id)
            #贪心解码
            output_ids=self._greedy_decode(src,src_mask)
            print(f"raw ids: {output_ids}")
            #解码成文本
            # 英文输出 → 用 tgt_tokenizer
            output_text = self.tgt_tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append(output_text)
        if is_str:
            return results[0]
        return results

def predict():
    """翻译预测入口"""
    config = TransformerConfig()

    # 1. 加载分词器
    src_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    tgt_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_EN_PATH)

    # 2. 加载训练好的模型
    model = build_model(config)
    model.load_state_dict(torch.load(MODELS_DIR / BEST_MODEL, map_location=config.device, weights_only=False))

    # 3. 创建预测器
    predictor = Predictor(model, src_tokenizer, tgt_tokenizer,config.device)

    # 4. 测试翻译
    test_sentences = [
        "你漏出鸡脚了",
        "伞兵一号准备就位",
        "你17张牌能秒我？",
    ]

    results = predictor.predict(test_sentences)
    for src, tgt in zip(test_sentences, results):
        print(f"中文: {src}")
        print(f"英文: {tgt}")
        print("---")

if __name__ == '__main__':
    predict()