'''
    服务层
'''
import torch
from torch._C import device
from transformers import AutoTokenizer

from config.config import *
from model.model import build_model
from runner.predict import Predictor

'''
 service.py                                          │
│  predict_service("你好世界")                         │
│                                                       │
│  1. 启动时已加载好：                                 │
│     - model（模型权重）                              │
│     - src_tokenizer（中文分词器）                     │
│     - tgt_tokenizer（英文分词器）                     │
│     - predictor（Predictor 实例）                    │
│                                                       │
│  2. 调 predictor.predict("你好世界")                  │
│     → 中文分词 → encoder → 贪心解码 → 英文文本       │
│     → 返回 "hello world"   
'''

def predictor_service():
    #加载配置
    config=TransformerConfig()
    device=config.device

    #模型加载
    print('加载模型')
    model=build_model(config)
    model.load_state_dict(
        torch.load(MODELS_DIR / BEST_MODEL, map_location=device, weights_only=False)
    )
    model.eval()
    print('模型加载完成')

    #分词器加载
    print('加载分词器。。。')
    src_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    tgt_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_EN_PATH)
    print('分词器加载完成！')

    #创建预测器
    predictor=Predictor(model,src_tokenizer,tgt_tokenizer,device)

    return predictor

def predict_service(text):
    predictor=predictor_service()
    return predictor.predict(text)

