'''
    定义数据结构
'''
from pydantic import BaseModel,Field

#定义预测请求 请求：中文文本
class PredictRequest(BaseModel):
    text: str = Field(examples=["中英翻译，请修改这段话为需要翻译的中文文本,例如：你好世界"])

#定义预测响应 原始中文\翻译英文
class PredictResponse(BaseModel):
    src: str
    tgt: str
