from fastapi import FastAPI
from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from web.schema import PredictRequest, PredictResponse
from web.service import predict_service

'''
 @predict_router.post("/predict") 匹配到这个路由     │
│                                                       │
│  1. 收到 JSON → 用 schemas.py 的 PredictRequest 校验  │
│     {"text": "你好世界"} → 必须有 text 字段且是 str   │
│     校验失败 → 自动返回 422 错误                      │
│                                                       │
│  2. 校验通过 → 调 service.py 的 predict_service(text) │
│                                                       │
│  3. 拿到翻译结果 → 用 schemas.py 的 PredictResponse   │
│     包装成 {"src": "你好世界", "tgt": "hello world"}  │
│     自动转 JSON 返回给用户  
'''
predict_router =APIRouter(tags=['translationapi'])

@predict_router.post('/predict')
def predict(request:PredictRequest)->PredictResponse:
    try:
        text=request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        tgt = predict_service(text)

        return PredictResponse(src=text,tgt=tgt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")

