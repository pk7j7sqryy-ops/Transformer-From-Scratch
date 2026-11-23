from fastapi import FastAPI

import uvicorn
from fastapi import FastAPI

from web.routers import predict_router
'''
用户发请求: POST http://localhost:8000/predict
  body: {"text": "你好世界"}
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  app.py                                              │
│  FastAPI 接收 HTTP 请求，路由到 /predict             │
│  app.include_router(predict_router) 把路由注册进来  │
'''
app = FastAPI(title='中英翻译API')

app.include_router(predict_router)

def run_app():
    uvicorn.run('web.app:app', host='127.0.0.1', port=9001, reload=True)


if __name__ == '__main__':
    run_app()
