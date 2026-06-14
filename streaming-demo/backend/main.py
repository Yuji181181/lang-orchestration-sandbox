"""
LLMチャットボット ストリーミングデモ バックエンド
FastAPI + Groq API を使用したSSEストリーミングサーバー
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# 環境変数読み込み
load_dotenv("../../.env")

app = FastAPI()

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq APIの設定
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class ChatRequest(BaseModel):
    """チャットリクエストの型定義"""
    message: str


async def stream_groq_response(message: str):
    """
    Groq APIからストリーミングでレスポンスを受け取り、
    SSE（Server-Sent Events）形式でフロントエンドに中継する

    【ストリーミングの流れ】
    フロントエンド → FastAPI → Groq API (stream=True)
                                  ↓
    フロントエンド ← SSE ← FastAPI ← チャンク受信
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": message}],
        "stream": True,  # ストリーミング有効化
    }

    # httpxでストリーミングリクエスト
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", GROQ_API_URL, json=payload, headers=headers
        ) as response:
            # レスポンスのチャンクを1つずつ処理
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # "data: " を除去
                    if data.strip() == "[DONE]":
                        # ストリーミング終了
                        yield f"data: [DONE]\n\n"
                        break
                    # チャンクをSSE形式でフロントエンドに送信
                    yield f"data: {data}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    チャットエンドポイント（SSEストリーミング）

    フロントエンドからのメッセージを受け取り、
    Groq APIのストリーミングレスポンスを中継する
    """
    return StreamingResponse(
        stream_groq_response(request.message),
        media_type="text/event-stream",
    )


@app.get("/")
async def root():
    return {"message": "LLM Chatbot Streaming Backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
