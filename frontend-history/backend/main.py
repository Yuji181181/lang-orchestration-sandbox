"""
会話履歴付きチャットボット バックエンド
FastAPI + Groq API を使用したSSEストリーミングサーバー

【学習ポイント】
- 会話履歴（Chat History）の管理方法
- ストリーミングによるリアルタイム応答
- SSE（Server-Sent Events）の実装
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# 環境変数読み込み（.envファイルからGROQ_API_KEYを取得）
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
    """
    チャットリクエストの型定義

    【学習ポイント】
    Pydanticでリクエストの型を定義することで、
    - バリデーション（不正なリクエストの拒否）
    - 自動ドキュメント生成（Swagger UI）
    が可能になる
    """
    messages: list  # 会話履歴全体（[{role, content}, ...]）


async def stream_groq_response(messages: list):
    """
    Groq APIからストリーミングでレスポンスを受け取り、
    SSE（Server-Sent Events）形式でフロントエンドに中継する

    【会話履歴の処理フロー】
    フロントエンドから会話履歴全体が送信される
    → Groq APIにそのまま渡す
    → ストリーミングレスポンスを中継

    【なぜ履歴全体を渡すのか？】
    LLMは「直前の会話」を見て回答する。
    履歴がないと「その詳細を教えて」のような
    前文脈に依存した質問に答えられない。
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        # 会話履歴全体をmessagesとして渡す
        # 例: [
        #   {"role": "user", "content": "Reactとは？"},
        #   {"role": "assistant", "content": "ReactはUIライブラリです"},
        #   {"role": "user", "content": "その詳細を教えて"}
        # ]
        "messages": messages,
        "stream": True,  # ストリーミング有効化
    }

    # httpxでストリーミングリクエスト
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", GROQ_API_URL, json=payload, headers=headers
        ) as response:
            # レスポンスのチャンクを1行ずつ処理
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # "data: " を除去
                    if data.strip() == "[DONE]":
                        # ストリーミング終了シグナル
                        yield f"data: [DONE]\n\n"
                        break
                    # チャンクをSSE形式でフロントエンドに送信
                    yield f"data: {data}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    チャットエンドポイント（SSEストリーミング）

    【学習ポイント】
    - リクエストボディに会話履歴全体が含まれる
    - レスポンスはSSE（text/event-stream）形式
    - フロントエンドはこのストリーミングを受信して1文字ずつ表示する
    """
    return StreamingResponse(
        stream_groq_response(request.messages),
        media_type="text/event-stream",
    )


@app.get("/")
async def root():
    return {"message": "Chat History Demo Backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
