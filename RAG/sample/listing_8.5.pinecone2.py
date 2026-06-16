"""
RAGサンプル - 検索・回答生成フェーズ
================================================================
listing_6.7で格納したWikipediaベクトルを検索し、
LLM（GPT）に渡して回答を生成する処理。

RAGの「ユーザーの質問に答える」側のコード。

前提知識:
- RetrievalQA: LangChainのQAチェーン。検索結果をLLMのプロンプトに注入して回答を生成する。
- chain_type="stuff": 検索結果をすべて結合して1つのプロンプトにまとめる方式。
  他の方式: "map_reduce"（各チャンクにLLMを適用後集約）、"refine"（逐次改善）など。
- similarity_search: ベクトルの類似度に基づき、クエリに近いドキュメントを検索する。
"""

# ==============================
# 標準ライブラリ
# ==============================
import os  # 環境変数の読み込み用

# ==============================
# LangChain関連
# ==============================
from langchain.chains import RetrievalQA               # 検索結果を使ったQAチェーン
from langchain.chains import RetrievalQAWithSourcesChain # 出典付きQAチェーン
from langchain.chat_models import ChatOpenAI             # OpenAIチャットモデル（GPT）
from langchain.embeddings.openai import OpenAIEmbeddings # OpenAI埋め込みモデル
from langchain.vectorstores import Pinecone as lc_Pinecone  # LangChain用のPineconeラッパー

# ==============================
# Pinecone SDK（直接操作用）
# ==============================
from pinecone import Pinecone

# ==============================
# APIキーの取得
# ==============================
# OpenAIのAPIキー（LLMの利用に必要）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PineconeのAPIキー（ベクトルDBの検索に必要）
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ==============================
# Pineconeクライアントの初期化
# ==============================
pc = Pinecone(api_key=PINECONE_API_KEY)


# ==============================
# ベクトルストアの接続
# ==============================
# listing_6.7で作成したインデックスに接続
index_name = "pincecone-llm-example"
index = pc.Index(index_name)

# OpenAIの埋め込みモデルを初期化
# ※ listing_6.7で書き込み時に使用したのと同じモデルを使用する必要がある
# （書き込みと検索でベクトルの次元数や意味空間が異なると正しく検索できない）
embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
)

# --- LangChainのPineconeラッパーを作成 ---
# LangChainが提供するPineconeクラスで、Pineconeインデックスをラップする
# これにより、LangChainの共通インターフェース（similarity_search等）が使えるようになる
#
# 引数の説明:
#   index: Pineconeのインデックスオブジェクト
#   embedder.embed_query: クエリ（検索テキスト）をベクトルに変換する関数
#     ※ embed_documents（ドキュメント用）ではなく embed_query（クエリ用）を使う
#     ※ 本质上同じモデルだが、内部的に微妙に異なる前処理が行われる場合がある
#   text_field: メタデータの中でテキスト本文が格納されているフィールド名
#     （listing_6.7で "text" として格納したフィールド）
text_field = "text"
vectorstore = lc_Pinecone(index, embedder.embed_query, text_field)


# ==============================
# 類似検索テスト
# ==============================
# "Who was Johannes Gutenberg?" という質問に最も関連するドキュメントを3件検索
# 以下の処理が自動的に行われる:
#   1. クエリテキストをembed_queryでベクトルに変換
#   2. Pinecone上でコサイン類似度を計算
#   3. 上位3件のドキュメント（テキスト＋メタデータ）を返す
query = "Who was Johannes Gutenberg?"
vectorstore.similarity_search(
    query, k=3  # k=3: 上位3件を返す
)


# ==============================
# LLM（大規模言語モデル）のセットアップ
# ==============================
# ChatOpenAI: OpenAIのチャットモデルをLangChain経由で利用するクラス
# GPT-3.5-turboを使用（コストパフォーマンスが良い）
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",  # チャットモデルの指定
    temperature=0.0,              # Temperature: 回答のランダム性を制御
                                 # 0.0 = 完全に決定的（同じ入力で同じ出力）
                                 # 値が大きいほど多様な回答になる
)


# ==============================
# RetrievalQA: 検索結果を使ったQAチェーン
# ==============================
# 内部で以下の処理が自動的に行われる:
#   1. ユーザーの質問をベクトル化
#   2. Pineconeで類似ドキュメントを検索（リトリーバー）
#   3. 検索結果をプロンプトに組み込んでLLMに渡す
#   4. LLMが回答を生成
#
# chain_type="stuff" の動作:
#   検索結果をすべて1つの文字列に結合し、1回のLLM呼び出しで回答を生成する
#   プロンプト例:
#     「以下の情報を参考に質問に答えてください:
#       [ドキュメント1]
#       [ドUIBarButtonItem]
#       ...
#     質問: Who was Johannes Gutenberg?」
qa = RetrievalQA.from_chain_type(
    llm=llm,                                      # 使用するLLM
    chain_type="stuff",                           # チェーンタイプ（全結果を結合）
    retriever=vectorstore.as_retriever()           # ベクトルストアをリトリーバーに変換
)

# QAチェーンを実行して回答を取得
qa.run(query)


# ==============================
# RetrievalQAWithSourcesChain: 出典付きQAチェーン
# ==============================
# RetrievalQAと同様だが、回答だけでなく「参照した出典（WikipediaのURL等）」も返す。
# 回答の根拠を確認できるため、透明性が高い。
#
# 戻り値の構造:
#   {
#     "question": "質問テキスト",
#     "answer": "LLMの回答",
#     "sources": "参照したドキュメントの出典URL"
#   }
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 出典付きで回答を生成
result = qa_with_sources(query)
print(result)

# ※ 出力例（コメントアウト）:
# {
#   'question': 'Who was Johannes Gutenberg?',
#   'answer': 'Johannes Gutenberg was a German metal-worker and inventor...',
#   'sources': 'https://simple.wikipedia.org/wiki/Johannes_Gutenberg'
# }

# ==============================
# RAG全体の流れまとめ（listing_6.7 + listing_8.5）
# ==============================
# 1. [listing_6.7] Wikipedia → チャンク分割 → ベクトル化 → Pineconeに格納
# 2. [listing_8.5] ユーザー質問 → ベクトル化 → Pineconeで類似検索
#                → 関連ドキュメントを取得 → LLMに渡して回答を生成
#
# これにより、LLMは事前学習データに含まれない知識（Wikipedia等）も参照して回答できる。
