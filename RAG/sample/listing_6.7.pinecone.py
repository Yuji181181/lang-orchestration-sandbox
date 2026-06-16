"""
RAGサンプル - データ格納（インデックス作成）フェーズ
================================================================
Wikipediaの記事をベクトル化し、Pinecone（クラウドベクトルDB）に格納する処理。
RAGの「データを準備する」側のコード。

前提知識:
- 埋め込み（Embedding）: テキストを数値のベクトル（配列）に変換する処理。
  例: "りんごは赤い" → [0.12, -0.45, 0.78, ...]（1536次元の浮動小数点数配列）
  これにより、テキストの「意味の近さ」を数値で比較できるようになる。
- Pinecone: クラウド上のベクトルDB。APIキーがあればサーバ構築不要で利用可能。
- チャンク分割: 長いドキュメントをLLMの処理に適した短いブロックに分割する処理。
"""

# ==============================
# 標準ライブラリ
# ==============================
import os              # 環境変数の読み込み用
from uuid import uuid4 # 一意のID生成用（ベクトルごとにユニークなIDが必要）

# ==============================
# 外部ライブラリ
# ==============================
import tiktoken                                    # OpenAI系のトークンカウンター
from datasets import load_dataset                  # HuggingFaceのデータセット読み込み
from langchain.text_splitter import RecursiveCharacterTextSplitter  # テキスト分割
from langchain.embeddings.openai import OpenAIEmbeddings           # OpenAI埋め込みモデル
from pinecone import Pinecone, ServerlessSpec     # Pineconeクライアント
from sentence_transformers import SentenceTransformer             # ローカル埋め込みモデル（代替用）
from tqdm.auto import tqdm                         # プログレスバー表示

# ==============================
# APIキーの取得
# ==============================
# OpenAIのAPIキー（埋め込みモデルの利用に必要）
# platform.openai.com の「API Keys」ページから取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PineconeのAPIキー（ベクトルDBの利用に必要）
# app.pinecone.io のコンソールから取得
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ==============================
# Pineconeクライアントの初期化
# ==============================
# APIキーを渡してPineconeサービスに接続するクライアントを作成
pc = Pinecone(api_key=PINECONE_API_KEY)


class WikiDataIngestion:
    """
    Wikipediaのデータを取得し、ベクトル化してPineconeに格納するクラス。

    処理の流れ:
      1. Wikipedia記事を取得（HuggingFace datasets）
      2. 各記事を小さなチャンク（ブロック）に分割
      3. 各チャンクを埋め込みモデルでベクトルに変換
      4. ベクトルとメタデータをPineconeにアップロード

    このクラスは「インデックス作成（データ格納）」を担当し、
    検索・回答生成は別のファイル（listing_8.5）で行う。
    """

    def __init__(
        self,
        index,             # Pineconeのインデックスオブジェクト（格納先のテーブルのようなもの）
        wikidata=None,     # Wikipediaデータセット（Noneなら自動でロード）
        embedder=None,     # 埋め込みモデル（NoneならOpenAIのデフォルトモデルを使用）
        tokenizer=None,    # トークナイザー（文字数カウント用）
        text_splitter=None,# テキスト分割器（Noneならデフォルト設定で作成）
        batch_limit=100,   # 1回のアップロードで送信するチャンク数の上限
    ):
        # --- Pineconeインデックス（書き込み先） ---
        self.index = index

        # --- Wikipediaデータセット ---
        # HuggingFaceのdatasetsライブラリから「simple wikipedia」を取得
        # split="train[:10000]" で最初の10,000件のみ使用（全件は数百万人件）
        # 各レコードには "id", "url", "title", "text" が含まれる
        self.wikidata = wikidata or load_dataset(
            "wikipedia", "20220301.simple", split="train[:10000]"
        )

        # --- 埋め込みモデル ---
        # OpenAIの「text-embedding-ada-002」を使用
        # 入力テキストを1536次元のベクトルに変換する
        # 同じ意味のテキストは近いベクトルになり、意味が離れたテキストは遠いベクトルになる
        self.embedder = embedder or OpenAIEmbeddings(
            model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
        )

        # --- トークナイザー ---
        # tiktokenの「cl100k_base」エンコーディングを使用
        # テキストを「トークン（LLMの最小処理単位）」に分割し、その数をカウントする
        # 例: "Hello world" → ["Hello", " world"] → 2トークン
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")

        # --- テキスト分割器 ---
        # RecursiveCharacterTextSplitter: 再帰的にテキストを分割する
        # chunk_size=400: 1チャンク最大400トークン
        # chunk_overlap=20: チャンク間で20トークン重複させる（文脈の連続性を保持）
        # separators: 改行→スペース→文字の順で分割ポイントを探す
        #   （\n\n で分けられなければ \n、それでもダメなら " "、最後は文字単位）
        self.text_splitter = (
            text_splitter
            or RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20,
                length_function=self.token_length,
                separators=["\n\n", "\n", " ", ""],
            )
        )

        # --- バッチサイズ ---
        # 1回のAPI呼び出しでPineconeに送信するチャンク数の上限
        # 大量データを一度に送るとAPIエラーになるため、バッチ単位で処理する
        self.batch_limit = batch_limit

    def token_length(self, text):
        """
        テキストのトークン数を返す関数。
        text_splitterの length_function として使用される。

        Args:
            text (str): カウント対象のテキスト
        Returns:
            int: トークン数
        """
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    def get_wiki_metadata(self, page):
        """
        Wikipedia記事からメタデータ（付随情報）を抽出する。

        Pineconeにベクトルを格納する際、ベクトルだけでなくメタデータも一緒に保存できる。
        検索結果としてベクトルが返されたときに、元の記事の情報を確認できる。

        Args:
            page (dict): 1件のWikipedia記事
        Returns:
            dict: メタデータ（記事ID、URL、タイトル）
        """
        return {
            "wiki-id": str(page["id"]),   # Wikipediaの固有ID
            "source": page["url"],         # 記事のURL（出典表示用）
            "title": page["title"],        # 記事のタイトル
        }

    def split_texts_and_metadatas(self, page):
        """
        1記事を複数チャンクに分割し、各チャンクに対応するメタデータを生成する。

        例:
          入力: "Johannes Gutenbergは..."（2000文字の記事）
          出力: texts = ["Johannes Gutenbergは...", "彼は活字を...", ...]（5チャンク）
                 metadatas = [{chunk:0, text:...}, {chunk:1, text:...}, ...]

        Args:
            page (dict): 1件のWikipedia記事
        Returns:
            tuple: (チャンクのリスト, メタデータのリスト)
        """
        # 基本メタデータを取得（記事ID、URL、タイトル）
        basic_metadata = self.get_wiki_metadata(page)

        # 記事の本文をチャンクに分割
        texts = self.text_splitter.split_text(page["text"])

        # 各チャンクにメタデータを付与
        # chunk: チャンク番号（0, 1, 2...）
        # text: チャンクのテキスト本文（検索結果確認用）
        # **basic_metadata: 記事ID、URL、タイトルを展開して結合
        metadatas = [
            {"chunk": j, "text": text, **basic_metadata}
            for j, text in enumerate(texts)
        ]
        return texts, metadatas

    def upload_batch(self, texts, metadatas):
        """
        チャンクのバッチをPineconeにアップロードする。

        処理内容:
          1. 各チャンクにユニークなIDを生成
          2. 埋め込みモデルでチャンクをベクトルに変換
          3. ID、ベクトル、メタデータをセットでPineconeに送信

        Args:
            texts (list[str]): チャンクのテキストリスト
            metadatas (list[dict]): 各チャンクに対応するメタデータリスト
        """
        # 各チャンクにユニークなIDを生成（UUID v4）
        # PineconeではIDで上書き（upsert）されるため、重複しないIDが必要
        ids = [str(uuid4()) for _ in range(len(texts))]

        # テキストのリストをベクトルのリストに変換
        # embed_documents: 複数テキストを一括でベクトル化する（1536次元×N件）
        embeddings = self.embedder.embed_documents(texts)

        # zipでID、ベクトル、メタデータを組み合わせ、Pineconeに一括送信
        # upsert: 既存IDなら上書き、新規IDなら挿入
        self.index.upsert(vectors=zip(ids, embeddings, metadatas))

    def batch_upload(self):
        """
        全データをバッチ単位でPineconeにアップロードする。

        10,000件のWikipedia記事を100件ずつバッチにまとめ、
        1バッチごとにembed → upsertを行う。
        これにより、大量データを効率的かつ安定的に格納できる。
        """
        batch_texts = []      # 現在のバッチに含まれるテキスト
        batch_metadatas = []  # 現在のバッチに含まれるメタデータ

        # tqdmで進捗バーを表示しながら全記事を処理
        for page in tqdm(self.wikidata):
            # 1記事をチャンクに分割し、メタデータを生成
            texts, metadatas = self.split_texts_and_metadatas(page)

            # バッチに追加
            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            # バッチが上限（100チャンク）に達したら、Pineconeに送信
            if len(batch_texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                # 送信後、バッチをクリアして次のバッチへ
                batch_texts = []
                batch_metadatas = []

        # 最後に残ったチャンク（100件未満）を送信
        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)


# ==============================
# メイン実行部分
# ==============================
if __name__ == "__main__":
    # --- Pineconeインデックスの作成 ---
    # インデックスはRDBで言う「テーブル」に相当
    index_name = "pincecone-llm-example"

    # 指定名のインデックスが存在しなければ作成
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            metric="cosine",       # 類似度計算方法: コサイン類似度（-1〜1の範囲）
            dimension=1536,        # ベクトルの次元数（text-embedding-ada-002の出力次元）
            spec=ServerlessSpec(
                cloud="aws",       # クラウドプロバイダ: AWS
                region="us-east-1" # リージョン: 米国東部
            ),
        )

    # インデックスへの接続を取得
    index = pc.Index(index_name)
    # インデックスの現在の状態を表示（格納件数など）
    print(index.describe_index_stats())

    # --- 埋め込みモデルの選択 ---
    # OpenAI APIキーが設定されている場合はOpenAIのモデルを使用
    # 設定されていない場合はローカルのSentenceTransformerを使用
    # （SentenceTransformerは無料でAPIキー不要だが、精度が異なる場合がある）
    embedder = None
    if not OPENAI_API_KEY:
        # ローカル埋め込みモデル（日本語・英語・韓国語対応）
        # 出力次元も1536（OpenAIと互換性あり）
        embedder = SentenceTransformer(
            "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"
        )
        # SentenceTransformerのencode()をLangChainのインターフェースに合わせてラップ
        # embed_documents()メソッドを追加し、戻り値をリスト型に変換
        embedder.embed_documents = lambda *args, **kwargs: embedder.encode(
            *args, **kwargs
        ).tolist()

    # --- データの取り込みと格納 ---
    # WikiDataIngestionクラスでWikipedia → Pineconeへの格納を実行
    wiki_data_ingestion = WikiDataIngestion(index, embedder=embedder)
    wiki_data_ingestion.batch_upload()

    # 格納後のインデックス状態を表示（格納件数が増えているはず）
    print(index.describe_index_stats())

    # --- 類似検索テスト ---
    # "Did Johannes Gutenberg invent the printing press?" という質問で検索
    # 1. 質問テキストをベクトルに変換
    # 2. Pinecone上で最も近いベクトルを上位3件取得
    # 3. メタデータ（記事本文、タイトル、URL等）も含めて返す
    query = "Did Johannes Gutenberg invent the printing press?"
    embeddings = wiki_data_ingestion.embedder.embed_documents(query)
    results = index.query(vector=embeddings, top_k=3, include_metadata=True)
    print(results)

    # 検索結果例（コメントアウト）:
    # - Johannes Gutenbergの記事（スコア: 0.871）→ 最も関連性が高い
    # - Pencil（鉛筆）の記事（スコア: 0.868）→ 「印刷」に関連する文脈でヒット
    # - Printing press（印刷機）の記事（スコア: 0.865）→ 直接関連
    #
    # スコアはコサイン類似度で、1に近いほど類似性が高い
