"""
ブレインストームエージェント - LangGraph学習用コード

このエージェントは、LangGraphの基本概念を学ぶためのサンプルです。
3つの世代（10代・20代・30代）の視点でプレゼン企画を並列生成し、
ファシリテーターが統合、ジャッジが評価するワークフローを実行します。

学習ポイント：
- LangGraphの状態管理（StateGraph）
- ノードとエッジの概念
- 並列処理と条件分岐
- LangChainのプロンプトテンプレート
"""

# ============================================================
# 1. インポート（必要なライブラリを読み込み）
# ============================================================

import os
import asyncio
from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
from operator import add

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# ============================================================
# 2. 環境変数の設定
# ============================================================
# .envファイルからAPIキーなどを読み込む
# load_dotenv()は現在のディレクトリから.envファイルを探し、
# その内容を環境変数として設定する
load_dotenv()

# ============================================================
# 3. LLM（大規模言語モデル）の定義
# ============================================================
# ChatGroq: Groq社のAPIを使用してLLMと通信するクラス
# LangChainの標準的なインターフェースで異なるLLMプロバイダにアクセスできる
#
# パラメータ説明：
# - model: 使用するモデル名（llama-3.3-70b-versatileは700億パラメータの汎用モデル）
# - temperature: 出力の多様性を制御（0=決定的、1=より創造的）
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)


# ============================================================
# 4. 状態管理（LangGraphの核となる概念）
# ============================================================
# LangGraphでは「状態（State）」を介してノード間でデータを共有する
# TypedDictはPythonの辞書を型安全に扱うためのクラス
#
# 【LangGraphの基本概念】
# - StateGraph: 状態の変化をグラフとして管理するコンテナ
# - Node（ノード）: 状態を受け取り、処理して新しい状態を返す関数
# - Edge（エッジ）: ノード間の接続を定義する（次のノードへの流れ）
# - START/END: グラフの開始点と終了点
#
# 【各フィールドの説明】
# - user_input: ユーザーからの入力テキスト
# - path: 実行されたノードの履歴（Annotatedでadd演算子を使うと値が追加される）
# - ideas_10s/20s/30s: 各世代が生成した企画リスト
# - merged_ideas: ファシリテーターが統合した企画
# - evaluation: ジャッジの評価結果
class BrainstormState(TypedDict, total=False):
    user_input: str
    path: Annotated[List[str], add]  # add演算子でリストを追加結合
    ideas_10s: List[Dict[str, Any]]
    ideas_20s: List[Dict[str, Any]]
    ideas_30s: List[Dict[str, Any]]
    merged_ideas: List[Dict[str, Any]]
    evaluation: Dict[str, Any]


# ============================================================
# 5. プロンプトテンプレート（LangChainの重要な概念）
# ============================================================
# ChatPromptTemplate: プロンプトを定型化し、変数を埋め込むためのクラス
# これにより、同じプロンプトを異なる入力で再利用できる
#
# 【LangChainのチェーン概念】
# - prompt | llm のようにパイプ演算子を使うとチェーンが作成される
# - チェーンはプロンプト → LLM → 出力の流れを自動で処理する
def persona_prompt(age_label: str) -> ChatPromptTemplate:
    # システムプロンプト: LLMの役割と行動を定義
    sys = (
        f"あなたは{age_label}の視点を代表する企画ブレインです。"
        "ユーザー入力に対して、プレゼンテーションの企画を5つ考えてください。"
    )
    # ユーザープロンプト: 実際の入力テンプレート（{user_input}は後で値が入る）
    user = (
        "【ユーザー入力】\n{user_input}"
    )
    # from_messages: システム・ユーザーメッセージのリストからプロンプトを作成
    return ChatPromptTemplate.from_messages([("system", sys), ("user", user)])


# ============================================================
# 6. LLM呼び出し（非同期処理）
# ============================================================
# async/await: 非同期プログラミング（LLM応答を待つ間他の処理を実行可能）
#
# 【非同期処理の利点】
# - 複数のLLM呼び出しを並列で実行できる
# - 3つの世代の企画を同時に生成するため必要
#
# 【パイプ演算子の動作】
# - prompt | llm でチェーンが作成される
# - .ainvoke(): 非同期でチェーンを実行するメソッド
async def run_persona(user_input: str, age_label: str) -> List[Dict[str, Any]]:
    prompt = persona_prompt(age_label)
    raw = (prompt | llm)  # プロンプトとLLMをチェーン化
    resp = await raw.ainvoke({"user_input": user_input})  # 非同期実行
    return resp.content  # LLMの応答テキストを返す


# ============================================================
# 7. ノード関数の定義（LangGraphのノード）
# ============================================================
# LangGraphのノードは以下のルールに従う：
# - 現在の状態（State）を受け取る
# - 処理を実行する
# - 変更した状態の辞書を返す（部分更新可能）
#
# 【並列処理の仕組み】
# - 各世代のノードは独立して実行される
# - LangGraphが自動的に並列で実行し、完了を待つ
# - 全ノード完了後に次のフェーズ（facilitator）に進む

async def brain_10s(state: BrainstormState) -> Dict[str, Any]:
    """10代の視点で企画を生成するノード"""
    # state["user_input"]で現在の状態からユーザー入力を取得
    ideas = await run_persona(state["user_input"], "10代")
    # 返す辞書のキーはBrainstormStateのフィールドと対応
    return {"ideas_10s": ideas, "path": ["brain_10s"]}


async def brain_20s(state: BrainstormState) -> Dict[str, Any]:
    """20代の視点で企画を生成するノード"""
    ideas = await run_persona(state["user_input"], "20代")
    return {"ideas_20s": ideas, "path": ["brain_20s"]}


async def brain_30s(state: BrainstormState) -> Dict[str, Any]:
    """30代の視点で企画を生成するノード"""
    ideas = await run_persona(state["user_input"], "30代")
    return {"ideas_30s": ideas, "path": ["brain_30s"]}


# ============================================================
# 8. 統合ノード（ファシリテーター）
# ============================================================
# ファシリテーター: 複数の入力を統合・整理するノード
#
# 【LangGraphのデータフロー】
# - ノードは前のノードの出力を受け取る
# - ここでは3世代の企画（ideas_10s, ideas_20s, ideas_30s）を統合する
# - 状態の各フィールドは前のノードが設定した値が入っている
async def facilitator(state: BrainstormState) -> Dict[str, Any]:
    """3世代の企画を統合するファシリテーターノード"""
    # 統合用のプロンプトテンプレートを定義
    merged_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "あなたは熟練のファシリテーターです。3年代の案を要素抽出・統合し、"
         "重複を排除しつつ多様性と実現可能性のバランスを取って5件にまとめてください。"),
        # {ideas_10s}などは後で実際の値に置き換えられる
        ("user",
         "10代案:\n{ideas_10s}\n\n20代案:\n{ideas_20s}\n\n30代案:\n{ideas_30s}")
    ])
    raw = (merged_prompt | llm)
    # stateから各世代の企画を取得してプロンプトに渡す
    resp = await raw.ainvoke({
        "ideas_10s": state["ideas_10s"],
        "ideas_20s": state["ideas_20s"],
        "ideas_30s": state["ideas_30s"],
    })
    return {"merged_ideas": resp, "path": ["facilitator"]}


# ============================================================
# 9. 評価ノード（ジャッジ）
# ============================================================
# ジャッジ: 統合された企画を評価・ランキングする最終ノード
#
# 【評価基準】
# - 市場性: その企画に市場ニーズがあるか
# - 独自性: 他の企画と差別化できているか
# - 実行可能性: 実現可能な範囲の企画か
async def judge(state: BrainstormState) -> Dict[str, Any]:
    """統合された企画を評価するジャッジノード"""
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "あなたは事業企画の審査AIです。各企画を市場性・独自性・実行可能性で10点満点評価し、"
         "合計点でランキングしてください。リスクと改善案も簡潔に。"),
        ("user", "{items}")
    ])
    raw = (judge_prompt | llm)
    # 統合された企画（merged_ideas）を評価に渡す
    resp = await raw.ainvoke({
        "items": state["merged_ideas"]
    })
    return {"evaluation": resp, "path": ["judge"]}


# ============================================================
# 10. グラフ構築（LangGraphの核心）
# ============================================================
# StateGraph: 状態管理とノードの接続を定義するグラフコンテナ
#
# 【LangGraphのワークフロー構築ステップ】
# 1. StateGraphを作成し、状態型を指定
# 2. add_node()でノードを追加（名前と関数のペア）
# 3. add_edge()でノード間の接続を定義
# 4. compile()で実行可能なグラフに変換
#
# 【エッジの種類】
# - 通常のエッジ: 単一のノードから単一のノードへ
# - 条件付きエッジ: 状態に基づいて次のノードを分岐
# - 並列エッジ: 複数のノードから同じノードへ（ファシリテーターのように）

# グラフビルダーの作成（状態型を指定）
builder = StateGraph(BrainstormState)

# ノードの追加（名前, 関数）のペアで定義
builder.add_node("brain_10s", brain_10s)    # 10代企画ノード
builder.add_node("brain_20s", brain_20s)    # 20代企画ノード
builder.add_node("brain_30s", brain_30s)    # 30代企画ノード
builder.add_node("facilitator", facilitator)  # 統合ノード
builder.add_node("judge", judge)            # 評価ノード

# ============================================================
# 11. エッジの定義（ノード間の接続）
# ============================================================
# STARTから各世代のノードへの接続（並列で実行される）
builder.add_edge(START, "brain_10s")
builder.add_edge(START, "brain_20s")
builder.add_edge(START, "brain_30s")

# 各世代のノードからファシリテーターへの接続
# リストで複数ノードを指定すると、全ノード完了後に次のノードが実行される
builder.add_edge(["brain_10s", "brain_20s", "brain_30s"], "facilitator")

# ファシリテーター → ジャッジ → END の順で接続
builder.add_edge("facilitator", "judge")
builder.add_edge("judge", END)

# ============================================================
# 12. グラフのコンパイル
# ============================================================
# compile(): グラフを実行可能な形式に変換
# 実行時に状態の管理やノード間のデータ渡しを自動で行う
graph = builder.compile()


# ============================================================
# 13. メイン実行
# ============================================================
async def main():
    """
    グラフを実行し、結果を表示するメイン関数

    【グラフの実行フロー】
    1. ainvoke()でグラフを非同期実行
    2. 初期状態（user_input, path）を渡す
    3. LangGraphが自動的にノードを順番に実行
    4. 最終的な状態が返される
    """
    # ユーザー入力（テスト用の入力データ）
    user_input = "プレゼンテーションの自動化ツールについて"

    print("=" * 60)
    print("ブレインストームエージェント開始")
    print("=" * 60)
    print(f"\n【入力】: {user_input}\n")

    # ============================================================
    # グラフの実行
    # ============================================================
    # ainvoke(): 非同期でグラフを実行するメソッド
    # 引数は初期状態（辞書形式）
    # LangGraphが自動的に：
    # - START → brain_10s, brain_20s, brain_30s（並列実行）
    # - 全ノード完了 → facilitator（統合）
    # - facilitator完了 → judge（評価）
    # - judge完了 → END
    result = await graph.ainvoke({
        "user_input": user_input,
        "path": []  # 実行パスの初期値（空リスト）
    })

    # ============================================================
    # 結果の表示
    # ============================================================
    # resultは最終的な状態の辞書（BrainstormStateの全フィールドを含む）
    print("\n" + "=" * 60)
    print("【10代の企画】")
    print("=" * 60)
    print(result.get("ideas_10s", "なし"))

    print("\n" + "=" * 60)
    print("【20代の企画】")
    print("=" * 60)
    print(result.get("ideas_20s", "なし"))

    print("\n" + "=" * 60)
    print("【30代の企画】")
    print("=" * 60)
    print(result.get("ideas_30s", "なし"))

    print("\n" + "=" * 60)
    print("【統合された企画】")
    print("=" * 60)
    print(result.get("merged_ideas", "なし"))

    print("\n" + "=" * 60)
    print("【審査結果】")
    print("=" * 60)
    print(result.get("evaluation", "なし"))

    print("\n" + "=" * 60)
    print("【実行パス】")
    print("=" * 60)
    # pathリストは各ノードが追加した値が順番に入っている
    print(" → ".join(result.get("path", [])))


# ============================================================
# 14. スクリプトのエントリーポイント
# ============================================================
# __name__ == "__main__": このファイルが直接実行された場合のみ実行
# asyncio.run(): 非同期関数を実行するためのヘルパー
if __name__ == "__main__":
    asyncio.run(main())


# ============================================================
# 【LangGraph学習サマリー】
# ============================================================
#
# ■ LangGraphとは？
# LangChainエコシステム内のライブラリで、AIエージェントの
# ワークフローを「グラフ」として定義・実行するためのツール
#
# ■ 核となる概念
# 1. StateGraph: 状態の変化を管理するグラフコンテナ
# 2. State（状態）: ノード間で共有されるデータ（TypedDictで定義）
# 3. Node（ノード）: 状態を受け取り処理する関数
# 4. Edge（エッジ）: ノード間の接続を定義
# 5. START/END: グラフの開始点と終了点
#
# ■ 実行フロー
# 1. StateGraphを作成し、状態型を指定
# 2. add_node()でノードを追加
# 3. add_edge()で接続を定義
# 4. compile()で実行可能に変換
# 5. ainvoke()で初期状態を渡して実行
#
# ■ このサンプルのポイント
# - 並列処理: 3世代の企画を同時に生成
# - 状態管理: 各ノードの出力を状態に格納
# - チェーン化: prompt | llm でプロンプトとLLMを結合
# - 非同期実行: async/await で効率的な処理
#
# ■ LangChainとの関係
# - LangChain: LLMアプリケーション開発の基盤ライブラリ
# - LangGraph: LangChain上のワークフロー管理ツール
# - ChatGroq: LangChainのLLMプロバイダの一つ
# - ChatPromptTemplate: LangChainのプロンプト管理クラス
# ============================================================
