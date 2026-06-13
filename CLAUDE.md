# Lang Orchestration Framework

## プロジェクト概要
LangChain/LangGraphを使用したAIエージェント開発プロジェクト（学習用）

## 現在の環境

| 項目 | 内容 |
|------|------|
| Python | 3.12 |
| パッケージ管理 | uv |
| 仮想環境 | .venv（uv venv） |

### インストール済みパッケージ
- langchain
- langgraph
- langchain-groq
- python-dotenv

## 環境変数（.env）

ルートディレクトリに `.env` ファイルを配置：

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

- `.gitignore` に含まれるためgitにコミットされない
- 新しいAPIキーが必要な場合は `.env` に追加

## ディレクトリ構成

```
lang-orchestration-flamework/
├── .env                    # 環境変数（APIキー等）
├── .gitignore              # git除外設定
├── .python-version         # Python 3.12
├── pyproject.toml          # プロジェクト設定
├── uv.lock                 # 依存関係ロックファイル
├── CLAUDE.md               # このファイル
```

## エージェント開発のルール

### 新規エージェント作成
1. ルートディレクトリに `{agent-name}/` フォルダを作成
2. `main.py` を作成
3. APIキーは `.env` から `os.getenv()` で取得

### パッケージ追加
```bash
uv add {package-name}
```

### コーディング規約
- 日本語コメントを付ける
- Pydantic BaseModelで出力構造を定義する
- `load_dotenv()` で環境変数を読み込む

## 使用中のAPI

| サービス | 用途 | 利用制限 |
|----------|------|----------|
| Groq | LLM推論 | アプリケーション開発可 |

## 実行コマンド
```bash
uv run {agent-name}/main.py
```
