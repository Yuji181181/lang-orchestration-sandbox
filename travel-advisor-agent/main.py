# LangChain モジュールの読み込み
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# 環境変数の読み込み
load_dotenv()


# 出力データの構造定義
class Destination(BaseModel):
    name: str = Field(description="観光地の名前")
    prefecture: str = Field(description="都道府県名")
    access: str = Field(description="主な移動手段（例：電車、車など）")
    highlight: str = Field(description="その観光地の見どころ")


# 出力パーサーの設定
parser = PydanticOutputParser(pydantic_object=Destination)

# 出力フォーマットの指示文取得
format_instructions = parser.get_format_instructions()

# プロンプトテンプレートの作成
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは旅行アドバイザーです"),
    ("human", "東京から1時間で行ける観光地と移動手段を教えて下さい。\n{format_instructions}")
]).partial(format_instructions=format_instructions)

# モデルの設定（Groq Llama 3.3 70Bを使用）
model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# チェーンの構築（Prompt → Model → Parser）
chain = prompt | model | parser

# 実行
result = chain.invoke({})

# 結果の表示
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("観光地の名前:", result.name)
print("都道府県名:", result.prefecture)
print("主な移動手段:", result.access)
print("観光地の見どころ:", result.highlight)
