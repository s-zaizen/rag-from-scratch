from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

SCRIPT_DIR = Path(__file__).resolve().parent

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = InMemoryVectorStore(embeddings)

# https://www.meti.go.jp/shingikai/mono_info_service/distribution_industry/pdf/004_03_00.pdf
# https://www.meti.go.jp/report/tsuhaku2023/2023honbun/i2220000.html
# https://www.chusho.meti.go.jp/pamflet/hakusyo/2023/chusho/excel/b1_1_34.xlsx
file_paths = [
    str(SCRIPT_DIR / "example_data/004_03_00.pdf"),
    str(SCRIPT_DIR / "example_data/i2220000.html"),
    str(SCRIPT_DIR / "example_data/b1_1_34.xlsx"),
]

loader = UnstructuredLoader(file_paths, chunking_strategy="basic", max_characters=1000000, include_orig_elements=False,)

docs = loader.load()

assert len(docs) > 0, "No documents loaded"
total_chars = sum(len(doc.page_content) for doc in docs)
print(f"Loaded {len(docs)} documents. Total characters: {total_chars}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, 
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(all_splits)

print(f"Split example data into {len(all_splits)} sub-documents.")

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

source_counts: dict[str, int] = {}
for _doc in all_splits:
    src = Path(_doc.metadata.get("source", "unknown")).name
    source_counts[src] = source_counts.get(src, 0) + 1
source_summary = "\n".join(
    f"  - {src} ({count} chunks)" for src, count in source_counts.items()
)

@tool
def retrieve_context(query: str) -> str:
    """全ソースを横断して、クエリに近いチャンクを検索して返す。

    Args:
        query: 検索クエリ。検索に適したキーワードやフレーズに言い換えると精度が上がる。example: "2022年の貿易赤字の要因分析"
    """
    docs = retriever.invoke(query)
    return "\n\n".join(
        f"[Source: {Path(doc.metadata.get('source', 'unknown')).name}]\n{doc.page_content}"
        for doc in docs
    )


tools = [retrieve_context]
prompt = (
    "あなたは複数の非構造化データを検索・要約できるアシスタントです。\n\n"
    "利用可能なソース:\n"
    f"{source_summary}\n\n"
    "ツールの使い分け:\n"
    "- retrieve_context: 全ソースを横断検索する\n"
    "回答は必ず取得したコンテキストに基づいてください。"
)

schema = {
    "type": "object",
    "description": "非構造化データの要約",
    "properties": {
        "summary": {"type": "string", "description": "要約した文書"},
    },
    "required": ["summary"]
}

agent = create_agent(model, tools, system_prompt=prompt, response_format=ProviderStrategy(schema))

query = (
    "非構造化データの要約をしてください。\n\n"
    "日本語で回答してください。"
)

result = agent.invoke({"messages": [{"role": "user", "content": query}]})

# TODO: Visualize decision with LangGraph
# TODO: Evaluate result with LangSmith
print(result["structured_response"]["summary"])