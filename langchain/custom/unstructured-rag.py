from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

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

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
prompt = (
    "あなたは非構造化データからコンテキストを取得できるツールにアクセスできます。"
    "ツールを使ってユーザーの質問に答えてください。"
)
agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "非構造化データの詳細は何ですか？\n\n"
    "回答を得たら、さらに非構造化データの詳細を要約して、日本語で回答してください。"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()