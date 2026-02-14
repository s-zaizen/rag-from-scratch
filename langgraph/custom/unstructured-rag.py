from pathlib import Path
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# モデル / Embeddings
# ---------------------------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.3,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore(embeddings)

# ---------------------------------------------------------------------------
# 1. ドキュメント前処理
# ---------------------------------------------------------------------------
file_paths = [
    str(SCRIPT_DIR / "example_data/004_03_00.pdf"),
    str(SCRIPT_DIR / "example_data/i2220000.html"),
    str(SCRIPT_DIR / "example_data/b1_1_34.xlsx"),
]

# PDFチャンキング
# - by_title: 見出しレベルでチャンク分割
# - combine_under_n_chars: 小さすぎるチャンクを結合
# - new_after_n_chars: 大きすぎるチャンクを分割
loader = UnstructuredLoader(
    file_paths,
    chunking_strategy="by_title",
    combine_under_n_chars=500,
    new_after_n_chars=4000,
    max_characters=8000,
    include_orig_elements=False,
)
docs = loader.load()
assert len(docs) > 0, "No documents loaded"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
_ = vector_store.add_documents(all_splits)

# ソース情報を集計（システムプロンプトに埋め込む）
source_counts: dict[str, int] = {}
for doc in all_splits:
    src = Path(doc.metadata.get("source", "unknown")).name
    source_counts[src] = source_counts.get(src, 0) + 1

source_summary = "\n".join(
    f"  - {src} ({count} chunks)" for src, count in source_counts.items()
)
print(f"Indexed {len(all_splits)} chunks from {len(source_counts)} sources.")

# ---------------------------------------------------------------------------
# 2. Retriever ツールの作成
# ---------------------------------------------------------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 6})


@tool
def retrieve_documents(query: str) -> str:
    """非構造化データ（PDF・HTML・Excel）を横断検索し、関連チャンクを返す。
    要約に必要な情報を取得するために使う。検索クエリは日本語でも英語でも可。"""
    retrieved = retriever.invoke(query)
    return "\n\n".join(
        f"[Source: {Path(doc.metadata.get('source', 'unknown')).name}]\n"
        f"{doc.page_content}"
        for doc in retrieved
    )


retriever_tool = retrieve_documents

# ---------------------------------------------------------------------------
# 3. カスタム State
# ---------------------------------------------------------------------------
class SummaryState(MessagesState):
    """要約ワークフロー用の拡張 State。

    MessagesState の messages に加え、カバレッジ判定結果を保持する。
    assess_coverage が verdict を書き込み、coverage_route が参照する。
    """

    coverage_verdict: str  # "sufficient" or "insufficient"


# ---------------------------------------------------------------------------
# 4. ノード定義
# ---------------------------------------------------------------------------

# --- plan_retrieval 用プロンプト ---
PLAN_RETRIEVAL_PROMPT = (
    "あなたは複数の非構造化データを要約する専門アシスタントです。\n\n"
    "利用可能なソース:\n"
    f"{source_summary}\n\n"
    "あなたの役割:\n"
    "- ユーザーのリクエストに応じて、要約に必要な情報を検索すること\n"
    "- 各ソースから主要テーマ・数値データ・結論を網羅的に取得すること\n"
    "- 1回の検索でカバーしきれない場合は、観点を変えて複数回検索すること\n\n"
    "会話履歴にカバレッジ評価が含まれている場合は、\n"
    "そこで指摘された不足点を補う検索を行ってください。\n\n"
    "十分な情報が集まっていると判断した場合は、ツールを呼ばずに回答してください。"
)


# ---------------------------------------------------------------------------
# Node 1: plan_retrieval — 検索戦略の計画（Agentic 判断 #1）
#
# LLM が要約に必要な情報を得るため、どんなクエリで検索するかを計画する。
# - 初回: ソース情報をもとに網羅的な検索を計画
# - 2回目以降: assess_coverage が残した分析を読み、不足点を狙って検索
# - 十分だと判断すれば検索せず直接回答 → generate_summary へルーティング
# ---------------------------------------------------------------------------
def plan_retrieval(state: SummaryState) -> dict:
    """検索戦略を LLM が計画する。

    assess_coverage の分析メッセージが会話履歴にある場合は、
    そこで指摘された不足ソース・不足テーマを補う検索を自律的に計画する。
    """
    response = (
        model
        .bind_tools([retriever_tool])
        .invoke(
            [{"role": "system", "content": PLAN_RETRIEVAL_PROMPT}]
            + state["messages"]
        )
    )
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Node 2: retrieve — ToolNode（ツール実行）
# ---------------------------------------------------------------------------
# グラフ組み立て時に ToolNode([retriever_tool]) として登録


# ---------------------------------------------------------------------------
# Node 3: assess_coverage — 累積カバレッジの構造的評価（Agentic 判断 #2）
#
# 旧 evaluate_coverage + refine_search を統合。
# - 直前の1件ではなく、全 ToolMessage を横断して評価
# - 不足がある場合は「何が足りないか」を構造的に分析
# - 分析結果をメッセージに追加 → 次の plan_retrieval が参照
# - verdict を SummaryState に書き込み → coverage_route がルーティング
# ---------------------------------------------------------------------------
ASSESS_COVERAGE_PROMPT = (
    "あなたは要約の品質管理を担当する専門家です。\n\n"
    "ユーザーのリクエスト: {question}\n\n"
    "利用可能なソース:\n{source_info}\n\n"
    "これまでに取得された全コンテキスト:\n{context}\n\n"
    "以下の観点で、要約を書くための情報が十分に揃っているか評価してください:\n"
    "1. ソース網羅性: 全ソースから情報が取得されているか？\n"
    "2. テーマ網羅性: 各ソースの主要テーマ・数値・結論がカバーされているか？\n"
    "3. 情報量: 包括的な要約を作成するのに十分な情報量があるか？\n\n"
    "評価結果を構造化して回答してください。"
)


class CoverageAssessment(BaseModel):
    """要約カバレッジの構造化評価。"""

    covered_sources: list[str] = Field(
        description="情報が取得できたソースのファイル名一覧"
    )
    missing_aspects: list[str] = Field(
        description="まだカバーされていないテーマ・ソース・観点の一覧（なければ空リスト）"
    )
    verdict: Literal["sufficient", "insufficient"] = Field(
        description="カバレッジ判定: 全ソース・主要テーマが網羅されていれば sufficient"
    )
    reasoning: str = Field(description="判定理由の簡潔な説明")


def assess_coverage(state: SummaryState) -> dict:
    """全 ToolMessage を横断してカバレッジを評価し、分析結果をメッセージに追加する。

    旧 evaluate_coverage との違い:
    - 直前の1件ではなく **累積された全 ToolMessage** を対象に評価
    - 不足がある場合は「どのソース・テーマが足りないか」を具体的に指摘
    - 分析結果はメッセージとして残り、次の plan_retrieval がそれを参照して
      不足を補う検索を計画できる（旧 refine_search の役割を吸収）
    """
    question = state["messages"][0].content

    # 全 ToolMessage を収集（累積コンテキスト）
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    all_context = "\n\n---\n\n".join(m.content for m in tool_messages)

    prompt = ASSESS_COVERAGE_PROMPT.format(
        question=question,
        source_info=source_summary,
        context=all_context,
    )

    assessment = (
        model
        .with_structured_output(CoverageAssessment)
        .invoke([{"role": "user", "content": prompt}])
    )

    analysis_parts = [
        f"【カバレッジ評価: {assessment.verdict}】",
        f"取得済みソース: {', '.join(assessment.covered_sources) or 'なし'}",
    ]
    if assessment.missing_aspects:
        analysis_parts.append(
            f"不足している観点: {', '.join(assessment.missing_aspects)}"
        )
    analysis_parts.append(f"理由: {assessment.reasoning}")

    return {
        "messages": [AIMessage(content="\n".join(analysis_parts))],
        "coverage_verdict": assessment.verdict,
    }


def coverage_route(
    state: SummaryState,
) -> Literal["generate_summary", "plan_retrieval"]:
    """assess_coverage が書き込んだ verdict に基づいてルーティングする。"""
    if state.get("coverage_verdict") == "sufficient":
        return "generate_summary"
    return "plan_retrieval"


# ---------------------------------------------------------------------------
# Node 4: generate_summary — 累積コンテキストから要約を生成（Agentic 判断 #3）
#
# 旧版との違い: state["messages"][-1] ではなく **全 ToolMessage** を使って要約。
# 複数ラウンドの検索で蓄積された情報をすべて活用する。
# ---------------------------------------------------------------------------
GENERATE_SUMMARY_PROMPT = (
    "あなたは複数の資料を横断的に分析・要約する専門家です。\n\n"
    "以下は複数回の検索で取得された全コンテキストです。\n"
    "これらを使って、ユーザーのリクエストに応じた包括的な要約を作成してください。\n\n"
    "要約の要件:\n"
    "- 各ソースの主要ポイントを網羅すること\n"
    "- ソース間の関連性や共通テーマがあれば言及すること\n"
    "- 重要な数値データを含めること\n"
    "- 重複を排除し、論理的な構造で整理すること\n"
    "- 日本語で回答すること\n\n"
    "リクエスト: {question}\n\n"
    "全取得コンテキスト:\n{context}"
)


def generate_summary(state: SummaryState) -> dict:
    """全 ToolMessage を横断して包括的な要約を生成する。

    直前のメッセージだけでなく、累積された全検索結果を使って要約する。
    """
    question = state["messages"][0].content

    # 全 ToolMessage を収集
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    all_context = "\n\n---\n\n".join(m.content for m in tool_messages)

    if not all_context:
        all_context = "（検索結果がありません。利用可能な情報に基づいて回答してください。）"

    prompt = GENERATE_SUMMARY_PROMPT.format(question=question, context=all_context)
    response = model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 5. グラフ組み立て
# ---------------------------------------------------------------------------
workflow = StateGraph(SummaryState)

workflow.add_node(plan_retrieval)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(assess_coverage)
workflow.add_node(generate_summary)

# START → LLM が要約に必要な検索を計画
workflow.add_edge(START, "plan_retrieval")

# Agentic 判断 #1: 検索が必要か？
# - tool_calls あり → retrieve（検索実行）
# - tool_calls なし → generate_summary（十分と判断して要約生成へ）
workflow.add_conditional_edges(
    "plan_retrieval",
    tools_condition,
    {"tools": "retrieve", END: "generate_summary"},
)

# retrieve → assess_coverage（累積カバレッジ評価）
workflow.add_edge("retrieve", "assess_coverage")

# Agentic 判断 #2: カバレッジは十分か？
# - sufficient   → generate_summary
# - insufficient → plan_retrieval（分析結果を踏まえて再検索）
workflow.add_conditional_edges(
    "assess_coverage",
    coverage_route,
)

# 要約生成 → 終了
workflow.add_edge("generate_summary", END)

graph = workflow.compile()

# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------
print("\n" + graph.get_graph().draw_mermaid())

# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------
query = "非構造化データの要約をしてください。"

print(f"\n=== Query: {query} ===\n")

for chunk in graph.stream(
    {"messages": [{"role": "user", "content": query}]},
    {"recursion_limit": 15},
):
    for node, update in chunk.items():
        print(f"\n--- {node} ---")
        if "messages" in update and update["messages"]:
            update["messages"][-1].pretty_print()
