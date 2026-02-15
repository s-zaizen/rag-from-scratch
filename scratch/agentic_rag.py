"""
Agentic RAG — フルスクラッチ実装

Basic RAG との違い:
  - Basic RAG: 1回の検索 → 回答生成（top_k=2 固定）
  - Agentic RAG: LLM がクエリ分解 → 反復検索 → カバレッジ評価 → 回答生成

Agent ループ:
  ┌─────────────────────────────────────────────────────┐
  │  Step 1: decompose_query                            │
  │    LLM がユーザー質問を検索用サブクエリに分解       │
  │                                                     │
  │  Step 2: retrieve (各サブクエリで FAISS 検索)       │
  │    重複チャンクを排除しながら結果を蓄積             │
  │                                                     │
  │  Step 3: assess_coverage                            │
  │    LLM が取得済み情報の網羅性を評価                 │
  │    ├─ sufficient → Step 4 へ                        │
  │    └─ insufficient → 不足点を指摘 → Step 2 へ戻る  │
  │                                                     │
  │  Step 4: generate_answer                            │
  │    全取得チャンクを使って包括的な回答を生成         │
  └─────────────────────────────────────────────────────┘
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Data Preparation
#    架空企業「エグザンプルデータパブリッシャー」の社内ナレッジベース
# ---------------------------------------------------------------------------
documents = [
    # --- 会社概要 ---
    "株式会社エグザンプルデータパブリッシャーは2015年に大阪で創業したAIスタートアップである。2022年3月に本社を横浜市みなとみらいのZetaタワー28階に移転した。代表取締役CEOは佐藤健一。",
    "エグザンプルデータパブリッシャーの従業員数は2023年度末時点で1,523名、2024年度末時点で1,847名である。エンジニア比率は全体の62%を占める。",

    # --- 財務情報 ---
    "エグザンプルデータパブリッシャーの2023年度の売上高は290億円、営業利益は32億円（営業利益率11.0%）であった。",
    "エグザンプルデータパブリッシャーの2024年度の売上高は342億円（前年比+18%）、営業利益は45億円（営業利益率13.2%）であった。主力製品ZetaCore v3の販売が好調で、海外売上比率は28%に達した。",
    "エグザンプルデータパブリッシャーの研究開発費は2023年度が38億円、2024年度が51億円で、売上高比率はそれぞれ13.1%、14.9%であった。",

    # --- 製品情報 ---
    "ZetaCore v3はエッジAI推論チップで、前モデルv2比で消費電力40%削減・推論速度2.1倍を実現した。主要顧客は自動車メーカーA社とロボットメーカーB社。2024年4月に出荷開始。",
    "ZetaGuardは量子暗号通信ミドルウェアで、2024年10月にβ版をリリースした。金融機関向けに2025年度の本格展開を予定している。PoC段階の顧客は3社（メガバンク2行、証券1社）。",
    "ZetaSenseは産業用IoTセンサープラットフォームで、2023年度に販売終了した旧製品である。後継はZetaCore v3のIoTエディション。",

    # --- 人事・組織 ---
    "エグザンプルデータパブリッシャーのCTOは田中美咲氏（2021年にGoogleから転職）。VP of Engineeringは李明浩氏（2023年にAmazonから転職）。AI研究部門の部長は鈴木拓也氏。",
    "エグザンプルデータパブリッシャーの有給休暇取得率は2024年度で87.3%。育児休業取得率は男性72%・女性100%。離職率は2023年度8.2%→2024年度5.1%に改善した。",
    "エグザンプルデータパブリッシャーの新卒採用人数は2024年度が85名（うちエンジニア63名）、2025年度計画は110名（うちエンジニア80名）。中途採用は2024年度が204名。",

    # --- 研究開発戦略 ---
    "エグザンプルデータパブリッシャーの中期経営計画（2024-2026年）では、重点投資分野としてエッジAI（投資額60億円）、量子暗号通信（投資額30億円）、生成AI応用（投資額20億円）の3領域を掲げている。",
    "エグザンプルデータパブリッシャーのAI研究部門は2024年度にNeurIPS 2本、ICML 1本の論文が採択された。特にエッジデバイス向け量子化手法の研究が高く評価されている。",

    # --- パートナーシップ・イベント ---
    "エグザンプルデータパブリッシャーは2024年8月にドイツのシュミットAG社と資本業務提携を締結した。シュミットAG社は欧州最大の産業用ロボットメーカーで、ZetaCore v3の欧州販売をシュミットAG社が担う。",
    "エグザンプルデータパブリッシャーは毎年11月に技術カンファレンス「ZetaConf」を開催している。2024年のZetaConf 2024には参加者2,300名が集まり、基調講演はCTO田中美咲氏が担当した。",

    # --- 知財・セキュリティ ---
    "エグザンプルデータパブリッシャーの特許保有数は2024年度末時点で国内287件、海外142件の計429件。2024年度の新規出願は国内52件、海外31件であった。",
    "エグザンプルデータパブリッシャーのISO 27001認証は2023年6月に取得。SOC 2 Type II報告書は2024年3月に初取得した。セキュリティ専任チームは12名体制。",
]

# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer('stsb-xlm-r-multilingual')
doc_embeddings = embedding_model.encode(documents, normalize_embeddings=True)

# ---------------------------------------------------------------------------
# 3. Indexing
# ---------------------------------------------------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings.astype(np.float32))

print(f"[Indexing] {len(documents)} documents indexed (dim={dimension})")

# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """FAISS でコサイン類似度検索を行い、上位 top_k 件を返す。"""
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)

    results = []
    for idx, (i, score) in enumerate(zip(indices[0], scores[0])):
        results.append({
            "rank": idx + 1,
            "doc_id": int(i),
            "score": float(score),
            "content": documents[i],
        })
    return results


# ---------------------------------------------------------------------------
# 5. Agentic RAG
# ---------------------------------------------------------------------------
client = genai.Client()
MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Step 1: クエリ分解（Agentic 判断 #1）
#   ユーザーの質問を、検索に適した複数のサブクエリに分解する。
#   単純な質問はそのまま 1 つのサブクエリとして返す。
# ---------------------------------------------------------------------------
def decompose_query(question: str) -> list[str]:
    """LLM がユーザー質問を検索用サブクエリに分解する。"""

    prompt = (
        "あなたは検索クエリの専門家です。\n"
        "ユーザーの質問に回答するために必要な情報を検索するため、"
        "質問を検索に適したサブクエリに分解してください。\n\n"
        "ルール:\n"
        "- 各サブクエリは1つの情報ニーズに対応すること\n"
        "- 検索エンジンに入力するような短いフレーズにすること\n"
        "- 単純な質問は1つのサブクエリで十分\n"
        "- 複雑な質問は2〜4つのサブクエリに分解すること\n\n"
        f"ユーザーの質問: {question}\n\n"
        "JSON形式で回答してください: {\"sub_queries\": [\"クエリ1\", \"クエリ2\", ...]}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "sub_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "検索用サブクエリのリスト",
                    }
                },
                "required": ["sub_queries"],
            },
        ),
    )

    result = json.loads(response.text)
    return result["sub_queries"]


# ---------------------------------------------------------------------------
# Step 2: 複数サブクエリでの検索と重複排除
# ---------------------------------------------------------------------------
def multi_retrieve(sub_queries: list[str], top_k: int = 3) -> list[dict]:
    """複数サブクエリで検索し、重複チャンクを排除して統合する。"""

    seen_doc_ids = set()
    all_results = []

    for query in sub_queries:
        results = retrieve(query, top_k=top_k)
        for r in results:
            if r["doc_id"] not in seen_doc_ids:
                seen_doc_ids.add(r["doc_id"])
                r["source_query"] = query
                all_results.append(r)

    return all_results


# ---------------------------------------------------------------------------
# Step 3: カバレッジ評価（Agentic 判断 #2）
#   取得済みチャンクがユーザーの質問に十分に答えられるか評価する。
#   不足がある場合は、追加検索のためのクエリを提案する。
# ---------------------------------------------------------------------------
def assess_coverage(
    question: str,
    retrieved_chunks: list[dict],
    attempt: int,
    max_attempts: int,
) -> dict:
    """LLM が取得済み情報の網羅性を評価し、不足があれば追加クエリを提案する。"""

    context = "\n\n".join(
        f"[Chunk {i+1} / doc_id={c['doc_id']}]\n{c['content']}"
        for i, c in enumerate(retrieved_chunks)
    )

    prompt = (
        "あなたは情報の網羅性を評価する専門家です。\n\n"
        f"ユーザーの質問: {question}\n\n"
        f"現在取得済みの情報:\n{context}\n\n"
        f"検索試行: {attempt}/{max_attempts}\n\n"
        "【評価基準】\n"
        "- 質問に回答するために必要な情報が揃っているか\n"
        "- 複数の観点（年度比較、製品と戦略の対応など）に対応できるか\n"
        "- 試行回数が多い場合は現状の情報で回答可能と判断してよい\n\n"
        "JSON形式で回答してください:\n"
        "{\n"
        '  "verdict": "sufficient" or "insufficient",\n'
        '  "reasoning": "判定理由",\n'
        '  "missing_aspects": ["不足している観点1", ...],\n'
        '  "additional_queries": ["追加検索クエリ1", ...]\n'
        "}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["sufficient", "insufficient"],
                    },
                    "reasoning": {"type": "string"},
                    "missing_aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "additional_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["verdict", "reasoning", "missing_aspects", "additional_queries"],
            },
        ),
    )

    return json.loads(response.text)


# ---------------------------------------------------------------------------
# Step 4: 回答生成
# ---------------------------------------------------------------------------
def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """全取得チャンクを使って包括的な回答を生成する。"""

    context = "\n\n".join(
        f"[情報{i+1}] {c['content']}" for i, c in enumerate(retrieved_chunks)
    )

    prompt = (
        "以下の参考情報のみに基づいて、質問に正確かつ包括的に回答してください。\n"
        "参考情報に含まれない内容は「情報がありません」と答えてください。\n"
        "数値の比較や計算が必要な場合は、具体的な数字を示して計算してください。\n\n"
        f"参考情報:\n{context}\n\n"
        f"質問: {question}\n\n"
        "答え:"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text


# ===========================================================================
# 5. Agentic RAG メインループ
# ===========================================================================
MAX_ATTEMPTS = 3


def agentic_rag(question: str) -> str:
    """Agentic RAG のメインループ。

    1. クエリ分解
    2. 検索
    3. カバレッジ評価 → 不足なら追加検索（最大 MAX_ATTEMPTS 回）
    4. 回答生成
    """

    print(f"\n  [Step 1: Query Decomposition]")
    sub_queries = decompose_query(question)
    for i, sq in enumerate(sub_queries):
        print(f"    Sub-query {i+1}: {sq}")

    # 初回検索
    print(f"\n  [Step 2: Initial Retrieval]")
    all_chunks = multi_retrieve(sub_queries, top_k=3)
    for c in all_chunks:
        print(f"    doc_id={c['doc_id']} (score={c['score']:.4f}) {c['content'][:50]}...")

    # 評価
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n  [Step 3: Coverage Assessment (attempt {attempt}/{MAX_ATTEMPTS})]")
        assessment = assess_coverage(question, all_chunks, attempt, MAX_ATTEMPTS)
        print(f"    Verdict: {assessment['verdict']}")
        print(f"    Reasoning: {assessment['reasoning']}")

        if assessment["verdict"] == "sufficient":
            print(f"    → カバレッジ十分。回答生成へ。")
            break

        if attempt >= MAX_ATTEMPTS:
            print(f"    → 最大試行回数到達。現在の情報で回答生成へ。")
            break

        # 追加検索
        additional_queries = assessment.get("additional_queries", [])
        if additional_queries:
            print(f"\n  [Step 2': Additional Retrieval (attempt {attempt})]")
            for q in additional_queries:
                print(f"    Additional query: {q}")
            new_chunks = multi_retrieve(
                additional_queries, top_k=3
            )
            # 既に取得済みのチャンクは追加しない
            existing_ids = {c["doc_id"] for c in all_chunks}
            added = 0
            for c in new_chunks:
                if c["doc_id"] not in existing_ids:
                    all_chunks.append(c)
                    existing_ids.add(c["doc_id"])
                    added += 1
                    print(f"    + doc_id={c['doc_id']} (score={c['score']:.4f}) {c['content'][:50]}...")
            if added == 0:
                print(f"    (新規チャンクなし)")
                print(f"    → これ以上取得できる情報がないため回答生成へ。")
                break
        else:
            print(f"    (追加クエリなし)")
            break

    # 回答生成
    print(f"\n  [Step 4: Answer Generation]")
    print(f"    Total chunks used: {len(all_chunks)}")
    answer = generate_answer(question, all_chunks)
    return answer


# ---------------------------------------------------------------------------
# 6. Execution
# ---------------------------------------------------------------------------
questions = [
    # 単純な検索（1チャンクで回答可能）
    "エグザンプルデータパブリッシャーのCTOは誰で、どこから転職してきましたか？",

    # 比較・計算（複数チャンクの横断が必要）
    "エグザンプルデータパブリッシャーの売上高は2023年度から2024年度でどれだけ成長しましたか？営業利益率の変化も教えてください。",

    # マルチホップ推論（製品 → パートナー → 戦略の連鎖）
    "ZetaCore v3の欧州展開はどのような体制で行われますか？また、この製品の技術的な強みは何ですか？",

    # 複数領域の統合（人事 + 財務 + 組織の横断）
    "エグザンプルデータパブリッシャーの従業員数の推移と、一人当たり売上高を2023年度・2024年度で比較してください。",

    # 深い推論（中期計画 + 財務 + 製品ロードマップの統合）
    "エグザンプルデータパブリッシャーの中期経営計画における投資配分と、各分野の現在の製品・成果を対応づけて説明してください。",
]

for q in questions:
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    answer = agentic_rag(q)
    print(f"\n  [Final Answer]")
    print(f"  {answer}")
