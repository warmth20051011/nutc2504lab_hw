import csv
import re
import numpy as np
from typing import List

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ============================================================
# 基本設定
# ============================================================

QA_PATH = "qa_data.txt"
QUESTION_PATH = "questions.csv"
OUTPUT_CSV = "day6_HW_questions.csv"

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.1
)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ============================================================
# 讀取 QA 資料
# ============================================================

def is_question(line: str) -> bool:
    return (
        len(line) < 80
        and ("？" in line or "?" in line)
        and not line.startswith("來源")
        and "發布日期" not in line
    )

def load_qa(path: str):
    qa_pairs = []

    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    current_q = None
    current_a = []

    for line in lines:
        if is_question(line):
            if current_q and current_a:
                qa_pairs.append({
                    "q": current_q,
                    "a": "\n".join(current_a)
                })
            current_q = line
            current_a = []
            continue

        if line.startswith("來源") or "發布日期" in line:
            continue

        if current_q:
            current_a.append(line)

    if current_q and current_a:
        qa_pairs.append({
            "q": current_q,
            "a": "\n".join(current_a)
        })

    return qa_pairs

# 真正載入 QA
qa_data = load_qa(QA_PATH)
print(f"📄 載入 QA 筆數：{len(qa_data)}")

# ============================================================
# Hybrid Search 建立
# ============================================================

corpus = [item["q"] + " " + item["a"] for item in qa_data]

# BM25 保命檢查（一定要有）
if not corpus:
    raise ValueError("❌ QA corpus 為空，請確認 qa_data.txt 是否有內容")

# BM25
tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

# Dense
corpus_embeddings = embed_model.encode(
    corpus, normalize_embeddings=True
)

# ============================================================
# Query Rewrite
# ============================================================

def rewrite_query(query: str, history: List[str]) -> str:
    prompt = f"""
你是 AI 客服助理，請根據對話歷史，將使用者問題改寫成清楚、完整的查詢句。

對話歷史：
{history}

使用者問題：
{query}

請只輸出改寫後的問題。
"""
    return llm.invoke(prompt).content.strip()

# ============================================================
# Hybrid Search
# ============================================================

def hybrid_search(query: str, top_k=5):
    bm25_scores = bm25.get_scores(query.split())

    q_emb = embed_model.encode([query], normalize_embeddings=True)
    dense_scores = cosine_similarity(
        q_emb, corpus_embeddings
    )[0]

    scores = 0.5 * bm25_scores + 0.5 * dense_scores
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [(i, corpus[i]) for i in top_idx]

# ============================================================
# Rerank（LLM）
# ============================================================

def rerank(query: str, docs: List[str]) -> str:
    context = "\n\n".join(docs)

    prompt = f"""
你是水務公司的 AI 客服助理，請根據下方提供的資料回答問題。

規則：
1. **只能根據提供的資料回答**
2. 如果資料「有相關但不完全一模一樣」，請用合理推論回答
3. 如果真的完全無關，才回答「資料中未提及」

使用者問題：
{query}

參考資料：
{context}

請直接給出完整、自然的客服回答，不要提到「資料中沒有」這種描述。
"""

    return llm.invoke(prompt).content.strip()


# ============================================================
# AI 客服（多輪）
# ============================================================

def chat():
    history = []

    while True:
        user_q = input("\n使用者：").strip()
        if user_q.lower() in ["q", "quit"]:
            break

        rewritten = rewrite_query(user_q, history)
        hits = hybrid_search(rewritten)
        docs = [d for _, d in hits]
        answer = rerank(rewritten, docs)

        history.append(f"使用者：{user_q}")
        history.append(f"助理：{answer}")

        print("\n🤖 助理：", answer)

# ============================================================
# 批次回答（產 CSV 給 DeepEval）
# ============================================================

def batch_answer():
    rows = []

    with open(QUESTION_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewritten = rewrite_query(row["questions"], [])
            hits = hybrid_search(rewritten)
            docs = [d for _, d in hits]
            answer = rerank(rewritten, docs)

            rows.append({
                "q_id": row["q_id"],
                "questions": row["questions"],
                "answer": answer,
                "Faithfulness": "",
                "Answer_Relevancy": "",
                "Contextual_Recall": "",
                "Contextual_Precision": "",
                "Contextual_Relevancy": ""
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("✅ day6_HW_questions.csv 已產生")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    mode = input("1: 互動客服  2: 產 CSV > ")
    if mode == "1":
        chat()
    else:
        batch_answer()

