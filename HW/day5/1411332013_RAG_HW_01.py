import os
import re
import csv
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# =========================================================
# 基本設定
# =========================================================
OUTPUT_CSV = f"1411332013_RAG_HW_01.csv"

DATA_FILES = [f"data_{i:02d}.txt" for i in range(1, 6)]
QUESTIONS_FILE = "questions.csv"

EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SCORE_API_URL = "https://hw-01.wade0426.me/submit_answer"
QDRANT_URL = "http://localhost:6333"

FIXED_CHUNK_SIZE = 500
SLIDING_CHUNK_SIZE = 500
SLIDING_OVERLAP = 50
SEMANTIC_CHUNK_SIZE = 500
SEMANTIC_OVERLAP = 50


# =========================================================
# 工具函式
# =========================================================
def load_text(fp):
    with open(fp, encoding="utf-8") as f:
        return re.sub(r"\s+", " ", f.read()).strip()


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_embedding(texts):
    payload = {"texts": texts, "normalize": True, "batch_size": 32}
    r = requests.post(EMBED_API_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["embeddings"], data["dimension"]


def submit_answer(q_id, answer):
    payload = {
        "q_id": int(q_id),
        "student_answer": answer.strip()[:2000]
    }
    r = requests.post(SCORE_API_URL, json=payload, timeout=60)
    if r.status_code == 200:
        return float(r.json().get("score", 0.0))
    return 0.0


# =========================================================
# Chunking 方法
# =========================================================
def fixed_chunking():
    chunks = []
    for fn in DATA_FILES:
        text = load_text(fn)
        for i in range(0, len(text), FIXED_CHUNK_SIZE):
            c = text[i:i + FIXED_CHUNK_SIZE]
            if len(c) > 150:
                chunks.append({"text": c, "source": fn})
    return chunks


def sliding_chunking():
    chunks = []
    step = SLIDING_CHUNK_SIZE - SLIDING_OVERLAP
    for fn in DATA_FILES:
        text = load_text(fn)
        for i in range(0, len(text), step):
            c = text[i:i + SLIDING_CHUNK_SIZE]
            if len(c) > 150:
                chunks.append({"text": c, "source": fn})
    return chunks


def semantic_chunking():
    chunks = []
    seps = ["。", "！", "？", "；"]
    for fn in DATA_FILES:
        text = load_text(fn)
        sents = [text]
        for sep in seps:
            tmp = []
            for s in sents:
                tmp.extend([p + sep for p in s.split(sep) if p.strip()])
            sents = tmp

        buf = ""
        for s in sents:
            if len(buf) + len(s) <= SEMANTIC_CHUNK_SIZE:
                buf += s
            else:
                if len(buf) > 150:
                    chunks.append({"text": buf, "source": fn})
                buf = buf[-SEMANTIC_OVERLAP:] + s

        if len(buf) > 150:
            chunks.append({"text": buf, "source": fn})

    return chunks


# =========================================================
# Qdrant 操作
# =========================================================
def build_collection(client, name, chunks, dim):
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    texts = [c["text"] for c in chunks]
    embs, _ = get_embedding(texts)

    points = []
    for i, (c, v) in enumerate(zip(chunks, embs)):
        points.append(PointStruct(
            id=i,
            vector=v,
            payload={"text": c["text"], "source": c["source"]}
        ))

    client.upsert(collection_name=name, points=points)


def retrieve_top1(client, collection, question):
    q_emb, _ = get_embedding([question])
    res = client.query_points(
        collection_name=collection,
        query=q_emb[0],
        limit=1
    )
    if res.points:
        p = res.points[0]
        return p.payload["text"], p.payload["source"]
    return "", ""


# =========================================================
# Main
# =========================================================
def main():
    questions = load_questions()

    methods = {
        "固定大小": fixed_chunking(),
        "滑動視窗": sliding_chunking(),
        "語意切塊": semantic_chunking()
    }

    client = QdrantClient(url=QDRANT_URL)
    _, dim = get_embedding(["test"])

    collections = {
        "固定大小": "fixed_chunks",
        "滑動視窗": "sliding_chunks",
        "語意切塊": "semantic_chunks"
    }

    for m in methods:
        build_collection(client, collections[m], methods[m], dim)
        print(f"  建立 collection：{m}")

    rows = []
    uid = 1

    for q in questions:
        print(f"\nQ{q['q_id']}：{q['questions'][:40]}...")
        for m in methods:
            text, src = retrieve_top1(client, collections[m], q["questions"])
            score = submit_answer(q["q_id"], text)

            rows.append({
                "id": uid,
                "q_id": q["q_id"],
                "method": m,
                "retrieve_text": text,
                "score": round(score, 6),
                "source": src
            })

            print(f"  {m} | score={score:.4f} | {src}")
            uid += 1
            time.sleep(0.3)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "q_id", "method", "retrieve_text", "score", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n完成！CSV 輸出：{OUTPUT_CSV}")
    
        # =====================================================
    # 平均分數統計
    # =====================================================
    print("\n 各方法平均分數")
    print("-" * 40)

    avg_scores = {}
    best_method = ""
    best_avg = 0.0

    for r in rows:
        avg_scores.setdefault(r["method"], []).append(float(r["score"]))

    for method, scores in avg_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{method}：平均 {avg:.6f}")

        if avg > best_avg:
            best_avg = avg
            best_method = method

    print(f"\n 最佳方法：{best_method}（平均 {best_avg:.6f}）")



if __name__ == "__main__":
    main()

