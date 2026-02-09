import csv
import os

DATA_FILES = [
    "data_01.txt",
    "data_02.txt",
    "data_03.txt",
    "data_04.txt",
    "data_05.txt",
]

OUTPUT_CSV = "1411332013_RAG_HW_01.csv"


# ---------- 讀題目（安全處理 BOM） ----------

def load_questions(path="questions.csv"):
    questions = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]

        qid_idx = header.index("q_id")
        question_idx = header.index("questions")

        for row in reader:
            questions.append({
                "q_id": int(row[qid_idx]),
                "question": row[question_idx]
            })

    return questions


# ---------- 切塊方法 ----------

def fixed_chunking(chunk_size=200):
    chunks = []
    for fn in DATA_FILES:
        with open(fn, encoding="utf-8") as f:
            text = f.read()
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append({"text": chunk, "source": fn})
    return chunks


def sliding_chunking(chunk_size=200, overlap=50):
    chunks = []
    step = chunk_size - overlap
    for fn in DATA_FILES:
        with open(fn, encoding="utf-8") as f:
            text = f.read()
        for i in range(0, len(text), step):
            chunk = text[i:i+chunk_size].strip()
            if len(chunk) > 20:
                chunks.append({"text": chunk, "source": fn})
    return chunks


def semantic_chunking():
    chunks = []
    for fn in DATA_FILES:
        with open(fn, encoding="utf-8") as f:
            for para in f.read().split("\n\n"):
                para = para.strip()
                if len(para) > 50:
                    chunks.append({"text": para, "source": fn})
    return chunks


# ---------- 每一題都重新找 chunk ----------

def retrieve_best_chunk(question, chunks):
    best = None
    best_score = -1

    for ch in chunks:
        score = sum(1 for c in question if c in ch["text"])
        if score > best_score:
            best = ch
            best_score = score

    return best["text"], best_score, best["source"]


# ---------- 主程式 ----------

def main():
    questions = load_questions()

    methods = [
        ("fixed", fixed_chunking()),
        ("sliding_window", sliding_chunking()),
        ("semantic", semantic_chunking()),
    ]

    rows = []
    uid = 1

    for method_name, chunks in methods:
        for q in questions:
            retrieve_text, score, source = retrieve_best_chunk(
                q["question"], chunks
            )

            rows.append({
                "id": uid,
                "q_id": q["q_id"],
                "method": method_name,
                "retrieve_text": retrieve_text,
                "score": score,
                "source": source
            })

            uid += 1

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "q_id", "method", "retrieve_text", "score", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ 已完成，共 {len(rows)} 筆（20 × 3）")
    print(f"📄 輸出檔案：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()

