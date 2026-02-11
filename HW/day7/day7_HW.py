import os
import re
import csv
import requests
from pathlib import Path

import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from docx import Document

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM


# ─────────────────────────────
# 基本設定
# ─────────────────────────────

BASE_DIR = Path(__file__).parent
EMBED_URL = "https://ws-04.wade0426.me/embed"

LLM_BASE_URL = "https://ws-06.huannago.com/v1"
LLM_MODEL = "gemma-3-27b-it"
LLM_API_KEY = "NoNeed"

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_SAMPLES = 5   # ⭐ 限制前 5 筆


# ═══════════════════════════════
# IDP 文件提取
# ═══════════════════════════════

def extract_pdf_text(path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n".join(texts)

def extract_pdf_ocr(path):
    images = convert_from_path(path, dpi=200)
    texts = []
    for img in images:
        texts.append(pytesseract.image_to_string(img, lang="chi_tra+eng"))
    return "\n".join(texts)

def extract_image(path):
    from PIL import Image
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang="chi_tra+eng")

def extract_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def load_documents():
    docs = {}
    for f in ["1.pdf", "2.pdf", "3.pdf", "4.png", "5.docx"]:
        path = BASE_DIR / f
        if not path.exists():
            continue

        if f.endswith(".pdf"):
            text = extract_pdf_text(path)
            if len(text.strip()) < 100:
                text = extract_pdf_ocr(path)
        elif f.endswith(".png"):
            text = extract_image(path)
        elif f.endswith(".docx"):
            text = extract_docx(path)

        docs[f] = text
    return docs


# ═══════════════════════════════
# Injection 偵測
# ═══════════════════════════════

PATTERNS = [
    r"ignore (previous|above) instructions",
    r"請忽略.*指示",
    r"你是一個.*LLM",
    r"system prompt",
    r"jailbreak",
    r"do not follow.*rules",
    r"不要遵守.*規則",
    r"act as .*",
    r"developer mode",
]

def detect_injection(text):
    for p in PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


# ═══════════════════════════════
# RAG
# ═══════════════════════════════

def split_text(text, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    return [{"text": c, "source": source} for c in chunks]

def embed(texts):
    r = requests.post(
        EMBED_URL,
        json={"texts": texts, "task_description": "qa", "normalize": True},
        timeout=150
    )
    return r.json()["embeddings"]

def build_index(chunks):
    client_q = QdrantClient(":memory:")
    test_vec = embed(["test"])[0]
    dim = len(test_vec)

    client_q.create_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    vectors = embed([c["text"] for c in chunks])

    points = [
        PointStruct(
            id=i,
            vector=v,
            payload={
                "text": chunks[i]["text"],
                "source": chunks[i]["source"]
            }
        )
        for i, v in enumerate(vectors)
    ]

    client_q.upsert("docs", points)
    return client_q

def search(client_q, query):
    q_vec = embed([query])[0]
    res = client_q.query_points("docs", query=q_vec, limit=TOP_K)
    return res.points

def generate_answer(query, contexts):
    ctx = "\n".join([c.payload["text"] for c in contexts])
    msg = [
        {"role": "system", "content": "根據資料回答問題，不可編造。"},
        {"role": "user", "content": f"資料:\n{ctx}\n\n問題:{query}"}
    ]
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msg,
        temperature=0
    )
    return resp.choices[0].message.content


# ═══════════════════════════════
# DeepEval
# ═══════════════════════════════

class CustomLLM(DeepEvalBaseLLM):

    def load_model(self):
        return None

    def generate(self, prompt: str):
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str):
        return self.generate(prompt)

    def get_model_name(self):
        return LLM_MODEL


# ═══════════════════════════════
# 主程式
# ═══════════════════════════════

def main():

    print("📄 載入文件...")
    docs = load_documents()

    print("🔍 檢測 Injection...")
    clean_docs = {}
    for name, text in docs.items():
        if detect_injection(text):
            print(f"❌ 發現惡意提示詞: {name} → 剃除")
        else:
            clean_docs[name] = text

    print("✂️ 切塊...")
    all_chunks = []
    for name, text in clean_docs.items():
        all_chunks += split_text(text, name)

    print("📦 建立向量庫...")
    qdrant_client = build_index(all_chunks)

    qa_data = []
    with open(BASE_DIR / "questions_answer.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= MAX_SAMPLES:
                break
            qa_data.append(row)

    custom_llm = CustomLLM()

    metrics = [
        FaithfulnessMetric(model=custom_llm),
        AnswerRelevancyMetric(model=custom_llm),
        ContextualRecallMetric(model=custom_llm),
        ContextualPrecisionMetric(model=custom_llm),
    ]

    print("🚀 執行 RAG + DeepEval...")

    for row in qa_data:

        query = row["questions"]
        contexts = search(qdrant_client, query)
        answer = generate_answer(query, contexts)

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=row["answer"],
            retrieval_context=[c.payload["text"] for c in contexts]
        )

        for m in metrics:
            m.measure(test_case)

        print("\n" + "=" * 60)
        print(f"Q: {query}")
        print(f"A: {answer}")
        print("Scores:")
        print("Faithfulness:", metrics[0].score)
        print("AnswerRelevancy:", metrics[1].score)
        print("ContextualRecall:", metrics[2].score)
        print("ContextualPrecision:", metrics[3].score)

    print("\n📄 產生 test_dataset.csv...")

    rows = []

    with open(BASE_DIR / "questions_answer.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):

            if i >= MAX_SAMPLES:
                break

            query = row["questions"]
            contexts = search(qdrant_client, query)
            answer = generate_answer(query, contexts)

            sources = list(set([c.payload["source"] for c in contexts]))

            rows.append({
                "q_id": row["id"],
                "questions": query,
                "answer": answer,
                "source": ",".join(sources)
            })

    with open(BASE_DIR / "test_dataset.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["q_id", "questions", "answer", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("✅ test_dataset.csv 產生完成")
    print("🎉 作業完成")


if __name__ == "__main__":
    main()

