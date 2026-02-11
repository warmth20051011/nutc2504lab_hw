import os
from pathlib import Path
import requests

import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from docx import Document
from PIL import Image

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# =========================
# 基本設定
# =========================

BASE_DIR = Path(__file__).parent

EMBED_URL = "https://ws-04.wade0426.me/embed"

LLM_BASE_URL = "https://ws-06.huannago.com/v1"
LLM_MODEL = "gemma-3-27b-it"

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="NoNeed"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5


# =========================
# IDP 文件讀取
# =========================

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
        texts.append(
            pytesseract.image_to_string(img, lang="chi_tra+eng")
        )
    return "\n".join(texts)


def extract_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang="chi_tra+eng")


def extract_docx(path):
    doc = Document(path)
    return "\n".join(
        [p.text for p in doc.paragraphs if p.text.strip()]
    )


def load_documents():
    docs = {}

    for f in ["1.pdf", "2.pdf", "3.pdf", "4.png", "5.docx"]:

        path = BASE_DIR / f
        if not path.exists():
            continue

        print(f"📄 讀取 {f}")

        if f.endswith(".pdf"):
            text = extract_pdf_text(path)

            # 如果 PDF 幾乎沒抓到字 → 改 OCR
            if len(text.strip()) < 100:
                print("   → 文字太少，改用 OCR")
                text = extract_pdf_ocr(path)

        elif f.endswith(".png"):
            text = extract_image(path)

        elif f.endswith(".docx"):
            text = extract_docx(path)

        else:
            text = ""

        print(f"   → OK ({len(text)} chars)")
        docs[f] = text

    return docs


# =========================
# 切塊
# =========================

def split_text(text, source):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_text(text)

    return [
        {"text": c, "source": source}
        for c in chunks
    ]


# =========================
# 向量嵌入
# =========================

def embed(texts):

    r = requests.post(
        EMBED_URL,
        json={
            "texts": texts,
            "task_description": "qa",
            "normalize": True
        },
        timeout=60
    )

    return r.json()["embeddings"]


# =========================
# 建立向量庫
# =========================

def build_index(chunks):

    print("📦 建立向量庫...")

    client_q = QdrantClient(":memory:")

    test_vec = embed(["test"])[0]
    dim = len(test_vec)

    client_q.create_collection(
        collection_name="docs",
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE
        )
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

    print("✅ 向量庫建立完成")

    return client_q


# =========================
# 搜尋
# =========================

def search(client_q, query):

    q_vec = embed([query])[0]

    res = client_q.query_points(
        "docs",
        query=q_vec,
        limit=TOP_K
    )

    return res.points


# =========================
# 生成回答
# =========================

def generate_answer(query, contexts):

    ctx = "\n".join(
        [c.payload["text"] for c in contexts]
    )

    messages = [
        {
            "role": "system",
            "content": "你是一個專業文件問答助手，僅能根據資料回答。"
        },
        {
            "role": "user",
            "content": f"資料:\n{ctx}\n\n問題:{query}"
        }
    ]

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0
    )

    return resp.choices[0].message.content


# =========================
# 主程式
# =========================

def main():

    print("📂 載入文件中...\n")

    docs = load_documents()

    print("\n✂️ 切塊中...")

    all_chunks = []

    for name, text in docs.items():
        all_chunks += split_text(text, name)

    print(f"   → 共 {len(all_chunks)} 個 chunks\n")

    qdrant_client = build_index(all_chunks)

    print("\n🚀 文件問答系統已啟動\n")

    while True:

        query = input("請輸入問題（輸入 exit 離開）：")

        if query.lower() == "exit":
            break

        if not query.strip():
            continue

        contexts = search(qdrant_client, query)

        answer = generate_answer(query, contexts)

        sources = list(
            set([c.payload["source"] for c in contexts])
        )

        print("\n==============================")
        print("回答：\n")
        print(answer)
        print("\n📚 來源文件：", ", ".join(sources))
        print("==============================\n")


if __name__ == "__main__":
    main()
