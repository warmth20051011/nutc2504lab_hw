import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from chunk_sentence_func import get_sentence_chunks
from test_02_embedding import get_embeddings

# =========================
# 基本設定
# =========================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "CW02_table_sentence"
DISTANCE = Distance.COSINE

TABLE_FILE = "table/table_txt.md"

# =========================
# 1. 讀取表格文字
# =========================
with open(TABLE_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# 2. 語意切塊（以列為單位）
# =========================
chunks = get_sentence_chunks(text)
print("表格切塊數量：", len(chunks))

# =========================
# 3. Embedding
# =========================
embeddings, VECTOR_SIZE = get_embeddings(chunks)
print("VECTOR_SIZE =", VECTOR_SIZE)

# =========================
# 4. 建立 Qdrant collection
# =========================
client = QdrantClient(url=QDRANT_URL)

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=DISTANCE
    )
)

# =========================
# 5. Upsert 到 VDB
# =========================
points = []

for i, vector in enumerate(embeddings):
    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "text": chunks[i],
                "source": "table_txt.md"
            }
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("✅ 表格資料已使用【語意切塊】嵌入到 VDB")
