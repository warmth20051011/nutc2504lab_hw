import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from chunk_fixed_func import get_fixed_chunks
from test_02_embedding import get_embeddings   

# =========================
# 基本設定
# =========================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "CW02_fixed"
DISTANCE = Distance.COSINE

# =========================
# 1. 讀取文字檔
# =========================
with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# 2. 固定切塊
# =========================
chunks = get_fixed_chunks(text)

print("切塊數量：", len(chunks))

# =========================
# 3. Embedding
# =========================
embeddings, VECTOR_SIZE = get_embeddings(chunks)

print("VECTOR_SIZE =", VECTOR_SIZE)

# =========================
# 4. 建立 Qdrant collection
# =========================
client = QdrantClient(url=QDRANT_URL)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=DISTANCE
    )
)

# =========================
# 5. 組成 points 並 upsert
# =========================
points = []

for i, vector in enumerate(embeddings):
    points.append(
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "text": chunks[i]
            }
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("✅ 已完成嵌入到 VDB：", COLLECTION_NAME)

