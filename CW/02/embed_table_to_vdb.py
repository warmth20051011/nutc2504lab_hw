from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from table_semantic_func import get_table_semantic_chunks
from test_02_embedding import get_embeddings

# =========================
# 基本設定
# =========================
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "CW02_table_semantic"
DISTANCE = Distance.COSINE

TABLE_FILE = "table/table_txt.md"

# =========================
#  語意切塊（表格）
# =========================
chunks = get_table_semantic_chunks(TABLE_FILE)
print("表格語意切塊數量：", len(chunks))

# =========================
#  Embedding
# =========================
embeddings, VECTOR_SIZE = get_embeddings(chunks)
print("VECTOR_SIZE =", VECTOR_SIZE)

# =========================
#  建立 Qdrant collection
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
#  Upsert
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

print("✅ 表格資料已用【語意切塊】嵌入 VDB：", COLLECTION_NAME)

